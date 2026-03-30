"""Ranking Agent — LLM-driven scoring and ranking of a StructuredTable."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from crawler.llm import replicate

from crawler.cost_tracker import tracker
from crawler.agents.structuring_agent import StructuredTable, StructuredRow


@dataclass
class CriterionWeight:
    column: str
    weight: float
    higher_is_better: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "weight": round(self.weight, 4), "higher_is_better": self.higher_is_better, "rationale": self.rationale}


@dataclass
class RankedRow:
    rank: int
    entity_name: str
    source_url: str
    composite_score: float
    criterion_scores: dict[str, float] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "entity_name": self.entity_name, "source_url": self.source_url, "composite_score": round(self.composite_score, 4), "criterion_scores": {k: round(v, 4) for k, v in self.criterion_scores.items()}, "fields": self.fields, "missing_keys": self.missing_keys}


@dataclass
class RankedTable:
    session_id: str
    user_query: str
    criteria: list[CriterionWeight] = field(default_factory=list)
    rows: list[RankedRow] = field(default_factory=list)
    ranking_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "user_query": self.user_query, "ranking_rationale": self.ranking_rationale, "criteria": [c.to_dict() for c in self.criteria], "rows": [r.to_dict() for r in self.rows]}


_CRITERIA_PROMPT = """\
You are a ranking expert. A user asked a ranking question and you have a structured data table.

User's ranking question: {query}
Available columns: {columns}
Sample data rows (first 3): {sample_rows}

Select the most relevant columns for ranking. Assign weights (summing to 1.0). Decide if higher is better.

Return ONLY a JSON object:
{{
  "ranking_rationale": "One sentence explaining the ranking logic.",
  "criteria": [
    {{"column": "Exact Column Name", "weight": 0.40, "higher_is_better": true, "rationale": "Why this matters."}}
  ]
}}"""


def _extract_number(value: Any) -> float | None:
    if value is None: return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a","null","none","unknown","not available","-","–",""}: return None
    cleaned = re.sub(r"[,$%\s]", "", text)
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "cr": 10_000_000}
    lower_clean = cleaned.lower()
    for suffix, mult in multipliers.items():
        if lower_clean.endswith(suffix):
            try: return float(lower_clean[:-len(suffix)]) * mult
            except ValueError: pass
    try: return float(cleaned)
    except ValueError: pass
    m = re.search(r"[-+]?\d+\.?\d*", text)
    if m:
        try: return float(m.group())
        except ValueError: pass
    return None


def _normalise_column(values: list[float | None], *, higher_is_better: bool) -> list[float]:
    numeric = [v for v in values if v is not None]
    if not numeric: return [0.0] * len(values)
    mn, mx = min(numeric), max(numeric)
    spread = mx - mn if mx != mn else 1.0
    result = []
    for v in values:
        if v is None: result.append(0.0)
        else:
            score = (v - mn) / spread
            result.append(1.0 - score if not higher_is_better else score)
    return result


class RankingAgent:

    def __init__(self, *, model: str = "meta-llama/llama-3-70b-instruct") -> None:
        self.model = model

    def _call_llm(self, prompt: str, *, node_label: str) -> str:
        t0 = time.time()
        output = replicate.run(self.model, input={"prompt": prompt, "max_tokens": 2048, "temperature": 0.2})
        raw = "".join(str(c) for c in output)
        tracker.record(node=node_label, model=self.model, input_tokens=len(prompt)//4, output_tokens=len(raw)//4, latency_s=time.time()-t0)
        return raw

    def _parse_llm_json(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(cleaned.strip())

    def _determine_criteria(self, table: StructuredTable) -> tuple[list[CriterionWeight], str]:
        sample_rows = [r.to_dict() for r in table.rows[:3]]
        prompt = _CRITERIA_PROMPT.format(query=table.user_query, columns=json.dumps(table.columns), sample_rows=json.dumps(sample_rows, ensure_ascii=True))
        raw = self._call_llm(prompt, node_label="ranking_agent_criteria")
        try:
            parsed = self._parse_llm_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[RankingAgent] Criteria parse failed: {exc}. Using equal weights.")
            return _fallback_criteria_parsed(table.columns)

        rationale = parsed.get("ranking_rationale", "")
        valid_columns = set(table.columns)
        criteria = []
        for item in parsed.get("criteria", []):
            col = item.get("column", "").strip()
            if col not in valid_columns: continue
            criteria.append(CriterionWeight(column=col, weight=float(item.get("weight", 0.0)), higher_is_better=bool(item.get("higher_is_better", True)), rationale=str(item.get("rationale", ""))))

        if not criteria:
            print("[RankingAgent] No valid criteria — falling back to equal weights.")
            return _fallback_criteria_parsed(table.columns)

        total_weight = sum(c.weight for c in criteria)
        if total_weight > 0:
            for c in criteria: c.weight /= total_weight
        return criteria, rationale

    def _compute_scores(self, rows, criteria) -> list[tuple[StructuredRow, float, dict[str, float]]]:
        if not criteria: return [(row, 0.0, {}) for row in rows]
        column_values: dict[str, list[float | None]] = {}
        for crit in criteria:
            column_values[crit.column] = [_extract_number(row.fields.get(crit.column)) for row in rows]
        normalised = {crit.column: _normalise_column(column_values[crit.column], higher_is_better=crit.higher_is_better) for crit in criteria}
        results = []
        for i, row in enumerate(rows):
            criterion_scores = {crit.column: normalised[crit.column][i] for crit in criteria}
            composite = sum(crit.weight * criterion_scores[crit.column] for crit in criteria)
            results.append((row, composite, criterion_scores))
        return results

    def _filter_relevant_rows(self, rows: list[StructuredRow], user_query: str) -> list[StructuredRow]:
        """Delegate geographic filtering to the shared utility."""
        from crawler.utils import geo_filter_entities

        # geo_filter_entities expects objects with .properties and .name;
        # StructuredRow uses .fields and .entity_name. Wrap temporarily.
        class _Adapter:
            def __init__(self, row: StructuredRow):
                self._row = row
                self.name = row.entity_name
                self.properties = row.fields

        adapted = [_Adapter(r) for r in rows]
        filtered = geo_filter_entities(adapted, user_query)
        return [a._row for a in filtered]

    def rank(self, table: StructuredTable) -> RankedTable:
        if not table.rows:
            return RankedTable(session_id=table.session_id, user_query=table.user_query, ranking_rationale="No entities to rank.")

        relevant_rows = self._filter_relevant_rows(table.rows, table.user_query)
        table = StructuredTable(session_id=table.session_id, user_query=table.user_query, columns=table.columns, rows=relevant_rows, missing_report=table.missing_report, round_number=table.round_number)

        print(f"[RankingAgent] Determining criteria for {len(table.rows)} entities across {len(table.columns)} columns...")
        criteria, rationale = self._determine_criteria(table)
        print(f"[RankingAgent] {len(criteria)} criteria: " + ", ".join(f"{c.column}(w={c.weight:.2f})" for c in criteria))

        scored = self._compute_scores(table.rows, criteria)
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked_rows = [RankedRow(rank=i+1, entity_name=row.entity_name, source_url=row.source_url, composite_score=composite, criterion_scores=crit_scores, fields=row.fields, missing_keys=row.missing_keys) for i, (row, composite, crit_scores) in enumerate(scored)]

        if ranked_rows: print(f"[RankingAgent] Top: {ranked_rows[0].entity_name!r} (score={ranked_rows[0].composite_score:.3f})")
        return RankedTable(session_id=table.session_id, user_query=table.user_query, criteria=criteria, rows=ranked_rows, ranking_rationale=rationale)


def _fallback_criteria_parsed(columns: list[str]) -> tuple[list[CriterionWeight], str]:
    n = len(columns) or 1
    return [CriterionWeight(column=col, weight=1.0/n, higher_is_better=True, rationale="Equal weight fallback.") for col in columns], "Equal weight assigned to all available columns."