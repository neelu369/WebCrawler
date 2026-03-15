"""Ranking Engine — deterministic + LLM-assisted ranking of Neo4j StructuredResults.

This is the authoritative ranking pipeline. It takes StructuredResult objects
produced by GraphStructurer (from Neo4j) and produces a fully ranked table.

Algorithm:
  1. Build a flat feature matrix from each entity's properties + relationships.
  2. Ask the LLM to select and weight the most relevant columns for the query.
  3. Parse numeric values from each cell (handles $1.2M, 45%, "3 years", etc.)
  4. Min-max normalise each column to [0, 1].
  5. Compute weighted composite score per entity.
  6. Sort descending → ranked table.
  7. If LLM criteria selection fails, fall back to equal-weight scoring.

Integration:
  Called directly from api.py after the LangGraph pipeline completes.
  Input:  list[StructuredResult]  (from state.structured_results)
  Output: RankingResult dataclass  (.to_dict() for JSON serialisation)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import replicate

from crawler.cost_tracker import tracker
from crawler.models import StructuredResult


# ── Data models ───────────────────────────────────────────────

@dataclass
class RankingCriterion:
    """One weighted column used in the composite score."""
    column: str
    weight: float           # normalised to sum to 1.0 across all criteria
    higher_is_better: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "weight": round(self.weight, 4),
            "higher_is_better": self.higher_is_better,
            "rationale": self.rationale,
        }


@dataclass
class RankedEntity:
    """One entity with its rank, scores, and all displayable fields."""
    rank: int
    name: str
    entity_type: str
    description: str
    composite_score: float
    criterion_scores: dict[str, float]  # per-column normalised score [0,1]
    properties: dict[str, str]          # raw property values from Neo4j
    relationships: list[dict[str, str]] # {type, target} from Neo4j
    source_urls: list[str]
    missing_criteria: list[str]         # criteria columns with no numeric value

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "composite_score": round(self.composite_score, 4),
            "criterion_scores": {k: round(v, 4) for k, v in self.criterion_scores.items()},
            "properties": self.properties,
            "relationships": self.relationships,
            "source_urls": self.source_urls,
            "missing_criteria": self.missing_criteria,
        }


@dataclass
class RankingResult:
    """Full output of the RankingEngine."""
    user_query: str
    session_id: str
    ranking_rationale: str
    criteria: list[RankingCriterion]
    entities: list[RankedEntity]
    all_columns: list[str]          # every property column across all entities
    total_entities: int
    algorithm: str = "weighted_minmax_composite"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "session_id": self.session_id,
            "ranking_rationale": self.ranking_rationale,
            "algorithm": self.algorithm,
            "criteria": [c.to_dict() for c in self.criteria],
            "entities": [e.to_dict() for e in self.entities],
            "all_columns": self.all_columns,
            "total_entities": self.total_entities,
        }


# ── LLM prompt ────────────────────────────────────────────────

_CRITERIA_PROMPT = """\
You are a ranking expert. A user asked a ranking question. You have structured entity data.

User's ranking question:
{query}

Available property columns (extracted from a knowledge graph):
{columns}

Sample entities (first 3) with their property values:
{sample_entities}

Task:
1. Select the columns that are MOST RELEVANT for answering this ranking question.
   Only pick columns that contain meaningful comparable data (skip names, URLs, descriptions).
2. Assign a weight (0.0–1.0) to each — weights MUST sum exactly to 1.0.
3. Set higher_is_better: true if a higher value means better (e.g. funding, rating, score).
   Set higher_is_better: false if lower means better (e.g. equity taken, cost, time to market).
4. Write one sentence explaining the overall ranking rationale.

Return ONLY valid JSON in this exact shape, no markdown, no explanation:
{{
  "ranking_rationale": "One sentence.",
  "criteria": [
    {{
      "column": "Exact Column Name",
      "weight": 0.35,
      "higher_is_better": true,
      "rationale": "Why this column matters for this query."
    }}
  ]
}}
"""


# ── Numeric extraction ────────────────────────────────────────

def _extract_number(value: Any) -> float | None:
    """
    Extract a float from heterogeneous cell values.
    Handles: "$1.2M", "45%", "₹500 Cr", "1,200", "~3 years", "Top 10" → 10
    Returns None if no numeric content found.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "null", "none", "unknown", "not available", "-", "–", ""}:
        return None

    # Strip currency symbols, commas, spaces, tildes
    cleaned = re.sub(r"[,$%₹€£¥~\s]", "", text)

    # Magnitude suffixes (Indian numbering too)
    multipliers = {
        "b":   1_000_000_000,
        "bn":  1_000_000_000,
        "m":   1_000_000,
        "mn":  1_000_000,
        "cr":  10_000_000,   # Indian crore
        "lac": 100_000,
        "lac": 100_000,
        "lakh":100_000,
        "k":   1_000,
    }
    lower_clean = cleaned.lower()
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if lower_clean.endswith(suffix):
            try:
                return float(lower_clean[: -len(suffix)]) * mult
            except ValueError:
                pass

    # Plain number
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Extract first number from mixed string e.g. "Top 10 startups" → 10
    m = re.search(r"[-+]?\d+\.?\d*", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass

    return None


# ── Normalisation ─────────────────────────────────────────────

def _minmax_normalise(
    values: list[float | None],
    *,
    higher_is_better: bool,
) -> list[float]:
    """
    Min-max normalise to [0, 1].
    None values → 0.0 (treated as worst possible score).
    If all values are the same, everyone scores 0.5.
    """
    numeric = [v for v in values if v is not None]
    if not numeric:
        return [0.0] * len(values)

    mn, mx = min(numeric), max(numeric)
    spread = mx - mn if mx != mn else None

    normalised: list[float] = []
    for v in values:
        if v is None:
            normalised.append(0.0)
        elif spread is None:
            # All values equal — neutral score
            normalised.append(0.5)
        else:
            score = (v - mn) / spread
            normalised.append(score if higher_is_better else 1.0 - score)
    return normalised


# ── Feature matrix builder ────────────────────────────────────

def _build_feature_matrix(
    entities: list[StructuredResult],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Collect all unique property keys across all entities.
    Also flatten relationships into property-style entries.
    Returns (all_columns, list_of_flat_dicts).
    """
    all_keys: set[str] = set()
    flat_rows: list[dict[str, str]] = []

    for entity in entities:
        row: dict[str, str] = {}

        # Primary: typed properties from Neo4j
        for k, v in entity.properties.items():
            all_keys.add(k)
            row[k] = str(v)

        # Secondary: flatten relationships as "Type: Target"
        # e.g. {"type": "Supports Industry", "target": "Fintech"} → "Supports Industry": "Fintech"
        for rel in entity.relationships:
            rel_type = rel.get("type", "")
            target = rel.get("target", "")
            if rel_type and target:
                all_keys.add(rel_type)
                # If multiple values for same type, concatenate
                if rel_type in row:
                    row[rel_type] = f"{row[rel_type]}, {target}"
                else:
                    row[rel_type] = target

        flat_rows.append(row)

    # Sort columns: shorter/simpler names first, then alphabetical
    all_columns = sorted(all_keys, key=lambda c: (len(c), c))
    return all_columns, flat_rows


# ── LLM criteria selector ─────────────────────────────────────

def _select_criteria_via_llm(
    user_query: str,
    all_columns: list[str],
    flat_rows: list[dict[str, str]],
    *,
    model: str,
) -> tuple[list[RankingCriterion], str]:
    """Ask the LLM to pick and weight the best columns for this query."""

    # Build a compact sample (first 3 entities, key fields only)
    sample = []
    for row in flat_rows[:3]:
        # Keep at most 10 fields per sample entity to avoid token bloat
        sample.append({k: v for k, v in list(row.items())[:10]})

    prompt = _CRITERIA_PROMPT.format(
        query=user_query,
        columns=json.dumps(all_columns, ensure_ascii=True),
        sample_entities=json.dumps(sample, ensure_ascii=True, indent=2),
    )

    t0 = time.time()
    try:
        output = replicate.run(
            model,
            input={"prompt": prompt, "max_tokens": 1024, "temperature": 0.15},
        )
        raw = "".join(str(c) for c in output)
        tracker.record(
            node="ranking_engine",
            model=model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(raw) // 4,
            latency_s=time.time() - t0,
        )
    except Exception as exc:
        print(f"[RankingEngine] LLM call failed: {exc}. Using equal weights.")
        return _equal_weight_criteria(all_columns), "Equal weight assigned to all available columns."

    # Parse the LLM JSON
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        # Find the JSON object
        idx = cleaned.find("{")
        if idx != -1:
            cleaned = cleaned[idx:]
        parsed = json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[RankingEngine] LLM JSON parse failed: {exc}. Using equal weights.")
        return _equal_weight_criteria(all_columns), "Equal weight assigned to all available columns."

    rationale = parsed.get("ranking_rationale", "")
    valid_cols = set(all_columns)
    criteria: list[RankingCriterion] = []

    for item in parsed.get("criteria", []):
        col = str(item.get("column", "")).strip()
        if col not in valid_cols:
            continue
        try:
            weight = float(item.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        if weight <= 0:
            continue
        criteria.append(
            RankingCriterion(
                column=col,
                weight=weight,
                higher_is_better=bool(item.get("higher_is_better", True)),
                rationale=str(item.get("rationale", "")),
            )
        )

    if not criteria:
        print("[RankingEngine] No valid criteria from LLM. Falling back to equal weights.")
        return _equal_weight_criteria(all_columns), rationale or "Equal weight fallback."

    # Normalise weights to exactly 1.0
    total = sum(c.weight for c in criteria)
    if total > 0:
        for c in criteria:
            c.weight = round(c.weight / total, 6)

    # Fix floating point drift: adjust last weight so sum == 1.0 exactly
    diff = 1.0 - sum(c.weight for c in criteria)
    if criteria:
        criteria[-1].weight = round(criteria[-1].weight + diff, 6)

    print(
        f"[RankingEngine] LLM selected {len(criteria)} criteria: "
        + ", ".join(f"{c.column}({c.weight:.2f})" for c in criteria)
    )
    return criteria, rationale


def _equal_weight_criteria(columns: list[str]) -> list[RankingCriterion]:
    """Fallback: every column gets equal weight."""
    # Only keep columns that look potentially numeric (skip pure-text ones)
    candidates = [c for c in columns if c not in {"Description", "Name", "Entity Type", "Source"}]
    if not candidates:
        candidates = columns[:5] if columns else ["Score"]
    n = len(candidates)
    weight = round(1.0 / n, 6)
    criteria = [RankingCriterion(column=col, weight=weight, higher_is_better=True, rationale="Equal weight fallback.") for col in candidates]
    # Fix sum drift
    diff = 1.0 - sum(c.weight for c in criteria)
    if criteria:
        criteria[-1].weight = round(criteria[-1].weight + diff, 6)
    return criteria


# ── Core scoring ──────────────────────────────────────────────

def _score_entities(
    entities: list[StructuredResult],
    flat_rows: list[dict[str, str]],
    criteria: list[RankingCriterion],
) -> list[tuple[StructuredResult, float, dict[str, float], list[str]]]:
    """
    For each entity compute:
      - per-criterion normalised score [0,1]
      - weighted composite score
      - list of missing criteria (columns with no parseable number)

    Returns list of (entity, composite_score, criterion_scores, missing_criteria).
    """
    if not criteria:
        return [(e, 0.0, {}, []) for e in entities]

    # Step 1: extract raw numeric values per criterion column
    raw_values: dict[str, list[float | None]] = {}
    for crit in criteria:
        raw_values[crit.column] = [
            _extract_number(row.get(crit.column))
            for row in flat_rows
        ]

    # Step 2: normalise each column
    normalised: dict[str, list[float]] = {
        crit.column: _minmax_normalise(
            raw_values[crit.column],
            higher_is_better=crit.higher_is_better,
        )
        for crit in criteria
    }

    # Step 3: compute composite scores
    results = []
    for i, entity in enumerate(entities):
        criterion_scores: dict[str, float] = {}
        missing: list[str] = []
        composite = 0.0

        for crit in criteria:
            norm_score = normalised[crit.column][i]
            criterion_scores[crit.column] = norm_score
            composite += crit.weight * norm_score

            if raw_values[crit.column][i] is None:
                missing.append(crit.column)

        results.append((entity, composite, criterion_scores, missing))

    return results


# ── Geographic filter ─────────────────────────────────────────

def _apply_geographic_filter(
    entities: list[StructuredResult],
    user_query: str,
) -> list[StructuredResult]:
    """
    If the query has a clear geographic focus, remove entities whose
    location properties explicitly contradict it.
    """
    query_lower = user_query.lower()

    india_keywords = ["india", "indian", "delhi", "mumbai", "bangalore",
                      "bengaluru", "hyderabad", "chennai", "pune", "kolkata"]
    india_query = any(kw in query_lower for kw in india_keywords)

    if not india_query:
        return entities

    _NON_INDIA = {"united states", "usa", "u.s.", "silicon valley",
                  "san francisco", "new york", "london", "uk", "singapore",
                  "europe", "australia", "canada"}
    _GLOBAL_ONLY_NAMES = {
        "openvc", "hf0", "hf0 residency", "soma capital fellowship",
        "y combinator", "techstars", "500 startups", "sequoia arc",
    }

    filtered, excluded = [], []
    for entity in entities:
        name_lower = entity.name.strip().lower()
        location = str(entity.properties.get("Located In")
                       or entity.properties.get("Headquartered In")
                       or entity.properties.get("Location", "")).lower()

        location_mismatch = (
            any(c in location for c in _NON_INDIA)
            and "india" not in location
            and bool(location)
        )
        name_mismatch = name_lower in _GLOBAL_ONLY_NAMES

        if location_mismatch or name_mismatch:
            excluded.append(entity.name)
        else:
            filtered.append(entity)

    if excluded:
        print(f"[RankingEngine] Filtered {len(excluded)} off-target entities: {excluded[:5]}")

    return filtered if filtered else entities


# ── Public API ────────────────────────────────────────────────

class RankingEngine:
    """
    Takes Neo4j StructuredResult objects → RankingResult.

    Full algorithm:
      1. Build flat feature matrix from entity properties + relationships.
      2. LLM selects and weights the most relevant columns for the query.
      3. Min-max normalise each column.
      4. Compute weighted composite score per entity.
      5. Sort descending → assign ranks.

    Falls back to equal-weight scoring if LLM call fails.
    """

    def __init__(
        self,
        *,
        model: str = "meta/meta-llama-3-70b-instruct",
    ) -> None:
        self.model = model

    def rank(
        self,
        *,
        user_query: str,
        session_id: str,
        structured_results: list[StructuredResult],
    ) -> RankingResult:
        """
        Main entry point.

        Args:
            user_query:         The original user question.
            session_id:         Pipeline session ID (for traceability).
            structured_results: Output of GraphStructurer from Neo4j.

        Returns:
            RankingResult with fully ranked entities + scoring breakdown.
        """
        if not structured_results:
            print("[RankingEngine] No structured results to rank.")
            return RankingResult(
                user_query=user_query,
                session_id=session_id,
                ranking_rationale="No entities found to rank.",
                criteria=[],
                entities=[],
                all_columns=[],
                total_entities=0,
            )

        # Step 1: geographic filter
        filtered = _apply_geographic_filter(structured_results, user_query)
        print(f"[RankingEngine] Ranking {len(filtered)} entities (after filter, was {len(structured_results)})")

        # Step 2: build flat feature matrix
        all_columns, flat_rows = _build_feature_matrix(filtered)
        print(f"[RankingEngine] Feature matrix: {len(filtered)} entities × {len(all_columns)} columns")
        print(f"[RankingEngine] Columns: {all_columns}")

        # Step 3: LLM criteria selection
        criteria, rationale = _select_criteria_via_llm(
            user_query, all_columns, flat_rows, model=self.model
        )

        # Step 4 + 5: normalise + score
        scored = _score_entities(filtered, flat_rows, criteria)

        # Step 6: sort descending by composite score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 7: build ranked entities
        ranked_entities: list[RankedEntity] = []
        for rank_idx, (entity, composite, crit_scores, missing) in enumerate(scored, start=1):
            ranked_entities.append(
                RankedEntity(
                    rank=rank_idx,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    composite_score=composite,
                    criterion_scores=crit_scores,
                    properties=entity.properties,
                    relationships=entity.relationships,
                    source_urls=entity.source_urls,
                    missing_criteria=missing,
                )
            )

        if ranked_entities:
            top = ranked_entities[0]
            print(
                f"[RankingEngine] #1: {top.name!r} "
                f"composite={top.composite_score:.4f}"
            )

        return RankingResult(
            user_query=user_query,
            session_id=session_id,
            ranking_rationale=rationale,
            criteria=criteria,
            entities=ranked_entities,
            all_columns=all_columns,
            total_entities=len(ranked_entities),
        )
