"""Ranking Engine — multi-algorithm ranking for any type of entity data.

Algorithms used (combined via ensemble):
─────────────────────────────────────────────────────────────────────────
1. TOPSIS  (Technique for Order of Preference by Similarity to Ideal Solution)
   - Finds the "ideal best" and "ideal worst" across all numeric columns
   - Scores each entity by its relative distance to ideal vs anti-ideal
   - Handles both higher-is-better and lower-is-better criteria correctly
   - Industry standard for multi-criteria decision making (MCDM)

2. Borda Count  (rank aggregation across individual columns)
   - For each criterion column, sort entities and assign rank positions
   - Sum rank positions across all criteria (lower = better)
   - Works on ANY data — numeric, ordinal, categorical
   - Robust to outliers and missing values

3. Weighted Frequency Score  (for categorical / text fields)
   - Counts meaningful non-null values per entity across all columns
   - Rewards entities with richer, more complete data
   - Acts as a completeness tiebreaker

Ensemble:
   final_score = w_topsis * topsis_score
               + w_borda  * borda_score
               + w_freq   * completeness_score

   Default weights: TOPSIS 55%, Borda 35%, Completeness 10%
   All weights sum to 1.0.

LLM role:
   - Selects WHICH columns are relevant for the query
   - Assigns per-column weights (how important each criterion is)
   - Sets higher_is_better direction per column
   - Does NOT do scoring — all scoring is deterministic math

Works for: movies, startups, universities, stocks, people, places, products —
anything that can be described with named properties.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import replicate

from crawler.cost_tracker import tracker
from crawler.models import StructuredResult


# ── Constants ─────────────────────────────────────────────────
TOPSIS_WEIGHT      = 0.55
BORDA_WEIGHT       = 0.35
COMPLETENESS_WEIGHT= 0.10

_MISSING_VALUES = frozenset({
    "", "n/a", "na", "null", "none", "not available", "not found",
    "unknown", "-", "–", "?", "not specified", "not disclosed",
    "not mentioned", "none mentioned",
})


# ── Data models ───────────────────────────────────────────────

@dataclass
class RankingCriterion:
    column: str
    weight: float
    higher_is_better: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "column":           self.column,
            "weight":           round(self.weight, 4),
            "higher_is_better": self.higher_is_better,
            "rationale":        self.rationale,
        }


@dataclass
class AlgorithmScores:
    """Per-algorithm scores for transparency."""
    topsis:       float = 0.0
    borda:        float = 0.0
    completeness: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "topsis":       round(self.topsis, 4),
            "borda":        round(self.borda, 4),
            "completeness": round(self.completeness, 4),
        }


@dataclass
class RankedEntity:
    rank:             int
    name:             str
    entity_type:      str
    description:      str
    composite_score:  float
    algorithm_scores: AlgorithmScores
    criterion_scores: dict[str, float]   # per-column normalised scores
    properties:       dict[str, str]
    relationships:    list[dict[str, str]]
    source_urls:      list[str]
    missing_criteria: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank":             self.rank,
            "name":             self.name,
            "entity_type":      self.entity_type,
            "description":      self.description,
            "composite_score":  round(self.composite_score, 4),
            "algorithm_scores": self.algorithm_scores.to_dict(),
            "criterion_scores": {k: round(v, 4) for k, v in self.criterion_scores.items()},
            "properties":       self.properties,
            "relationships":    self.relationships,
            "source_urls":      self.source_urls,
            "missing_criteria": self.missing_criteria,
        }


@dataclass
class RankingResult:
    user_query:        str
    session_id:        str
    ranking_rationale: str
    criteria:          list[RankingCriterion]
    entities:          list[RankedEntity]
    all_columns:       list[str]
    total_entities:    int
    algorithm:         str = "topsis+borda+completeness_ensemble"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query":        self.user_query,
            "session_id":        self.session_id,
            "ranking_rationale": self.ranking_rationale,
            "algorithm":         self.algorithm,
            "criteria":          [c.to_dict() for c in self.criteria],
            "entities":          [e.to_dict() for e in self.entities],
            "all_columns":       self.all_columns,
            "total_entities":    self.total_entities,
        }


# ── Value extraction ──────────────────────────────────────────

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _MISSING_VALUES


def _extract_number(value: Any) -> float | None:
    """
    Extract a float from any cell value.
    Handles: "$1.2M", "₹500 Cr", "45%", "9.2/10", "3 years", "Top 10", "1,200"
    Also handles ordinal text: "Very High" → 5, "High" → 4, "Medium" → 3, etc.
    """
    if _is_missing(value):
        return None
    text = str(value).strip()

    # Ordinal/categorical text mappings — common across all domains
    _ORDINAL_MAP = {
        # Difficulty / intensity
        "extremely high": 6.0, "very high": 5.0, "high": 4.0,
        "moderate": 3.0, "medium": 3.0, "average": 3.0,
        "low": 2.0, "very low": 1.0, "minimal": 1.0,
        # Ratings
        "excellent": 5.0, "very good": 4.0, "good": 3.5,
        "fair": 2.5, "poor": 1.5, "very poor": 1.0,
        # Frequency
        "always": 5.0, "usually": 4.0, "often": 3.5,
        "sometimes": 2.5, "rarely": 1.5, "never": 1.0,
        # Size
        "very large": 5.0, "large": 4.0, "medium": 3.0,
        "small": 2.0, "very small": 1.0,
        # Tier
        "tier 1": 5.0, "tier 2": 4.0, "tier 3": 3.0, "tier 4": 2.0,
        # Yes/No
        "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
    }
    text_lower = text.lower().strip()
    if text_lower in _ORDINAL_MAP:
        return _ORDINAL_MAP[text_lower]

    # Handle "X/Y" rating format — convert to percentage
    m = re.match(r"^(\d+\.?\d*)\s*/\s*(\d+\.?\d*)$", text)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass

    # Strip currency, commas, percent, whitespace, tildes
    cleaned = re.sub(r"[,$%₹€£¥~\s]", "", text)

    # Magnitude suffixes — longest first to avoid partial matches
    multipliers = [
        ("bn",    1_000_000_000),
        ("b",     1_000_000_000),
        ("mn",    1_000_000),
        ("m",     1_000_000),
        ("lakh",  100_000),
        ("lac",   100_000),
        ("cr",    10_000_000),
        ("k",     1_000),
    ]
    lower = cleaned.lower()
    for suffix, mult in multipliers:
        if lower.endswith(suffix):
            try:
                return float(lower[: -len(suffix)]) * mult
            except ValueError:
                pass

    # Plain number
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Extract first number from mixed text e.g. "Rank 5", "Top 10", "3 years"
    m = re.search(r"[-+]?\d+\.?\d*", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass

    return None


def _cell_has_value(value: Any) -> bool:
    """True if the cell contains any meaningful content."""
    if _is_missing(value):
        return False
    return bool(str(value).strip())


# ── Feature matrix ────────────────────────────────────────────

def _build_feature_matrix(
    entities: list[StructuredResult],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Collect all property keys + flattened relationship types.
    Returns (all_columns, flat_row_per_entity).
    """
    all_keys: set[str] = set()
    flat_rows: list[dict[str, str]] = []

    for entity in entities:
        row: dict[str, str] = {}

        for k, v in entity.properties.items():
            all_keys.add(k)
            row[k] = str(v)

        for rel in entity.relationships:
            rel_type = rel.get("type", "")
            target   = rel.get("target", "")
            if rel_type and target:
                all_keys.add(rel_type)
                if rel_type in row:
                    row[rel_type] = f"{row[rel_type]}, {target}"
                else:
                    row[rel_type] = target

        flat_rows.append(row)

    all_columns = sorted(all_keys, key=lambda c: (len(c), c))
    return all_columns, flat_rows


# ── Algorithm 1: TOPSIS ───────────────────────────────────────

def _run_topsis(
    flat_rows: list[dict[str, str]],
    criteria:  list[RankingCriterion],
) -> tuple[list[float], dict[str, list[float]]]:
    """
    TOPSIS: Technique for Order Preference by Similarity to Ideal Solution.

    Steps:
      1. Build decision matrix (numeric values per entity per criterion)
      2. Normalise using vector normalisation: r_ij = x_ij / sqrt(sum(x^2))
      3. Apply weights: v_ij = w_j * r_ij
      4. Find ideal best (A+) and ideal worst (A-) per criterion
      5. Compute Euclidean distance from each entity to A+ and A-
      6. Score = d- / (d+ + d-)  → 1.0 means closest to ideal

    Returns:
      scores          list[float] — TOPSIS score per entity (0..1)
      criterion_scores dict       — per-criterion normalised score per entity
    """
    n = len(flat_rows)
    if n == 0 or not criteria:
        return [0.5] * n, {}

    # Step 1: raw numeric matrix
    raw: dict[str, list[float | None]] = {}
    for crit in criteria:
        raw[crit.column] = [_extract_number(row.get(crit.column)) for row in flat_rows]

    # Step 2: vector normalisation per column
    normalised: dict[str, list[float]] = {}
    for crit in criteria:
        vals = raw[crit.column]
        numeric = [v for v in vals if v is not None]
        if not numeric:
            normalised[crit.column] = [0.0] * n
            continue
        col_norm = math.sqrt(sum(v * v for v in numeric))
        if col_norm == 0:
            normalised[crit.column] = [0.0] * n
            continue
        normalised[crit.column] = [
            (v / col_norm if v is not None else 0.0)
            for v in vals
        ]

    # Step 3: weighted normalised matrix
    weighted: dict[str, list[float]] = {
        crit.column: [v * crit.weight for v in normalised[crit.column]]
        for crit in criteria
    }

    # Step 4: ideal best (A+) and ideal worst (A-)
    ideal_best:  dict[str, float] = {}
    ideal_worst: dict[str, float] = {}
    for crit in criteria:
        vals = [v for v in weighted[crit.column] if v != 0.0] or [0.0]
        if crit.higher_is_better:
            ideal_best[crit.column]  = max(vals)
            ideal_worst[crit.column] = min(vals)
        else:
            ideal_best[crit.column]  = min(vals)
            ideal_worst[crit.column] = max(vals)

    # Step 5: distances
    d_plus  = [0.0] * n
    d_minus = [0.0] * n
    for i in range(n):
        dp = dm = 0.0
        for crit in criteria:
            v  = weighted[crit.column][i]
            dp += (v - ideal_best[crit.column])  ** 2
            dm += (v - ideal_worst[crit.column]) ** 2
        d_plus[i]  = math.sqrt(dp)
        d_minus[i] = math.sqrt(dm)

    # Step 6: TOPSIS score
    topsis_scores = [
        (d_minus[i] / (d_plus[i] + d_minus[i]))
        if (d_plus[i] + d_minus[i]) > 0
        else 0.5
        for i in range(n)
    ]

    # Per-criterion contribution scores (for display)
    criterion_scores: dict[str, list[float]] = {}
    for crit in criteria:
        col_max = max(normalised[crit.column]) if any(v > 0 for v in normalised[crit.column]) else 1.0
        if col_max == 0:
            criterion_scores[crit.column] = [0.0] * n
        else:
            raw_norm = [v / col_max for v in normalised[crit.column]]
            criterion_scores[crit.column] = raw_norm if crit.higher_is_better else [1.0 - v for v in raw_norm]

    return topsis_scores, criterion_scores


# ── Algorithm 2: Borda Count ──────────────────────────────────

def _run_borda(
    flat_rows: list[dict[str, str]],
    criteria:  list[RankingCriterion],
) -> list[float]:
    """
    Borda Count: rank aggregation across all criteria.

    For each criterion:
      - Sort entities by that criterion (numeric if possible, else lexicographic)
      - Assign Borda points: best entity gets N-1 points, worst gets 0
      - Weight the points by criterion weight

    Final score = sum(weighted borda points) / max_possible → [0, 1]

    Works for ANY data type — numeric, text, categorical, mixed.
    """
    n = len(flat_rows)
    if n == 0 or not criteria:
        return [0.5] * n

    total_points = [0.0] * n

    for crit in criteria:
        col_vals = [flat_rows[i].get(crit.column) for i in range(n)]

        # Try numeric sort first
        numeric_vals = [(i, _extract_number(v)) for i, v in enumerate(col_vals)]
        has_numeric  = any(v is not None for _, v in numeric_vals)

        if has_numeric:
            # Entities with no numeric value get 0 points (treated as worst)
            def sort_key(iv: tuple[int, float | None]) -> float:
                return iv[1] if iv[1] is not None else -math.inf
            sorted_indices = sorted(numeric_vals, key=sort_key, reverse=crit.higher_is_better)
        else:
            # Lexicographic sort for text/categorical
            def lex_key(iv: tuple[int, Any]) -> str:
                return str(iv[1]).lower() if not _is_missing(iv[1]) else ""
            sorted_indices = sorted(enumerate(col_vals), key=lex_key, reverse=crit.higher_is_better)

        # Assign Borda points — handle ties by giving tied entities the same score
        borda_points = [0.0] * n
        prev_val     = object()
        prev_pts     = 0.0
        for rank_pos, (i, val) in enumerate(sorted_indices):
            pts = (n - 1 - rank_pos) * crit.weight
            if val == prev_val:
                pts = prev_pts  # tied — same points
            borda_points[i] = pts
            prev_val = val
            prev_pts = pts

        for i in range(n):
            total_points[i] += borda_points[i]

    # Normalise to [0, 1]
    max_pts = max(total_points) if any(p > 0 for p in total_points) else 1.0
    return [p / max_pts for p in total_points]


# ── Algorithm 3: Completeness score ──────────────────────────

def _run_completeness(
    flat_rows:   list[dict[str, str]],
    all_columns: list[str],
) -> list[float]:
    """
    Completeness score: fraction of non-missing cells across all columns.
    Rewards entities with richer, more complete data.
    Acts as tiebreaker when other scores are equal.
    """
    n = len(flat_rows)
    if n == 0 or not all_columns:
        return [0.0] * n

    scores = []
    for row in flat_rows:
        filled = sum(1 for col in all_columns if _cell_has_value(row.get(col)))
        scores.append(filled / len(all_columns))
    return scores


# ── LLM criteria selection ────────────────────────────────────

_CRITERIA_PROMPT = """\
You are a ranking expert. A user wants to rank entities based on a question.

User's ranking question:
{query}

Available data columns (extracted from a knowledge graph):
{columns}

Sample entities (first 3) showing their actual data values:
{sample_entities}

Task — select the best columns for ranking this specific query:
1. Pick columns with meaningful comparable data. Skip name, URL, description columns.
2. Assign a weight (0.0–1.0) to each selected column. Weights MUST sum exactly to 1.0.
3. Set higher_is_better:
   - true  → higher value = better (e.g. rating, funding, score, revenue)
   - false → lower value = better (e.g. equity taken, cost, rank number, years to exit)
4. Write one sentence explaining the overall ranking logic.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "ranking_rationale": "One sentence.",
  "criteria": [
    {{
      "column": "Exact Column Name As Listed Above",
      "weight": 0.40,
      "higher_is_better": true,
      "rationale": "Why this column matters for this query."
    }}
  ]
}}
"""


def _select_criteria_llm(
    user_query:  str,
    all_columns: list[str],
    flat_rows:   list[dict[str, str]],
    *,
    model: str,
) -> tuple[list[RankingCriterion], str]:
    """Call LLM to select and weight ranking criteria."""

    # Build compact sample showing real values
    sample = []
    for row in flat_rows[:3]:
        # Only show columns that have actual values
        sample.append({k: v for k, v in list(row.items())[:12] if _cell_has_value(v)})

    prompt = _CRITERIA_PROMPT.format(
        query=user_query,
        columns=json.dumps(all_columns, ensure_ascii=True),
        sample_entities=json.dumps(sample, ensure_ascii=True, indent=2),
    )

    t0 = time.time()
    try:
        output = replicate.run(
            model,
            input={"prompt": prompt, "max_tokens": 1024, "temperature": 0.1},
        )
        raw = "".join(str(c) for c in output)
        tracker.record(
            node="ranking_engine", model=model,
            input_tokens=len(prompt) // 4, output_tokens=len(raw) // 4,
            latency_s=time.time() - t0,
        )
    except Exception as exc:
        print(f"[RankingEngine] LLM call failed: {exc}. Using equal weights.")
        return _equal_weight_criteria(all_columns), "Equal weight assigned to all columns (LLM unavailable)."

    # Parse JSON
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        idx = cleaned.find("{")
        if idx != -1:
            cleaned = cleaned[idx:]
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[RankingEngine] LLM JSON parse failed: {exc}. Using equal weights.")
        return _equal_weight_criteria(all_columns), "Equal weight fallback."

    rationale   = parsed.get("ranking_rationale", "")
    valid_cols  = set(all_columns)
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
        criteria.append(RankingCriterion(
            column=col,
            weight=weight,
            higher_is_better=bool(item.get("higher_is_better", True)),
            rationale=str(item.get("rationale", "")),
        ))

    if not criteria:
        return _equal_weight_criteria(all_columns), rationale or "Equal weight fallback."

    # Normalise weights to exactly 1.0
    total = sum(c.weight for c in criteria)
    if total > 0:
        for c in criteria:
            c.weight = round(c.weight / total, 6)
    diff = 1.0 - sum(c.weight for c in criteria)
    if criteria:
        criteria[-1].weight = round(criteria[-1].weight + diff, 6)

    print(
        f"[RankingEngine] LLM selected {len(criteria)} criteria: "
        + ", ".join(f"{c.column}({c.weight:.2f})" for c in criteria)
    )
    return criteria, rationale


def _equal_weight_criteria(columns: list[str]) -> list[RankingCriterion]:
    """Fallback: equal weight on all columns."""
    # Skip pure-text columns unlikely to be rankable
    candidates = [
        c for c in columns
        if c.lower() not in {"name", "description", "source", "url", "type", "entity type"}
    ] or columns[:6]
    n      = len(candidates) or 1
    weight = round(1.0 / n, 6)
    crits  = [RankingCriterion(column=col, weight=weight, higher_is_better=True, rationale="Equal weight fallback.") for col in candidates]
    diff   = 1.0 - sum(c.weight for c in crits)
    if crits:
        crits[-1].weight = round(crits[-1].weight + diff, 6)
    return crits


# ── Geographic filter ─────────────────────────────────────────

def _apply_geographic_filter(
    entities:   list[StructuredResult],
    user_query: str,
) -> list[StructuredResult]:
    """Remove entities whose location contradicts the query's geographic focus."""
    q = user_query.lower()

    geo_filters = {
        "india":     ["united states", "usa", "u.s.", "silicon valley", "london", "uk", "singapore", "europe", "australia", "canada"],
        "us":        ["india", "china", "europe", "australia"],
        "uk":        ["india", "usa", "china", "australia"],
        "europe":    ["india", "usa", "china", "australia"],
    }

    target_region = None
    for region in geo_filters:
        keywords = [region] if region != "india" else ["india", "indian", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai", "pune"]
        if any(kw in q for kw in keywords):
            target_region = region
            break

    if not target_region:
        return entities

    exclude = set(geo_filters[target_region])
    filtered, removed = [], []

    for entity in entities:
        location = str(
            entity.properties.get("Located In")
            or entity.properties.get("Headquartered In")
            or entity.properties.get("Location", "")
        ).lower()

        contradicts = any(place in location for place in exclude) and target_region not in location and bool(location)

        if contradicts:
            removed.append(entity.name)
        else:
            filtered.append(entity)

    if removed:
        print(f"[RankingEngine] Geo-filtered {len(removed)} off-target entities: {removed[:5]}")

    return filtered if filtered else entities


# ── Public API ────────────────────────────────────────────────

class RankingEngine:
    """
    Multi-algorithm ranking engine for any type of entity data.

    Ensemble of three algorithms:
      TOPSIS      (55%) — multi-criteria distance from ideal solution
      Borda Count (35%) — rank aggregation, works on any data type
      Completeness(10%) — data richness tiebreaker

    LLM selects relevant columns and weights; all scoring is deterministic math.
    """

    def __init__(self, *, model: str = "meta/meta-llama-3-70b-instruct") -> None:
        self.model = model

    def rank(
        self,
        *,
        user_query:         str,
        session_id:         str,
        structured_results: list[StructuredResult],
    ) -> RankingResult:

        if not structured_results:
            return RankingResult(
                user_query=user_query, session_id=session_id,
                ranking_rationale="No entities found to rank.",
                criteria=[], entities=[], all_columns=[], total_entities=0,
            )

        # Step 1: geographic filter
        entities = _apply_geographic_filter(structured_results, user_query)
        print(f"[RankingEngine] {len(entities)} entities after geo-filter (was {len(structured_results)})")

        # Step 2: feature matrix
        all_columns, flat_rows = _build_feature_matrix(entities)
        print(f"[RankingEngine] Feature matrix: {len(entities)} × {len(all_columns)} columns")

        # Step 3: LLM selects criteria + weights
        criteria, rationale = _select_criteria_llm(
            user_query, all_columns, flat_rows, model=self.model
        )
        print(f"[RankingEngine] Criteria: {[(c.column, c.weight) for c in criteria]}")

        n = len(entities)

        # Step 4: TOPSIS
        topsis_scores, criterion_scores_per_col = _run_topsis(flat_rows, criteria)
        print(f"[RankingEngine] TOPSIS scores: {[round(s, 3) for s in topsis_scores]}")

        # Step 5: Borda Count
        borda_scores = _run_borda(flat_rows, criteria)
        print(f"[RankingEngine] Borda scores: {[round(s, 3) for s in borda_scores]}")

        # Step 6: Completeness
        completeness_scores = _run_completeness(flat_rows, all_columns)

        # Step 7: Ensemble composite
        composite_scores = [
            TOPSIS_WEIGHT       * topsis_scores[i]
            + BORDA_WEIGHT      * borda_scores[i]
            + COMPLETENESS_WEIGHT * completeness_scores[i]
            for i in range(n)
        ]

        # Step 8: Sort and assign ranks
        order = sorted(range(n), key=lambda i: composite_scores[i], reverse=True)

        ranked_entities: list[RankedEntity] = []
        for rank_pos, i in enumerate(order, start=1):
            entity = entities[i]
            row    = flat_rows[i]

            # Per-criterion scores for display
            crit_display = {
                col: round(criterion_scores_per_col.get(col, [0.0] * n)[i], 4)
                for col in criterion_scores_per_col
            }

            # Missing criteria = criteria columns with no parseable number
            missing = [
                crit.column for crit in criteria
                if _extract_number(row.get(crit.column)) is None
            ]

            ranked_entities.append(RankedEntity(
                rank=rank_pos,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                composite_score=composite_scores[i],
                algorithm_scores=AlgorithmScores(
                    topsis=topsis_scores[i],
                    borda=borda_scores[i],
                    completeness=completeness_scores[i],
                ),
                criterion_scores=crit_display,
                properties=entity.properties,
                relationships=entity.relationships,
                source_urls=entity.source_urls,
                missing_criteria=missing,
            ))

        if ranked_entities:
            top = ranked_entities[0]
            print(
                f"[RankingEngine] #1: {top.name!r} "
                f"composite={top.composite_score:.4f} "
                f"(topsis={top.algorithm_scores.topsis:.3f}, "
                f"borda={top.algorithm_scores.borda:.3f})"
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