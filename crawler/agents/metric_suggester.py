"""Metric suggestion helpers for agent-to-agent crawling."""

from __future__ import annotations

import re


def _normalize_metric(metric: str) -> str:
    return re.sub(r"\s+", " ", metric).strip().lower()


_DOMAIN_METRICS: list[tuple[list[str], list[str]]] = [
    (
        ["hollywood", "movie", "movies", "film", "films", "imdb", "cinema"],
        [
            "Release Year",
            "IMDb Score",
            "Rotten Tomatoes Score",
            "Box Office Collection",
            "Budget",
            "Genre",
            "Director",
            "Lead Cast",
            "Runtime",
        ],
    ),
    (
        ["startup", "startups", "incubator", "accelerator", "venture", "founder"],
        [
            "Location",
            "Funding Amount",
            "Valuation",
            "Founding Year",
            "Investors",
            "Sector",
            "Revenue",
            "Employee Count",
        ],
    ),
    (
        ["stock", "equity", "company", "market", "finance", "financial"],
        [
            "Market Cap",
            "Revenue",
            "Net Income",
            "PE Ratio",
            "Debt",
            "EPS",
            "Dividend Yield",
            "52 Week High Low",
        ],
    ),
    (
        ["university", "college", "school", "education", "course"],
        [
            "Location",
            "Ranking",
            "Tuition Fees",
            "Acceptance Rate",
            "Program Duration",
            "Placement Rate",
            "Scholarships",
        ],
    ),
]

_GENERIC_METRICS = [
    "Location",
    "Rating",
    "Cost",
    "Date",
    "Category",
]


def suggest_metrics_for_query(query: str, *, max_suggestions: int = 8) -> list[str]:
    """Suggest relevant metrics based on query keywords."""
    normalized_query = query.strip().lower()
    if not normalized_query:
        return []

    suggested: list[str] = []
    for keywords, metrics in _DOMAIN_METRICS:
        if any(keyword in normalized_query for keyword in keywords):
            suggested.extend(metrics)

    if not suggested:
        suggested.extend(_GENERIC_METRICS)

    deduped: list[str] = []
    seen: set[str] = set()
    for metric in suggested:
        norm = _normalize_metric(metric)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(metric)
        if len(deduped) >= max_suggestions:
            break
    return deduped


def merge_metrics(
    *,
    suggested_metrics: list[str],
    user_metrics: list[str],
) -> list[str]:
    """Merge suggested metrics with user-specified metrics (user metrics preserved)."""
    merged: list[str] = []
    seen: set[str] = set()

    for source in (suggested_metrics, user_metrics):
        for metric in source:
            cleaned = metric.strip()
            if not cleaned:
                continue
            norm = _normalize_metric(cleaned)
            if norm in seen:
                continue
            seen.add(norm)
            merged.append(cleaned)
    return merged
