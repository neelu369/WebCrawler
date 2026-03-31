"""Intent Parser node — extracts search queries AND target metrics from user input.

Also detects top_n requests ("top 10", "best 5") and generates queries that
target individual entity detail pages — not just list/overview pages.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from crawler.llm import replicate
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import SearchQuery
from crawler.state import State


def _extract_top_n(text: str) -> int | None:
    """Extract 'top N' or 'best N' from query text."""
    m = re.search(r"\b(?:top|best|rank(?:ing)?|list)\s+(\d+)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


_SYSTEM_PROMPT = """\
You are a search-query generation expert for a web research pipeline.

Given a user's ranking/research question, return a JSON object with:
1. "target_metrics": list of 3-6 measurable criteria relevant to ranking these entities
2. "search_queries": list of 15-25 search queries with varied angles

Each search query MUST be a JSON object with:
- "query"       : specific search string — target INDIVIDUAL ENTITY pages
- "topic"       : what this query is looking for
- "preferences" : list of preferences
- "priority"    : "low", "medium", or "high"

Return ONLY valid JSON, no markdown, no explanation.
"""


_RETRY_PROMPT = """\
Previous search found these entities but is missing data for: {missing}

Generate 12-16 NEW search queries specifically to find values for those missing metrics.
Search across ALL available sources — blogs, news articles, startup directories,
government databases, company profiles, interviews, and official pages.
Do NOT limit to any particular source type.

Target these entities: {entities}

Return ONLY a JSON array of query objects:
[{{"query": "...", "topic": "...", "preferences": [], "priority": "high"}}]
"""


def _augment_for_broad_collection(user_query: str, query_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand query set for broad-coverage catalog questions (high recall mode)."""
    ql = user_query.lower()
    is_india_incubator_query = (
        "india" in ql
        and any(k in ql for k in ("incubator", "incubators", "accelerator", "accelerators", "startup"))
    )
    if not is_india_incubator_query:
        return query_dicts

    expansions = [
        "list of startup incubators in india by city",
        "state-wise startup incubators in india",
        "government recognized incubators in india",
        "india incubator directory site:startupindia.gov.in",
        "startup incubators site:aim.gov.in india",
        "technology business incubators india university",
        "biotech incubators india list",
        "fintech incubators india list",
        "social impact incubators india list",
        "women startup incubators india",
        "startup accelerators in bengaluru",
        "startup accelerators in mumbai",
        "startup accelerators in delhi",
        "startup accelerators in hyderabad",
        "startup accelerators in chennai",
        "startup accelerators in pune",
        "startup accelerators in ahmedabad",
        "incubation centres in iit iisc india",
        "best startup incubators in india with portfolio",
        "startup incubator cohort india application",
    ]

    for query in expansions:
        query_dicts.append(
            {
                "query": query,
                "topic": "India incubator discovery",
                "preferences": ["directory", "official", "city-specific", "portfolio"],
                "priority": "high",
            }
        )

    return query_dicts


def _dedupe_query_dicts(query_dicts: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for q in query_dicts:
        text = str(q.get("query", "")).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out


async def parse_intent(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:

    configuration = Configuration.from_runnable_config(config)

    if not state.user_query.strip():
        print("[IntentParser] Empty user_query received; skipping query generation.")
        return {"search_queries": [], "target_metrics": []}

    # Retry mode for missing metrics
    if state.retry_count > 0 and state.missing_data_targets:
        pairs: list[tuple[str, str]] = []
        for target in state.missing_data_targets[:8]:
            if "::" in target:
                entity, metric = target.split("::", 1)
                entity = entity.strip()
                metric = metric.strip()
                if entity and metric:
                    pairs.append((entity, metric))

        entity_names = list({entity for entity, _ in pairs})
        missing_metrics = list({metric for _, metric in pairs})

        if not entity_names or not missing_metrics:
            # Backward-compatible fallback for old target formatting.
            entity_names = list({t.split(" ")[0] for t in state.missing_data_targets[:8]})
            missing_metrics = list(
                {" ".join(t.split(" ")[1:]) for t in state.missing_data_targets[:8]}
            )

        prompt = _RETRY_PROMPT.format(
            missing=", ".join(missing_metrics[:5]),
            entities=", ".join(entity_names[:8]),
        )

        t0 = time.time()

        output = replicate.run(
            configuration.model,
            input={"prompt": prompt, "max_tokens": 1024, "temperature": 0.5},
        )

        raw_text = "".join(str(c) for c in output)

        tracker.record(
            node="intent_parser",
            model=configuration.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(raw_text) // 4,
            latency_s=time.time() - t0,
        )

        try:
            cleaned = raw_text.strip()

            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]

            idx = cleaned.find("[")
            if idx != -1:
                cleaned = cleaned[idx:]

            raw_queries = json.loads(cleaned)
            if not isinstance(raw_queries, list):
                raw_queries = []
            raw_queries = _augment_for_broad_collection(state.user_query, [q for q in raw_queries if isinstance(q, dict)])
            raw_queries = _dedupe_query_dicts(
                raw_queries,
                limit=max(20, int(configuration.max_retry_queries)),
            )
            queries = [SearchQuery(**q) for q in raw_queries]

            print(f"[IntentParser] Retry: {len(queries)} targeted queries")

            return {"search_queries": queries}

        except Exception as exc:
            print(f"[IntentParser] Retry parse failed: {exc}")

    # First pass — full intent parsing
    prompt = f"{_SYSTEM_PROMPT}\n\nUser question:\n{state.user_query}"

    t0 = time.time()

    try:
        output = replicate.run(
            configuration.model,
            input={"prompt": prompt, "max_tokens": 2048, "temperature": 0.6},
        )

        raw_text = "".join(str(c) for c in output)

        tracker.record(
            node="intent_parser",
            model=configuration.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(raw_text) // 4,
            latency_s=time.time() - t0,
        )

        cleaned = raw_text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]

        idx = cleaned.find("{")
        if idx != -1:
            cleaned = cleaned[idx:]

        parsed = json.loads(cleaned)

        target_metrics: list[str] = parsed.get("target_metrics", [])
        query_dicts: list[dict] = parsed.get("search_queries", [])

        if not query_dicts and isinstance(parsed, list):
            query_dicts = parsed

    except Exception as exc:

        print(f"[IntentParser] Parse failed: {exc}. Using fallback.")

        target_metrics = []

        query_dicts = [
            {
                "query": state.user_query,
                "topic": state.user_query,
                "preferences": [],
                "priority": "high",
            }
        ]

    query_dicts = _augment_for_broad_collection(state.user_query, [q for q in query_dicts if isinstance(q, dict)])
    query_dicts = _dedupe_query_dicts(
        query_dicts,
        limit=max(20, int(configuration.max_intent_queries)),
    )
    queries = [SearchQuery(**q) for q in query_dicts]

    detected_top_n = _extract_top_n(state.user_query)

    print(f"[IntentParser] {len(queries)} queries, {len(target_metrics)} metrics")

    if detected_top_n:
        print(f"[IntentParser] Detected top_n={detected_top_n}")

    for q in queries[:5]:
        print(f"  -> {q.query} (priority={q.priority})")

    result: dict[str, Any] = {
        "search_queries": queries,
        "target_metrics": target_metrics,
    }

    return result