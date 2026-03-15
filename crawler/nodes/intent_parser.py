"""Intent Parser node — extracts structured search queries and target metrics.

Uses the Replicate LLM to analyse the user query and produce a list of
SearchQuery objects along with a set of domain-specific target metrics.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import replicate
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import SearchQuery
from crawler.state import State

_SYSTEM_PROMPT = """\
You are a research-query and metric generation expert. Given a user's research request,
analyze it and produce a JSON object with two keys:
- "target_metrics": an array of strings representing the essential, domain-agnostic properties or metrics that matter most for the entities being asked about (e.g. ["bed count", "specialties", "funding amount", "pricing model"]).
- "search_queries": an array of search query objects to send to a web search API.

Each element in "search_queries" MUST be a JSON object with these keys:
- "query"       : the search string (be specific, diverse, use different phrasings)
- "topic"       : high-level topic this query relates to
- "preferences" : list of user-stated preferences (e.g. ["recent", "peer-reviewed"])
- "priority"    : "low", "medium", or "high"

Generate 3-5 queries that cover different angles of the user's request.
Return ONLY the JSON object, no markdown fences, no commentary.
"""


async def parse_intent(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Analyse the user intent and generate structured SearchQuery objects."""
    configuration = Configuration.from_runnable_config(config)

    user_msg = state.user_query
    is_retry_for_missing = state.retry_count > 0 and state.missing_data_targets
    
    if is_retry_for_missing:
        user_msg = (
            "We have already found entities for the original query, but are missing specific metrics. "
            "Please generate highly precise search queries to find the following missing data:\n"
            + json.dumps(state.missing_data_targets, indent=2)
            + f"\n\nOriginal request context for reference: {state.user_query}"
        )
    elif state.retry_count > 0:
        user_msg += (
            "\n\n[RETRY NOTE: Previous search yielded too few results. "
            "Try broader or alternative queries.]"
        )

    t0 = time.time()

    output = replicate.run(
        configuration.model,
        input={
            "prompt": f"{_SYSTEM_PROMPT}\n\nUser request:\n{user_msg}",
            "max_tokens": 1024,
            "temperature": 0.7,
        },
    )

    # Replicate streaming returns an iterator of string chunks
    raw_text = "".join(str(chunk) for chunk in output)
    latency = time.time() - t0

    # Estimate tokens (rough: ~4 chars per token)
    input_tokens = len(f"{_SYSTEM_PROMPT}\n\nUser request:\n{user_msg}") // 4
    output_tokens = len(raw_text) // 4

    tracker.record(
        node="intent_parser",
        model=configuration.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency,
    )

    # Parse LLM response into SearchQuery objects
    target_metrics = []
    queries_data = []
    try:
        # Strip potential markdown fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
            
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            target_metrics = parsed.get("target_metrics", [])
            queries_data = parsed.get("search_queries", [])
        else:
            # Fallback if it mistakenly returned an array (old schema)
            queries_data = parsed
    except json.JSONDecodeError:
        # Fallback: create a single generic query
        queries_data = [
            {
                "query": state.user_query,
                "topic": state.user_query,
                "preferences": [],
                "priority": "high",
            }
        ]

    queries = [SearchQuery(**item) for item in queries_data]
    print(f"[Intent Parser] Generated {len(queries)} search queries")
    for q in queries:
        print(f"  -> {q.query} (priority={q.priority})")
    
    if target_metrics:
        print(f"[Intent Parser] Identified target metrics: {target_metrics}")

    if is_retry_for_missing:
        # Preserve existing target metrics on retries
        return {"search_queries": queries}
    else:
        return {"search_queries": queries, "target_metrics": target_metrics}
