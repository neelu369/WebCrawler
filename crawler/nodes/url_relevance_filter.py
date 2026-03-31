"""Pre-crawl URL relevance filter node.

This node keeps only URLs that look relevant to the active user query.
It runs immediately after URL discovery to avoid crawling unrelated pages.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from crawler.agents.url_relevance_agent import URLRelevanceAgent
from crawler.config import Configuration
from crawler.models import DiscoveredURL
from crawler.state import State


async def filter_relevant_urls(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    agent = URLRelevanceAgent(
        model=configuration.model,
        min_overlap_score=configuration.min_url_relevance_score,
        enable_llm_tiebreak=configuration.enable_llm_url_relevance,
    )

    relevant: list[DiscoveredURL] = []
    irrelevant: list[DiscoveredURL] = []
    dropped_ranked: list[tuple[float, float, DiscoveredURL]] = []

    for candidate in state.discovered_urls:
        decision = agent.assess(
            query=state.user_query,
            url=candidate.url,
            title=candidate.title,
            snippet=candidate.snippet,
        )

        if decision.is_relevant:
            relevant.append(candidate)
            verdict = "PASS"
        else:
            irrelevant.append(candidate)
            dropped_ranked.append((decision.overlap_score, decision.confidence, candidate))
            verdict = "DROP"

        print(
            f"[URL Relevance] {verdict} {candidate.url[:80]} | "
            f"method={decision.method} conf={decision.confidence:.2f} "
            f"overlap={decision.overlap_score:.2f} reason={decision.reason}"
        )

    # High-recall rescue: keep top borderline dropped URLs so broad queries do not collapse.
    target_keep = min(len(state.discovered_urls), max(0, int(configuration.url_filter_min_keep)))
    rescued = 0
    if len(relevant) < target_keep and dropped_ranked:
        needed = target_keep - len(relevant)
        dropped_ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        rescue_candidates = [item[2] for item in dropped_ranked[:needed]]
        rescued = len(rescue_candidates)
        if rescued:
            relevant.extend(rescue_candidates)
            rescued_urls = {u.url for u in rescue_candidates}
            irrelevant = [u for u in irrelevant if u.url not in rescued_urls]

    print(
        f"[URL Relevance] Kept {len(relevant)}/{len(state.discovered_urls)} URLs "
        f"for crawling (dropped {len(irrelevant)}, rescued={rescued})"
    )

    return {"discovered_urls": relevant, "irrelevant_urls": irrelevant}
