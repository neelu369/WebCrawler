"""
LangGraph node for incubator discovery.

Integrates with the existing pipeline to discover Indian incubators.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.sources.india_incubator_discovery import (
    IndiaIncubatorDiscovery,
    IncubatorEntity,
)
from crawler.state import State
from crawler.models import DiscoveredURL


async def discover_incubators(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Node: Discover Indian incubators from multiple sources.
    
    This is a specialized discovery node that targets the specific
    goal of finding all ~1100-1200 incubators in India.
    
    Returns state update with discovered_urls.
    """
    configuration = Configuration.from_runnable_config(config)
    
    print(f"[IncubatorDiscovery] Starting discovery for query: {state.user_query}")
    
    # Initialize discovery
    discovery = IndiaIncubatorDiscovery()
    
    # Discover all sources
    entities = await discovery.discover_all(max_concurrent=configuration.crawler_concurrency)
    
    # Convert IncubatorEntity to DiscoveredURL for pipeline compatibility
    discovered_urls = []
    for entity in entities:
        if entity.website:
            discovered_urls.append(
                DiscoveredURL(
                    url=entity.website,
                    title=entity.name,
                    snippet=entity.official_name or f"{entity.type} incubator in {entity.city}, {entity.state}",
                    search_query=state.user_query,
                )
            )
    
    # Also include institutional pages for crawling
    institution_urls = []
    for category, urls in discovery.INSTITUTION_PATTERNS.items():
        if isinstance(urls, list):
            for url in urls:
                institution_urls.append(
                    DiscoveredURL(
                        url=url,
                        title=f"{category.upper()} - Incubator Page",
                        snippet=f"Institutional incubator directory - {category}",
                        search_query=f"{state.user_query} {category}",
                    )
                )
    
    all_urls = discovered_urls + institution_urls
    
    print(f"[IncubatorDiscovery] Discovered {len(all_urls)} URLs to crawl")
    print(f"  - Entity websites: {len(discovered_urls)}")
    print(f"  - Institution pages: {len(institution_urls)}")
    
    # Store the entities for later enrichment
    return {
        "discovered_urls": all_urls,
        "extracted_entities": entities,  # Pass full entities to state
    }


async def enrich_incubator_entity(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Node: Enrich incubator entities with detailed information.
    
    Iteratively crawls entity websites to fill missing fields.
    """
    from crawler.sources.india_incubator_discovery import IncubatorEnricher
    
    configuration = Configuration.from_runnable_config(config)
    entities = state.extracted_entities if hasattr(state, 'extracted_entities') else []
    
    if not entities:
        print("[Enrichment] No entities to enrich")
        return {"extracted_entities": []}
    
    print(f"[Enrichment] Enriching {len(entities)} incubator entities...")
    
    enricher = IncubatorEnricher()
    enriched = []
    
    for entity in entities:
        if isinstance(entity, IncubatorEntity):
            enriched_entity = await enricher.enrich_entity(entity)
            enriched.append(enriched_entity)
    
    # Calculate overall completeness
    avg_completeness = sum(e.data_completeness for e in enriched) / len(enriched) if enriched else 0
    
    print(f"[Enrichment] Complete. Average data completeness: {avg_completeness:.1%}")
    print(f"  - Complete entities: {sum(1 for e in enriched if e.data_completeness == 1.0)}")
    print(f"  - Partially complete: {sum(1 for e in enriched if 0.5 <= e.data_completeness < 1.0)}")
    print(f"  - Minimal data: {sum(1 for e in enriched if e.data_completeness < 0.5)}")
    
    return {"extracted_entities": enriched}
