"""LangGraph state definitions for the crawler pipeline."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from crawler.models import (
    CrawledDoc,
    DiscoveredURL,
    ExtractedEntity,
    GraphEntity,
    SearchQuery,
    StructuredResult,
    VerifiedSource,
)


@dataclass(kw_only=True)
class InputState:
    user_query: str = ""
    # session_id can be provided as input (e.g. A2A pipeline reusing a session)
    # and is updated by mongo_logger when a new session is created.
    session_id: str = ""


@dataclass(kw_only=True)
class State(InputState):
    # Intent Parser
    search_queries: list[SearchQuery] = field(default_factory=list)

    # URL Discovery
    discovered_urls: list[DiscoveredURL] = field(default_factory=list)
    irrelevant_urls: list[DiscoveredURL] = field(default_factory=list)

    # Web Crawler
    crawled_docs: list[CrawledDoc] = field(default_factory=list)
    preloaded_crawled_docs: list[CrawledDoc] = field(default_factory=list)

    # Source Verifier
    verified_sources: list[VerifiedSource] = field(default_factory=list)

    # MongoDB Logger
    raw_doc_ids: list[str] = field(default_factory=list)
    raw_vector_ids: list[str] = field(default_factory=list)
    # session_id is inherited from InputState — no re-declaration needed

    # Preprocessor (flat entities → ChromaDB for StructuringAgent)
    extracted_entities: list[ExtractedEntity] = field(default_factory=list)
    entity_vector_ids: list[str] = field(default_factory=list)

    # Knowledge Graph pipeline (triples → Neo4j)
    graph_entities: list[GraphEntity] = field(default_factory=list)
    structured_results: list[StructuredResult] = field(default_factory=list)

    # Insights / explainability
    insights_summary: str = ""
    insights_items: list[dict[str, Any]] = field(default_factory=list)
    insights_metadata: dict[str, Any] = field(default_factory=dict)

    # Iterative enrichment
    target_metrics: list[str] = field(default_factory=list)
    missing_data_targets: list[str] = field(default_factory=list)
    investigator_findings: list[dict[str, Any]] = field(default_factory=list)

    # Retry / loop control
    retry_count: Annotated[int, operator.add] = 0
    max_retries: int = 2

    # Cost tracking
    cost_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class OutputState:
    extracted_entities: list[ExtractedEntity]
    structured_results: list[StructuredResult] = field(default_factory=list)
    insights_summary: str = ""
    insights_items: list[dict[str, Any]] = field(default_factory=list)
    insights_metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    raw_doc_ids: list[str] = field(default_factory=list)
    raw_vector_ids: list[str] = field(default_factory=list)
    entity_vector_ids: list[str] = field(default_factory=list)
    cost_summary: dict[str, Any] = field(default_factory=dict)
    missing_data_targets: list[str] = field(default_factory=list)
    investigator_findings: list[dict[str, Any]] = field(default_factory=list)
