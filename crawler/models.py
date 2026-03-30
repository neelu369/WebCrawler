"""Pydantic v2 data models used across all pipeline nodes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(description="The search string to send to the search API.")
    topic: str = Field(description="High-level topic this query relates to.")
    preferences: list[str] = Field(default_factory=list)
    priority: str = Field(default="medium")


class DiscoveredURL(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    search_query: str = Field(default="")


class CrawledDoc(BaseModel):
    url: str
    content: str
    word_count: int = 0
    crawl_method: str = Field(default="crawl4ai")


class VerifiedSource(BaseModel):
    url: str
    content: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    is_trusted: bool = False


class ExtractedEntity(BaseModel):
    name: str
    description: str
    metrics: dict[str, str] = Field(default_factory=dict)
    source_url: str
    priority_score: float = Field(default=0.0)
    original_content: str = ""


# ── Knowledge-graph models ───────────────────────────────────

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence_snippet: str = ""
    source_url: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class GraphEntity(BaseModel):
    name: str
    entity_type: str = "Entity"
    description: str = ""
    triples: list[Triple] = Field(default_factory=list)
    source_url: str = ""
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CitationMetadata(BaseModel):
    value: str
    evidence: str = ""
    source: str = ""


class StructuredResult(BaseModel):
    name: str
    entity_type: str = "Entity"
    description: str = ""
    properties: dict[str, str] = Field(default_factory=dict)
    relationships: list[dict[str, str]] = Field(default_factory=list)
    citations: dict[str, CitationMetadata] = Field(default_factory=dict)
    source_urls: list[str] = Field(default_factory=list)
    priority_score: float = 0.0

class DiscoveredMetric(BaseModel):
    entity_name: str
    metric_name: str
    value: str
    source_url: str

class InvestigatorResponse(BaseModel):
    findings: list[DiscoveredMetric] = Field(default_factory=list)
