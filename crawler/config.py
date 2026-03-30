"""Configurable parameters for the crawler pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Optional

from langchain_core.runnables import RunnableConfig, ensure_config


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(kw_only=True)
class Configuration:
    """Runtime configuration — values can be overridden via RunnableConfig."""

    model: str = field(
        default="meta/meta-llama-3-70b-instruct",
        metadata={"description": "Replicate model identifier used for all LLM calls."},
    )

    max_search_results: int = field(
        default=15,
        metadata={"description": "Max results per SearXNG search query."},
    )

    min_url_relevance_score: float = field(
        default=0.2,
        metadata={
            "description": "Minimum lexical overlap to keep a URL heuristics-only."
        },
    )

    enable_serper_search: bool = field(
        default=False,
        metadata={
            "description": "Legacy flag (currently unused). Discovery runs through SearXNG."
        },
    )

    enable_llm_url_relevance: bool = field(
        default=True,
        metadata={
            "description": "If true, use an LLM tie-breaker for borderline URL relevance decisions."
        },
    )

    min_word_count: int = field(
        default=100,
        metadata={
            "description": "Minimum word count to pass the crawler quality gate."
        },
    )

    crawler_concurrency: int = field(
        default=5,
        metadata={"description": "Maximum number of concurrent crawl workers."},
    )

    enable_playwright_mcp: bool = field(
        default_factory=lambda: _env_bool("PLAYWRIGHT_MCP_ENABLED", True),
        metadata={
            "description": "If true, use Playwright MCP fallback for dynamic pages."
        },
    )

    playwright_mcp_command: str = field(
        default_factory=lambda: os.getenv("PLAYWRIGHT_MCP_COMMAND", "npx"),
        metadata={"description": "Command used to launch Playwright MCP server."},
    )

    playwright_mcp_args: list[str] = field(
        default_factory=lambda: _env_csv(
            "PLAYWRIGHT_MCP_ARGS", ["-y", "@playwright/mcp@latest", "--headless"]
        ),
        metadata={"description": "Arguments passed to Playwright MCP launch command."},
    )

    playwright_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "20000")),
        metadata={
            "description": "Navigation timeout (milliseconds) for Playwright MCP."
        },
    )

    playwright_domain_allowlist: list[str] = field(
        default_factory=lambda: _env_csv("PLAYWRIGHT_DOMAIN_ALLOWLIST", []),
        metadata={
            "description": "Optional host allowlist for Playwright MCP fallback."
        },
    )

    min_credibility: float = field(
        default=0.25,
        metadata={"description": "Minimum credibility score (0-1) to keep a source."},
    )

    min_relevance: float = field(
        default=0.3,
        metadata={
            "description": "Minimum relevance score (0-1) to keep a source. Filters off-topic pages."
        },
    )

    max_retries: int = field(
        default=3,
        metadata={
            "description": "How many times the pipeline may loop back to Intent Parser."
        },
    )

    enable_react_investigator: bool = field(
        default=True,
        metadata={
            "description": "Enable ReAct agent for targeted gap filling without full pipeline re-crawls."
        },
    )

    react_investigator_model: str = field(
        default="meta/meta-llama-3-70b-instruct",
        metadata={
            "description": "Model to use for the ReAct gap-filling agent. Should be strong at tool calling."
        },
    )

    enable_scraperapi: bool = field(
        default=False,
        metadata={
            "description": "Use ScraperAPI to prevent IP bans and handle remote JavaScript rendering."
        },
    )

    enable_searxng_search: bool = field(
        default=True,
        metadata={"description": "Use self-hosted SearXNG instead of commercial APIs."},
    )

    mongo_db_name: str = field(
        default="langgraph_crawler",
        metadata={"description": "MongoDB database name."},
    )

    min_processed_docs: int = field(
        default=3,
        metadata={
            "description": "Minimum processed docs before the pipeline is satisfied."
        },
    )

    enable_chroma_sink: bool = field(
        default=True,
        metadata={
            "description": "If true, write verified sources/entities to ChromaDB."
        },
    )

    chroma_persist_dir: str = field(
        default="./chroma_db",
        metadata={"description": "Directory used by Chroma PersistentClient."},
    )

    chroma_raw_collection: str = field(
        default="crawler_raw_sources",
        metadata={"description": "Chroma collection for verified source documents."},
    )

    chroma_entity_collection: str = field(
        default="crawler_entities",
        metadata={"description": "Chroma collection for extracted entities."},
    )

    chroma_embedding_dim: int = field(
        default=384,
        metadata={"description": "Embedding dimension for local Chroma vectors."},
    )

    neo4j_database: str = field(
        default="neo4j",
        metadata={"description": "Neo4j database name (default or Aura DB name)."},
    )

    enable_insights_node: bool = field(
        default=True,
        metadata={
            "description": "If true, generate explainability insights from structured results and verified sources."
        },
    )

    enable_insights_llm_synthesis: bool = field(
        default=False,
        metadata={
            "description": "If true, use an LLM pass to polish the insights summary text."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Instantiate from a LangGraph RunnableConfig, using defaults for missing keys."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
