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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except Exception:
        return default


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
        default=75,
        metadata={"description": "Max results per SearXNG search query."},
    )

    searxng_pages: int = field(
        default=4,
        metadata={"description": "How many SearXNG result pages to fetch per query."},
    )

    max_searxng_pages: int = field(
        default=20,
        metadata={"description": "Safety upper bound for SearXNG pages per query."},
    )

    min_url_relevance_score: float = field(
        default=0.08,
        metadata={
            "description": "Minimum lexical overlap to keep a URL heuristics-only."
        },
    )

    url_filter_min_keep: int = field(
        default=200,
        metadata={
            "description": "Minimum URLs to keep after relevance filtering (rescues borderline URLs for high recall)."
        },
    )

    max_intent_queries: int = field(
        default=80,
        metadata={"description": "Maximum unique intent queries to keep on first pass."},
    )

    max_retry_queries: int = field(
        default=80,
        metadata={"description": "Maximum unique retry queries to keep during iterative enrichment."},
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
        default=40,
        metadata={
            "description": "Minimum word count to pass the crawler quality gate."
        },
    )

    crawler_concurrency: int = field(
        default=10,
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
        default=0.15,
        metadata={"description": "Minimum credibility score (0-1) to keep a source."},
    )

    min_relevance: float = field(
        default=0.15,
        metadata={
            "description": "Minimum relevance score (0-1) to keep a source. Filters off-topic pages."
        },
    )

    max_retries: int = field(
        default=5,
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

    enable_openclaw: bool = field(
        default=False,
        metadata={
            "description": "If true, fetch crawl documents from an OpenClaw instance instead of direct URL crawling."
        },
    )

    openclaw_base_url: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_BASE_URL", "http://localhost:3100"),
        metadata={"description": "Base URL for the OpenClaw API endpoint."},
    )

    openclaw_search_path: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_SEARCH_PATH", "/api/v1/search"),
        metadata={"description": "Relative API path used for OpenClaw search requests."},
    )

    openclaw_mode: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_MODE", "auto"),
        metadata={
            "description": "OpenClaw request mode: auto, search, or gateway (session-based)."
        },
    )

    openclaw_session_key: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY", "agent:main"),
        metadata={"description": "Session key used for gateway session endpoints."},
    )

    openclaw_api_key: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_API_KEY", ""),
        metadata={"description": "Optional API key used for OpenClaw Authorization header."},
    )

    openclaw_max_docs_per_query: int = field(
        default_factory=lambda: int(os.getenv("OPENCLAW_MAX_DOCS_PER_QUERY", "150")),
        metadata={"description": "Maximum OpenClaw documents to retrieve per query."},
    )

    openclaw_timeout_s: int = field(
        default_factory=lambda: _env_int("OPENCLAW_TIMEOUT_S", 45),
        metadata={"description": "Timeout in seconds for OpenClaw API calls."},
    )

    openclaw_enable_cli_fallback: bool = field(
        default_factory=lambda: _env_bool("OPENCLAW_ENABLE_CLI_FALLBACK", True),
        metadata={
            "description": "If true, fall back to OpenClaw CLI agent calls when REST endpoint is unavailable."
        },
    )

    openclaw_cli_timeout_s: int = field(
        default_factory=lambda: _env_int("OPENCLAW_CLI_TIMEOUT_S", 90),
        metadata={"description": "Timeout in seconds for OpenClaw CLI fallback calls."},
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
