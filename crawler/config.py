"""Configurable parameters for the crawler pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class Configuration:
    """Runtime configuration — values can be overridden via RunnableConfig."""

    model: str = field(
        default="meta/meta-llama-3-70b-instruct",
        metadata={"description": "Replicate model identifier used for all LLM calls."},
    )

    max_search_results: int = field(
        default=15,
        metadata={"description": "Max results per Tavily search query."},
    )

    min_word_count: int = field(
        default=100,
        metadata={"description": "Minimum word count to pass the crawler quality gate."},
    )

    min_credibility: float = field(
        default=0.5,
        metadata={"description": "Minimum credibility score (0-1) to keep a source."},
    )

    max_retries: int = field(
        default=2,
        metadata={"description": "How many times the pipeline may loop back to Intent Parser."},
    )

    mongo_db_name: str = field(
        default="langgraph_crawler",
        metadata={"description": "MongoDB database name."},
    )

    min_processed_docs: int = field(
        default=3,
        metadata={"description": "Minimum processed docs before the pipeline is satisfied."},
    )

    enable_chroma_sink: bool = field(
        default=True,
        metadata={"description": "If true, write verified sources/entities to ChromaDB."},
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

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Instantiate from a LangGraph RunnableConfig, using defaults for missing keys."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})