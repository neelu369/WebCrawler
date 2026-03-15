"""Pipeline Orchestrator — single entry point for all agent coordination.

Architecture:
  Orchestrator
  ├── CrawlerAgent       — runs LangGraph pipeline, writes to MongoDB + ChromaDB
  ├── ValidatorAgent     — checks ChromaDB for required metric coverage
  ├── StructuringAgent   — reads ChromaDB → builds clean StructuredTable (LLM)
  ├── RankingAgent       — scores StructuredTable → RankedTable (LLM, ChromaDB path)
  └── RankingEngine      — scores Neo4j StructuredResults directly (primary path)

Two pipeline modes:
  1. rank()    — primary path used by api.py
                 LangGraph → Neo4j StructuredResults → RankingEngine → RankingResult
                 Fast, no extra LLM structuring call needed.

  2. a2a_run() — agent-to-agent validation path used by POST /crawl/a2a
                 CrawlerAgent → ValidatorAgent loop → StructuringAgent → RankingAgent
                 Used when caller specifies required_metrics to validate against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from crawler.graph import graph
from crawler.ranking_engine import RankingEngine, RankingResult
from crawler.models import StructuredResult


# ── Shared message type ───────────────────────────────────────

@dataclass
class AgentMessage:
    round_number: int
    from_agent: str
    to_agent: str
    content: str


# ── Result types ─────────────────────────────────────────────

@dataclass
class A2AResult:
    """Result from the agent-to-agent validation pipeline."""
    status: str                                    # "sufficient" | "no_data_available"
    message: str
    session_id: str
    query: str
    required_metrics: list[str]
    available_metrics: list[str] = field(default_factory=list)
    missing_metrics: list[str]   = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    communication_log: list[AgentMessage] = field(default_factory=list)
    rounds_used: int = 0
    cost_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status":            self.status,
            "message":           self.message,
            "session_id":        self.session_id,
            "query":             self.query,
            "required_metrics":  self.required_metrics,
            "available_metrics": self.available_metrics,
            "missing_metrics":   self.missing_metrics,
            "entities":          self.entities,
            "communication_log": [asdict(m) for m in self.communication_log],
            "rounds_used":       self.rounds_used,
            "cost_summary":      self.cost_summary,
        }


# ── Individual agents (used internally by Orchestrator) ──────

class CrawlerAgent:
    """Runs the LangGraph crawl pipeline and writes results to storage."""

    def __init__(
        self,
        *,
        chroma_persist_dir: str = "./chroma_db",
        chroma_raw_collection: str = "crawler_raw_sources",
        chroma_entity_collection: str = "crawler_entities",
        chroma_embedding_dim: int = 384,
    ) -> None:
        self.chroma_persist_dir       = chroma_persist_dir
        self.chroma_raw_collection    = chroma_raw_collection
        self.chroma_entity_collection = chroma_entity_collection
        self.chroma_embedding_dim     = chroma_embedding_dim

    async def crawl(
        self,
        *,
        base_query: str,
        missing_metrics: list[str] | None = None,
        session_id: str | None = None,
        max_retries: int = 0,
        min_credibility: float = 0.65,
    ) -> dict[str, Any]:
        """
        Run the full LangGraph pipeline.
        Returns the raw state dict from graph.ainvoke().
        """
        if missing_metrics:
            focused = ", ".join(missing_metrics)
            query = (
                f"{base_query}. Focus only on finding explicit values for: {focused}."
            )
        else:
            query = base_query

        payload: dict[str, Any] = {"user_query": query}
        if session_id:
            payload["session_id"] = session_id

        result = await graph.ainvoke(
            payload,
            config={
                "configurable": {
                    "max_retries":             max_retries,
                    "min_credibility":         min_credibility,
                    "enable_chroma_sink":      True,
                    "chroma_persist_dir":      self.chroma_persist_dir,
                    "chroma_raw_collection":   self.chroma_raw_collection,
                    "chroma_entity_collection":self.chroma_entity_collection,
                    "chroma_embedding_dim":    self.chroma_embedding_dim,
                }
            },
        )
        return result


class ValidatorAgent:
    """Reads ChromaDB to check if required metrics are covered."""

    def __init__(
        self,
        *,
        chroma_persist_dir: str = "./chroma_db",
        chroma_entity_collection: str = "crawler_entities",
        chroma_embedding_dim: int = 384,
        max_scan_records: int = 1000,
    ) -> None:
        from crawler.vector import ChromaKnowledgeBase
        self.kb = ChromaKnowledgeBase(
            persist_dir=chroma_persist_dir,
            collection_name=chroma_entity_collection,
            embedding_dimensions=chroma_embedding_dim,
        )
        self.max_scan_records = max_scan_records

    def validate(
        self,
        *,
        session_id: str,
        required_metrics: list[str],
    ) -> dict[str, Any]:
        """
        Returns:
          sufficient        bool
          no_data_available bool
          available_metrics list[str]
          missing_metrics   list[str]
          entity_count      int
        """
        def _norm(s: str) -> str:
            return " ".join(s.strip().lower().split())

        if not session_id:
            return {"sufficient": False, "no_data_available": True,
                    "available_metrics": [], "missing_metrics": required_metrics, "entity_count": 0}

        records = self.kb.get_records(where={"session_id": session_id}, limit=self.max_scan_records)
        entity_records = [r for r in records if (r.get("metadata") or {}).get("record_type") == "entity"]

        if not entity_records:
            return {"sufficient": False, "no_data_available": True,
                    "available_metrics": [], "missing_metrics": required_metrics, "entity_count": 0}

        available_norm: set[str] = set()
        for record in entity_records:
            for raw_key in str((record.get("metadata") or {}).get("metric_keys_csv", "")).split(","):
                norm = _norm(raw_key)
                if norm:
                    available_norm.add(norm)

        required_map = {_norm(m): m.strip() for m in required_metrics if _norm(m)}
        missing = [original for norm, original in required_map.items() if norm not in available_norm]

        return {
            "sufficient":        len(missing) == 0,
            "no_data_available": False,
            "available_metrics": sorted(available_norm),
            "missing_metrics":   missing,
            "entity_count":      len(entity_records),
        }


class StructuringAgent_:
    """Reads ChromaDB entities → clean StructuredTable via LLM."""

    def __init__(self, *, chroma_persist_dir: str = "./chroma_db",
                 chroma_entity_collection: str = "crawler_entities",
                 chroma_embedding_dim: int = 384,
                 model: str = "meta/meta-llama-3-70b-instruct") -> None:
        from crawler.agents.structuring_agent import StructuringAgent as _SA
        self._agent = _SA(
            chroma_persist_dir=chroma_persist_dir,
            chroma_entity_collection=chroma_entity_collection,
            chroma_embedding_dim=chroma_embedding_dim,
            model=model,
        )

    def structure(self, *, session_id: str, user_query: str, round_number: int = 1):
        return self._agent.structure(session_id=session_id, user_query=user_query, round_number=round_number)

    def patch(self, *, table, patch_entities: list[dict[str, Any]]):
        return self._agent.patch(table=table, patch_entities=patch_entities)


class RankingAgent_:
    """Scores a StructuredTable → RankedTable via LLM (ChromaDB path)."""

    def __init__(self, *, model: str = "meta/meta-llama-3-70b-instruct") -> None:
        from crawler.agents.ranking_agent import RankingAgent as _RA
        self._agent = _RA(model=model)

    def rank(self, table):
        return self._agent.rank(table)


# ── Orchestrator ──────────────────────────────────────────────

class Orchestrator:
    """
    Single orchestrator that coordinates all agents.

    Primary path (api.py → POST /crawl/rank):
        result = await orchestrator.rank(query, structured_results)

    A2A validation path (api.py → POST /crawl/a2a):
        result = await orchestrator.a2a_run(query, required_metrics)
    """

    def __init__(
        self,
        *,
        model: str = "meta/meta-llama-3-70b-instruct",
        chroma_persist_dir: str = "./chroma_db",
        chroma_raw_collection: str = "crawler_raw_sources",
        chroma_entity_collection: str = "crawler_entities",
        chroma_embedding_dim: int = 384,
        max_a2a_rounds: int = 3,
        max_patch_rounds: int = 2,
    ) -> None:
        self.model              = model
        self.max_a2a_rounds     = max_a2a_rounds
        self.max_patch_rounds   = max_patch_rounds

        # Shared Chroma config
        self._chroma_cfg = dict(
            chroma_persist_dir=chroma_persist_dir,
            chroma_entity_collection=chroma_entity_collection,
            chroma_embedding_dim=chroma_embedding_dim,
        )

        # Agents
        self.crawler   = CrawlerAgent(
            chroma_persist_dir=chroma_persist_dir,
            chroma_raw_collection=chroma_raw_collection,
            chroma_entity_collection=chroma_entity_collection,
            chroma_embedding_dim=chroma_embedding_dim,
        )
        self.validator  = ValidatorAgent(**self._chroma_cfg)
        self.structurer = StructuringAgent_(model=model, **self._chroma_cfg)
        self.ranker     = RankingAgent_(model=model)
        self.engine     = RankingEngine(model=model)

    # ── Primary path ─────────────────────────────────────────

    def rank(
        self,
        *,
        user_query: str,
        session_id: str,
        structured_results: list[StructuredResult],
    ) -> RankingResult:
        """
        Score and rank Neo4j StructuredResults.
        Called directly from api.py after LangGraph pipeline completes.
        No extra LLM structuring call — Neo4j data is already clean.
        """
        return self.engine.rank(
            user_query=user_query,
            session_id=session_id,
            structured_results=structured_results,
        )

    # ── A2A validation path ───────────────────────────────────

    async def a2a_run(
        self,
        *,
        query: str,
        required_metrics: list[str],
    ) -> A2AResult:
        """
        Agent-to-agent loop:
          Round 1..N:
            CrawlerAgent crawls → ValidatorAgent checks metrics
            If sufficient → StructuringAgent + RankingAgent → return
            If not → targeted recrawl with missing metrics
          After max rounds → return no_data_available
        """
        required = [m.strip() for m in required_metrics if m.strip()]
        if not required:
            return A2AResult(status="no_data_available", message="no required metrics specified",
                             session_id="", query=query, required_metrics=[])

        session_id = ""
        missing    = list(required)
        log: list[AgentMessage] = []
        latest_entities: list[dict[str, Any]] = []
        cost_summary: dict[str, Any] = {}

        for round_num in range(1, self.max_a2a_rounds + 1):

            # ── Orchestrator → CrawlerAgent ──────────────────
            log.append(AgentMessage(
                round_number=round_num, from_agent="orchestrator", to_agent="crawler_agent",
                content=f"Crawl round {round_num}: fetch data for [{', '.join(missing)}]",
            ))

            result      = await self.crawler.crawl(base_query=query, missing_metrics=missing,
                                                    session_id=session_id or None)
            session_id  = str(result.get("session_id") or session_id or "")
            cost_summary= dict(result.get("cost_summary") or {})

            entities_raw = result.get("extracted_entities", [])
            latest_entities = [
                e.model_dump() if hasattr(e, "model_dump") else dict(e)
                for e in entities_raw
            ]

            # ── CrawlerAgent → ValidatorAgent ────────────────
            log.append(AgentMessage(
                round_number=round_num, from_agent="crawler_agent", to_agent="validator_agent",
                content=f"Crawl complete: {len(latest_entities)} entities, session={session_id}",
            ))

            validation = self.validator.validate(session_id=session_id, required_metrics=required)

            if validation["no_data_available"]:
                log.append(AgentMessage(
                    round_number=round_num, from_agent="validator_agent", to_agent="orchestrator",
                    content="No entity data found in vector store.",
                ))
                return A2AResult(
                    status="no_data_available", message="no data available",
                    session_id=session_id, query=query, required_metrics=required,
                    available_metrics=validation["available_metrics"],
                    missing_metrics=validation["missing_metrics"],
                    entities=latest_entities, communication_log=log,
                    rounds_used=round_num, cost_summary=cost_summary,
                )

            if validation["sufficient"]:
                log.append(AgentMessage(
                    round_number=round_num, from_agent="validator_agent", to_agent="orchestrator",
                    content=f"All {len(required)} required metrics found. Proceeding to rank.",
                ))
                return A2AResult(
                    status="sufficient", message="sufficient data available",
                    session_id=session_id, query=query, required_metrics=required,
                    available_metrics=validation["available_metrics"], missing_metrics=[],
                    entities=latest_entities, communication_log=log,
                    rounds_used=round_num, cost_summary=cost_summary,
                )

            # Still missing — feed back to crawler next round
            missing = validation["missing_metrics"]
            log.append(AgentMessage(
                round_number=round_num, from_agent="validator_agent", to_agent="orchestrator",
                content=f"Still missing {len(missing)} metrics: [{', '.join(missing)}]. Retrying.",
            ))

        # Max rounds exhausted
        log.append(AgentMessage(
            round_number=self.max_a2a_rounds, from_agent="orchestrator", to_agent="orchestrator",
            content=f"Max rounds ({self.max_a2a_rounds}) reached. Returning with partial data.",
        ))
        return A2AResult(
            status="no_data_available", message="no data available after max rounds",
            session_id=session_id, query=query, required_metrics=required,
            missing_metrics=missing, entities=latest_entities,
            communication_log=log, rounds_used=self.max_a2a_rounds, cost_summary=cost_summary,
        )


# ── Backwards-compat alias ────────────────────────────────────
# api.py still imports AgentToAgentPipeline — point it at Orchestrator
class AgentToAgentPipeline(Orchestrator):
    """Backwards-compatible alias for Orchestrator."""
    async def run(self, *, query: str, required_metrics: list[str]) -> A2AResult:
        return await self.a2a_run(query=query, required_metrics=required_metrics)
