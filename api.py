"""FastAPI server for the WebCrawler Ranking Pipeline.

Endpoints:
  POST /crawl/rank             — Start pipeline (returns job_id)
  GET  /crawl/rank/{id}/stream — SSE stream of live events
  GET  /crawl/rank/{id}        — Poll for final result
  POST /crawl/a2a              — Agent-to-agent validation pipeline
  GET  /health
  GET  /cost-summary

Run:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ── Startup env validation ────────────────────────────────────
def _validate_env() -> None:
    """Fail fast with a clear message if required env vars are wrong."""
    errors = []

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    if not mongo_uri.startswith(("mongodb://", "mongodb+srv://")):
        errors.append(
            f"  MONGO_URI={mongo_uri!r} — must start with 'mongodb://' or 'mongodb+srv://'\n"
            "  (Did you accidentally paste the Neo4j bolt URI here?)"
        )

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    if not neo4j_uri.startswith(("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")):
        errors.append(
            f"  NEO4J_URI={neo4j_uri!r} — must start with 'bolt://' or 'neo4j://'\n"
            "  (Did you accidentally paste the MongoDB URI here?)"
        )

    if errors:
        msg = "❌ Environment variable errors — fix your .env file:\n\n" + "\n\n".join(errors)
        raise RuntimeError(msg)

    print("✅ Env validated — MONGO_URI and NEO4J_URI look correct.")

_validate_env()

from crawler.graph import graph                   # noqa: E402
from crawler.cost_tracker import tracker, new_tracker  # noqa: E402
from crawler.agents import Orchestrator            # noqa: E402


# ── Lifespan: clean up Neo4j driver on shutdown ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        from crawler.neo4j_client import close as close_neo4j
        await close_neo4j()
        print("[API] Neo4j driver closed.")
    except Exception as exc:
        print(f"[API] Neo4j cleanup error: {exc}")


app = FastAPI(title="WebCrawler Ranking Pipeline", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}
_JOB_TTL = timedelta(hours=6)


def _normalize_graph_result(payload: Any) -> dict[str, Any]:
    """Normalize LangGraph output state to a dictionary payload."""
    if isinstance(payload, dict):
        return payload

    if payload is None:
        return {}

    # Pydantic v2/v1 models
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
        if isinstance(data, dict):
            return data

    dict_method = getattr(payload, "dict", None)
    if callable(dict_method):
        data = dict_method()
        if isinstance(data, dict):
            return data

    if is_dataclass(payload):
        data = asdict(payload)
        if isinstance(data, dict):
            return data

    if hasattr(payload, "__dict__"):
        data = {
            k: v
            for k, v in vars(payload).items()
            if not k.startswith("_")
        }
        if data:
            return data

    return {}


def _parse_iso_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _cleanup_stale_jobs() -> None:
    now = datetime.now(timezone.utc)
    stale_ids: list[str] = []
    for job_id, job in _jobs.items():
        completed_at = _parse_iso_ts(job.get("completed_at"))
        started_at = _parse_iso_ts(job.get("started_at"))
        anchor = completed_at or started_at
        if anchor and (now - anchor) > _JOB_TTL:
            stale_ids.append(job_id)

    for job_id in stale_ids:
        _jobs.pop(job_id, None)

    if stale_ids:
        print(f"[API] Cleaned up {len(stale_ids)} stale jobs (ttl={_JOB_TTL}).")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# ── Node metadata for SSE events ─────────────────────────────

_NODE_LABELS: dict[str, str] = {
    "intent_parser":        "Parsing intent & generating search queries",
    "url_discovery":        "Discovering URLs via SearXNG search",
    "url_relevance_filter": "Filtering URLs by query relevance",
    "web_crawler":          "Crawling pages (crawl4ai + Playwright MCP + httpx)",
    "source_verifier":      "Verifying source credibility",
    "mongo_logger":         "Persisting to MongoDB + ChromaDB",
    "entity_extractor":     "Extracting knowledge graph triples",
    "neo4j_ingester":       "Ingesting entities into Neo4j",
    "graph_structurer":     "Querying Neo4j → StructuredResults",
    "insights_generator":   "Generating explainability insights",
    "metrics_evaluator":    "Checking for missing metric gaps",
    "investigator":         "Investigating missing metrics via autonomous agent",
}

# Keys to count in node outputs for richer SSE payloads
_COUNT_KEYS: dict[str, str] = {
    "discovered_urls":      "urls",
    "crawled_docs":         "pages",
    "verified_sources":     "sources",
    "graph_entities":       "entities",
    "structured_results":   "results",
    "insights_items":       "insights",
    "missing_data_targets": "gaps",
    "investigator_findings":"findings",
}


class RankRequest(BaseModel):
    query: str = Field(..., description="The ranking question to research.")
    top_n: int = Field(default=10, ge=1, le=100, description="How many top results to return.")
    max_retries: int = Field(default=2, ge=0, le=5)
    min_credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum relevance score to keep a source.")
    use_searxng_search: bool = Field(default=True, description="Enable local SearXNG metasearch.")
    use_playwright_mcp: bool = Field(default=True, description="Enable Playwright MCP fallback for dynamic pages.")
    use_openclaw: bool = Field(default=False, description="Use OpenClaw instance as crawl backend.")
    openclaw_max_docs_per_query: int = Field(default=150, ge=1, le=2000, description="Max docs per query from OpenClaw.")
    playwright_domain_allowlist: list[str] = Field(
        default_factory=list,
        description="Optional domain allowlist for Playwright MCP fallback.",
    )
    crawler_concurrency: int = Field(default=5, ge=1, le=15)
    playwright_timeout_ms: int = Field(default=20000, ge=1000, le=120000)


class A2ACrawlRequest(BaseModel):
    query: str = Field(..., description="The research query to process.")
    required_metrics: list[str] = Field(..., min_length=1)
    max_rounds: int = Field(default=3, ge=1, le=10)


class A2ACrawlResponse(BaseModel):
    status: str
    message: str
    session_id: str
    query: str
    required_metrics: list[str]
    available_metrics: list[str]
    missing_metrics: list[str]
    entities: list[dict[str, Any]]
    communication_log: list[dict[str, Any]]
    rounds_used: int
    cost_summary: dict[str, Any]


async def _run_rank_pipeline(job_id: str, query: str, config: dict, top_n: int = 10) -> None:
    """
    Full pipeline:
      Phase 1 — LangGraph: crawl → extract → Neo4j → StructuredResults
      Phase 2 — RankingEngine: StructuredResults → weighted composite → RankingResult

    Uses graph.astream(stream_mode="updates") to emit real-time SSE events
    as each graph node completes, instead of fake pre-emitted progress.
    """
    # Create a fresh per-job cost tracker for this async task
    job_tracker = new_tracker()

    job = _jobs[job_id]
    events: list[dict] = job["events"]

    def push(event_type: str, payload: dict) -> None:
        entry = {"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        events.append(entry)

    try:
        # ── Phase 1: LangGraph ────────────────────────────────
        push("phase_start", {"phase": "crawl", "label": "Starting web crawl pipeline"})

        # Stream real node-by-node updates instead of faking progress
        result: dict[str, Any] = {}
        async for updates in graph.astream(
            {"user_query": query},
            config={"configurable": config},
            stream_mode="updates",
        ):
            for node_name, node_output in updates.items():
                if node_name.startswith("__"):
                    continue

                label = _NODE_LABELS.get(node_name, node_name)
                event_payload: dict[str, Any] = {"node": node_name, "label": label}

                # Add counts for richer SSE payloads
                if isinstance(node_output, dict):
                    for out_key, count_label in _COUNT_KEYS.items():
                        if out_key in node_output:
                            val = node_output[out_key]
                            if isinstance(val, list):
                                event_payload["count"] = len(val)
                                event_payload["count_label"] = count_label

                push("node_complete", event_payload)

                # Accumulate outputs — later nodes overwrite earlier ones
                # (correct for replace-semantics state fields)
                if isinstance(node_output, dict):
                    result.update(node_output)

        session_id         = result.get("session_id", "")
        structured_results = result.get("structured_results", [])
        insights_summary   = result.get("insights_summary", "")
        insights_items     = result.get("insights_items", [])
        insights_metadata  = result.get("insights_metadata", {})
        missing_targets    = result.get("missing_data_targets", [])

        if missing_targets:
            push("agent_message", {
                "from": "metrics_evaluator",
                "to": "intent_parser",
                "content": f"Triggered retry loop — {len(missing_targets)} missing metric targets",
                "missing": missing_targets,
            })

        push("phase_complete", {"phase": "crawl", "label": f"Crawl complete — {len(structured_results)} entities ready for ranking", "entities": len(structured_results)})

        # ── Phase 2: Ranking Engine ───────────────────────────
        push("phase_start", {"phase": "ranking", "label": "Starting ranking engine"})

        if not structured_results:
            msg = "No structured results were produced. Ranking cannot proceed. Verify Neo4j connectivity and source coverage."
            push("error", {"message": msg})
            job.update({
                "status": "failed",
                "ranking_result": {},
                "insights": {
                    "summary": insights_summary,
                    "items": insights_items,
                    "metadata": insights_metadata,
                },
                "cost_summary": result.get("cost_summary", job_tracker.get_summary()),
                "error": msg,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            push("done", {"status": "failed", "job_id": job_id})
            return

        # Build feature matrix summary for the log
        all_props: set[str] = set()
        for sr in structured_results:
            all_props.update(sr.properties.keys())
            for rel in sr.relationships:
                if rel.get("type"):
                    all_props.add(rel["type"])

        push("agent_message", {
            "from": "orchestrator",
            "to": "ranking_engine",
            "content": f"Ranking {len(structured_results)} entities — {len(all_props)} feature columns available",
        })

        engine = Orchestrator()
        ranking_result = engine.rank(
            user_query=query,
            session_id=session_id,
            structured_results=structured_results,
        )

        # Apply top_n limit — keep only the top N ranked entities
        if top_n and len(ranking_result.entities) > top_n:
            ranking_result.entities = ranking_result.entities[:top_n]
            ranking_result.total_entities = top_n

        top_name = ranking_result.entities[0].name if ranking_result.entities else "none"
        push("agent_message", {
            "from": "ranking_engine",
            "to": "orchestrator",
            "content": f"Ranking complete — #{1}: {top_name} (score={ranking_result.entities[0].composite_score:.4f})" if ranking_result.entities else "Ranking complete — no entities scored",
            "criteria":  [c.to_dict() for c in ranking_result.criteria],
            "rationale": ranking_result.ranking_rationale,
        })

        push("phase_complete", {"phase": "ranking", "label": f"Ranked {ranking_result.total_entities} entities", "top_entity": top_name})

        job.update({
            "status":         "completed",
            "session_id":     session_id,
            "ranking_result": ranking_result.to_dict(),
            "insights": {
                "summary": insights_summary,
                "items": insights_items,
                "metadata": insights_metadata,
            },
            "cost_summary":   result.get("cost_summary", job_tracker.get_summary()),
            "completed_at":   datetime.now(timezone.utc).isoformat(),
        })
        push("done", {"status": "completed", "job_id": job_id})

    except Exception as exc:
        import traceback
        print(f"[Pipeline] ERROR job={job_id}:\n{traceback.format_exc()}")
        push("error", {"message": str(exc)})
        job.update({"status": "failed", "error": str(exc), "completed_at": datetime.now(timezone.utc).isoformat()})


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/crawl/rank")
async def start_rank(request: RankRequest):
    _cleanup_stale_jobs()

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id": job_id, "status": "running", "query": request.query,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None, "events": [], "ranking_result": None,
        "insights": {"summary": "", "items": [], "metadata": {}},
        "cost_summary": {}, "session_id": "", "error": None,
    }
    config = {
        "max_retries": request.max_retries,
        "min_credibility": request.min_credibility,
        "min_relevance": request.min_relevance,
        "enable_openclaw": request.use_openclaw,
        "openclaw_max_docs_per_query": request.openclaw_max_docs_per_query,
        "enable_searxng_search": request.use_searxng_search,
        "enable_playwright_mcp": request.use_playwright_mcp,
        "playwright_domain_allowlist": request.playwright_domain_allowlist,
        "crawler_concurrency": request.crawler_concurrency,
        "playwright_timeout_ms": request.playwright_timeout_ms,
    }
    asyncio.create_task(_run_rank_pipeline(job_id, request.query, config, top_n=request.top_n))
    return {"job_id": job_id, "status": "running"}


@app.get("/crawl/rank/{job_id}/stream")
async def stream_rank_events(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        job = _jobs[job_id]
        sent_idx = 0
        while True:
            events = job["events"]
            while sent_idx < len(events):
                ev = events[sent_idx]
                yield _sse(ev["type"], ev)
                sent_idx += 1
            if job["status"] in ("completed", "failed"):
                yield _sse("status", {"status": job["status"], "job_id": job_id})
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/crawl/rank/{job_id}")
async def get_rank_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.post("/crawl/a2a", response_model=A2ACrawlResponse)
async def crawl_agent_to_agent(request: A2ACrawlRequest):
    pipeline = Orchestrator(max_a2a_rounds=request.max_rounds)
    result = await pipeline.a2a_run(query=request.query, required_metrics=request.required_metrics)
    return A2ACrawlResponse(**result.to_dict())


@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": sum(1 for j in _jobs.values() if j["status"] == "running")}


@app.get("/cost-summary")
async def cost_summary():
    return tracker.get_summary()
