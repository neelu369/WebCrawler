"""LangGraph StateGraph — full 9-node pipeline.

Flow:
  START → intent_parser → url_discovery
        →(URLs?)→ web_crawler
        →(docs?)→ source_verifier
        →(verified?)→ mongo_logger
        → entity_extractor → neo4j_ingester
        → graph_structurer → metrics_evaluator
        →(gaps & retries left?)→ intent_parser (retry)
        →(done)→ END

The preprocessor node still runs in parallel via the mongo_logger
to keep ChromaDB populated for the StructuringAgent / A2A pipeline.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph

from crawler.config import Configuration
from crawler.state import InputState, OutputState, State

from crawler.nodes.intent_parser import parse_intent
from crawler.nodes.url_discovery import discover_urls
from crawler.nodes.web_crawler import crawl_pages
from crawler.nodes.source_verifier import verify_sources
from crawler.nodes.mongo_logger import log_to_mongo
from crawler.nodes.preprocessor import preprocess
from crawler.nodes.entity_extractor import extract_entities
from crawler.nodes.neo4j_ingester import ingest_to_neo4j
from crawler.nodes.graph_structurer import structure_from_graph
from crawler.nodes.metrics_evaluator import evaluate_metrics


# ── Routing ──────────────────────────────────────────────────

def route_after_discovery(state: State) -> Literal["web_crawler", "__end__"]:
    if state.discovered_urls:
        return "web_crawler"
    print("[Router] No URLs discovered — ending pipeline.")
    return "__end__"


def route_after_crawl(state: State) -> Literal["source_verifier", "__end__"]:
    if state.crawled_docs:
        return "source_verifier"
    print("[Router] No documents crawled — ending pipeline.")
    return "__end__"


def route_after_verify(state: State) -> Literal["mongo_logger", "__end__"]:
    if state.verified_sources:
        return "mongo_logger"
    print("[Router] All sources rejected — ending pipeline.")
    return "__end__"


def route_after_evaluation(state: State) -> Literal["__end__", "intent_parser"]:
    """Retry with targeted queries if metrics are missing and retries remain."""
    if state.missing_data_targets and state.retry_count < state.max_retries:
        print(
            f"[Router] {len(state.missing_data_targets)} gaps found — "
            f"looping to intent_parser ({state.retry_count + 1}/{state.max_retries})."
        )
        return "intent_parser"
    return "__end__"


# ── Combined mongo_logger + preprocessor node ────────────────
# We run preprocessor right after mongo_logger so ChromaDB stays
# populated for the StructuringAgent. The entity_extractor runs
# separately for the Neo4j knowledge graph path.

async def log_and_preprocess(state: State, config=None):
    """Run mongo_logger then preprocessor sequentially.
    Both are optional — if MongoDB is misconfigured or unreachable,
    log a warning and continue with a generated session_id.
    """
    import uuid as _uuid

    # Step 1: mongo_logger (handles its own errors internally)
    try:
        log_result = await log_to_mongo(state, config)
    except Exception as exc:
        print(f"[Graph] mongo_logger failed: {exc}. Continuing with generated session_id.")
        log_result = {
            "raw_doc_ids":    [src.url for src in state.verified_sources],
            "raw_vector_ids": [],
            "session_id":     _uuid.uuid4().hex[:24],
        }

    # Step 2: preprocessor — update state with session_id, then run
    try:
        from dataclasses import replace as dc_replace
        updated    = dc_replace(state, **{k: v for k, v in log_result.items() if hasattr(state, k)})
        pre_result = await preprocess(updated, config)
    except Exception as exc:
        print(f"[Graph] preprocessor failed: {exc}. ChromaDB skipped — pipeline continues.")
        pre_result = {}

    return {**log_result, **pre_result}


# ── Build graph ──────────────────────────────────────────────
workflow = StateGraph(
    State,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)

workflow.add_node("intent_parser", parse_intent)
workflow.add_node("url_discovery", discover_urls)
workflow.add_node("web_crawler", crawl_pages)
workflow.add_node("source_verifier", verify_sources)
workflow.add_node("mongo_logger", log_and_preprocess)   # logs + chroma write
workflow.add_node("entity_extractor", extract_entities)
workflow.add_node("neo4j_ingester", ingest_to_neo4j)
workflow.add_node("graph_structurer", structure_from_graph)
workflow.add_node("metrics_evaluator", evaluate_metrics)

workflow.add_edge("__start__", "intent_parser")
workflow.add_edge("intent_parser", "url_discovery")
workflow.add_conditional_edges("url_discovery", route_after_discovery)
workflow.add_conditional_edges("web_crawler", route_after_crawl)
workflow.add_conditional_edges("source_verifier", route_after_verify)
workflow.add_edge("mongo_logger", "entity_extractor")
workflow.add_edge("entity_extractor", "neo4j_ingester")
workflow.add_edge("neo4j_ingester", "graph_structurer")
workflow.add_edge("graph_structurer", "metrics_evaluator")
workflow.add_conditional_edges("metrics_evaluator", route_after_evaluation)

graph = workflow.compile()
graph.name = "CrawlerPipeline"
