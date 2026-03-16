"""Neo4j Ingester node — MERGEs extracted entities and triples into Neo4j.

Neo4j is REQUIRED for the ranking pipeline (structured results come from it).
If Neo4j is unreachable this node logs a clear error and returns early —
the pipeline will complete but produce no ranking results.
"""
from __future__ import annotations
import re
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from crawler.config import Configuration
from crawler.neo4j_client import get_driver, check_neo4j_available
from crawler.state import State

_KNOWN_PREDICATES = {
    "LOCATED_IN","HEADQUARTERED_IN","HAS_FUNDING","FUNDING_AMOUNT","FOUNDED_IN",
    "FOUNDED_BY","SUPPORTS_INDUSTRY","OPERATES_IN","HAS_FEATURE","HAS_STRENGTH",
    "HAS_WEAKNESS","BUILT_BY","DEVELOPED_BY","MAINTAINED_BY","SUPPORTS_LANGUAGE",
    "WRITTEN_IN","HAS_PRICING","PRICING_MODEL","HAS_COMMUNITY_SIZE","RELATED_TO",
    "COMPETES_WITH","INTEGRATES_WITH","HAS_USE_CASE","BEST_FOR","HAS_DIFFICULTY_LEVEL",
    "IS_TYPE_OF","DIRECTED_BY","RELEASED_IN","HAS_IMDB_RATING","HAS_BOX_OFFICE",
    "HAS_GENRE","HAS_RUNTIME","HAS_RATING","WON_AWARD","PRODUCED_BY","SET_IN",
    "STARRING","HAS_ROTTEN_TOMATOES","HAS_METACRITIC","HAS_BUDGET",
    # Exam-specific
    "HAS_PASS_RATE","HAS_APPLICANTS","HAS_SEATS","HAS_CUTOFF","CONDUCTED_BY",
    "HAS_DIFFICULTY","HELD_IN","REQUIRES_ELIGIBILITY","RANKED_AT","HAS_SCORE",
    "HAS_RANK","HAS_SYLLABUS","HAS_EXAM_PATTERN","HAS_ELIGIBILITY",
    # General comparative
    "HAS_PROPERTY","HAS_SCORE","HAS_RANK","RANKED_AT",
}
_SAFE_REL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

def _safe_predicate(pred: str) -> tuple[str, bool]:
    n = pred.upper().replace(" ", "_")
    if n in _KNOWN_PREDICATES: return n, True
    if _SAFE_REL_RE.match(n) and len(n) <= 40: return n, True
    return "HAS_PROPERTY", False

_MERGE_ENTITY = (
    "MERGE (e:Entity {normalized_name: $norm_name}) "
    "ON CREATE SET e.name=$name, e.entity_type=$entity_type, e.description=$description, "
    "e.priority_score=$priority_score, e.session_id=$session_id, e.created_at=datetime() "
    "ON MATCH SET "
    "e.description=CASE WHEN size(e.description)<size($description) THEN $description ELSE e.description END, "
    "e.priority_score=CASE WHEN e.priority_score<$priority_score THEN $priority_score ELSE e.priority_score END, "
    "e.updated_at=datetime()"
)
_MERGE_SOURCE = "MERGE (s:Source {url: $url}) ON CREATE SET s.created_at=datetime()"
_LINK_SRC     = "MATCH (e:Entity {normalized_name:$norm_name}) MATCH (s:Source {url:$url}) MERGE (e)-[:MENTIONED_IN]->(s)"
_MERGE_ATTR   = "MERGE (a:Attribute {normalized_name:$attr_norm}) ON CREATE SET a.name=$attr_name"

def _rel_query(rel_type: str, is_known: bool) -> str:
    if is_known:
        return (
            f"MATCH (e:Entity {{normalized_name:$norm_name}}) "
            f"MATCH (a:Attribute {{normalized_name:$attr_norm}}) "
            f"MERGE (e)-[r:{rel_type} {{source:$source_url}}]->(a) "
            "ON CREATE SET r.confidence=$confidence, r.evidence=$evidence "
            "ON MATCH SET r.evidence=$evidence"
        )
    return (
        "MATCH (e:Entity {normalized_name:$norm_name}) "
        "MATCH (a:Attribute {normalized_name:$attr_norm}) "
        "MERGE (e)-[r:HAS_PROPERTY {predicate:$original_pred, source:$source_url}]->(a) "
        "ON CREATE SET r.confidence=$confidence, r.evidence=$evidence "
        "ON MATCH SET r.evidence=$evidence"
    )


async def ingest_to_neo4j(state: State, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    entities = state.graph_entities

    if not entities:
        print("[Neo4jIngester] No entities to ingest.")
        return {}

    # Check connectivity before attempting any writes
    available = await check_neo4j_available()
    if not available:
        print(
            f"[Neo4jIngester] ⚠️  Neo4j unreachable — skipping ingestion of {len(entities)} entities.\n"
            "[Neo4jIngester]    The ranking step will produce no results without Neo4j.\n"
            "[Neo4jIngester]    Fix: Start Neo4j Desktop → click ▶ on your instance, then rerun."
        )
        return {}

    db_name     = configuration.neo4j_database
    driver      = get_driver()
    total_nodes = total_rels = 0

    try:
        async with driver.session(database=db_name) as session:
            for ge in entities:
                norm_name = ge.name.lower().strip()
                try:
                    await session.run(_MERGE_ENTITY, {
                        "norm_name": norm_name, "name": ge.name,
                        "entity_type": ge.entity_type, "description": ge.description,
                        "priority_score": ge.priority_score, "session_id": state.session_id,
                    })
                    total_nodes += 1

                    source_urls = [u.strip() for u in ge.source_url.split(",") if u.strip()]
                    for url in source_urls:
                        await session.run(_MERGE_SOURCE, {"url": url})
                        await session.run(_LINK_SRC, {"norm_name": norm_name, "url": url})

                    for triple in ge.triples:
                        attr_name = triple.object.strip()
                        attr_norm = attr_name.lower()
                        await session.run(_MERGE_ATTR, {"attr_norm": attr_norm, "attr_name": attr_name})
                        rel_type, is_known = _safe_predicate(triple.predicate)
                        await session.run(
                            _rel_query(rel_type, is_known),
                            {
                                "norm_name":    norm_name,
                                "attr_norm":    attr_norm,
                                "source_url":   triple.source_url or (source_urls[0] if source_urls else ""),
                                "confidence":   triple.confidence,
                                "evidence":     triple.evidence_snippet,
                                "original_pred":triple.predicate,
                            }
                        )
                        total_rels += 1

                except Exception as exc:
                    print(f"[Neo4jIngester] Failed to ingest entity {ge.name!r}: {exc}")

    except Exception as exc:
        print(f"[Neo4jIngester] Session error: {exc}")
        return {}

    print(f"[Neo4jIngester] ✅ {total_nodes} nodes, {total_rels} relationships written to Neo4j")
    return {}