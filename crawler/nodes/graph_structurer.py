"""Graph Structurer node — queries Neo4j → StructuredResult objects."""
from __future__ import annotations
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import StructuredResult, CitationMetadata
from crawler.neo4j_client import get_driver
from crawler.state import State

_QUERY = """\
MATCH (e:Entity) WHERE e.session_id=$session_id
OPTIONAL MATCH (e)-[r]->(t) WHERE NOT type(r)='MENTIONED_IN'
OPTIONAL MATCH (e)-[:MENTIONED_IN]->(s:Source)
RETURN e.name AS name, e.entity_type AS entity_type, e.description AS description,
       e.priority_score AS priority_score,
       collect(DISTINCT {type:type(r),predicate:CASE WHEN 'predicate' IN keys(r) THEN r.predicate ELSE type(r) END,value:t.name,evidence:r.evidence,source:r.source}) AS relationships,
       collect(DISTINCT s.url) AS source_urls
ORDER BY e.priority_score DESC
"""

_PROPERTY_PREDICATES = {
    # General
    "LOCATED_IN","HEADQUARTERED_IN","HAS_FUNDING","FUNDING_AMOUNT","FOUNDED_IN",
    "HAS_PRICING","PRICING_MODEL","HAS_COMMUNITY_SIZE","HAS_DIFFICULTY_LEVEL",
    "WRITTEN_IN","SUPPORTS_LANGUAGE","HAS_PROPERTY",
    # Movie-specific
    "DIRECTED_BY","RELEASED_IN","HAS_IMDB_RATING","HAS_BOX_OFFICE","HAS_GENRE",
    "HAS_RUNTIME","HAS_RATING","WON_AWARD","PRODUCED_BY","SET_IN","STARRING",
    "HAS_ROTTEN_TOMATOES","HAS_METACRITIC","HAS_BUDGET",
}

def _categorise(raw_rels):
    properties, relationships, citations = {}, [], {}
    for rel in raw_rels:
        if not rel or not rel.get("value"): continue
        rel_type = rel.get("type", "")
        predicate = rel.get("predicate") or rel_type
        value = str(rel["value"])
        display_key = predicate.replace("_", " ").title()
        citation = CitationMetadata(value=value, evidence=rel.get("evidence",""), source=rel.get("source",""))
        if rel_type in _PROPERTY_PREDICATES:
            if display_key in properties:
                if value.lower() != properties[display_key].lower() and value not in properties[display_key]:
                    properties[display_key] = f"{properties[display_key]}, {value}"
            else:
                properties[display_key] = value
            citations[display_key] = citation
        else:
            relationships.append({"type": display_key, "target": value})
            citations[f"{display_key}:{value}"] = citation
    return properties, relationships, citations

async def structure_from_graph(state: State, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    driver = get_driver()
    results = []
    async with driver.session(database=configuration.neo4j_database) as session:
        cursor = await session.run(_QUERY, {"session_id": state.session_id})
        records = [r.data() async for r in cursor]
    for record in records:
        name = record.get("name","")
        if not name: continue
        properties, relationships, citations = _categorise(record.get("relationships",[]))
        results.append(StructuredResult(name=name, entity_type=record.get("entity_type","Entity"), description=record.get("description",""), properties=properties, relationships=relationships, citations=citations, source_urls=[u for u in record.get("source_urls",[]) if u], priority_score=float(record.get("priority_score",0.0))))
    cost_summary = tracker.get_summary()
    print(f"[GraphStructurer] {len(results)} structured results from Neo4j")
    tracker.print_report()
    return {"structured_results": results, "cost_summary": cost_summary}
