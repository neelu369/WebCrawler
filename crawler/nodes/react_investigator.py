"""ReAct Agent Investigator node — targeted gap-filling for missing metrics.

Uses LangGraph's prebuilt ReAct agent with specific tools to:
1. Search the web for specific missing metrics.
2. Scrape promising URLs.
3. Extract and save findings directly to Neo4j.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from crawler.config import Configuration
from crawler.state import State
from crawler.utils import clean_text


# ── Global state for tools ───────────────────────────────────
# LangChain @tool decorators don't easily accept runtime dependencies 
# without complex binding. We use module-level state injected per-run.
_active_session_id = ""
_active_db_name = ""


def _make_skip_finding(reason: str) -> dict[str, Any]:
    """Emit a structured skip record so API/UI can surface investigator behavior."""
    import time

    return {
        "status": "skipped",
        "reason": reason,
        "timestamp": time.time(),
    }


@tool
async def search_web(query: str) -> str:
    """Search the web for specific information.
    Use targeted queries like 'UPSC pass rate 2025' rather than broad ones.
    Returns the top 3 results text.
    """
    base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    url = f"{base_url.rstrip('/')}/search"
    params = {"q": query, "format": "json"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])[:3]
                    output = []
                    for i, res in enumerate(results, 1):
                        output.append(
                            f"Result {i}:\n"
                            f"Title: {res.get('title', '')}\n"
                            f"URL: {res.get('url', '')}\n"
                            f"Preview: {res.get('content', '')}\n"
                        )
                    return "\n".join(output) if output else "No results found."
                return f"Search error (HTTP {response.status})"
    except Exception as exc:
        return f"Search failed: {str(exc)}"


@tool
async def scrape_page(url: str) -> str:
    """Fetch a webpage and return its main text content.
    Call this after search_web returns a promising URL to find the exact metric.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as response:
                if response.status != 200:
                    return f"Failed to fetch page (HTTP {response.status})"
                
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                
                # Remove junk
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()
                
                text = clean_text(soup.get_text(separator=" "))
                
                # Limit to 8000 chars to save context window
                return text[:8000] + ("..." if len(text) > 8000 else "")
    except Exception as exc:
        return f"Scrape failed: {str(exc)}"


_SAFE_REL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

@tool
async def save_finding(entity_name: str, metric_name: str, value: str, source_url: str) -> str:
    """Save a discovered data point to the knowledge graph.
    Call this when you have successfully found a missing metric.
    
    Args:
        entity_name: The target entity (e.g., "UPSC")
        metric_name: The metric you were asked to find (e.g., "pass_rate")
        value: The extracted value (e.g., "0.2%")
        source_url: The URL where you found this data
    """
    from crawler.neo4j_client import get_driver, check_neo4j_available
    
    available = await check_neo4j_available()
    if not available:
        return "Error: Neo4j database is unreachable. Cannot save."

    driver = get_driver()
    
    entity_norm = entity_name.lower().strip()
    attr_name = str(value).strip()
    attr_norm = attr_name.lower()
    metric = metric_name.strip()

    # Convert metric name to Neo4j relationship type
    predicate = metric.upper().replace(" ", "_")
    if not predicate.startswith("HAS_"):
        predicate = f"HAS_{predicate}"
    if not _SAFE_REL_RE.match(predicate) or len(predicate) > 40:
        predicate = "HAS_PROPERTY"

    merge_attr = (
        "MERGE (a:Attribute {normalized_name: $attr_norm}) "
        "ON CREATE SET a.name = $attr_name"
    )
    rel_query = f"""
        MATCH (e:Entity)
        WHERE coalesce(e.normalized_name, e.norm_name) = $norm_name
        MATCH (a:Attribute)
        WHERE coalesce(a.normalized_name, a.norm_name) = $attr_norm
        MERGE (e)-[r:{predicate}]->(a)
        ON CREATE SET r.original_pred = $original_pred,
                      r.source = $source_url,
                      r.confidence = 0.8,
                      r.evidence = "Found by ReAct Investigator"
        RETURN id(r)
    """

    try:
        async with driver.session(database=_active_db_name) as session:
            # Note: We don't MERGE the Entity here because it should already exist
            # from the main pipeline. We just link to it.
            await session.run(merge_attr, {"attr_norm": attr_norm, "attr_name": attr_name})
            result = await session.run(
                rel_query, 
                {
                    "norm_name": entity_norm,
                    "attr_norm": attr_norm,
                    "original_pred": metric_name,
                    "source_url": source_url
                }
            )
            records = [r.data() async for r in result]
            if not records:
                return f"Warning: Entity '{entity_name}' not found in database. Could not save."
            return f"Successfully saved {metric_name} = {value} for {entity_name}."
    except Exception as exc:
        return f"Database error: {str(exc)}"


# ── Main Node ────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are a precise data investigator agent.
Your mission is to find missing metrics for entities in a knowledge graph.

Missing data to find:
{missing_data}

For each missing item:
1. Search the web with a targeted query (e.g. 'UPSC pass rate 2025 statistics') using `search_web`.
2. Evaluate search previews. If needed, use `scrape_page` to read the full content of the most promising URL.
3. Once you find the exact metric value, use `save_finding` to write it to the database.

CRITICAL RULES:
- Do NOT hallucinate data. If you cannot find a metric after a couple of search attempts, skip it.
- Keep your tool calls focused and step-by-step.
- After saving a finding, move to the next missing item immediately.
- If you have successfully addressed (or exhausted) all missing items, return a final summary of what you found.
"""


async def run_react_investigator(state: State, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """LangGraph node that runs the ReAct gap-filling agent."""
    configuration = Configuration.from_runnable_config(config)
    
    if not configuration.enable_react_investigator:
        print("[ReActInvestigator] Agent disabled in config. Skipping.")
        return {
            "retry_count": state.retry_count + 1,
            "investigator_findings": [_make_skip_finding("disabled_in_config")],
        }
        
    gaps = state.missing_data_targets
    if not gaps:
        print("[ReActInvestigator] No missing_data_targets. Skipping.")
        return {
            "retry_count": state.retry_count + 1,
            "investigator_findings": [_make_skip_finding("no_missing_targets")],
        }

    # Set globals for the tools
    global _active_session_id, _active_db_name
    _active_session_id = state.session_id
    _active_db_name = configuration.neo4j_database

    budget = max(3, min(15, len(gaps) * 3))
    print(f"\n[ReActInvestigator] Triggered to fix {len(gaps)} gaps (budget: {budget} tool steps)")
    for gap in gaps[:5]:
        print(f"  - {gap}")
    
    import time
    
    api_key = os.getenv("REPLICATE_API_TOKEN")
    if not api_key:
        print("[ReActInvestigator] Error: REPLICATE_API_TOKEN not found.")
        return {
            "retry_count": state.retry_count + 1,
            "investigator_findings": [_make_skip_finding("missing_replicate_api_token")],
        }

    # Initialize the LLM (using LangChain's OpenAI wrapper pointing to Replicate's compatibility endpoint)
    llm = ChatOpenAI(
        base_url="https://api.replicate.com/v1",
        api_key=api_key,
        model=configuration.react_investigator_model,
        temperature=0.1,
        max_tokens=1024
    )

    tools = [search_web, scrape_page, save_finding]
    
    try:
        # Build the dynamic system prompt
        formatted_gaps = "\n".join(f"- {gap}" for gap in gaps)
        sys_msg = SystemMessage(content=_SYSTEM_PROMPT.format(missing_data=formatted_gaps))
        
        # Instantiate the agent
        agent = create_react_agent(
            llm,
            tools=tools,
            state_modifier=sys_msg,
        )

        # Run the agent (use recursion_limit for safety)
        inputs = {"messages": [HumanMessage(content="Start investigating the missing data targets.")]}
        result = await agent.ainvoke(
            inputs, 
            config={"recursion_limit": budget + 2} # +2 for intro/outro
        )
        
        # Extract the final response
        messages = result.get("messages", [])
        if messages:
            final_msg = messages[-1].content
            print(f"[ReActInvestigator] Agent finished: {final_msg[:200]}...")
            
            # Log finding to state so SSE can pick it up
            # We don't strictly need to pass the raw data in state since save_finding writes to DB,
            # but tracking it is good for the UI.
            findings = [
                {
                    "status": "completed",
                    "reason": "ran",
                    "agent_summary": final_msg,
                    "timestamp": time.time(),
                }
            ]
            
            return {
                "investigator_findings": findings,
                "retry_count": state.retry_count + 1
            }
            
    except Exception as exc:
        print(f"[ReActInvestigator] Agent execution failed: {exc}")
        return {
            "retry_count": state.retry_count + 1,
            "investigator_findings": [
                {
                    "status": "failed",
                    "reason": "agent_execution_failed",
                    "error": str(exc),
                    "timestamp": time.time(),
                }
            ],
        }

    return {
        "retry_count": state.retry_count + 1,
        "investigator_findings": [_make_skip_finding("no_agent_messages")],
    }
