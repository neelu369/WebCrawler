import sys
import os
import asyncio
import pandas as pd
import streamlit as st

# Ensure we can import crawler module from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crawler.graph import graph

st.set_page_config(page_title="WebCrawler Dashboard", layout="wide")

st.title("🕷️ WebCrawler Data Collection Team")
st.markdown("Enter a research query below to crawl, extract entities, patch missing metrics via the ReAct Investigator, and build a local dataset. **(Ranking is bypassed in this prototype)**.")

query = st.text_input("Research Query", placeholder="e.g. top 10 competitive exams in India")

# Create a mapping for user-friendly node names
NODE_LABELS = {
    "intent_parser": "Parsing Intent & Generating Search Queries",
    "url_discovery": "Discovering URLs (SearXNG)",
    "url_relevance_filter": "Filtering URL Relevance",
    "web_crawler": "Crawling Pages (httpx + ScraperAPI)",
    "source_verifier": "Verifying Source Credibility",
    "mongo_logger": "Persisting Raw Docs",
    "entity_extractor": "Extracting Graph Entities",
    "neo4j_ingester": "Ingesting to Neo4j Graph DB",
    "graph_structurer": "Querying Structured Results from Graph",
    "insights_generator": "Generating Explainability Insights",
    "metrics_evaluator": "Evaluating Entity Completeness",
    "investigator": "ReAct Agent: Patching Missing Data via Web Search"
}

async def run_pipeline(user_query: str, status_container):
    """Run the LangGraph pipeline and update the status container."""
    final_state = {}
    
    async for event in graph.astream({"user_query": user_query}, stream_mode="updates"):
        for node_name, state_update in event.items():
            # Merge updates into final_state
            final_state.update(state_update)
            
            label = NODE_LABELS.get(node_name, node_name)
            context = ""
            
            if node_name == "metrics_evaluator" and state_update.get("missing_data_targets"):
                gaps = len(state_update["missing_data_targets"])
                context = f" — Found {gaps} missing metrics. Routing to Investigator."
            elif node_name == "investigator" and state_update.get("investigator_findings"):
                findings = len(state_update["investigator_findings"])
                context = f" — Agent found {findings} answers."
                
            status_container.write(f"✅ **{label}**{context}")
            
    return final_state

if st.button("Start Data Collection", type="primary"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        # Before we run, check API token
        if not os.getenv("REPLICATE_API_TOKEN"):
            st.error("⚠️ **REPLICATE_API_TOKEN** environment variable is missing. Please add it to your .env file or environment.")
            st.stop()
            
        with st.status("Initializing WebCrawler Pipeline...", expanded=True) as status:
            try:
                final_results = asyncio.run(run_pipeline(query, status))
                status.update(label="Pipeline Complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label=f"Pipeline Failed: {e}", state="error", expanded=True)
                st.exception(e)
                st.stop()
                
        # ── Pipeline Done: Process the Data ─────────────────────
        st.success("Successfully built dataset!")
        
        # We need structured_results from the final state output
        results = final_results.get("structured_results", [])
        
        if not results:
            st.warning("No structured data was extracted. The query might have been too narrow or the sources lacked tabular facts.")
        else:
            # Flatten into a Pandas DataFrame
            rows = []
            for r in results:
                row = {
                    "Entity Name": getattr(r, "name", ""),
                    "Type": getattr(r, "entity_type", ""),
                    "Description": getattr(r, "description", "")[:100] + "..." if getattr(r, "description", "") else "",
                    "Priority": getattr(r, "priority_score", 0.0),
                }
                # Add all properties dynamically as columns
                if hasattr(r, "properties") and r.properties:
                    row.update(r.properties)
                rows.append(row)
                
            df = pd.DataFrame(rows)
            
            st.subheader("📊 Extracted Dataset")
            st.dataframe(df, use_container_width=True)
            
            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"webcrawler_dataset_{query.replace(' ', '_')[:30]}.csv",
                mime="text/csv",
            )
            
            with st.expander("Show Operation Costs"):
                cost = final_results.get("cost_summary", {})
                if cost:
                    st.json(cost)
                else:
                    st.write("No cost data recorded.")
