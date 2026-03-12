"""Streamlit dashboard for the LangGraph Crawler Pipeline.

Provides an interactive UI for:
  - Submitting research queries
  - Real-time pipeline status
  - Viewing processed documents
  - Cost breakdown dashboard

Run:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _merge_csv_values(existing: str, incoming: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for blob in [existing, incoming]:
        for part in str(blob).split("|"):
            value = part.strip()
            if not value:
                continue
            norm = value.lower()
            if norm in seen:
                continue
            seen.add(norm)
            values.append(value)
    return " | ".join(values)


def _merge_url_values(existing: str, incoming: str) -> str:
    urls: list[str] = []
    seen: set[str] = set()
    for blob in [existing, incoming]:
        for part in str(blob).split(","):
            value = part.strip()
            if not value:
                continue
            if value in seen:
                continue
            seen.add(value)
            urls.append(value)
    return ", ".join(urls)


# ── Helper: Fetch History ────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_history():
    """Fetch past sessions and their processed documents from MongoDB."""
    import os
    from motor.motor_asyncio import AsyncIOMotorClient

    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "langgraph_crawler")
    client = AsyncIOMotorClient(uri)
    db = client[db_name]

    async def _fetch():
        # Get last 20 sessions, sorted newest first
        cursor = db.sessions.find().sort("created_at", -1).limit(20)
        sessions = await cursor.to_list(length=20)

        history_data = []
        for sess in sessions:
            sess_id = str(sess["_id"])
            query = sess.get("user_query", "Unknown Query")
            date = sess.get("created_at")
            if date:
                date_str = date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = "Unknown"

            # Count entities for this session
            entity_count = await db.extracted_entities.count_documents(
                {"session_id": sess_id}
            )
            a2a_runs = await db.a2a_runs.count_documents({"session_id": sess_id})

            history_data.append(
                {
                    "Session ID": sess_id,
                    "Date": date_str,
                    "Query": query,
                    "Entities Found": entity_count,
                    "A2A Runs": a2a_runs,
                }
            )

        return history_data

    return run_async(_fetch())


def fetch_session_docs(session_id: str):
    """Fetch processed docs for a specific session."""
    import os
    from motor.motor_asyncio import AsyncIOMotorClient

    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "langgraph_crawler")
    client = AsyncIOMotorClient(uri)
    db = client[db_name]

    async def _fetch():
        cursor = db.extracted_entities.find({"session_id": session_id}).sort(
            "priority_score", -1
        )
        entities = await cursor.to_list(length=100)

        table_data_map: dict[str, dict[str, Any]] = {}
        for ent in entities:
            entity_name = str(ent.get("name", "Unknown")).strip() or "Unknown"
            key = _normalize_name(entity_name)

            row = table_data_map.get(key)
            if row is None:
                row = {
                    "Priority": round(float(ent.get("priority_score", 0.0)), 2),
                    "Entity Name": entity_name,
                    "Source URL": ent.get("source_url", ""),
                    "Description": ent.get("description", ""),
                }
                table_data_map[key] = row
            else:
                row["Priority"] = max(
                    float(row.get("Priority", 0.0)),
                    float(ent.get("priority_score", 0.0)),
                )
                existing_desc = str(row.get("Description", ""))
                incoming_desc = str(ent.get("description", ""))
                if len(incoming_desc) > len(existing_desc):
                    row["Description"] = incoming_desc
                row["Source URL"] = _merge_url_values(
                    str(row.get("Source URL", "")),
                    str(ent.get("source_url", "")),
                )

            # Flatten metrics into the row
            metrics = ent.get("metrics", {})
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    key_name = str(k)
                    if key_name in row:
                        row[key_name] = _merge_csv_values(str(row[key_name]), str(v))
                    else:
                        row[key_name] = str(v)
        table_data = list(table_data_map.values())
        table_data.sort(key=lambda item: float(item.get("Priority", 0.0)), reverse=True)
        return table_data

    return run_async(_fetch())


def fetch_a2a_runs(session_id: str):
    """Fetch prior A2A runs for a specific crawler session."""
    import os
    from motor.motor_asyncio import AsyncIOMotorClient

    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "langgraph_crawler")
    client = AsyncIOMotorClient(uri)
    db = client[db_name]

    async def _fetch():
        cursor = db.a2a_runs.find({"session_id": session_id}).sort("created_at", -1).limit(20)
        runs = await cursor.to_list(length=20)
        formatted: list[dict[str, Any]] = []
        for run in runs:
            run_copy = dict(run)
            run_copy["_id"] = str(run_copy.get("_id", ""))
            created_at = run_copy.get("created_at")
            if hasattr(created_at, "strftime"):
                run_copy["created_at"] = created_at.strftime("%Y-%m-%d %H:%M:%S")
            elif created_at is not None:
                run_copy["created_at"] = str(created_at)
            formatted.append(run_copy)
        return formatted

    return run_async(_fetch())


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="🕷️ Crawler Pipeline",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-title {
        color: #2b6cb0;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #4a5568;
        font-size: 1rem;
        margin-top: -10px;
    }
    .node-step {
        padding: 8px 16px;
        border-left: 3px solid #3182ce;
        margin-bottom: 8px;
        background: #ebf8ff;
        border-radius: 0 8px 8px 0;
        color: #2d3748;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helper: run async in streamlit ───────────────────────────
def run_async(coro):
    """Run an async coroutine from streamlit (sync context)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def parse_metric_csv(raw: str) -> list[str]:
    """Parse comma-separated metrics from text input."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def entities_to_table_rows(entities: list[Any]) -> list[dict[str, Any]]:
    """Flatten extracted entities into tabular rows."""
    rows: list[dict[str, Any]] = []
    for entity in entities:
        if hasattr(entity, "model_dump"):
            payload = entity.model_dump()
        elif isinstance(entity, dict):
            payload = dict(entity)
        else:
            continue

        row = {
            "Priority": round(float(payload.get("priority_score", 0.0)), 2),
            "Entity Name": payload.get("name", "Unknown"),
            "Source URL": payload.get("source_url", ""),
            "Description": payload.get("description", ""),
        }
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                row[str(key)] = str(value)
        rows.append(row)
    return rows


# ── Sidebar: Configuration ───────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    model = st.selectbox(
        "LLM Model",
        [
            "meta/meta-llama-3-8b-instruct",
            "meta/meta-llama-3-70b-instruct",
        ],
        index=0,
    )

    max_search = st.slider("Max search results per query", 3, 20, 10)
    min_words = st.slider("Min word count (quality gate)", 50, 500, 200)
    min_cred = st.slider("Min credibility score", 0.0, 1.0, 0.6, 0.05)
    max_retries = st.slider("Max retries", 0, 5, 2)
    min_docs = st.slider("Min processed docs target", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 📊 Node Pipeline")
    nodes = [
        ("🧠", "Intent Parser", "LLM extracts search queries"),
        ("🔍", "URL Discovery", "Tavily search API"),
        ("🕷️", "Web Crawler", "crawl4ai + httpx fallback"),
        ("✅", "Source Verifier", "Credibility scoring"),
        ("💾", "MongoDB Logger", "Async upsert"),
        ("📝", "Preprocessor", "Summarise & extract"),
    ]
    for icon, name, desc in nodes:
        st.markdown(
            f'<div class="node-step">{icon} <strong>{name}</strong><br/>'
            f'<small style="color: #a0aec0">{desc}</small></div>',
            unsafe_allow_html=True,
        )

# ── Main content ─────────────────────────────────────────────
st.markdown('<p class="main-title">🕷️ Crawler Pipeline</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">LangGraph-powered web research • Replicate LLM • Tavily Search • MongoDB</p>',
    unsafe_allow_html=True,
)

# ── Layout: Tabs ─────────────────────────────────────────────
tab1, tab_a2a, tab2 = st.tabs(["🔍 New Search", "🤝 A2A Search", "🕰️ History"])

with tab1:
    # ── Query input ──────────────────────────────────────────────
    query = st.text_area(
        "🔎 Research Query",
        placeholder="e.g., Latest advancements in AI agents and multi-agent systems 2025",
        height=80,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    # ── Pipeline execution ───────────────────────────────────────
    if run_btn and query:
        from crawler.graph import graph
        from crawler.cost_tracker import tracker

        # Reset motor clients so they bind to the fresh event loop
        from crawler.nodes import mongo_logger as _ml, preprocessor as _pp

        _ml._client = None
        _pp._client = None

        # Reset cost tracker for this run
        tracker._calls.clear()

        config = {
            "configurable": {
                "model": model,
                "max_search_results": max_search,
                "min_word_count": min_words,
                "min_credibility": min_cred,
                "max_retries": max_retries,
                "min_processed_docs": min_docs,
            }
        }

        t0 = time.time()

        try:
            with st.spinner("🚀 Running pipeline... (this may take 1-3 minutes)"):
                result = run_async(graph.ainvoke({"user_query": query}, config=config))

            elapsed = time.time() - t0

            entities = result.get("extracted_entities", [])
            cost = result.get("cost_summary", tracker.get_summary())

            # ── Metrics row ──────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Results Summary")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📦 Entities Found", len(entities))
            with m2:
                st.metric("⏱️ Duration", f"{elapsed:.1f}s")
            with m3:
                total_cost = cost.get("total_cost_usd", 0)
                st.metric("💰 Est. Cost", f"${total_cost:.6f}")
            with m4:
                st.metric("🔢 LLM Calls", cost.get("total_calls", 0))

            # ── Cost breakdown ───────────────────────────────
            if cost.get("by_node"):
                st.markdown("### 💰 Cost Breakdown by Node")
                cost_data = []
                for node, data in cost["by_node"].items():
                    cost_data.append(
                        {
                            "Node": node.replace("_", " ").title(),
                            "Calls": data["calls"],
                            "Input Tokens": data["input_tokens"],
                            "Output Tokens": data["output_tokens"],
                            "Cost (USD)": f"${data['cost_usd']:.6f}",
                            "Latency (s)": f"{data['latency_s']:.2f}",
                        }
                    )
                st.table(cost_data)

            # ── Entities (Structured Dataframe) ──────────────
            if entities:
                st.markdown("### 📦 Extracted Entities (Data Table)")

                # Format data into a list of dicts for the dataframe
                table_data = []
                for ent in entities:
                    row = {
                        "Priority": round(ent.priority_score, 2),
                        "Entity Name": ent.name,
                        "Source URL": ent.source_url,
                        "Description": ent.description,
                    }
                    if isinstance(ent.metrics, dict):
                        for k, v in ent.metrics.items():
                            row[k] = str(v)
                    table_data.append(row)

                # Dynamic column config
                col_config = {
                    "Priority": st.column_config.NumberColumn(
                        "Priority", format="%.2f"
                    ),
                    "Source URL": st.column_config.LinkColumn("Source URL"),
                    "Description": st.column_config.TextColumn(
                        "Description", width="large"
                    ),
                }

                # Build Pandas DataFrame to handle dynamic metrics columns safely
                df = pd.DataFrame(table_data)

                st.dataframe(
                    df,
                    width="stretch",
                    column_config=col_config,
                    hide_index=True,
                )

                # Allow user to download the structured data
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Data as CSV",
                    data=csv,
                    file_name=f"crawled_entities_{int(time.time())}.csv",
                    mime="text/csv",
                )

            else:
                st.warning(
                    "No entities were found. Try a broader query or lower the credibility threshold."
                )

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            import traceback

            st.code(traceback.format_exc())

    elif run_btn and not query:
        st.warning("Please enter a research query first.")


with tab_a2a:
    st.markdown("### 🤝 Agent-to-Agent Strict Pipeline")
    st.caption(
        "Flow: Crawler Agent stores to MongoDB + ChromaDB → Validator Agent checks "
        "required metrics → one targeted recrawl if needed → else `no data available`."
    )

    a2a_query = st.text_area(
        "🔎 A2A Research Query",
        placeholder="e.g., Top Hollywood movies in 2025",
        height=80,
        key="a2a_query",
    )

    selected_suggested_metrics: list[str] = []
    custom_metrics = parse_metric_csv(
        st.text_input(
            "✍️ Additional User Metrics (comma-separated)",
            placeholder="e.g., IMDb Score, Runtime",
            key="a2a_custom_metrics",
        )
    )

    if a2a_query.strip():
        from crawler.agents.metric_suggester import merge_metrics, suggest_metrics_for_query

        suggested_metrics = suggest_metrics_for_query(a2a_query)
        selected_suggested_metrics = st.multiselect(
            "🤖 Suggested Metrics (select what you want to enforce)",
            options=suggested_metrics,
            default=suggested_metrics,
            key="a2a_suggested_metrics",
        )
        final_metrics = merge_metrics(
            suggested_metrics=selected_suggested_metrics,
            user_metrics=custom_metrics,
        )
    else:
        suggested_metrics = []
        final_metrics = custom_metrics
        st.info("Enter a query to get automatic metric suggestions.")

    st.markdown("**Final Metrics Enforced By Validator**")
    if final_metrics:
        st.code(", ".join(final_metrics))
    else:
        st.warning("No metrics selected yet.")

    a2a_rounds = st.slider(
        "A2A Rounds (strict mode uses one recrawl; recommended: 2)",
        min_value=1,
        max_value=4,
        value=2,
        key="a2a_rounds",
    )

    run_a2a = st.button(
        "🚀 Run A2A Pipeline",
        type="primary",
        use_container_width=True,
        key="run_a2a_btn",
    )

    if run_a2a and not a2a_query.strip():
        st.warning("Please enter a research query first.")
    elif run_a2a and not final_metrics:
        st.warning("Please select/add at least one metric.")
    elif run_a2a:
        from crawler.agents import AgentToAgentPipeline
        from crawler.cost_tracker import tracker
        from crawler.nodes import mongo_logger as _ml, preprocessor as _pp

        # Reset stateful clients for this fresh async run in Streamlit.
        _ml._client = None
        _pp._client = None
        tracker._calls.clear()

        t0 = time.time()
        try:
            with st.spinner("🤝 Running A2A pipeline... (this may take 1-3 minutes)"):
                pipeline = AgentToAgentPipeline(max_rounds=a2a_rounds)
                result_obj = run_async(
                    pipeline.run(
                        query=a2a_query.strip(),
                        required_metrics=final_metrics,
                    )
                )
            elapsed = time.time() - t0
            result = result_obj.to_dict()
            result["suggested_metrics"] = suggested_metrics
            result["user_metrics"] = custom_metrics
            result["final_metrics"] = final_metrics

            from crawler.agents.a2a_store import save_a2a_run

            try:
                run_id = run_async(save_a2a_run(payload=result, source="dashboard"))
            except Exception as store_exc:
                run_id = ""
                st.warning(f"Could not persist A2A run log: {store_exc}")
            result["run_id"] = run_id

            st.markdown("---")
            st.markdown("### 📊 A2A Summary")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Status", result.get("status", "unknown"))
            with m2:
                st.metric("Rounds Used", result.get("rounds_used", 0))
            with m3:
                st.metric("Entities", len(result.get("entities", [])))
            with m4:
                st.metric("Duration", f"{elapsed:.1f}s")
            st.caption(f"A2A Run ID: {result.get('run_id', '')}")

            st.markdown("### 🎯 Metric Validation")
            st.write("Required:", result.get("required_metrics", []))
            st.write("Available:", result.get("available_metrics", []))
            st.write("Missing:", result.get("missing_metrics", []))

            missing_data_details = result.get("missing_data_details", [])
            if missing_data_details:
                st.markdown("### ⚠️ Entities With Missing/Placeholder Metrics")
                issue_rows: list[dict[str, Any]] = []
                for detail in missing_data_details:
                    entity_name = detail.get("entity_name", "Unknown Entity")
                    for metric in detail.get("missing_metrics", []):
                        issue_rows.append(
                            {
                                "Entity Name": entity_name,
                                "Metric": metric,
                                "Issue Type": "Missing",
                                "Value": "",
                            }
                        )
                    for metric, value in detail.get("placeholder_metrics", {}).items():
                        issue_rows.append(
                            {
                                "Entity Name": entity_name,
                                "Metric": metric,
                                "Issue Type": "Placeholder",
                                "Value": str(value),
                            }
                        )
                if issue_rows:
                    st.dataframe(pd.DataFrame(issue_rows), width="stretch", hide_index=True)

            st.markdown("### 💬 Agent Communication Log")
            comm_log = result.get("communication_log", [])
            if comm_log:
                df_log = pd.DataFrame(comm_log)
                df_log = df_log.rename(
                    columns={
                        "round_number": "Round",
                        "from_agent": "From",
                        "to_agent": "To",
                        "content": "Message",
                    }
                )
                st.dataframe(df_log, width="stretch", hide_index=True)
            else:
                st.info("No communication logs found.")

            entities = result.get("entities", [])
            if entities:
                st.markdown("### 📦 A2A Extracted Entities")
                table_data = entities_to_table_rows(entities)
                df = pd.DataFrame(table_data)
                col_config = {
                    "Priority": st.column_config.NumberColumn("Priority", format="%.2f"),
                    "Source URL": st.column_config.LinkColumn("Source URL"),
                    "Description": st.column_config.TextColumn(
                        "Description", width="large"
                    ),
                }
                st.dataframe(
                    df,
                    width="stretch",
                    column_config=col_config,
                    hide_index=True,
                )
            else:
                st.warning(result.get("message", "no data available"))

        except Exception as exc:
            st.error(f"A2A pipeline error: {exc}")
            import traceback

            st.code(traceback.format_exc())


with tab2:
    st.markdown("### 🕰️ Past Research Sessions")

    if st.button("🔄 Refresh History"):
        fetch_history.clear()

    history = fetch_history()

    if not history:
        st.info("No past sessions found in the database. Run a search first!")
    else:
        # Show recent sessions
        df_history = pd.DataFrame(history)
        st.dataframe(
            df_history,
            width="stretch",
            hide_index=True,
        )

        st.markdown("### 📂 View Session Results")
        session_options = [
            h["Session ID"]
            for h in history
            if h["Entities Found"] > 0 or h.get("A2A Runs", 0) > 0
        ]
        if not session_options:
            st.info("No sessions with extracted entities or A2A runs yet.")
            selected_session = None
        else:
            selected_session = st.selectbox(
                "Select a session to view its extracted entities:",
                options=session_options,
                format_func=lambda x: f"{next(h['Date'] for h in history if h['Session ID'] == x)} — {next(h['Query'] for h in history if h['Session ID'] == x)[:50]}...",
            )

        if selected_session:
            entities_data = fetch_session_docs(selected_session)
            st.markdown(f"**Results for Session:** `{selected_session}`")

            if entities_data:
                col_config = {
                    "Priority": st.column_config.NumberColumn(
                        "Priority", format="%.2f"
                    ),
                    "Source URL": st.column_config.LinkColumn("Source URL"),
                    "Description": st.column_config.TextColumn(
                        "Description", width="large"
                    ),
                }

                df_hist = pd.DataFrame(entities_data)
                st.dataframe(
                    df_hist,
                    width="stretch",
                    column_config=col_config,
                    hide_index=True,
                )

                csv_hist = df_hist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Session Data as CSV",
                    data=csv_hist,
                    file_name=f"session_{selected_session}.csv",
                    mime="text/csv",
                    key="download_hist",
                )
            else:
                st.info("No extracted entities found for this session.")

            a2a_runs = fetch_a2a_runs(selected_session)
            st.markdown("### 🤝 Past A2A Runs")
            if not a2a_runs:
                st.info("No A2A runs stored for this session.")
            else:
                run_labels = [
                    f"{run.get('created_at', 'Unknown')} | {run.get('status', 'unknown')} | rounds={run.get('rounds_used', 0)}"
                    for run in a2a_runs
                ]
                selected_idx = st.selectbox(
                    "Select an A2A run to inspect:",
                    options=list(range(len(a2a_runs))),
                    format_func=lambda i: run_labels[i],
                    key="history_a2a_run_select",
                )
                selected_run = a2a_runs[selected_idx]

                st.write("Final Metrics:", selected_run.get("final_metrics", []))
                st.write("Available Metrics:", selected_run.get("available_metrics", []))
                st.write("Missing Metrics:", selected_run.get("missing_metrics", []))

                missing_details = selected_run.get("missing_data_details", [])
                if missing_details:
                    st.markdown("**Missing / Placeholder Metric Details**")
                    detail_rows: list[dict[str, Any]] = []
                    for detail in missing_details:
                        entity_name = detail.get("entity_name", "Unknown Entity")
                        for metric in detail.get("missing_metrics", []):
                            detail_rows.append(
                                {
                                    "Entity Name": entity_name,
                                    "Metric": metric,
                                    "Issue Type": "Missing",
                                    "Value": "",
                                }
                            )
                        for metric, value in detail.get("placeholder_metrics", {}).items():
                            detail_rows.append(
                                {
                                    "Entity Name": entity_name,
                                    "Metric": metric,
                                    "Issue Type": "Placeholder",
                                    "Value": str(value),
                                }
                            )
                    if detail_rows:
                        st.dataframe(
                            pd.DataFrame(detail_rows),
                            width="stretch",
                            hide_index=True,
                        )

                comm_log = selected_run.get("communication_log", [])
                st.markdown("**Communication Log**")
                if comm_log:
                    df_log = pd.DataFrame(comm_log).rename(
                        columns={
                            "round_number": "Round",
                            "from_agent": "From",
                            "to_agent": "To",
                            "content": "Message",
                        }
                    )
                    st.dataframe(df_log, width="stretch", hide_index=True)
                else:
                    st.info("No communication logs found for this run.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #4a5568; font-size: 0.85rem;">'
    "LangGraph Crawler Pipeline v0.1.0 • Replicate + Tavily + MongoDB + ChromaDB + A2A"
    "</p>",
    unsafe_allow_html=True,
)
