from __future__ import annotations

import time
from typing import Any

import httpx
import pandas as pd
import streamlit as st


st.set_page_config(page_title="WebCrawler Streamlit", layout="wide")
st.title("WebCrawler Ranking Pipeline")
st.caption("Run the existing FastAPI crawler pipeline from Streamlit.")


DEFAULTS = {
    "top_n": 100,
    "max_retries": 5,
    "min_credibility": 0.15,
    "min_relevance": 0.15,
    "use_searxng_search": True,
    "use_playwright_mcp": True,
    "use_openclaw": False,
    "openclaw_max_docs_per_query": 150,
    "crawler_concurrency": 10,
    "playwright_timeout_ms": 20000,
}


def _start_job(api_base: str, payload: dict[str, Any]) -> tuple[str | None, str | None]:
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{api_base}/crawl/rank", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("job_id"), None
    except Exception as exc:
        return None, str(exc)


def _poll_job(api_base: str, job_id: str, poll_interval_s: float = 1.0, timeout_s: int = 900) -> tuple[dict[str, Any] | None, str | None]:
    start = time.time()
    with st.status("Pipeline running...", expanded=True) as status:
        while True:
            if time.time() - start > timeout_s:
                status.update(label="Timed out", state="error")
                return None, f"Timed out after {timeout_s} seconds"

            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.get(f"{api_base}/crawl/rank/{job_id}")
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:
                status.update(label="Polling error", state="error")
                return None, str(exc)

            job_status = data.get("status", "unknown")
            event_count = len(data.get("events") or [])
            status.write(f"status={job_status} | events={event_count}")

            if job_status in {"completed", "failed"}:
                final_state = "complete" if job_status == "completed" else "error"
                status.update(label=f"Pipeline {job_status}", state=final_state)
                return data, None

            time.sleep(poll_interval_s)


def _render_result(job: dict[str, Any]) -> None:
    st.subheader("Job Result")
    st.json({
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "error": job.get("error"),
        "session_id": job.get("session_id"),
    })

    if job.get("status") != "completed":
        return

    ranking = job.get("ranking_result") or {}
    entities = ranking.get("entities") or []
    criteria = ranking.get("criteria") or []
    insights = (job.get("insights") or {}).get("items") or []

    st.subheader("Ranked Entities")
    if entities:
        rows = []
        for idx, e in enumerate(entities, start=1):
            rows.append(
                {
                    "rank": idx,
                    "name": e.get("name", ""),
                    "score": e.get("composite_score", 0.0),
                    "type": e.get("entity_type", ""),
                    "sources": len(e.get("source_urls") or []),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("No ranked entities returned.")

    st.subheader("Criteria")
    if criteria:
        st.dataframe(pd.DataFrame(criteria), use_container_width=True)
    else:
        st.info("No criteria returned.")

    st.subheader("Insights")
    if insights:
        st.dataframe(pd.DataFrame(insights), use_container_width=True)
    else:
        st.info("No insights returned.")


with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", value="http://127.0.0.1:8000")
    top_n = st.number_input("Top N", min_value=1, max_value=500, value=DEFAULTS["top_n"], step=1)
    max_retries = st.number_input("Max retries", min_value=0, max_value=10, value=DEFAULTS["max_retries"], step=1)
    min_credibility = st.slider("Min credibility", min_value=0.0, max_value=1.0, value=float(DEFAULTS["min_credibility"]), step=0.01)
    min_relevance = st.slider("Min relevance", min_value=0.0, max_value=1.0, value=float(DEFAULTS["min_relevance"]), step=0.01)
    crawler_concurrency = st.number_input("Crawler concurrency", min_value=1, max_value=20, value=DEFAULTS["crawler_concurrency"], step=1)
    use_openclaw = st.checkbox("Use OpenClaw", value=DEFAULTS["use_openclaw"])
    openclaw_max_docs_per_query = st.number_input(
        "OpenClaw max docs/query",
        min_value=1,
        max_value=2000,
        value=DEFAULTS["openclaw_max_docs_per_query"],
        step=10,
    )
    use_searxng_search = st.checkbox("Use SearXNG", value=DEFAULTS["use_searxng_search"])
    use_playwright_mcp = st.checkbox("Use Playwright MCP", value=DEFAULTS["use_playwright_mcp"])

query = st.text_input("Query", value="incubators in India")
run = st.button("Run Pipeline", type="primary")

if run:
    payload = {
        "query": query,
        "top_n": int(top_n),
        "max_retries": int(max_retries),
        "min_credibility": float(min_credibility),
        "min_relevance": float(min_relevance),
        "use_openclaw": bool(use_openclaw),
        "openclaw_max_docs_per_query": int(openclaw_max_docs_per_query),
        "use_searxng_search": bool(use_searxng_search),
        "use_playwright_mcp": bool(use_playwright_mcp),
        "crawler_concurrency": int(crawler_concurrency),
        "playwright_timeout_ms": DEFAULTS["playwright_timeout_ms"],
    }

    job_id, err = _start_job(api_base, payload)
    if err:
        st.error(f"Failed to start job: {err}")
    elif not job_id:
        st.error("API did not return a job_id")
    else:
        st.success(f"Started job: {job_id}")
        job, poll_err = _poll_job(api_base, job_id)
        if poll_err:
            st.error(f"Polling failed: {poll_err}")
        elif job is not None:
            _render_result(job)
