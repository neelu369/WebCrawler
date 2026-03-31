from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


DEFAULT_QUERY_BANK = [
    "startup incubators in India",
    "government backed incubators in India",
    "publicly funded incubators in India",
    "academic startup incubators in India",
    "state-wise startup incubators in India",
    "startup incubators site:startupindia.gov.in",
    "Atal incubation centers in India",
    "DST funded incubators in India",
    "university incubators in India",
    "technology business incubators in India",
    "biotech incubators in India",
    "fintech incubators in India",
    "women-focused incubators in India",
    "startup accelerators in Bengaluru",
    "startup accelerators in Mumbai",
    "startup accelerators in Delhi",
    "startup accelerators in Hyderabad",
    "IIT incubators in India",
    "IIM incubators in India",
    "NIT incubators in India",
]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_existing_keys(path: Path, key_field: str) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get(key_field, "")
            if key:
                keys.add(key)
    return keys


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]], key_field: str) -> int:
    if not rows:
        return 0
    existing = _read_existing_keys(path, key_field)
    new_rows = [r for r in rows if r.get(key_field, "") and r[key_field] not in existing]
    if not new_rows:
        return 0

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def _make_entity_key(query: str, name: str, sources: list[str]) -> str:
    payload = f"{query.lower()}|{name.lower()}|{'|'.join(sorted(sources))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _check_openclaw(config_base_url: str, search_path: str, timeout_s: int) -> None:
    base = config_base_url.rstrip("/")
    path = search_path if search_path.startswith("/") else f"/{search_path}"
    endpoint = f"{base}{path}"

    async with httpx.AsyncClient(timeout=max(5, timeout_s)) as client:
        try:
            resp = await client.post(endpoint, json={"query": "startup incubators in India", "limit": 1})
            if resp.status_code >= 400:
                raise RuntimeError(f"OpenClaw endpoint responded {resp.status_code}: {resp.text[:200]}")
            return
        except Exception as exc:
            # Detect if only Ollama is up to provide a clearer hint.
            try:
                tags = await client.get("http://127.0.0.1:11434/api/tags")
                if tags.status_code == 200:
                    raise RuntimeError(
                        "OpenClaw API is not reachable, but Ollama is running on 11434. "
                        "`ollama launch openclaw` starts a chat model, not the OpenClaw crawl API service. "
                        f"Expected OpenClaw HTTP endpoint at {endpoint}. Original error: {exc}"
                    )
            except Exception:
                pass
            if _env_bool("OPENCLAW_ENABLE_CLI_FALLBACK", True) and shutil.which("openclaw"):
                print(
                    "[Scheduler] OpenClaw REST endpoint unavailable; proceeding with OPENCLAW_ENABLE_CLI_FALLBACK=true."
                )
                return
            raise RuntimeError(f"OpenClaw API preflight failed for {endpoint}: {exc}")


async def _start_job(client: httpx.AsyncClient, api_base: str, payload: dict[str, Any]) -> str:
    resp = await client.post(f"{api_base.rstrip('/')}/crawl/rank", json=payload)
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.text
        except Exception:
            detail = "<no body>"
        raise RuntimeError(
            f"/crawl/rank failed ({resp.status_code}) for payload={payload}. Response={detail}"
        )
    data = resp.json()
    job_id = str(data.get("job_id", "")).strip()
    if not job_id:
        raise RuntimeError("/crawl/rank did not return job_id")
    return job_id


async def _wait_job(client: httpx.AsyncClient, api_base: str, job_id: str, timeout_s: int) -> dict[str, Any]:
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"job {job_id} timed out after {timeout_s}s")
        resp = await client.get(f"{api_base.rstrip('/')}/crawl/rank/{job_id}")
        resp.raise_for_status()
        job = resp.json()
        status = job.get("status", "")
        if status in {"completed", "failed"}:
            return job
        await asyncio.sleep(1)


def _build_payload(query: str) -> dict[str, Any]:
    top_n = _env_int("SCHEDULER_TOP_N", 100)
    top_n = max(1, min(100, top_n))
    return {
        "query": query,
        "top_n": top_n,
        "max_retries": _env_int("SCHEDULER_MAX_RETRIES", 2),
        "min_credibility": float(os.getenv("SCHEDULER_MIN_CREDIBILITY", "0.05")),
        "min_relevance": float(os.getenv("SCHEDULER_MIN_RELEVANCE", "0.05")),
        "crawler_concurrency": _env_int("SCHEDULER_CRAWLER_CONCURRENCY", 10),
        "playwright_timeout_ms": _env_int("SCHEDULER_PLAYWRIGHT_TIMEOUT_MS", 20000),
        "use_searxng_search": _env_bool("SCHEDULER_USE_SEARXNG", True),
        "use_playwright_mcp": _env_bool("SCHEDULER_USE_PLAYWRIGHT", True),
        "use_openclaw": True,
        "openclaw_max_docs_per_query": _env_int("OPENCLAW_MAX_DOCS_PER_QUERY", 150),
    }


def _extract_entity_rows(job: dict[str, Any], query: str, captured_at: str) -> list[dict[str, str]]:
    ranking = job.get("ranking_result") or {}
    entities = ranking.get("entities") or []
    session_id = str(job.get("session_id", ""))

    rows: list[dict[str, str]] = []
    for e in entities:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        source_urls = e.get("source_urls") or []
        if not isinstance(source_urls, list):
            source_urls = []
        key = _make_entity_key(query, name, [str(u) for u in source_urls])
        rows.append(
            {
                "entity_key": key,
                "name": name,
                "entity_type": str(e.get("entity_type", "")),
                "composite_score": str(e.get("composite_score", "")),
                "priority_score": str(e.get("priority_score", "")),
                "query": query,
                "session_id": session_id,
                "source_urls_json": json.dumps(source_urls, ensure_ascii=False),
                "captured_at": captured_at,
            }
        )
    return rows


async def run_scheduler(args: argparse.Namespace) -> None:
    load_dotenv()

    api_base = os.getenv("SCHEDULER_API_BASE", "http://127.0.0.1:8000")
    dataset_dir = Path(os.getenv("SCHEDULER_DATASET_DIR", "datasets"))
    dataset_dir.mkdir(parents=True, exist_ok=True)

    openclaw_base = os.getenv("OPENCLAW_BASE_URL", "http://127.0.0.1:3100")
    openclaw_path = os.getenv("OPENCLAW_SEARCH_PATH", "/api/v1/search")
    openclaw_timeout = _env_int("OPENCLAW_TIMEOUT_S", 45)

    await _check_openclaw(openclaw_base, openclaw_path, openclaw_timeout)

    cycles = max(0, args.cycles)
    queries_per_cycle = max(1, args.queries_per_cycle)
    cycle_minutes = max(1, args.cycle_minutes)

    cycle_index = 0
    while cycles == 0 or cycle_index < cycles:
        cycle_index += 1
        started = time.time()
        captured_at = datetime.now(timezone.utc).isoformat()

        queries = DEFAULT_QUERY_BANK[:]
        random.shuffle(queries)
        selected = queries[:queries_per_cycle]

        all_rows: list[dict[str, str]] = []
        run_summary: dict[str, Any] = {
            "timestamp": captured_at,
            "cycle": cycle_index,
            "queries": selected,
            "jobs": [],
        }

        async with httpx.AsyncClient(timeout=60) as client:
            for query in selected:
                payload = _build_payload(query)
                job_id = await _start_job(client, api_base, payload)
                job = await _wait_job(client, api_base, job_id, timeout_s=_env_int("SCHEDULER_JOB_TIMEOUT_S", 1800))

                run_summary["jobs"].append(
                    {
                        "query": query,
                        "job_id": job_id,
                        "status": job.get("status"),
                        "session_id": job.get("session_id", ""),
                        "error": job.get("error", ""),
                    }
                )

                if job.get("status") == "completed":
                    all_rows.extend(_extract_entity_rows(job, query, captured_at))

        entities_csv = dataset_dir / "incubators_india_ranked_entities.csv"
        inserted = _append_csv(
            entities_csv,
            [
                "entity_key",
                "name",
                "entity_type",
                "composite_score",
                "priority_score",
                "query",
                "session_id",
                "source_urls_json",
                "captured_at",
            ],
            all_rows,
            "entity_key",
        )

        run_summary["rows_seen"] = len(all_rows)
        run_summary["rows_inserted"] = inserted

        runs_jsonl = dataset_dir / "incubator_scheduler_runs.jsonl"
        with runs_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")

        print(
            f"[Scheduler] cycle={cycle_index} queries={len(selected)} "
            f"rows_seen={len(all_rows)} rows_inserted={inserted}"
        )

        if cycles != 0 and cycle_index >= cycles:
            break

        elapsed = time.time() - started
        sleep_seconds = max(0, cycle_minutes * 60 - int(elapsed))
        print(f"[Scheduler] sleeping {sleep_seconds}s before next cycle")
        time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate incubator crawl runs via API + OpenClaw backend")
    parser.add_argument("--cycles", type=int, default=_env_int("SCHEDULER_CYCLES", 0), help="0 means run forever")
    parser.add_argument("--queries-per-cycle", type=int, default=_env_int("SCHEDULER_QUERIES_PER_CYCLE", 5))
    parser.add_argument("--cycle-minutes", type=int, default=_env_int("SCHEDULER_CYCLE_MINUTES", 180))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_scheduler(parse_args()))
