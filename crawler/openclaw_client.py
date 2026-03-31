"""OpenClaw API adapter used by crawler nodes.

This module normalizes varied OpenClaw-like response shapes into a stable list
of URL/content documents for downstream processing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import re
import shutil
import subprocess
from typing import Any

import httpx

from crawler.config import Configuration


@dataclass
class OpenClawDocument:
    url: str
    title: str
    content: str
    snippet: str
    query: str


def _coerce_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("results", "documents", "items", "data", "hits"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    if isinstance(payload.get("result"), dict):
        obj = payload["result"]
        for key in ("results", "documents", "items", "data", "hits"):
            value = obj.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    return []


def _pick_first(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _endpoint_candidates(configuration: Configuration) -> list[str]:
    base = configuration.openclaw_base_url.rstrip("/")

    configured_path = configuration.openclaw_search_path.strip()
    if not configured_path.startswith("/"):
        configured_path = f"/{configured_path}"

    mode = (configuration.openclaw_mode or "auto").strip().lower()
    out: list[str] = []

    def _add(path: str) -> None:
        endpoint = f"{base}{path}"
        if endpoint not in out:
            out.append(endpoint)

    if mode in {"search", "auto"}:
        _add(configured_path)

    if mode in {"gateway", "auto"}:
        _add("/sessions/send")
        _add("/api/sessions/send")
        _add("/v1/sessions/send")

    return out


def _payload_candidates(configuration: Configuration, query: str, limit: int) -> list[dict[str, Any]]:
    session_key = (configuration.openclaw_session_key or "agent:main").strip()
    return [
        {
            "query": query,
            "q": query,
            "limit": max(1, int(limit)),
            "top_k": max(1, int(limit)),
        },
        {
            "session": session_key,
            "query": query,
            "limit": max(1, int(limit)),
        },
        {
            "session_key": session_key,
            "input": query,
            "limit": max(1, int(limit)),
        },
        {
            "key": session_key,
            "message": {
                "type": "crawl_search",
                "query": query,
                "limit": max(1, int(limit)),
            },
        },
    ]


def _extract_json_array_from_text(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    cleaned = text.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            arr = _coerce_list(parsed)
            if arr:
                return arr
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        maybe = match.group(0)
        try:
            parsed = json.loads(maybe)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            pass

    return []


async def _search_via_cli(configuration: Configuration, query: str, limit: int) -> list[OpenClawDocument]:
    if not configuration.openclaw_enable_cli_fallback:
        return []
    cli_cmd: list[str] | None = None
    if shutil.which("openclaw"):
        cli_cmd = ["openclaw"]
    elif os.name == "nt" and shutil.which("openclaw.cmd"):
        cli_cmd = ["openclaw.cmd"]
    elif os.name == "nt":
        appdata = os.getenv("APPDATA", "")
        if appdata:
            candidate = os.path.join(appdata, "npm", "openclaw.cmd")
            if os.path.exists(candidate):
                cli_cmd = [candidate]

    if cli_cmd is None:
        print("[OpenClaw] CLI fallback unavailable: openclaw command not found in PATH.")
        return []

    prompt = (
        "You are collecting incubator crawl sources. "
        f"Find up to {max(1, int(limit))} high-quality web sources for this query: {query}. "
        "Return ONLY valid JSON as an array of objects with keys: "
        "url, title, snippet, content."
    )

    session_key = (configuration.openclaw_session_key or "agent:main").strip()
    agent_id = "main"
    if session_key.startswith("agent:"):
        parts = session_key.split(":")
        if len(parts) >= 2 and parts[1].strip():
            agent_id = parts[1].strip()

    args = [
        *cli_cmd,
        "agent",
        "--agent",
        agent_id,
        "--json",
        "--message",
        prompt,
        "--thinking",
        "minimal",
        "--timeout",
        str(max(10, int(configuration.openclaw_cli_timeout_s))),
    ]

    try:
        command = args
        if os.name == "nt":
            command = ["cmd", "/c", *args]
        proc = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        print(f"[OpenClaw] CLI fallback invocation failed: {exc}")
        return []

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        print(f"[OpenClaw] CLI fallback failed (code={proc.returncode}): {stderr[:300]}")
        return []

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return []

    payload_texts: list[str] = [stdout]
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            for key in ("response", "reply", "message", "output", "content"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    payload_texts.append(val)
            maybe = parsed.get("result")
            if isinstance(maybe, str) and maybe.strip():
                payload_texts.append(maybe)
    except Exception:
        pass

    items: list[dict[str, Any]] = []
    for text in payload_texts:
        items = _extract_json_array_from_text(text)
        if items:
            break

    docs: list[OpenClawDocument] = []
    seen: set[str] = set()
    for item in items:
        url = _pick_first(item, "url", "source_url", "link", "source")
        if not url or url in seen:
            continue
        seen.add(url)

        title = _pick_first(item, "title", "name", "headline")
        content = _pick_first(item, "content", "text", "markdown", "body")
        snippet = _pick_first(item, "snippet", "summary", "description", "content", "text")

        docs.append(
            OpenClawDocument(
                url=url,
                title=title,
                content=content,
                snippet=(snippet or content)[:800],
                query=query,
            )
        )
        if len(docs) >= max(1, int(limit)):
            break

    if docs:
        print(f"[OpenClaw] CLI fallback returned {len(docs)} docs for query={query!r}.")
    return docs


async def search_documents(
    configuration: Configuration,
    query: str,
    limit: int,
) -> list[OpenClawDocument]:
    """Query OpenClaw and normalize documents for crawler pipeline use."""
    endpoints = _endpoint_candidates(configuration)

    headers = {"Content-Type": "application/json"}
    if configuration.openclaw_api_key.strip():
        headers["Authorization"] = f"Bearer {configuration.openclaw_api_key.strip()}"

    raw: Any = None
    last_exc: Exception | None = None
    payloads = _payload_candidates(configuration, query, limit)

    try:
        async with httpx.AsyncClient(timeout=max(5, int(configuration.openclaw_timeout_s))) as client:
            for endpoint in endpoints:
                for payload in payloads:
                    try:
                        resp = await client.post(endpoint, json=payload, headers=headers)
                        resp.raise_for_status()
                        raw = resp.json()
                        break
                    except Exception as exc:
                        last_exc = exc
                        continue
                if raw is not None:
                    break
    except Exception as exc:
        last_exc = exc

    if raw is None:
        print(f"[OpenClaw] API request failed for query={query!r}: {last_exc}")
        return await _search_via_cli(configuration, query, limit)

    items = _coerce_list(raw)
    docs: list[OpenClawDocument] = []
    seen: set[str] = set()

    for item in items:
        url = _pick_first(item, "url", "source_url", "link", "source")
        if not url or url in seen:
            continue
        seen.add(url)

        title = _pick_first(item, "title", "name", "headline")
        content = _pick_first(item, "content", "text", "markdown", "body")
        snippet = _pick_first(item, "snippet", "summary", "description", "content", "text")

        docs.append(
            OpenClawDocument(
                url=url,
                title=title,
                content=content,
                snippet=(snippet or content)[:800],
                query=query,
            )
        )
        if len(docs) >= max(1, int(limit)):
            break

    return docs
