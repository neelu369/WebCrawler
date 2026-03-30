"""URL Discovery node — searches the web and gathers candidate URLs.

Replaces commercial search APIs (Tavily, Serper) with a self-hosted SearXNG
container. Automatically staggers requests to prevent local IP blocking
from upstream engines like Google and Bing.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import aiohttp
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.models import DiscoveredURL
from crawler.state import State


async def _search_searxng(query: str, num: int, sem: asyncio.Semaphore) -> list[dict]:
    """Execute a single query against a local SearXNG instance."""
    base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    url = f"{base_url.rstrip('/')}/search"
    params = {"q": query, "format": "json"}
    
    # Use semaphore to stagger concurrent requests
    async with sem:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        # Extract first `num` results
                        return [
                            {
                                "url": item.get("url", ""),
                                "title": item.get("title", ""),
                                "content": item.get("content", ""),
                            }
                            for item in results[:num]
                        ]
                    else:
                        text = await response.text()
                        print(f"[URL Discovery] SearXNG error {response.status}: {text}")
                        return []
        except Exception as exc:
            print(f"[URL Discovery] SearXNG call failed for '{query}': {exc}")
            return []


async def discover_urls(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Search the web via SearXNG for each query and collect unique URLs."""
    configuration = Configuration.from_runnable_config(config)

    if not getattr(configuration, "enable_searxng_search", True):
        print("[URL Discovery] SearXNG disabled. No URLs found.")
        return {"discovered_urls": []}

    seen_urls: set[str] = set()
    urls: list[DiscoveredURL] = []

    # Semaphore to prevent DDOSing the local SearXNG container 
    # and to slow down requests to avoid upstream IP bans.
    sem = asyncio.Semaphore(2)

    async def _run_search(query: str):
        return query, await _search_searxng(query, configuration.max_search_results, sem)

    tasks = [_run_search(sq.query) for sq in state.search_queries]
    if not tasks:
        return {"discovered_urls": []}

    print(f"[URL Discovery] Dispatching {len(tasks)} queries to SearXNG...")
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results_list:
        if isinstance(res, Exception):
            print(f"[URL Discovery] Task failed: {res}")
            continue
            
        query, items = res
        for item in items:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
                
            seen_urls.add(url)
            urls.append(
                DiscoveredURL(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("content", "")[:500],
                    search_query=query,
                )
            )

    print(
        f"[URL Discovery] Found {len(urls)} unique URLs from {len(state.search_queries)} queries via SearXNG."
    )
    return {"discovered_urls": urls}
