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
from crawler.models import CrawledDoc
from crawler.openclaw_client import search_documents
from crawler.state import State


async def _search_searxng(
    query: str,
    num: int,
    sem: asyncio.Semaphore,
    pages: int,
    max_pages: int,
) -> list[dict]:
    """Execute a single query against a local SearXNG instance."""
    base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    base = base_url.rstrip("/")
    candidate_urls = [f"{base}/search", base]
    safe_pages = max(1, min(int(pages), max(1, int(max_pages))))
    per_page = max(1, num)
    seen_urls: set[str] = set()
    collected: list[dict] = []
    
    # Use semaphore to stagger concurrent requests
    async with sem:
        try:
            async with aiohttp.ClientSession() as session:
                for page in range(1, safe_pages + 1):
                    page_results: list[dict] = []
                    page_ok = False
                    for url in candidate_urls:
                        params = {"q": query, "format": "json", "pageno": page}
                        async with session.get(url, params=params, timeout=20) as response:
                            if response.status != 200:
                                text = await response.text()
                                print(
                                    f"[URL Discovery] SearXNG error {response.status} "
                                    f"for '{query}' page={page} endpoint={url}: {text}"
                                )

                                # 404 on first page means endpoint mismatch; try fallback endpoint.
                                # 404 on subsequent pages usually means no pagination route support.
                                if response.status == 404 and page > 1:
                                    return collected
                                continue

                            try:
                                data = await response.json()
                            except Exception:
                                continue

                            page_results = data.get("results", [])
                            page_ok = True
                            break

                    if not page_ok:
                        # No compatible endpoint for this page.
                        if page == 1:
                            return collected
                        break

                    if not page_results:
                        break

                    for item in page_results[:per_page]:
                        item_url = item.get("url", "")
                        if not item_url or item_url in seen_urls:
                            continue
                        seen_urls.add(item_url)
                        collected.append(
                            {
                                "url": item_url,
                                "title": item.get("title", ""),
                                "content": item.get("content", ""),
                            }
                        )
        except Exception as exc:
            print(f"[URL Discovery] SearXNG call failed for '{query}': {exc}")
            return []

    return collected


async def discover_urls(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Search the web via SearXNG for each query and collect unique URLs."""
    configuration = Configuration.from_runnable_config(config)

    if getattr(configuration, "enable_openclaw", False):
        seen_urls: set[str] = set()
        urls: list[DiscoveredURL] = []
        preloaded: list[CrawledDoc] = []

        if not state.search_queries:
            return {"discovered_urls": [], "preloaded_crawled_docs": []}

        limit = max(1, int(configuration.openclaw_max_docs_per_query))
        print(
            f"[URL Discovery] OpenClaw mode enabled: queries={len(state.search_queries)} "
            f"max_docs_per_query={limit}"
        )

        for sq in state.search_queries:
            docs = await search_documents(configuration, sq.query, limit)
            for doc in docs:
                if doc.url in seen_urls:
                    continue
                seen_urls.add(doc.url)
                urls.append(
                    DiscoveredURL(
                        url=doc.url,
                        title=doc.title,
                        snippet=doc.snippet[:500],
                        search_query=sq.query,
                    )
                )
                if doc.content.strip():
                    preloaded.append(
                        CrawledDoc(
                            url=doc.url,
                            content=doc.content,
                            word_count=len(doc.content.split()),
                            crawl_method="openclaw",
                        )
                    )

        print(
            f"[URL Discovery] OpenClaw provided {len(urls)} unique URLs and {len(preloaded)} preloaded docs."
        )
        return {
            "discovered_urls": urls,
            "preloaded_crawled_docs": preloaded,
        }

    if not getattr(configuration, "enable_searxng_search", True):
        print("[URL Discovery] SearXNG disabled. No URLs found.")
        return {"discovered_urls": [], "preloaded_crawled_docs": []}

    seen_urls: set[str] = set()
    urls: list[DiscoveredURL] = []

    # Semaphore to prevent DDOSing local SearXNG while allowing broader recall.
    sem_size = max(2, min(8, configuration.crawler_concurrency // 2 or 2))
    sem = asyncio.Semaphore(sem_size)

    async def _run_search(query: str):
        return query, await _search_searxng(
            query,
            configuration.max_search_results,
            sem,
            configuration.searxng_pages,
            configuration.max_searxng_pages,
        )

    tasks = [_run_search(sq.query) for sq in state.search_queries]
    if not tasks:
        return {"discovered_urls": [], "preloaded_crawled_docs": []}

    print(
        f"[URL Discovery] Dispatching {len(tasks)} queries to SearXNG "
        f"(pages={configuration.searxng_pages}, per_page={configuration.max_search_results}, parallel={sem_size})..."
    )
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
    return {"discovered_urls": urls, "preloaded_crawled_docs": []}
