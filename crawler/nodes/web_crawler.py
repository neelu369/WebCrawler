"""Web Crawler node — fetches page content using crawl4ai with httpx fallback.

Crawls all discovered URLs in parallel using crawl4ai's AsyncWebCrawler.
If a page fails, falls back to Playwright MCP (shared session pool), then
a simple httpx GET.  Applies a word-count quality gate to filter out thin
pages.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from typing import Any, Optional
import re

import httpx
from crawl4ai import AsyncWebCrawler
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.models import CrawledDoc
from crawler.state import State


_ANTI_BOT_PATTERNS = (
    re.compile(r"\bcaptcha\b", re.IGNORECASE),
    re.compile(r"\bcloudflare\b", re.IGNORECASE),
    re.compile(r"\baccess denied\b", re.IGNORECASE),
    re.compile(r"\bblocked\b", re.IGNORECASE),
    re.compile(r"\bverify you are human\b", re.IGNORECASE),
)


def _looks_js_heavy(text: str) -> bool:
    sample = (text or "")[:5000].lower()
    if not sample:
        return False
    return any(
        marker in sample
        for marker in (
            "enable javascript",
            "requires javascript",
            "application/javascript",
            "react-root",
            "__next",
            "hydration",
        )
    )


def _looks_antibot_text(text: str) -> bool:
    sample = (text or "")[:8000]
    return any(p.search(sample) for p in _ANTI_BOT_PATTERNS)


def _domain_allowed(url: str, allowlist: list[str]) -> bool:
    """Return true when URL host is allowed for Playwright MCP fallback."""
    if not allowlist:
        return True

    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False

    for allowed in allowlist:
        d = allowed.strip().lower()
        if not d:
            continue
        if host == d or host.endswith(f".{d}"):
            return True
    return False


def _extract_mcp_text(result: Any) -> str:
    """Extract text payload from common MCP call_tool return shapes."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        texts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        texts.append(item["content"])
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(t for t in texts if t)

    # Python MCP SDK objects usually expose .content
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            item_text = getattr(item, "text", None)
            if isinstance(item_text, str):
                texts.append(item_text)
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)

    return str(result)


# ── Shared Playwright MCP session pool ───────────────────────
# Reuses a single npx subprocess + MCP ClientSession across all
# URLs in one pipeline run, avoiding cold-start overhead per URL.

class _PlaywrightMCPPool:
    """Manages a shared Playwright MCP subprocess and ClientSession."""

    def __init__(self, configuration: Configuration):
        self._configuration = configuration
        self._session: Any = None
        self._available = False

    @asynccontextmanager
    async def open(self):
        """Open the MCP subprocess and yield the pool.

        Usage::

            pool = _PlaywrightMCPPool(config)
            async with pool.open():
                text = await pool.navigate_and_snapshot(url)
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:
            safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
            print(f"[Web Crawler] Playwright MCP unavailable: {safe_exc}")
            yield self
            return

        server = StdioServerParameters(
            command=self._configuration.playwright_mcp_command,
            args=self._configuration.playwright_mcp_args,
        )

        try:
            async with stdio_client(server) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._available = True
                    print("[Web Crawler] Playwright MCP session pool ready.")
                    yield self
        except Exception as exc:
            safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
            print(f"[Web Crawler] Playwright MCP pool failed to start: {safe_exc}")
            yield self
        finally:
            self._session = None
            self._available = False

    async def navigate_and_snapshot(self, url: str) -> str:
        """Navigate to *url* and return the accessibility snapshot text.

        Wrapped in ``asyncio.wait_for`` so the caller is never blocked
        indefinitely if the MCP server or browser hangs.
        """
        if not self._available or self._session is None:
            return ""

        timeout_s = self._configuration.playwright_timeout_ms / 1000 + 10

        try:
            return await asyncio.wait_for(
                self._navigate_and_snapshot_inner(url),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            print(f"[Web Crawler] Playwright MCP timed out for {url}")
            return ""

    async def _navigate_and_snapshot_inner(self, url: str) -> str:
        session = self._session
        nav = await session.call_tool(
            "browser_navigate",
            {"url": url, "timeout": self._configuration.playwright_timeout_ms},
        )
        nav_text = _extract_mcp_text(nav)
        if "error" in nav_text.lower() and "timeout" in nav_text.lower():
            return ""

        snapshot = await session.call_tool("browser_snapshot", {})
        return _extract_mcp_text(snapshot)


async def _crawl_single(
    url: str,
    configuration: Configuration,
    mcp_pool: _PlaywrightMCPPool,
    shared_crawler: AsyncWebCrawler | None,
    crawl4ai_sem: asyncio.Semaphore,
) -> CrawledDoc | None:
    """Try crawl4ai, then Playwright MCP, then httpx, then ScraperAPI for anti-bot pages."""
    min_words = configuration.min_word_count
    anti_bot_suspected = False
    js_heavy_suspected = False

    # ── Attempt 1: crawl4ai (primary) ───────────────────────
    if shared_crawler is not None:
        try:
            # Bound crawl4ai calls independently from overall task concurrency.
            async with crawl4ai_sem:
                result = await shared_crawler.arun(url=url)
            text = result.markdown or result.extracted_content or ""
            js_heavy_suspected = _looks_js_heavy(text)
            anti_bot_suspected = anti_bot_suspected or _looks_antibot_text(text)
            word_count = len(text.split())
            if word_count >= min_words:
                return CrawledDoc(
                    url=url,
                    content=text,
                    word_count=word_count,
                    crawl_method="crawl4ai",
                )
            print(
                f"[Web Crawler] crawl4ai thin content for {url} "
                f"(words={word_count} < min_words={min_words})"
            )
        except Exception as exc:
            safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
            low_exc = safe_exc.lower()
            if "403" in low_exc or "429" in low_exc or "captcha" in low_exc or "cloudflare" in low_exc:
                anti_bot_suspected = True
            print(f"[Web Crawler] crawl4ai failed for {url}: {safe_exc}")

    # ── Attempt 2: Playwright MCP fallback for JS/thin pages ─
    should_try_playwright = configuration.enable_playwright_mcp and _domain_allowed(
        url, configuration.playwright_domain_allowlist
    )
    if should_try_playwright:
        try:
            text = await mcp_pool.navigate_and_snapshot(url)
            word_count = len(text.split())
            anti_bot_suspected = anti_bot_suspected or _looks_antibot_text(text)
            if word_count >= min_words:
                return CrawledDoc(
                    url=url,
                    content=text,
                    word_count=word_count,
                    crawl_method="playwright_mcp",
                )
            if js_heavy_suspected or word_count > 0:
                print(
                    f"[Web Crawler] Playwright returned thin content for {url} "
                    f"(words={word_count} < min_words={min_words})"
                )
        except Exception as exc:
            safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
            low_exc = safe_exc.lower()
            if "403" in low_exc or "429" in low_exc or "captcha" in low_exc or "cloudflare" in low_exc:
                anti_bot_suspected = True
            print(f"[Web Crawler] Playwright MCP failed for {url}: {safe_exc}")

    # ── Attempt 3: httpx fallback ───────────────────────────
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code in (403, 429):
                anti_bot_suspected = True
            resp.raise_for_status()
            text = resp.text
            anti_bot_suspected = anti_bot_suspected or _looks_antibot_text(text)
            word_count = len(text.split())
            if word_count >= min_words:
                return CrawledDoc(
                    url=url,
                    content=text,
                    word_count=word_count,
                    crawl_method="httpx",
                )
            print(
                f"[Web Crawler] httpx thin content for {url} "
                f"(words={word_count} < min_words={min_words})"
            )
    except Exception as exc:
        safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
        low_exc = safe_exc.lower()
        if "403" in low_exc or "429" in low_exc or "captcha" in low_exc or "cloudflare" in low_exc:
            anti_bot_suspected = True
        print(f"[Web Crawler] httpx fallback failed for {url}: {safe_exc}")

    # ── Attempt 4: ScraperAPI (anti-bot / challenge fallback) ─
    if getattr(configuration, "enable_scraperapi", False) and anti_bot_suspected:
        import os
        import urllib.parse

        try:
            api_key = os.getenv("SCRAPERAPI_KEY", "")
            if api_key:
                encoded_url = urllib.parse.quote(url)
                scraper_url = f"http://api.scraperapi.com?api_key={api_key}&url={encoded_url}&render=true"
                async with httpx.AsyncClient(timeout=45.0) as client:
                    resp = await client.get(scraper_url)
                    resp.raise_for_status()
                    raw_html = resp.text
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url, html=raw_html)
                    text = result.markdown or result.extracted_content or ""
                    word_count = len(text.split())
                    if word_count >= min_words:
                        return CrawledDoc(
                            url=url,
                            content=text,
                            word_count=word_count,
                            crawl_method="scraperapi",
                        )
                    print(
                        f"[Web Crawler] ScraperAPI thin content for {url} "
                        f"(words={word_count} < min_words={min_words})"
                    )
            else:
                print("[Web Crawler] ScraperAPI enabled but SCRAPERAPI_KEY is missing.")
        except Exception as exc:
            safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
            print(f"[Web Crawler] ScraperAPI failed for {url}: {safe_exc}")

    return None


async def crawl_pages(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Crawl all discovered URLs in parallel with a quality gate."""
    configuration = Configuration.from_runnable_config(config)
    min_words = max(1, int(configuration.min_word_count))

    preloaded_docs = [d for d in (state.preloaded_crawled_docs or []) if d.word_count >= min_words]
    preloaded_urls = {d.url for d in preloaded_docs}
    remaining_urls = [u for u in state.discovered_urls if u.url not in preloaded_urls]

    if preloaded_docs:
        print(
            f"[Web Crawler] Using {len(preloaded_docs)} preloaded docs "
            f"(method=openclaw, min_words={min_words})."
        )

    # Open a single shared Playwright MCP session for the entire batch
    mcp_pool = _PlaywrightMCPPool(configuration)

    async with mcp_pool.open():
        # Launch all crawls concurrently (with a semaphore to be polite)
        sem = asyncio.Semaphore(max(1, configuration.crawler_concurrency))
        crawl4ai_sem = asyncio.Semaphore(min(4, max(1, configuration.crawler_concurrency // 2)))

        shared_crawler: AsyncWebCrawler | None = None
        try:
            shared_crawler_ctx = AsyncWebCrawler()
            shared_crawler = await shared_crawler_ctx.__aenter__()
        except Exception as exc:
            safe_exc = str(exc).encode("ascii", errors="replace").decode("ascii")
            print(f"[Web Crawler] Failed to initialize shared crawl4ai session: {safe_exc}")
            shared_crawler = None

        try:
            async def _bounded(url: str) -> CrawledDoc | None:
                async with sem:
                    return await _crawl_single(
                        url,
                        configuration,
                        mcp_pool,
                        shared_crawler,
                        crawl4ai_sem,
                    )

            tasks = [_bounded(u.url) for u in remaining_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if shared_crawler is not None:
                await shared_crawler_ctx.__aexit__(None, None, None)

    # Collect successful results and log method distribution
    docs: list[CrawledDoc] = list(preloaded_docs)
    method_counts: dict[str, int] = {}
    if preloaded_docs:
        method_counts["openclaw"] = len(preloaded_docs)
    for r in results:
        if isinstance(r, CrawledDoc):
            docs.append(r)
            method_counts[r.crawl_method] = method_counts.get(r.crawl_method, 0) + 1

    methods_str = ", ".join(f"{m}={c}" for m, c in sorted(method_counts.items()))
    print(
        f"[Web Crawler] Crawled {len(docs)} pages "
        f"(of {len(state.discovered_urls)} discovered, {len(remaining_urls)} attempted crawl, "
        f"min_words={configuration.min_word_count}, "
        f"playwright_mcp={configuration.enable_playwright_mcp}, "
        f"methods=[{methods_str}])"
    )
    return {"crawled_docs": docs}
