"""Source Verifier node — scores credibility AND quality of crawled docs.

Key improvement over the old version:
- Detects LIST PAGES (pages listing 10+ entities) and scores them lower for ranking quality
- Rewards DETAIL PAGES (pages focused on one specific entity) with higher relevance
- Applies domain-specific trusted domain list
- Gives clear per-doc logging so you can see what passed/failed and why
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse

import replicate
from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import VerifiedSource
from crawler.state import State


# ── Trusted domains ───────────────────────────────────────────
TRUSTED_DOMAINS: set[str] = {
    ".gov", ".edu", ".ac.in", ".gov.in", ".nic.in", ".ac.uk", ".gov.uk",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nature.com", "science.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
    "github.com", "stackoverflow.com", "wikipedia.org", "britannica.com",
    # India education
    "timesofindia.com", "thehindu.com", "ndtv.com", "hindustantimes.com",
    "indianexpress.com", "livemint.com", "economictimes.com",
    "shiksha.com", "careers360.com", "collegedunia.com", "getmyuni.com",
    "iitb.ac.in", "iitd.ac.in", "iitm.ac.in", "iisc.ac.in",
    # Finance / business
    "moneycontrol.com", "businesstoday.in", "yourstory.com",
    # Movies/entertainment
    "imdb.com", "boxofficemojo.com",
}

# List-page signals — if content has these many entity mentions it's a list page
_LIST_PAGE_PATTERNS = [
    re.compile(r"\b\d+\s*\.\s+[A-Z]"),         # "1. JEE Advanced"
    re.compile(r"(?:top|best|toughest)\s+\d+",  re.IGNORECASE),
    re.compile(r"here\s+(?:is|are)\s+(?:the\s+)?(?:top|best|\d+)", re.IGNORECASE),
    re.compile(r"list\s+of\s+(?:top|best|\d+)", re.IGNORECASE),
]


def _is_trusted_domain(url: str) -> bool:
    host = urlparse(url).hostname or ""
    for td in TRUSTED_DOMAINS:
        if td.startswith("."):
            if host.endswith(td): return True
        elif host == td or host.endswith(f".{td}"):
            return True
    return False


def _detect_list_page(content: str, url: str) -> tuple[bool, int]:
    """
    Returns (is_list_page, entity_count_estimate).
    List pages contain many ranked items — good for discovering entities
    but bad for getting per-entity detail data for ranking.
    """
    sample = content[:3000]
    list_signals = sum(1 for p in _LIST_PAGE_PATTERNS if p.search(sample))

    # Count numbered items
    numbered = len(re.findall(r"^\s*\d+[\.\)]\s+[A-Z]", sample, re.MULTILINE))

    is_list = list_signals >= 2 or numbered >= 5
    return is_list, numbered


_VERIFY_PROMPT = """\
You are evaluating a web page for a research pipeline.

User's research question: {query}
URL: {url}
Page type: {page_type}

Content (first 1500 chars):
{content}

Score this page on:
1. credibility_score (0.0-1.0): Is this a trustworthy, authoritative source?
   - Official sites, established media, academic sources = 0.8-1.0
   - General education/career sites = 0.6-0.8
   - Unknown blogs, low-quality sites = 0.2-0.5
2. relevance_score (0.0-1.0): Does this page contain useful COMPARATIVE DATA for ranking?
   - Has specific numbers, statistics, pass rates, scores, dates = 0.8-1.0
   - Has qualitative comparisons or difficulty assessments = 0.6-0.8
   - Just lists names without data = 0.2-0.5
   - Off-topic = 0.0-0.2

Return ONLY a JSON object, no markdown:
{{"credibility_score": 0.0, "relevance_score": 0.0}}
"""


async def verify_sources(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    verified: list[VerifiedSource] = []

    list_page_count   = 0
    detail_page_count = 0
    rejected_count    = 0

    for doc in state.crawled_docs:
        is_trusted             = _is_trusted_domain(doc.url)
        is_list_page, n_items  = _detect_list_page(doc.content, doc.url)
        page_type              = f"LIST PAGE (~{n_items} items)" if is_list_page else "DETAIL PAGE"

        if is_list_page:
            list_page_count += 1
        else:
            detail_page_count += 1

        prompt = _VERIFY_PROMPT.format(
            query=state.user_query,
            url=doc.url,
            page_type=page_type,
            content=doc.content[:1500],
        )

        t0 = time.time()
        try:
            output   = replicate.run(configuration.model, input={"prompt": prompt, "max_tokens": 128, "temperature": 0.1})
            raw_text = "".join(str(c) for c in output)
            tracker.record(node="source_verifier", model=configuration.model,
                           input_tokens=len(prompt)//4, output_tokens=len(raw_text)//4,
                           latency_s=time.time()-t0)

            cleaned = raw_text.strip()
            if cleaned.startswith("```"): cleaned = cleaned.split("\n",1)[1].rsplit("```",1)[0]
            idx = cleaned.find("{")
            if idx != -1: cleaned = cleaned[idx:]
            scores = json.loads(cleaned)
            cred = float(scores.get("credibility_score", 0.55))
            rel  = float(scores.get("relevance_score", 0.55))

        except Exception as exc:
            print(f"[SourceVerifier] LLM scoring failed for {doc.url}: {exc}")
            cred = 0.7 if is_trusted else 0.55
            rel  = 0.4 if is_list_page else 0.6

        # Trusted domain boost
        if is_trusted:
            cred = min(1.0, cred + 0.1)

        # List pages: keep if credible (they help discover entity names)
        # but cap relevance so detail pages score higher
        if is_list_page:
            rel = min(rel, 0.65)

        passed = cred >= configuration.min_credibility
        status = "✓" if passed else "✗"
        print(
            f"[SourceVerifier] {status} {doc.url[:60]} | "
            f"{page_type} | cred={cred:.2f} rel={rel:.2f} trusted={is_trusted}"
        )

        if passed:
            verified.append(VerifiedSource(
                url=doc.url,
                content=doc.content,
                credibility_score=round(cred, 3),
                relevance_score=round(rel, 3),
                is_trusted=is_trusted,
            ))
        else:
            rejected_count += 1

    print(
        f"[SourceVerifier] {len(verified)}/{len(state.crawled_docs)} passed | "
        f"list_pages={list_page_count} detail_pages={detail_page_count} "
        f"rejected={rejected_count} (min_cred={configuration.min_credibility})"
    )
    return {"verified_sources": verified}