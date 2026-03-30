"""URL relevance agent for pre-crawl filtering.

The agent evaluates whether a discovered URL is relevant to the active
user query before expensive crawling begins.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlparse

from crawler.llm import replicate

from crawler.cost_tracker import tracker

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "about", "into", "your", "their", "have", "has", "had", "what", "when",
    "where", "which", "will", "would", "could", "should", "than", "then", "them",
    "they", "you", "our", "its", "who", "why", "how", "all", "any", "each",
    "can", "may", "get", "use", "using", "used", "more", "most", "latest",
}

_BINARY_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".mp3", ".mp4", ".avi", ".mov",
)

_RELEVANCE_PROMPT = """\
You are a strict URL relevance classifier for a web crawler.

User query:
{query}

Candidate URL:
{url}

Title:
{title}

Snippet:
{snippet}

Return ONLY valid JSON:
{{"relevant": true/false, "confidence": 0.0-1.0, "reason": "short reason"}}

Mark relevant=true only if the URL is likely about the same topic and useful for extracting factual data.
"""


@dataclass(frozen=True)
class URLRelevanceDecision:
    is_relevant: bool
    confidence: float
    method: str
    overlap_score: float = 0.0
    reason: str = ""


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _url_to_text(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").replace(".", " ")
    path = unquote(parsed.path).replace("/", " ")
    return f"{host} {path}".strip()


def _looks_like_binary_url(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return any(path.endswith(ext) for ext in _BINARY_EXTENSIONS)


class URLRelevanceAgent:
    """Generic URL relevance scorer driven by the active user query."""

    def __init__(
        self,
        *,
        model: str,
        min_overlap_score: float = 0.2,
        enable_llm_tiebreak: bool = True,
        uncertain_low: float = 0.08,
    ) -> None:
        self.model = model
        self.min_overlap_score = min_overlap_score
        self.enable_llm_tiebreak = enable_llm_tiebreak
        self.uncertain_low = uncertain_low

    def _lexical_overlap(self, *, query: str, url: str, title: str, snippet: str) -> float:
        query_terms = _tokenize(query)
        if not query_terms:
            return 0.0

        candidate_terms = _tokenize(f"{title} {snippet} {_url_to_text(url)}")
        if not candidate_terms:
            return 0.0

        overlap = len(query_terms.intersection(candidate_terms))
        denominator = max(1, min(len(query_terms), 10))
        return overlap / denominator

    def _heuristic_decision(
        self, *, query: str, url: str, title: str, snippet: str
    ) -> Optional[URLRelevanceDecision]:
        if _looks_like_binary_url(url):
            return URLRelevanceDecision(
                is_relevant=False,
                confidence=0.95,
                method="heuristic",
                reason="binary_or_media_url",
            )

        overlap = self._lexical_overlap(query=query, url=url, title=title, snippet=snippet)

        if overlap >= self.min_overlap_score:
            confidence = min(0.95, 0.6 + overlap)
            return URLRelevanceDecision(
                is_relevant=True,
                confidence=confidence,
                method="heuristic",
                overlap_score=overlap,
                reason="query_term_overlap",
            )

        if overlap <= self.uncertain_low:
            return URLRelevanceDecision(
                is_relevant=False,
                confidence=max(0.55, 0.85 - overlap),
                method="heuristic",
                overlap_score=overlap,
                reason="very_low_overlap",
            )

        return None

    def _llm_decision(
        self, *, query: str, url: str, title: str, snippet: str, overlap: float
    ) -> Optional[URLRelevanceDecision]:
        if not self.enable_llm_tiebreak:
            return None

        prompt = _RELEVANCE_PROMPT.format(
            query=query.strip(),
            url=url.strip(),
            title=(title or "").strip()[:300],
            snippet=(snippet or "").strip()[:600],
        )

        t0 = time.time()
        try:
            output = replicate.run(
                self.model,
                input={"prompt": prompt, "max_tokens": 120, "temperature": 0.0},
            )
            raw_text = "".join(str(chunk) for chunk in output).strip()

            tracker.record(
                node="url_relevance_agent",
                model=self.model,
                input_tokens=len(prompt) // 4,
                output_tokens=len(raw_text) // 4,
                latency_s=time.time() - t0,
            )

            cleaned = raw_text
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            idx = cleaned.find("{")
            if idx != -1:
                cleaned = cleaned[idx:]

            parsed = json.loads(cleaned)
            is_relevant = bool(parsed.get("relevant", False))
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(parsed.get("reason", "")).strip()

            return URLRelevanceDecision(
                is_relevant=is_relevant,
                confidence=confidence,
                method="llm",
                overlap_score=overlap,
                reason=reason,
            )
        except Exception as exc:
            print(f"[URLRelevanceAgent] LLM relevance check failed for {url}: {exc}")
            return None

    def assess(self, *, query: str, url: str, title: str = "", snippet: str = "") -> URLRelevanceDecision:
        heuristic = self._heuristic_decision(query=query, url=url, title=title, snippet=snippet)
        if heuristic is not None:
            return heuristic

        overlap = self._lexical_overlap(query=query, url=url, title=title, snippet=snippet)
        llm = self._llm_decision(
            query=query, url=url, title=title, snippet=snippet, overlap=overlap
        )
        if llm is not None:
            return llm

        # Conservative fallback when LLM tie-breaker is unavailable.
        return URLRelevanceDecision(
            is_relevant=overlap >= self.min_overlap_score,
            confidence=0.5,
            method="fallback",
            overlap_score=overlap,
            reason="fallback_overlap_threshold",
        )
