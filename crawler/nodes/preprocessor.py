"""Preprocessor node — extracts structured entities with metrics from verified documents.

Performs text cleaning and LLM-driven entity extraction. Instructs the LLM to identify
the underlying entities related to the user's query and their public metrics (funding, location, etc.).
Writes results to the `extracted_entities` MongoDB collection.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import replicate
from langchain_core.runnables import RunnableConfig
from motor.motor_asyncio import AsyncIOMotorClient

from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import ExtractedEntity
from crawler.state import State

_client: AsyncIOMotorClient | None = None
_chroma_kb_cache: dict[tuple[str, str, int], Any] = {}
_PLACEHOLDER_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "unknown",
    "not available",
    "not specified",
    "unspecified",
    "tbd",
    "-",
    "--",
}


def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(uri)
    return _client


def _get_chroma_kb(configuration: Configuration):
    """Lazy-initialise a Chroma knowledge-base client for extracted entities."""
    if not configuration.enable_chroma_sink:
        return None

    key = (
        configuration.chroma_persist_dir,
        configuration.chroma_entity_collection,
        configuration.chroma_embedding_dim,
    )
    kb = _chroma_kb_cache.get(key)
    if kb is not None:
        return kb

    try:
        from crawler.vector import ChromaKnowledgeBase
    except ModuleNotFoundError as exc:
        if exc.name == "chromadb":
            raise RuntimeError(
                "Chroma sink is enabled but 'chromadb' is not installed. "
                "Install it with: pip install -r requirements-kb.txt"
            ) from exc
        raise

    kb = ChromaKnowledgeBase(
        persist_dir=configuration.chroma_persist_dir,
        collection_name=configuration.chroma_entity_collection,
        embedding_dimensions=configuration.chroma_embedding_dim,
    )
    _chroma_kb_cache[key] = kb
    return kb


# ── Text cleaning ────────────────────────────────────────────
def _clean_text(text: str) -> str:
    """Strip leftover HTML artifacts and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # HTML entities
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _is_placeholder(value: Any) -> bool:
    return " ".join(str(value).strip().lower().split()) in _PLACEHOLDER_VALUES


def _merge_source_urls(existing: str, new_value: str) -> str:
    urls: list[str] = []
    seen: set[str] = set()
    for chunk in [existing, new_value]:
        for part in str(chunk).split(","):
            url = part.strip()
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return ", ".join(urls)


def _merge_metric_values(existing: Any, new_value: Any) -> str:
    entries: list[str] = []
    seen_norm: set[str] = set()

    for raw in [existing, new_value]:
        text = str(raw).strip()
        if not text:
            continue
        parts = [p.strip() for p in text.split("|")]
        for part in parts:
            if not part:
                continue
            norm = part.lower()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            entries.append(part)

    non_placeholder = [item for item in entries if not _is_placeholder(item)]
    if non_placeholder:
        return " | ".join(non_placeholder)
    return " | ".join(entries)


def _merge_metrics(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for key, value in existing.items():
        merged[str(key)] = str(value)

    for key, value in incoming.items():
        key_str = str(key)
        if key_str in merged:
            merged[key_str] = _merge_metric_values(merged[key_str], value)
        else:
            merged[key_str] = str(value)
    return merged


_EXTRACT_PROMPT = """\
You are an expert data extraction analyst building a comparison engine.
Given the user's search query and the text of a webpage, extract all specific ENTITIES 
that match the intent of the search (e.g., if the user searches for "startup incubators in India", 
extract each incubator mentioned).

For each entity, extract relevant public METRICS as a flat dictionary of strings.
Example metrics: "Location", "Funding Amount", "Equity Taken", "Industries", "Notable Startups".

Return a JSON array of objects. Each object MUST have exactly these keys:
- "name": String (Entity name)
- "description": String (1-2 sentence description)
- "metrics": Object (Key-value pairs of extracted metrics/data points. Keys should be Title Case.)
- "priority_score": Float 0.0-1.0 (How well this entity matches the user's core intent)

User query: {query}

Document Content (truncated to 4000 chars):
{content}

Return ONLY the JSON array, no markdown brackets ```json, no explanation. If no relevant entities are found, return an empty array [].
"""


async def preprocess(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Extract structured entities from verified documents."""
    configuration = Configuration.from_runnable_config(config)

    extracted: list[ExtractedEntity] = []
    entity_aggregator: dict[str, ExtractedEntity] = {}

    for src in state.verified_sources:
        clean_content = _clean_text(src.content)

        prompt = _EXTRACT_PROMPT.format(
            query=state.user_query,
            content=clean_content[:4000],
        )

        t0 = time.time()
        try:
            output = replicate.run(
                configuration.model,
                input={
                    "prompt": prompt,
                    "max_tokens": 1024,
                    "temperature": 0.1,
                },
            )
            raw_text = "".join(str(chunk) for chunk in output)
            latency = time.time() - t0

            input_tokens = len(prompt) // 4
            output_tokens = len(raw_text) // 4
            tracker.record(
                node="preprocessor",
                model=configuration.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency,
            )

            # Parse expected JSON array
            cleaned_resp = raw_text.strip()
            if cleaned_resp.startswith("```"):
                cleaned_resp = cleaned_resp.split("\n", 1)[1]
                cleaned_resp = cleaned_resp.rsplit("```", 1)[0]

            entities_data = json.loads(cleaned_resp)
            if not isinstance(entities_data, list):
                if isinstance(entities_data, dict) and "name" in entities_data:
                    entities_data = [entities_data]
                else:
                    entities_data = []

            for data in entities_data:
                name = data.get("name", "Unknown Entity")
                norm_name = name.lower().strip()
                if not norm_name:
                    continue

                desc = data.get("description", "")
                metrics = data.get("metrics", {})
                priority = float(data.get("priority_score", 0.5))

                if norm_name in entity_aggregator:
                    existing = entity_aggregator[norm_name]
                    # Keep the higher priority
                    if priority > existing.priority_score:
                        existing.priority_score = priority

                    # Keep the longest description
                    if len(desc) > len(existing.description):
                        existing.description = desc

                    # Merge strings for source URL just to keep track
                    if src.url not in existing.source_url:
                        existing.source_url += f", {src.url}"

                    # Merge metrics
                    for k, v in metrics.items():
                        if k in existing.metrics:
                            # If overlapping, just keep the longest/most descriptive, or concat
                            if str(v) not in str(existing.metrics[k]):
                                existing.metrics[k] = f"{existing.metrics[k]} | {v}"
                        else:
                            existing.metrics[k] = v
                else:
                    entity_aggregator[norm_name] = ExtractedEntity(
                        name=name,
                        description=desc,
                        metrics=metrics,
                        source_url=src.url,
                        priority_score=priority,
                        original_content=clean_content,
                    )
        except Exception as exc:
            print(f"[Preprocessor] Failed extraction for {src.url}: {exc}")

    # Transfer aggregated entities to the final list
    extracted = list(entity_aggregator.values())

    entity_vector_ids: list[str] = []

    async def _write_mongo_entities() -> None:
        if not extracted:
            return
        client = _get_client()
        db = client[configuration.mongo_db_name]
        proc_col = db["extracted_entities"]
        now = datetime.now(timezone.utc)
        for entity in extracted:
            payload = entity.model_dump()
            norm_name = _normalize_name(entity.name)
            existing = await proc_col.find_one(
                {
                    "session_id": state.session_id,
                    "normalized_name": norm_name,
                }
            )

            if existing:
                merged_metrics = _merge_metrics(
                    existing.get("metrics", {}) if isinstance(existing.get("metrics"), dict) else {},
                    payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {},
                )
                merged_source_url = _merge_source_urls(
                    str(existing.get("source_url", "")),
                    str(payload.get("source_url", "")),
                )
                existing_desc = str(existing.get("description", ""))
                incoming_desc = str(payload.get("description", ""))
                merged_desc = incoming_desc if len(incoming_desc) > len(existing_desc) else existing_desc
                existing_content = str(existing.get("original_content", ""))
                incoming_content = str(payload.get("original_content", ""))
                merged_content = (
                    incoming_content
                    if len(incoming_content) > len(existing_content)
                    else existing_content
                )
                merged_priority = max(
                    float(existing.get("priority_score", 0.0)),
                    float(payload.get("priority_score", 0.0)),
                )

                await proc_col.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$set": {
                            "description": merged_desc,
                            "metrics": merged_metrics,
                            "source_url": merged_source_url,
                            "priority_score": merged_priority,
                            "original_content": merged_content,
                            "updated_at": now,
                        }
                    },
                )
            else:
                await proc_col.insert_one(
                    {
                        "name": entity.name,
                        "normalized_name": norm_name,
                        **payload,
                        "session_id": state.session_id,
                        "updated_at": now,
                        "created_at": now,
                    }
                )

    async def _write_chroma_entities() -> list[str]:
        kb = _get_chroma_kb(configuration)
        if kb is None or not extracted:
            return []
        return await asyncio.to_thread(
            kb.upsert_extracted_entities,
            extracted,
            session_id=state.session_id,
            user_query=state.user_query,
        )

    if extracted:
        try:
            if configuration.enable_chroma_sink:
                _, entity_vector_ids = await asyncio.gather(
                    _write_mongo_entities(),
                    _write_chroma_entities(),
                )
            else:
                await _write_mongo_entities()
        except Exception as exc:
            print(f"[Preprocessor] Storage write failed: {exc}")
            raise

    # ── Attach cost summary to state ─────────────────────
    cost_summary = tracker.get_summary()

    print(
        "[Preprocessor] Extracted "
        f"{len(extracted)} distinct entities, indexed {len(entity_vector_ids)} vectors."
    )
    tracker.print_report()

    return {
        "extracted_entities": extracted,
        "entity_vector_ids": entity_vector_ids,
        "cost_summary": cost_summary,
    }
