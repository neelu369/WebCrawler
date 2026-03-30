"""MongoDB Logger node — persists verified documents to MongoDB + ChromaDB.

MongoDB is OPTIONAL. If MONGO_URI is missing, wrong, or the server is
unreachable, the node logs a clear warning and continues the pipeline using
a UUID as the session_id. The rest of the pipeline (Neo4j, ranking) is
unaffected.

Common misconfiguration: MONGO_URI set to the Neo4j bolt URI (bolt://...)
Fix: MONGO_URI=mongodb://localhost:27017
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from crawler.config import Configuration
from crawler.state import State

_client: Any = None           # AsyncIOMotorClient or None
_mongo_ok: bool | None = None # cached connection check result
_chroma_kb_cache: dict[tuple[str, str, int], Any] = {}


# ── MongoDB client ────────────────────────────────────────────

def _validate_mongo_uri(uri: str) -> str | None:
    """Return an error message if URI is invalid, else None."""
    if not uri.startswith(("mongodb://", "mongodb+srv://")):
        if uri.startswith(("bolt://", "neo4j://", "bolt+s://", "neo4j+s://")):
            return (
                f"MONGO_URI looks like a Neo4j bolt URI: {uri!r}\n"
                "Fix your .env:\n"
                "  MONGO_URI=mongodb://localhost:27017\n"
                "  NEO4J_URI=bolt://localhost:7687"
            )
        return (
            f"MONGO_URI is invalid: {uri!r}\n"
            "It must start with 'mongodb://' or 'mongodb+srv://'\n"
            "Example: MONGO_URI=mongodb://localhost:27017"
        )
    return None


def _get_client() -> Any:
    """Return a motor AsyncIOMotorClient, or raise with a clear message."""
    global _client
    if _client is None:
        from motor.motor_asyncio import AsyncIOMotorClient
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        err = _validate_mongo_uri(uri)
        if err:
            raise ValueError(f"[MongoLogger] {err}")
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    return _client


async def _check_mongo_reachable() -> bool:
    """Ping MongoDB. Returns False instead of raising.

    Only caches a positive result. If MongoDB was previously down,
    retries on the next call (matching neo4j_client behavior).
    """
    global _mongo_ok
    # Cache only positive state — retry if previously failed.
    if _mongo_ok is True:
        return True
    try:
        client = _get_client()
        await client.admin.command("ping")
        _mongo_ok = True
        print("[MongoLogger] ✅ MongoDB connection OK.")
        return True
    except Exception as exc:
        _mongo_ok = None  # Don't cache False — allow retry
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        err = _validate_mongo_uri(uri)
        if err:
            print(f"[MongoLogger] ❌ Config error — {err}")
        else:
            print(f"[MongoLogger] ❌ Cannot reach MongoDB at {uri!r}: {exc}")
            print("[MongoLogger]    Is MongoDB running? Start it with: mongod --dbpath /data/db")
        return False


# ── Chroma KB ─────────────────────────────────────────────────

def _get_chroma_kb(configuration: Configuration) -> Any:
    if not configuration.enable_chroma_sink:
        return None
    key = (configuration.chroma_persist_dir, configuration.chroma_raw_collection, configuration.chroma_embedding_dim)
    if key in _chroma_kb_cache:
        return _chroma_kb_cache[key]
    try:
        from crawler.vector import ChromaKnowledgeBase
        kb = ChromaKnowledgeBase(
            persist_dir=configuration.chroma_persist_dir,
            collection_name=configuration.chroma_raw_collection,
            embedding_dimensions=configuration.chroma_embedding_dim,
        )
        _chroma_kb_cache[key] = kb
        return kb
    except Exception as exc:
        print(f"[MongoLogger] ChromaDB unavailable: {exc}. Skipping vector sink.")
        return None


# ── Main node ─────────────────────────────────────────────────

async def log_to_mongo(
    state: State, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Persist verified documents to MongoDB + ChromaDB.
    MongoDB is optional — if unreachable, pipeline continues with a generated session_id.
    """
    configuration = Configuration.from_runnable_config(config)

    # Always generate a session_id even if Mongo is down
    session_id = state.session_id or str(uuid.uuid4()).replace("-", "")[:24]

    mongo_available = await _check_mongo_reachable()
    raw_doc_ids: list[str] = []

    if mongo_available:
        try:
            client  = _get_client()
            db      = client[configuration.mongo_db_name]
            raw_col = db["raw_documents"]
            ses_col = db["sessions"]
            now     = datetime.now(timezone.utc)

            # Create or update session document
            if not state.session_id:
                result     = await ses_col.insert_one({"user_query": state.user_query, "status": "crawling", "created_at": now, "updated_at": now})
                session_id = str(result.inserted_id)
            else:
                try:
                    from bson import ObjectId
                    await ses_col.update_one({"_id": ObjectId(session_id)}, {"$set": {"status": "crawling", "updated_at": now}})
                except Exception:
                    pass

            # Upsert crawled documents
            for src in state.verified_sources:
                try:
                    result = await raw_col.update_one(
                        {"url": src.url},
                        {"$set": {"url": src.url, "content": src.content, "credibility_score": src.credibility_score, "relevance_score": src.relevance_score, "is_trusted": src.is_trusted, "session_id": session_id, "updated_at": now}, "$setOnInsert": {"created_at": now}},
                        upsert=True,
                    )
                    raw_doc_ids.append(str(result.upserted_id) if result.upserted_id else src.url)
                except Exception as exc:
                    print(f"[MongoLogger] Failed to upsert {src.url}: {exc}")
                    raw_doc_ids.append(src.url)

            print(f"[MongoLogger] Stored {len(raw_doc_ids)} docs. session_id={session_id}")

        except Exception as exc:
            print(f"[MongoLogger] MongoDB write failed: {exc}. Continuing without persistence.")
    else:
        print(f"[MongoLogger] MongoDB unavailable — skipping persistence. session_id={session_id}")
        raw_doc_ids = [src.url for src in state.verified_sources]

    # ChromaDB (also optional)
    raw_vector_ids: list[str] = []
    if configuration.enable_chroma_sink:
        try:
            kb = _get_chroma_kb(configuration)
            if kb is not None:
                raw_vector_ids = await asyncio.to_thread(
                    kb.upsert_verified_sources,
                    state.verified_sources,
                    session_id=session_id,
                    user_query=state.user_query,
                )
        except Exception as exc:
            print(f"[MongoLogger] ChromaDB write failed: {exc}. Continuing.")

    return {
        "raw_doc_ids":    raw_doc_ids,
        "raw_vector_ids": raw_vector_ids,
        "session_id":     session_id,
    }
