"""Persistence helpers for agent-to-agent run logs and outputs."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient


async def save_a2a_run(
    *,
    payload: dict[str, Any],
    source: str = "api",
    mongo_uri: str | None = None,
    mongo_db_name: str | None = None,
) -> str:
    """Persist one A2A run payload for history inspection."""
    uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = mongo_db_name or os.getenv("MONGO_DB_NAME", "langgraph_crawler")
    client = AsyncIOMotorClient(uri)
    try:
        db = client[db_name]
        collection = db["a2a_runs"]

        now = datetime.now(timezone.utc)
        doc = {
            "session_id": str(payload.get("session_id", "")),
            "query": payload.get("query", ""),
            "status": payload.get("status", ""),
            "message": payload.get("message", ""),
            "required_metrics": list(payload.get("required_metrics", []) or []),
            "suggested_metrics": list(payload.get("suggested_metrics", []) or []),
            "user_metrics": list(payload.get("user_metrics", []) or []),
            "final_metrics": list(payload.get("final_metrics", []) or []),
            "available_metrics": list(payload.get("available_metrics", []) or []),
            "missing_metrics": list(payload.get("missing_metrics", []) or []),
            "missing_data_details": list(payload.get("missing_data_details", []) or []),
            "communication_log": list(payload.get("communication_log", []) or []),
            "entities": list(payload.get("entities", []) or []),
            "rounds_used": int(payload.get("rounds_used", 0) or 0),
            "cost_summary": dict(payload.get("cost_summary", {}) or {}),
            "source": source,
            "created_at": now,
            "updated_at": now,
        }

        result = await collection.insert_one(doc)
        return str(result.inserted_id)
    finally:
        client.close()
