"""Async Neo4j driver wrapper — with connectivity check and clear error messages."""

from __future__ import annotations
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

_driver = None
_neo4j_ok: bool | None = None   # cached ping result


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import AsyncGraphDatabase
        uri      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user     = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        _driver  = AsyncGraphDatabase.driver(uri, auth=(user, password))
    return _driver


async def check_neo4j_available() -> bool:
    """Ping Neo4j. Returns False instead of raising. Caches result."""
    global _neo4j_ok
    # Cache only positive state. If Neo4j was previously down, retry on next call.
    if _neo4j_ok is True:
        return True
    try:
        driver = get_driver()
        await driver.verify_connectivity()
        _neo4j_ok = True
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        print(f"[Neo4j] ✅ Connected to {uri}")
        return True
    except Exception as exc:
        _neo4j_ok = None
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        print(f"[Neo4j] ❌ Cannot connect to {uri}: {exc}")
        print("[Neo4j]    → Open Neo4j Desktop and click ▶ START on your instance")
        print("[Neo4j]    → Or use Neo4j Aura: https://console.neo4j.io")
        return False


async def close() -> None:
    global _driver, _neo4j_ok
    if _driver is not None:
        await _driver.close()
        _driver = None
    _neo4j_ok = None