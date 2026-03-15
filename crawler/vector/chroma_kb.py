"""ChromaDB-backed knowledge base helpers for crawler data."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable

import chromadb


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization suitable for a small local embedding."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize_metric_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def hash_embed(text: str, *, dimensions: int = 384) -> list[float]:
    """Generate a deterministic embedding vector from text.

    This is not a replacement for SOTA embedding models, but it is fast,
    local, and sufficient for a local vector-search baseline.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * dimensions

    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if (digest[4] & 1) else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "model_dump"):
        return dict(record.model_dump())
    if hasattr(record, "__dict__"):
        return dict(record.__dict__)
    raise TypeError(f"Unsupported record type: {type(record)!r}")


def _build_entity_document(entity: dict[str, Any]) -> str:
    metrics = entity.get("metrics") if isinstance(entity.get("metrics"), dict) else {}
    metrics_text = "\n".join(
        f"- {str(k).strip()}: {str(v).strip()}" for k, v in metrics.items()
    )

    return (
        f"Entity: {entity.get('name', '')}\n"
        f"Description: {entity.get('description', '')}\n"
        f"Source URL: {entity.get('source_url', '')}\n"
        f"User Query: {entity.get('user_query', '')}\n"
        "Metrics:\n"
        f"{metrics_text}"
    ).strip()


def _build_mongo_entity_id(entity: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(entity.get("session_id", "")),
            str(entity.get("name", "")),
            str(entity.get("source_url", "")),
            str(entity.get("description", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _build_entity_metadata(entity: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"record_type": "entity"}

    if entity.get("name"):
        metadata["entity_name"] = str(entity["name"])
    if entity.get("source_url"):
        metadata["source_url"] = str(entity["source_url"])
    if entity.get("session_id"):
        metadata["session_id"] = str(entity["session_id"])
    if entity.get("_id"):
        metadata["mongo_id"] = str(entity["_id"])
    if entity.get("created_at"):
        metadata["created_at"] = str(entity["created_at"])
    if entity.get("updated_at"):
        metadata["updated_at"] = str(entity["updated_at"])

    priority = entity.get("priority_score")
    if isinstance(priority, (int, float)):
        metadata["priority_score"] = float(priority)

    metrics = entity.get("metrics")
    if isinstance(metrics, dict):
        norm_keys = sorted(
            {
                _normalize_metric_key(str(metric_key))
                for metric_key in metrics.keys()
                if str(metric_key).strip()
            }
        )
        metadata["metric_count"] = len(norm_keys)
        if norm_keys:
            metadata["metric_keys_csv"] = ",".join(norm_keys)

    return metadata


def _build_verified_source_document(source: dict[str, Any], *, user_query: str) -> str:
    return (
        f"Source URL: {source.get('url', '')}\n"
        f"User Query: {user_query}\n"
        f"Credibility Score: {source.get('credibility_score', '')}\n"
        f"Relevance Score: {source.get('relevance_score', '')}\n"
        f"Is Trusted: {source.get('is_trusted', '')}\n"
        f"Content:\n{source.get('content', '')}"
    ).strip()


def _build_verified_source_id(source: dict[str, Any], *, session_id: str) -> str:
    raw = "|".join(
        [
            str(session_id),
            str(source.get("url", "")),
            str(source.get("credibility_score", "")),
            str(source.get("relevance_score", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _build_verified_source_metadata(
    source: dict[str, Any],
    *,
    session_id: str,
    user_query: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "record_type": "verified_source",
        "session_id": session_id,
        "user_query": user_query,
    }

    if source.get("url"):
        metadata["source_url"] = str(source["url"])

    credibility = source.get("credibility_score")
    if isinstance(credibility, (int, float)):
        metadata["credibility_score"] = float(credibility)

    relevance = source.get("relevance_score")
    if isinstance(relevance, (int, float)):
        metadata["relevance_score"] = float(relevance)

    if isinstance(source.get("is_trusted"), bool):
        metadata["is_trusted"] = bool(source["is_trusted"])

    return metadata


class ChromaKnowledgeBase:
    """Persistent Chroma collection with upsert/query helpers."""

    def __init__(
        self,
        *,
        persist_dir: str = "./chroma_db",
        collection_name: str = "crawler_kb",
        embedding_dimensions: int = 384,
    ) -> None:
        self.embedding_dimensions = embedding_dimensions
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert_records(self, records: Iterable[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []

        for record in records:
            item_id = str(record["id"])
            document = str(record["document"])
            metadata = dict(record.get("metadata", {}))

            ids.append(item_id)
            docs.append(document)
            metadatas.append(metadata)
            embeddings.append(hash_embed(document, dimensions=self.embedding_dimensions))

        if not ids:
            return []

        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return ids

    def upsert_entities(self, entities: Iterable[dict[str, Any]]) -> int:
        """Upsert entities from MongoDB-shaped dictionaries."""
        records: list[dict[str, Any]] = []
        for entity in entities:
            records.append(
                {
                    "id": _build_mongo_entity_id(entity),
                    "document": _build_entity_document(entity),
                    "metadata": _build_entity_metadata(entity),
                }
            )
        return len(self._upsert_records(records))

    def upsert_extracted_entities(
        self,
        entities: Iterable[Any],
        *,
        session_id: str,
        user_query: str,
    ) -> list[str]:
        """Upsert entity models generated in the current pipeline run."""
        records: list[dict[str, Any]] = []

        for entity_obj in entities:
            entity = _as_dict(entity_obj)
            entity["session_id"] = session_id
            entity["user_query"] = user_query

            raw_id = "|".join(
                [
                    session_id,
                    str(entity.get("name", "")),
                    str(entity.get("source_url", "")),
                    str(entity.get("description", "")),
                ]
            )
            records.append(
                {
                    "id": hashlib.sha1(raw_id.encode("utf-8")).hexdigest(),
                    "document": _build_entity_document(entity),
                    "metadata": _build_entity_metadata(entity),
                }
            )

        return self._upsert_records(records)

    def upsert_verified_sources(
        self,
        sources: Iterable[Any],
        *,
        session_id: str,
        user_query: str,
    ) -> list[str]:
        """Upsert verified source documents from the crawler run."""
        records: list[dict[str, Any]] = []

        for source_obj in sources:
            source = _as_dict(source_obj)
            records.append(
                {
                    "id": _build_verified_source_id(source, session_id=session_id),
                    "document": _build_verified_source_document(
                        source,
                        user_query=user_query,
                    ),
                    "metadata": _build_verified_source_metadata(
                        source,
                        session_id=session_id,
                        user_query=user_query,
                    ),
                }
            )

        return self._upsert_records(records)

    def query(
        self,
        *,
        query_text: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query the knowledge base and return ranked matches."""
        query_embedding = hash_embed(
            query_text,
            dimensions=self.embedding_dimensions,
        )
        where = {"session_id": session_id} if session_id else None

        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        ids = response.get("ids", [[]])[0]

        results: list[dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            results.append(
                {
                    "id": ids[idx] if idx < len(ids) else "",
                    "distance": distances[idx] if idx < len(distances) else None,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "document": doc,
                }
            )
        return results

    def peek(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Inspect a few stored records without semantic querying."""
        data = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        records: list[dict[str, Any]] = []
        for idx, item_id in enumerate(ids):
            records.append(
                {
                    "id": item_id,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "document": docs[idx] if idx < len(docs) else "",
                }
            )
        return records

    def get_records(
        self,
        *,
        where: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch records from the collection by metadata filter."""
        data = self.collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        records: list[dict[str, Any]] = []
        for idx, item_id in enumerate(ids):
            records.append(
                {
                    "id": item_id,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "document": docs[idx] if idx < len(docs) else "",
                }
            )
        return records

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        return self.collection.count()