"""Structuring Agent — fetches raw entity data from ChromaDB, calls the LLM
to produce a clean structured table, detects missing/null values, and
communicates gaps back to the Crawler Agent for targeted re-crawls.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from crawler.llm import replicate

from crawler.cost_tracker import tracker


def _get_chroma_class():
    """Lazy import of ChromaKnowledgeBase — avoids DLL load at startup."""
    try:
        from crawler.vector.chroma_kb import ChromaKnowledgeBase
        return ChromaKnowledgeBase
    except (ImportError, Exception) as exc:
        raise RuntimeError(
            f"ChromaDB is not available: {exc}\n"
            "The A2A structuring path requires chromadb. "
            "The primary ranking path (Neo4j) works without it."
        ) from exc

_MAX_DATA_CHARS = 8000 * 4
_ENTITY_SUMMARY_CHARS = 400
_PATCH_ROW_CHARS = 200
_PATCH_ENTITY_CHARS = 400
_BATCH_MAX_ENTITIES = 20

_MISSING_SENTINELS: frozenset[str] = frozenset({
    "", "n/a", "na", "null", "none", "not available",
    "not found", "unknown", "-", "–", "?",
})


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _MISSING_SENTINELS


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _slim_entity(entity: dict[str, Any], max_chars: int = _ENTITY_SUMMARY_CHARS) -> dict[str, Any]:
    slim = {"name": entity.get("name", ""), "metrics": entity.get("metrics") or {}}
    serialised = json.dumps(slim, ensure_ascii=True)
    if len(serialised) <= max_chars:
        return slim
    metrics = dict(slim["metrics"])
    while metrics and len(json.dumps({"name": slim["name"], "metrics": metrics})) > max_chars:
        metrics.popitem()
    slim["metrics"] = metrics
    return slim


@dataclass
class StructuredRow:
    entity_name: str
    source_url: str
    fields: dict[str, str] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"entity_name": self.entity_name, "source_url": self.source_url, "fields": self.fields, "missing_keys": self.missing_keys}


@dataclass
class MissingFieldsReport:
    session_id: str
    user_query: str
    missing_by_entity: dict[str, list[str]] = field(default_factory=dict)
    missing_columns: list[str] = field(default_factory=list)
    total_missing_cells: int = 0

    def is_complete(self) -> bool:
        return self.total_missing_cells == 0

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "user_query": self.user_query, "missing_by_entity": self.missing_by_entity, "missing_columns": self.missing_columns, "total_missing_cells": self.total_missing_cells}


@dataclass
class StructuredTable:
    session_id: str
    user_query: str
    columns: list[str] = field(default_factory=list)
    rows: list[StructuredRow] = field(default_factory=list)
    missing_report: MissingFieldsReport | None = None
    round_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "user_query": self.user_query, "columns": self.columns, "rows": [r.to_dict() for r in self.rows], "missing_report": self.missing_report.to_dict() if self.missing_report else None, "round_number": self.round_number}


_COLUMN_DISCOVERY_PROMPT = """\
You are a data analyst. Given entity names and their raw metrics, identify the complete set of column names for a unified comparison table.

User query: {query}

Entities sample (names + metric keys only):
{entity_sample}

Return ONLY a JSON array of column names in Title Case. Example:
["Location", "Funding Amount", "Equity Taken", "Industries"]
No markdown, no explanation."""

_BATCH_STRUCTURE_PROMPT = """\
You are a data structuring expert. Unify these entities into structured rows for a comparison table.

User query: {query}
Required columns: {columns}

Entities to structure (batch {batch_num}/{total_batches}):
{batch_entities}

Return ONLY a JSON array. Each element:
{{"entity_name":"...","source_url":"...","fields":{{"Column":"value or null",...}}}}
No markdown, no explanation. If a column value is unknown use null."""

_PATCH_PROMPT = """\
You are a data-fill expert. Fill missing cells using new crawled data.

User query: {query}
Missing columns to fill: {missing_columns}

Incomplete rows (up to {max_rows} shown):
{incomplete_rows}

New patch entities:
{patch_entities}

Instructions:
1. Match patch entities to rows by entity name.
2. Fill only the missing columns where data exists in patch.
3. Do NOT invent values — use null if not found.

Return ONLY a JSON array with the same row shape:
{{"entity_name":"...","source_url":"...","fields":{{"Column":"value or null",...}}}}
No markdown, no explanation."""


class StructuringAgent:

    def __init__(self, *, chroma_persist_dir: str = "./chroma_db", chroma_entity_collection: str = "crawler_entities", chroma_embedding_dim: int = 384, model: str = "meta-llama/llama-3-70b-instruct", max_chroma_records: int = 500) -> None:
        ChromaKnowledgeBase = _get_chroma_class()
        self.kb = ChromaKnowledgeBase(persist_dir=chroma_persist_dir, collection_name=chroma_entity_collection, embedding_dimensions=chroma_embedding_dim)
        self.model = model
        self.max_chroma_records = max_chroma_records

    def _fetch_raw_entities(self, session_id: str) -> list[dict[str, Any]]:
        records = self.kb.get_records(where={"session_id": session_id}, limit=self.max_chroma_records)
        entities = []
        for record in records:
            meta = record.get("metadata") or {}
            if meta.get("record_type") != "entity":
                continue
            parsed = _parse_entity_document(record.get("document", ""))
            parsed["source_url"] = meta.get("source_url", "")
            parsed["session_id"] = session_id
            entities.append(parsed)
        return entities

    def _call_llm(self, prompt: str, *, node_label: str, max_tokens: int = 1024) -> str:
        max_prompt_chars = 7000 * 4
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars] + "\n[TRUNCATED]"
        t0 = time.time()
        output = replicate.run(self.model, input={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.1})
        raw = "".join(str(c) for c in output)
        tracker.record(node=node_label, model=self.model, input_tokens=len(prompt)//4, output_tokens=len(raw)//4, latency_s=time.time()-t0)
        return raw

    def _parse_llm_json(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        for start_char in ("[", "{"):
            idx = cleaned.find(start_char)
            if idx != -1:
                cleaned = cleaned[idx:]; break
        return json.loads(cleaned.strip())

    def _discover_columns(self, raw_entities: list[dict[str, Any]], user_query: str) -> list[str]:
        sample = [{"name": e.get("name", ""), "metric_keys": list((e.get("metrics") or {}).keys())} for e in raw_entities[:30]]
        prompt = _COLUMN_DISCOVERY_PROMPT.format(query=user_query, entity_sample=json.dumps(sample, ensure_ascii=True))
        raw = self._call_llm(prompt, node_label="structuring_agent_cols", max_tokens=256)
        try:
            cols = self._parse_llm_json(raw)
            if isinstance(cols, list) and cols:
                return [str(c).strip() for c in cols if str(c).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        all_keys: set[str] = set()
        for e in raw_entities:
            for k in (e.get("metrics") or {}).keys():
                all_keys.add(k.strip().title())
        return sorted(all_keys)

    def _structure_batch(self, batch, columns, user_query, batch_num, total_batches) -> list[dict]:
        slim_batch = [_slim_entity(e) for e in batch]
        prompt = _BATCH_STRUCTURE_PROMPT.format(query=user_query, columns=json.dumps(columns), batch_entities=json.dumps(slim_batch, ensure_ascii=True), batch_num=batch_num, total_batches=total_batches)
        raw = self._call_llm(prompt, node_label="structuring_agent", max_tokens=4096)
        try:
            result = self._parse_llm_json(raw)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[StructuringAgent] Batch {batch_num} parse failed: {exc}. Using fallback.")
        return _fallback_batch(batch, columns)

    def _scan_missing(self, rows, columns, session_id, user_query) -> MissingFieldsReport:
        report = MissingFieldsReport(session_id=session_id, user_query=user_query)
        missing_columns_set: set[str] = set()
        for row in rows:
            missing_for_row = []
            for col in columns:
                if _is_missing(row.fields.get(col)):
                    missing_for_row.append(col); missing_columns_set.add(col)
            if missing_for_row:
                row.missing_keys = missing_for_row
                report.missing_by_entity[row.entity_name] = missing_for_row
                report.total_missing_cells += len(missing_for_row)
        report.missing_columns = sorted(missing_columns_set)
        return report

    def structure(self, *, session_id: str, user_query: str, round_number: int = 1) -> StructuredTable:
        print(f"[StructuringAgent] Fetching ChromaDB for session={session_id!r}")
        raw_entities = self._fetch_raw_entities(session_id)
        if not raw_entities:
            print("[StructuringAgent] No entity records found.")
            return StructuredTable(session_id=session_id, user_query=user_query, missing_report=MissingFieldsReport(session_id=session_id, user_query=user_query, total_missing_cells=-1), round_number=round_number)

        print(f"[StructuringAgent] {len(raw_entities)} raw entities.")
        columns = self._discover_columns(raw_entities, user_query)
        print(f"[StructuringAgent] Columns: {columns}")

        batches, current_batch, current_size = [], [], 0
        for entity in raw_entities:
            slim = _slim_entity(entity)
            sz = len(json.dumps(slim, ensure_ascii=True))
            if (current_size + sz > _MAX_DATA_CHARS or len(current_batch) >= _BATCH_MAX_ENTITIES) and current_batch:
                batches.append(current_batch); current_batch = []; current_size = 0
            current_batch.append(entity); current_size += sz
        if current_batch: batches.append(current_batch)

        print(f"[StructuringAgent] {len(batches)} batch(es)...")
        all_raw_rows = []
        for i, batch in enumerate(batches, 1):
            print(f"[StructuringAgent] Batch {i}/{len(batches)} ({len(batch)} entities)...")
            all_raw_rows.extend(self._structure_batch(batch, columns, user_query, i, len(batches)))

        seen: dict[str, dict] = {}
        for r in all_raw_rows:
            name = r.get("entity_name", "").strip().lower()
            if not name: continue
            if name not in seen:
                seen[name] = r
            else:
                existing_filled = sum(1 for v in seen[name].get("fields", {}).values() if not _is_missing(v))
                new_filled = sum(1 for v in r.get("fields", {}).values() if not _is_missing(v))
                if new_filled > existing_filled: seen[name] = r

        rows = [StructuredRow(entity_name=r.get("entity_name","Unknown"), source_url=r.get("source_url",""), fields={col: r.get("fields",{}).get(col) for col in columns}) for r in seen.values()]
        print(f"[StructuringAgent] {len(rows)} rows, {len(columns)} columns.")
        missing_report = self._scan_missing(rows, columns, session_id, user_query)
        print(f"[StructuringAgent] Missing cells: {missing_report.total_missing_cells}")
        return StructuredTable(session_id=session_id, user_query=user_query, columns=columns, rows=rows, missing_report=missing_report, round_number=round_number)

    def _patch_batch(self, batch_rows, patch_entities, missing_cols, user_query) -> dict[str, dict]:
        slim_rows = [{"entity_name": r.entity_name, "source_url": r.source_url, "fields": {k: r.fields.get(k) for k in r.missing_keys}} for r in batch_rows]
        slim_patches = [_slim_entity(e, max_chars=_PATCH_ENTITY_CHARS) for e in patch_entities]
        prompt = _PATCH_PROMPT.format(query=user_query, missing_columns=json.dumps(missing_cols), incomplete_rows=json.dumps(slim_rows, ensure_ascii=True), patch_entities=json.dumps(slim_patches, ensure_ascii=True), max_rows=len(slim_rows))
        raw_response = self._call_llm(prompt, node_label="structuring_agent_patch", max_tokens=4096)
        try:
            updated_rows_data: list[dict] = self._parse_llm_json(raw_response)
            if not isinstance(updated_rows_data, list): raise ValueError("Expected JSON array.")
            return {item.get("entity_name","").strip().lower(): item for item in updated_rows_data if item.get("entity_name","").strip()}
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[StructuringAgent] Patch batch parse failed: {exc}. Skipping."); return {}

    def patch(self, *, table: StructuredTable, patch_entities: list[dict[str, Any]]) -> StructuredTable:
        if not patch_entities: return table
        incomplete_rows = [r for r in table.rows if r.missing_keys]
        if not incomplete_rows: return table
        missing_cols = table.missing_report.missing_columns if table.missing_report else []
        BATCH_SIZE = 10
        print(f"[StructuringAgent] Patching {len(incomplete_rows)} rows with {len(patch_entities)} new entities...")
        all_updates: dict[str, dict] = {}
        for i in range(0, len(incomplete_rows), BATCH_SIZE):
            batch = incomplete_rows[i: i + BATCH_SIZE]
            all_updates.update(self._patch_batch(batch, patch_entities, missing_cols, table.user_query))
        for row in table.rows:
            key = row.entity_name.strip().lower()
            if key in all_updates:
                for col, val in all_updates[key].get("fields", {}).items():
                    if col in row.fields and _is_missing(row.fields.get(col)):
                        row.fields[col] = val
        new_report = self._scan_missing(table.rows, table.columns, table.session_id, table.user_query)
        table.missing_report = new_report; table.round_number += 1
        print(f"[StructuringAgent] After patch: {new_report.total_missing_cells} missing cells remain")
        return table


def _parse_entity_document(doc_text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"name": "", "description": "", "metrics": {}}
    in_metrics = False
    for line in doc_text.splitlines():
        if line.startswith("Entity:"): result["name"] = line[len("Entity:"):].strip()
        elif line.startswith("Description:"): result["description"] = line[len("Description:"):].strip()
        elif line.startswith("Metrics:"): in_metrics = True
        elif in_metrics and line.startswith("- "):
            inner = line[2:]
            if ":" in inner:
                k, _, v = inner.partition(":"); result["metrics"][k.strip()] = v.strip()
    return result


def _fallback_batch(batch: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for ent in batch:
        raw_metrics = ent.get("metrics") or {}
        metric_lookup = {k.strip().title(): v for k, v in raw_metrics.items()}
        rows.append({"entity_name": ent.get("name","Unknown"), "source_url": ent.get("source_url",""), "fields": {col: metric_lookup.get(col, None) for col in columns}})
    return rows