"""Test script — runs the pipeline and prints every stage's output.

Usage:
    python test_pipeline.py

Requires: Neo4j running, .env configured.
MongoDB is optional (pipeline continues without it).
"""

import asyncio
import csv
import json
import os
from dotenv import load_dotenv

load_dotenv()

from crawler.graph import graph


QUERY = "top startup incubators and accelerators in India"
CSV_OUTPUT = os.path.join(os.path.dirname(__file__), "crawl_results.csv")


def _divider(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_table(headers: list[str], rows: list[list[str]], max_col: int = 30) -> None:
    """Print a formatted ASCII table to the terminal."""
    # Truncate cells
    truncated = []
    for row in rows:
        truncated.append([str(c)[:max_col] for c in row])

    # Column widths
    widths = [len(h) for h in headers]
    for row in truncated:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Header
    hdr = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * w for w in widths)
    print(f"  {hdr}")
    print(f"  {sep}")

    # Rows
    for row in truncated:
        line = " | ".join(
            (row[i] if i < len(row) else "").ljust(widths[i])
            for i in range(len(headers))
        )
        print(f"  {line}")


async def main():
    _divider("PIPELINE START")
    print(f"Query: {QUERY}")

    result = await graph.ainvoke(
        {"user_query": QUERY},
        config={"configurable": {}},   # uses defaults from config.py
    )

    # ── Stage 1: Crawled docs ────────────────────────────────
    _divider("STAGE 1 - Crawled Documents")
    raw_ids = result.get("raw_doc_ids", [])
    print(f"Documents logged: {len(raw_ids)}")
    for i, doc_id in enumerate(raw_ids, 1):
        print(f"  {i}. {doc_id}")

    # ── Stage 2: Extracted entities ──────────────────────────
    _divider("STAGE 2 - Extracted Entities")
    entities = result.get("extracted_entities", [])
    print(f"Total entities: {len(entities)}")
    for i, e in enumerate(entities, 1):
        print(f"\n  [{i}] {e.name}")
        print(f"      Description : {e.description[:120]}...")
        print(f"      Source      : {e.source_url}")
        print(f"      Priority    : {e.priority_score}")
        if e.metrics:
            print(f"      Metrics     :")
            for k, v in e.metrics.items():
                print(f"        - {k}: {v}")

    # ── Stage 3: Structured results ──────────────────────────
    _divider("STAGE 3 - Structured Results (from Neo4j)")
    structured = result.get("structured_results", [])
    print(f"Total structured results: {len(structured)}")
    for i, s in enumerate(structured, 1):
        print(f"\n  [{i}] {s.name} ({s.entity_type})")
        if s.description:
            print(f"      Description   : {s.description[:120]}...")
        if s.properties:
            print(f"      Properties    :")
            for k, v in s.properties.items():
                print(f"        - {k}: {v}")
        if s.relationships:
            print(f"      Relationships : {len(s.relationships)}")
            for rel in s.relationships[:5]:
                print(f"        -> {rel}")
        if s.citations:
            print(f"      Citations     : {len(s.citations)}")
        if s.source_urls:
            print(f"      Sources       : {', '.join(s.source_urls[:3])}")

    # ── Stage 4: Insights ────────────────────────────────────
    _divider("STAGE 4 - Insights")
    summary = result.get("insights_summary", "")
    items = result.get("insights_items", [])
    print(f"Summary: {summary[:300] if summary else '(empty)'}")
    print(f"Insight items: {len(items)}")
    for item in items[:5]:
        print(f"  - {json.dumps(item, default=str)[:200]}")

    # ── Stage 5: Gaps / missing data ─────────────────────────
    _divider("STAGE 5 - Missing Data Targets")
    missing = result.get("missing_data_targets", [])
    if missing:
        print(f"Gaps found ({len(missing)}):")
        for m in missing:
            print(f"  ! {m}")
    else:
        print("No gaps -- pipeline is satisfied with the data collected.")

    # ── SUMMARY TABLE — all entities with metrics ────────────
    _divider("SUMMARY TABLE - Entities & Metrics (Unranked)")

    # Collect all unique metric keys across entities
    all_metric_keys: list[str] = []
    seen_keys: set[str] = set()

    # Use structured results if available, fall back to extracted entities
    data_source = structured if structured else entities
    source_label = "structured_results" if structured else "extracted_entities"
    print(f"  (from {source_label}: {len(data_source)} entities)\n")

    if structured:
        for s in structured:
            for k in s.properties:
                if k not in seen_keys:
                    all_metric_keys.append(k)
                    seen_keys.add(k)
    else:
        for e in entities:
            for k in e.metrics:
                if k not in seen_keys:
                    all_metric_keys.append(k)
                    seen_keys.add(k)

    # Limit to first 6 metric columns for readability
    display_keys = all_metric_keys[:6]
    headers = ["#", "Entity", "Type"] + display_keys
    rows: list[list[str]] = []

    if structured:
        for i, s in enumerate(structured, 1):
            row = [str(i), s.name, s.entity_type]
            for k in display_keys:
                row.append(s.properties.get(k, "-"))
            rows.append(row)
    else:
        for i, e in enumerate(entities, 1):
            row = [str(i), e.name, "Entity"]
            for k in display_keys:
                row.append(e.metrics.get(k, "-"))
            rows.append(row)

    if rows:
        _print_table(headers, rows)
    else:
        print("  (no entities found)")

    if len(all_metric_keys) > 6:
        print(f"\n  ... {len(all_metric_keys) - 6} more metric columns in CSV")

    # ── CSV export ───────────────────────────────────────────
    _divider("CSV EXPORT")

    csv_headers = ["Entity", "Type", "Description", "Priority", "Sources"] + all_metric_keys
    csv_rows: list[list[str]] = []

    if structured:
        for s in structured:
            row = [
                s.name,
                s.entity_type,
                s.description[:200],
                str(s.priority_score),
                "; ".join(s.source_urls[:3]),
            ]
            for k in all_metric_keys:
                row.append(s.properties.get(k, ""))
            csv_rows.append(row)
    else:
        for e in entities:
            row = [
                e.name,
                "Entity",
                e.description[:200],
                str(e.priority_score),
                e.source_url,
            ]
            for k in all_metric_keys:
                row.append(e.metrics.get(k, ""))
            csv_rows.append(row)

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)

    print(f"  Saved {len(csv_rows)} rows x {len(csv_headers)} columns to:")
    print(f"  {CSV_OUTPUT}")

    # ── Cost summary ─────────────────────────────────────────
    _divider("COST SUMMARY")
    cost = result.get("cost_summary", {})
    if cost:
        print(json.dumps(cost, indent=2, default=str))
    else:
        print("(no cost data)")

    # ── Session info ─────────────────────────────────────────
    _divider("SESSION")
    print(f"Session ID     : {result.get('session_id', 'N/A')}")
    print(f"Vector IDs     : {len(result.get('entity_vector_ids', []))}")
    print(f"Raw vector IDs : {len(result.get('raw_vector_ids', []))}")

    _divider("PIPELINE COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
