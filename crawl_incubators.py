#!/usr/bin/env python3
"""
Stand-alone script to crawl and build comprehensive dataset of Indian incubators.

Usage:
    python crawl_incubators.py --mode discovery      # Phase 1: Discover all incubators
    python crawl_incubators.py --mode enrich        # Phase 2: Enrich with details
    python crawl_incubators.py --mode full          # Run full pipeline
    python crawl_incubators.py --output ./datasets  # Custom output directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def run_discovery(output_dir: str = "./datasets") -> dict:
    """
    Phase 1: Discover all incubator sources.
    """
    from crawler.sources.india_incubator_discovery import IndiaIncubatorDiscovery
    
    print("="*80)
    print("PHASE 1: DISCOVERING INDIAN INCUBATORS")
    print("="*80)
    print(f"Target: ~1100-1200 incubators")
    print(f"Expected breakdown:")
    print(f"  - Government-backed: ~400")
    print(f"  - Academic (IITs/IIMs/IISc/NITs): ~300")
    print(f"  - Private/Corporate: ~500")
    print()
    
    discovery = IndiaIncubatorDiscovery()
    
    # Discover all sources
    start_time = datetime.now()
    entities = await discovery.discover_all(max_concurrent=10)
    end_time = datetime.now()
    
    # Save discovery results
    os.makedirs(output_dir, exist_ok=True)
    
    discovery_file = f"{output_dir}/incubators_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(discovery_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "discovery_date": datetime.now().isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "total_discovered": len(entities),
            },
            "entities": [
                {
                    "name": e.name,
                    "website": e.website,
                    "city": e.city,
                    "state": e.state,
                    "type": e.type,
                    "backing": e.backing,
                    "sources": e.sources,
                }
                for e in entities
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Discovery complete: {len(entities)} incubators found")
    print(f"✓ Saved to: {discovery_file}")
    
    return {
        "entities": entities,
        "discovery_file": discovery_file,
        "count": len(entities),
    }


async def run_enrichment(input_file: str = None, output_dir: str = "./datasets") -> dict:
    """
    Phase 2: Enrich discovered incubators with detailed data.
    """
    from crawler.sources.india_incubator_discovery import IncubatorEnricher, IncubatorEntity
    import csv
    
    print("="*80)
    print("PHASE 2: ENRICHING INCUBATOR DATA")
    print("="*80)
    
    # Load entities from discovery
    if input_file and os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            entities = [IncubatorEntity(**e) for e in data.get('entities', [])]
    else:
        print("Error: No discovery file found. Run discovery phase first.")
        return {}
    
    enricher = IncubatorEnricher()
    enriched = []
    
    print(f"\nEnriching {len(entities)} entities...")
    print("This may take 30-60 minutes depending on connection speed.\n")
    
    for i, entity in enumerate(entities, 1):
        print(f"[{i}/{len(entities)}] Enriching: {entity.name}", end=" ")
        
        enriched_entity = await enricher.enrich_entity(entity)
        enriched.append(enriched_entity)
        
        completeness = enriched_entity.data_completeness
        if completeness >= 0.8:
            print(f"✓ ({completeness:.0%} complete)")
        elif completeness >= 0.5:
            print(f"⚠ ({completeness:.0%} complete)")
        else:
            print(f"✗ ({completeness:.0%} complete)")
    
    # Calculate statistics
    completeness_scores = [e.data_completeness for e in enriched]
    avg_completeness = sum(completeness_scores) / len(completeness_scores)
    
    # Group by completeness
    fully_complete = sum(1 for s in completeness_scores if s >= 0.9)
    mostly_complete = sum(1 for s in completeness_scores if 0.6 <= s < 0.9)
    partially_complete = sum(1 for s in completeness_scores if 0.3 <= s < 0.6)
    minimal_data = sum(1 for s in completeness_scores if s < 0.3)
    
    # Group by type and state
    by_type = {}
    by_state = {}
    for e in enriched:
        by_type[e.type] = by_type.get(e.type, 0) + 1
        if e.state:
            by_state[e.state] = by_state.get(e.state, 0) + 1
    
    # Save enriched data
    enriched_file = f"{output_dir}/incubators_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(enriched_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Extended header with all fields
        writer.writerow([
            'id', 'name', 'official_name', 'short_name', 'website', 'email', 'phone',
            'city', 'state', 'address', 'pincode', 'type', 'backing', 'funding_type',
            'investment_range', 'equity_taken', 'focus_sectors', 'programs',
            'duration_months', 'virtual_available', 'established_year', 'alumni_count',
            'active_startups', 'total_investment_made', 'team_size', 'mentor_count',
            'data_completeness', 'sources', 'missing_fields'
        ])
        
        for e in enriched:
            writer.writerow([
                e.id, e.name, e.official_name, e.short_name, e.website, e.email, e.phone,
                e.city, e.state, e.address, e.pincode, e.type, e.backing, e.funding_type,
                e.investment_range, e.equity_taken, '|'.join(e.focus_sectors), '|'.join(e.programs),
                e.duration_months, e.virtual_available, e.established_year, e.alumni_count,
                e.active_startups, e.total_investment_made, e.team_size, e.mentor_count,
                e.data_completeness, '|'.join(e.sources), '|'.join(e.missing_fields)
            ])
    
    # Also save JSON with full details
    json_file = enriched_file.replace('.csv', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "enrichment_date": datetime.now().isoformat(),
                "total_entities": len(enriched),
                "avg_completeness": avg_completeness,
                "by_type": by_type,
                "by_state": dict(sorted(by_state.items(), key=lambda x: x[1], reverse=True)[:15]),
            },
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "official_name": e.official_name,
                    "website": e.website,
                    "email": e.email,
                    "city": e.city,
                    "state": e.state,
                    "type": e.type,
                    "backing": e.backing,
                    "focus_sectors": e.focus_sectors,
                    "programs": e.programs,
                    "established_year": e.established_year,
                    "alumni_count": e.alumni_count,
                    "team_size": e.team_size,
                    "data_completeness": e.data_completeness,
                    "missing_fields": e.missing_fields,
                }
                for e in enriched
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("ENRICHMENT COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Total entities: {len(enriched)}")
    print(f"  Average completeness: {avg_completeness:.1%}")
    print(f"  Fully complete (≥90%): {fully_complete}")
    print(f"  Mostly complete (60-89%): {mostly_complete}")
    print(f"  Partially complete (30-59%): {partially_complete}")
    print(f"  Minimal data (<30%): {minimal_data}")
    
    print(f"\nBy type:")
    for t, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t}: {count}")
    
    print(f"\nTop states:")
    for state, count in sorted(by_state.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {state}: {count}")
    
    print(f"\n✓ CSV saved: {enriched_file}")
    print(f"✓ JSON saved: {json_file}")
    
    return {
        "enriched": enriched,
        "csv_file": enriched_file,
        "json_file": json_file,
        "stats": {
            "total": len(enriched),
            "avg_completeness": avg_completeness,
            "by_type": by_type,
            "by_state": by_state,
        }
    }


async def run_full_pipeline(output_dir: str = "./datasets"):
    """
    Run complete discovery + enrichment pipeline.
    """
    print("="*80)
    print("INDIAN INCUBATORS - FULL DATASET BUILD")
    print("="*80)
    print()
    
    # Phase 1: Discovery
    discovery_result = await run_discovery(output_dir)
    
    if discovery_result.get("count", 0) == 0:
        print("\n✗ Discovery failed - no entities found")
        return
    
    # Phase 2: Enrichment
    enrich_result = await run_enrichment(
        input_file=discovery_result["discovery_file"],
        output_dir=output_dir
    )
    
    # Final summary
    print("\n" + "="*80)
    print("FULL PIPELINE COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Discovery: {discovery_result['discovery_file']}")
    print(f"  Enriched CSV: {enrich_result.get('csv_file', 'N/A')}")
    print(f"  Enriched JSON: {enrich_result.get('json_file', 'N/A')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Crawl and build dataset of Indian incubators"
    )
    parser.add_argument(
        "--mode",
        choices=["discovery", "enrich", "full"],
        default="full",
        help="Pipeline mode to run"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file for enrichment mode (JSON from discovery)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datasets",
        help="Output directory for dataset files"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent crawlers"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Run appropriate mode
    if args.mode == "discovery":
        asyncio.run(run_discovery(args.output))
    elif args.mode == "enrich":
        if not args.input:
            print("Error: --input required for enrich mode")
            sys.exit(1)
        asyncio.run(run_enrichment(args.input, args.output))
    else:  # full
        asyncio.run(run_full_pipeline(args.output))


if __name__ == "__main__":
    main()
