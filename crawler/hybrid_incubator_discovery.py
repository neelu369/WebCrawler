"""
Hybrid Incubator Discovery Pipeline

Approach:
1. DISCOVERY: Use OpenClaw for fast, scalable discovery
   - OpenClaw searches pre-crawled web content
   - Returns URLs + pre-crawled content
   - Fast parallel queries

2. ENRICHMENT: Use direct crawling for fresh, detailed data
   - Crawl discovered websites directly
   - Extract structured data (email, phone, programs, etc.)
   - Fall back to OpenClaw if direct crawl fails

3. CONTINUOUS: OpenClaw keeps running
   - Can re-query for updates
   - Discovers new incubators as they're added to web
   - Refreshes existing data

This gives us:
- Speed: OpenClaw for discovery
- Freshness: Direct crawling for enrichment
- Resilience: OpenClaw fallback when crawling fails
- Scale: Can discover 1000+ incubators efficiently
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from crawler.config import Configuration
from crawler.openclaw_client import search_documents, OpenClawDocument
from crawler.sources.incubator_discovery_v3_complete import IncubatorEntity
from crawler.sources.retry_crawler import RetryableCrawler, RetryConfig
from crawl4ai import AsyncWebCrawler


@dataclass
class DiscoverySource:
    """Track data source for each entity."""
    method: str  # "openclaw", "direct_crawl", "fallback"
    query: Optional[str] = None
    confidence: float = 1.0
    freshness: str = "unknown"  # "realtime", "cached", "unknown"


class HybridIncubatorDiscovery:
    """
    Hybrid discovery pipeline combining OpenClaw and direct crawling.
    
    Flow:
    Phase 1 (Discovery):
        OpenClaw Search → Pre-crawled Docs → Extract Entities → Deduplicate
    
    Phase 2 (Enrichment):
        For each Entity:
            Try Direct Crawl → Success? Extract detailed data
                ↓ No
            OpenClaw Fallback → Extract from cached content
                ↓ No
            Mark as minimal data
    
    Phase 3 (Continuous):
        OpenClaw keeps running
        Periodic refresh of existing entities
        Discovery of new incubators
    """
    
    # Discovery queries optimized for OpenClaw
    DISCOVERY_QUERIES = [
        # Government schemes
        "India startup incubators government funded DST NIDHI",
        "Atal Incubation Centres India AIM government",
        "MeitY TIDE technology incubators India",
        "Startup India recognized incubators list",
        "STPI software technology parks incubators",
        
        # Academic
        "IIT incubators India technology business incubator",
        "IIM entrepreneurship centres India startup",
        "NIT incubators India national institutes",
        "IIIT incubators India information technology",
        "Central university incubators India",
        
        # State-wise
        "Karnataka startup incubators Bangalore",
        "Maharashtra startup incubators Mumbai Pune",
        "Telangana startup incubators Hyderabad T-Hub",
        "Tamil Nadu startup incubators Chennai",
        "Kerala startup incubators Kochi KSUM",
        "Gujarat startup incubators Ahmedabad",
        "Rajasthan startup incubators Jaipur",
        
        # Sector-specific
        "Fintech incubators India",
        "Healthcare biotech incubators India BioNEST",
        "Agriculture agri incubators India",
        "Cleantech sustainability incubators India",
        "Edtech education incubators India",
        "Social impact incubators India Villgro",
        "Women entrepreneurship incubators India",
    ]
    
    def __init__(
        self,
        openclaw_base_url: str = "http://localhost:3100",
        output_dir: str = "./datasets",
        max_openclaw_results: int = 30,  # Results per query
        enable_direct_crawl: bool = True,
        enable_openclaw_fallback: bool = True
    ):
        self.openclaw_base_url = openclaw_base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_openclaw_results = max_openclaw_results
        self.enable_direct_crawl = enable_direct_crawl
        self.enable_openclaw_fallback = enable_openclaw_fallback
        
        self.entities: List[IncubatorEntity] = []
        self.seen_websites: set = set()
        self.seen_names: set = set()
        
        # Initialize components
        self.retry_crawler = RetryableCrawler(RetryConfig(max_retries=3))
        
    async def run_hybrid_pipeline(self) -> List[IncubatorEntity]:
        """
        Run complete hybrid pipeline.
        
        Returns list of enriched IncubatorEntity objects.
        """
        print("="*80)
        print("HYBRID INCUBATOR DISCOVERY PIPELINE")
        print("="*80)
        print("Strategy:")
        print("  Phase 1: OpenClaw Discovery (fast, scalable)")
        print("  Phase 2: Direct Crawl Enrichment (fresh, detailed)")
        print("  Phase 3: OpenClaw Fallback (reliable backup)")
        print()
        
        # Phase 1: Discovery via OpenClaw
        await self._phase1_discovery()
        
        # Phase 2: Enrichment via direct crawling
        if self.enable_direct_crawl:
            await self._phase2_enrichment()
        
        # Phase 3: Fallback to OpenClaw for failed enrichments
        if self.enable_openclaw_fallback:
            await self._phase3_fallback()
        
        # Phase 4: Export results
        await self._phase4_export()
        
        return self.entities
    
    # ==================================================================
    # PHASE 1: Discovery via OpenClaw
    # ==================================================================
    
    async def _phase1_discovery(self):
        """
        Phase 1: Discover incubators using OpenClaw.
        
        OpenClaw provides:
        - Fast parallel queries
        - Pre-crawled content
        - Scalable to 1000+ results
        """
        print("[Phase 1] DISCOVERY via OpenClaw")
        print("-" * 40)
        
        config = Configuration(
            enable_openclaw=True,
            openclaw_base_url=self.openclaw_base_url,
            openclaw_max_docs_per_query=self.max_openclaw_results
        )
        
        total_discovered = 0
        
        for i, query in enumerate(self.DISCOVERY_QUERIES, 1):
            print(f"[{i}/{len(self.DISCOVERY_QUERIES)}] Querying OpenClaw: {query[:50]}...")
            
            try:
                # Query OpenClaw for pre-crawled documents
                docs = await search_documents(config, query, limit=self.max_openclaw_results)
                
                # Extract entities from documents
                new_entities = 0
                for doc in docs:
                    entity = self._extract_entity_from_openclaw(doc, query)
                    if entity and self._is_unique(entity):
                        entity.sources.append(f"openclaw_discovery:{query}")
                        self.entities.append(entity)
                        self._mark_seen(entity)
                        new_entities += 1
                
                total_discovered += new_entities
                print(f"  Found {new_entities} new entities (total: {len(self.entities)})")
                
            except Exception as e:
                print(f"  [ERROR] OpenClaw query failed: {e}")
        
        print(f"\n[OK] Phase 1 Complete: {len(self.entities)} unique entities discovered")
        print(f"     From {len(self.DISCOVERY_QUERIES)} queries")
        print()
    
    def _extract_entity_from_openclaw(
        self, 
        doc: OpenClawDocument, 
        query: str
    ) -> Optional[IncubatorEntity]:
        """
        Extract incubator entity from OpenClaw document.
        
        OpenClaw docs contain:
        - url: Website
        - title: Page title  
        - content: Pre-crawled text
        - snippet: Search snippet
        """
        # Check if it's an incubator-related page
        content_lower = doc.content.lower()
        incubator_indicators = [
            "incubator", "accelerator", "tbi", "startup centre",
            "technology business incubator", "nidhi", "aim",
            "atal incubation", "startup india", "entrepreneurship"
        ]
        
        is_incubator = any(ind in content_lower for ind in incubator_indicators)
        
        if not is_incubator:
            return None
        
        # Create entity from OpenClaw data
        entity = IncubatorEntity(
            name=self._clean_title(doc.title),
            website=doc.url,
            sources=[f"openclaw:{query}"],
        )
        
        # Extract location from content
        entity = self._extract_location(entity, doc.content)
        
        # Extract type from content
        entity = self._extract_type(entity, doc.content)
        
        return entity
    
    def _is_unique(self, entity: IncubatorEntity) -> bool:
        """Check if entity is unique."""
        normalized_name = self._normalize_name(entity.name)
        return (
            normalized_name not in self.seen_names and 
            entity.website not in self.seen_websites
        )
    
    def _mark_seen(self, entity: IncubatorEntity):
        """Mark entity as seen."""
        self.seen_names.add(self._normalize_name(entity.name))
        if entity.website:
            self.seen_websites.add(entity.website)
    
    # ==================================================================
    # PHASE 2: Enrichment via Direct Crawling
    # ==================================================================
    
    async def _phase2_enrichment(self):
        """
        Phase 2: Enrich entities via direct crawling.
        
        Direct crawling provides:
        - Fresh data (not cached)
        - Full page rendering
        - Detailed extraction (forms, dynamic content)
        """
        print("[Phase 2] ENRICHMENT via Direct Crawling")
        print("-" * 40)
        
        # Sort entities by data completeness (enrich least complete first)
        sorted_entities = sorted(
            self.entities, 
            key=lambda e: e.calculate_completeness()
        )
        
        enriched_count = 0
        failed_count = 0
        
        # Process in batches to avoid overwhelming servers
        batch_size = 10
        for i in range(0, len(sorted_entities), batch_size):
            batch = sorted_entities[i:i+batch_size]
            
            tasks = [self._enrich_single_entity(e) for e in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for entity, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed_count += 1
                    print(f"  [FAIL] {entity.name}: {result}")
                else:
                    enriched_count += 1
                    print(f"  [OK] {entity.name}: {entity.data_completeness:.0%} complete")
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        print(f"\n[OK] Phase 2 Complete:")
        print(f"     Enriched: {enriched_count}")
        print(f"     Failed: {failed_count}")
        print()
    
    async def _enrich_single_entity(self, entity: IncubatorEntity):
        """
        Enrich a single entity via direct crawling.
        
        Strategy:
        1. Crawl main website
        2. Extract structured data
        3. Update entity fields
        4. Track source
        """
        if not entity.website:
            return
        
        try:
            # Use retry crawler for resilience
            async with AsyncWebCrawler() as crawler:
                result = await self.retry_crawler.crawl_with_retry(
                    entity.website,
                    crawler=crawler
                )
                
                if result and hasattr(result, 'markdown'):
                    content = result.markdown
                    
                    # Extract all fields from content
                    entity = self._extract_all_fields(entity, content)
                    entity.sources.append("direct_crawl")
                    entity.calculate_completeness()
                    
        except Exception as e:
            # Don't raise - let fallback handle it
            entity.sources.append(f"direct_crawl_failed:{str(e)[:50]}")
    
    def _extract_all_fields(self, entity: IncubatorEntity, content: str) -> IncubatorEntity:
        """Extract all possible fields from content."""
        import re
        
        # Email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', content)
        if email_match and not entity.email:
            entity.email = email_match.group(0)
        
        # Phone
        phone_patterns = [
            r'\+91[\s\-]?\d{10}',
            r'\d{3}[\s\-]?\d{8}',
            r'\(\d{3}\)\s?\d{8}',
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, content)
            if phone_match and not entity.phone:
                entity.phone = phone_match.group(0)
                break
        
        # Established year
        year_match = re.search(r'established\s+(?:in\s+)?(19|20)\d{2}', content, re.IGNORECASE)
        if year_match and not entity.established_year:
            entity.established_year = int(year_match.group(0).split()[-1])
        
        # Alumni count
        alumni_match = re.search(r'(\d+)\s*(?:startups?|companies?)\s*(?:graduated|alumni)', content, re.IGNORECASE)
        if alumni_match and not entity.alumni_count:
            entity.alumni_count = int(alumni_match.group(1))
        
        # Team size
        team_match = re.search(r'(\d+)\s*(?:team|people|staff)', content, re.IGNORECASE)
        if team_match and not entity.team_size:
            entity.team_size = int(team_match.group(1))
        
        return entity
    
    # ==================================================================
    # PHASE 3: OpenClaw Fallback
    # ==================================================================
    
    async def _phase3_fallback(self):
        """
        Phase 3: Use OpenClaw for entities where direct crawl failed.
        
        OpenClaw fallback provides:
        - Reliable alternative when direct crawl fails
        - Cached content may have data from previous crawls
        - No rate limiting issues
        """
        print("[Phase 3] FALLBACK to OpenClaw")
        print("-" * 40)
        
        # Find entities with low completeness
        low_completeness = [e for e in self.entities if e.calculate_completeness() < 0.3]
        
        if not low_completeness:
            print("  No entities need fallback - all well enriched!")
            print()
            return
        
        print(f"  {len(low_completeness)} entities need fallback enrichment")
        
        config = Configuration(
            enable_openclaw=True,
            openclaw_base_url=self.openclaw_base_url
        )
        
        fallback_count = 0
        
        for entity in low_completeness[:20]:  # Limit for demo
            try:
                # Search OpenClaw for entity-specific data
                query = f"{entity.name} contact email phone programs"
                docs = await search_documents(config, query, limit=3)
                
                if docs:
                    # Extract from OpenClaw content
                    entity = self._extract_all_fields(entity, docs[0].content)
                    entity.sources.append("openclaw_fallback")
                    fallback_count += 1
                    print(f"  [OK] Fallback for {entity.name}")
                    
            except Exception as e:
                print(f"  [FAIL] Fallback failed for {entity.name}: {e}")
        
        print(f"\n[OK] Phase 3 Complete: {fallback_count} entities enriched via fallback")
        print()
    
    # ==================================================================
    # PHASE 4: Export Results
    # ==================================================================
    
    async def _phase4_export(self):
        """Export final dataset."""
        print("[Phase 4] EXPORTING DATASET")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate final statistics
        total = len(self.entities)
        avg_completeness = sum(e.calculate_completeness() for e in self.entities) / total
        
        by_source = {}
        for e in self.entities:
            for source in e.sources:
                by_source[source] = by_source.get(source, 0) + 1
        
        # Export JSON
        json_file = self.output_dir / f"incubators_hybrid_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "pipeline": "hybrid",
                    "timestamp": datetime.now().isoformat(),
                    "total_entities": total,
                    "avg_completeness": avg_completeness,
                    "by_source": by_source,
                    "phases": {
                        "discovery": "openclaw",
                        "enrichment": "direct_crawl",
                        "fallback": "openclaw"
                    }
                },
                "entities": [e.to_dict() for e in self.entities]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Exported to: {json_file}")
        print()
        print("="*60)
        print("HYBRID PIPELINE COMPLETE")
        print("="*60)
        print(f"Total entities: {total}")
        print(f"Average completeness: {avg_completeness:.1%}")
        print(f"\nData sources:")
        for source, count in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
        print()
    
    # ==================================================================
    # HELPER METHODS
    # ==================================================================
    
    def _clean_title(self, title: str) -> str:
        """Clean page title for use as entity name."""
        if not title:
            return "Unknown"
        # Remove common suffixes
        title = re.sub(r'\s*[\|\-–]\s*(Home|Incubator|About).*$', '', title)
        return title.strip()
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for deduplication."""
        if not name:
            return ""
        name = name.lower()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'[^\w\s]', '', name)
        name = name.replace('incubator', '').replace('centre', '').replace('center', '').strip()
        return name
    
    def _extract_location(self, entity: IncubatorEntity, content: str) -> IncubatorEntity:
        """Extract city and state from content."""
        # Common Indian cities and states
        cities = [
            "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
            "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
            "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara",
            "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut",
            "Rajkot", "Kalyan", "Vasai", "Varanasi", "Srinagar", "Aurangabad",
            "Dhanbad", "Amritsar", "Navi Mumbai", "Allahabad", "Ranchi",
            "Coimbatore", "Jabalpur", "Gwalior", "Vijayawada", "Jodhpur",
            "Madurai", "Raipur", "Kota", "Guwahati", "Chandigarh"
        ]
        
        states = [
            "Maharashtra", "Delhi", "Karnataka", "Telangana", "Tamil Nadu",
            "West Bengal", "Gujarat", "Rajasthan", "Uttar Pradesh", "Madhya Pradesh",
            "Kerala", "Punjab", "Haryana", "Bihar", "Odisha", "Jharkhand",
            "Assam", "Chhattisgarh", "Himachal Pradesh", "Uttarakhand",
            "Jammu and Kashmir", "Goa", "Tripura", "Meghalaya", "Manipur",
            "Nagaland", "Arunachal Pradesh", "Mizoram", "Sikkim"
        ]
        
        content_lower = content.lower()
        
        # Find city
        for city in cities:
            if city.lower() in content_lower:
                entity.city = city
                break
        
        # Find state
        for state in states:
            if state.lower() in content_lower:
                entity.state = state
                break
        
        return entity
    
    def _extract_type(self, entity: IncubatorEntity, content: str) -> IncubatorEntity:
        """Extract incubator type from content."""
        content_lower = content.lower()
        
        type_indicators = {
            "government": ["government", "govt", "ministry", "department", "public sector"],
            "academic": ["iit", "iim", "university", "college", "institute", "campus"],
            "corporate": ["microsoft", "google", "amazon", "reliance", "tata", "corporate"],
            "social": ["social", "impact", "villgro", "non-profit", "ngo"],
        }
        
        for type_, indicators in type_indicators.items():
            if any(ind in content_lower for ind in indicators):
                entity.type = type_
                break
        
        if not entity.type:
            entity.type = "private"
        
        return entity


async def main():
    """Run hybrid discovery pipeline."""
    discovery = HybridIncubatorDiscovery(
        openclaw_base_url="http://localhost:3100",
        output_dir="./datasets",
        enable_direct_crawl=True,
        enable_openclaw_fallback=True
    )
    
    entities = await discovery.run_hybrid_pipeline()
    return entities


if __name__ == "__main__":
    asyncio.run(main())
