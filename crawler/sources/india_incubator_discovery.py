"""
India Incubator Discovery - Multi-source entity collection.

Sources prioritized by reliability:
1. Government directories (Startup India, DST, MeitY, AIM)
2. Institutional websites (IITs, IIMs, IISc, NITs)
3. Industry associations (ISBA)
4. Commercial databases (YourStory, Inc42)
5. Academic papers and reports
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from crawl4ai import AsyncWebCrawler


@dataclass
class IncubatorSeed:
    """Minimal data for initial discovery."""
    name: str
    source_url: str
    source_type: str  # gov_list, institution, commercial, secondary
    confidence: float = 0.5
    discovered_at: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


@dataclass
class IncubatorEntity:
    """Full incubator entity with all fields."""
    id: str = ""  # UUID after deduplication
    name: str = ""
    official_name: str = ""
    short_name: str = ""
    
    # Contact
    website: str = ""
    email: str = ""
    phone: str = ""
    
    # Location
    city: str = ""
    state: str = ""
    address: str = ""
    pincode: str = ""
    
    # Classification
    type: str = ""  # government, private, academic, corporate, social
    backing: str = ""  # dst, meity, aim, self-funded, corporate
    
    # Financial
    funding_type: str = ""  # grant, equity, debt, hybrid
    investment_range: str = ""
    equity_taken: str = ""
    
    # Programs
    focus_sectors: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    duration_months: int = 0
    virtual_available: bool = False
    
    # Stats
    established_year: int = 0
    alumni_count: int = 0
    active_startups: int = 0
    total_investment_made: str = ""
    
    # Team
    team_size: int = 0
    mentor_count: int = 0
    
    # Metadata
    data_completeness: float = 0.0  # 0-1
    sources: list[str] = field(default_factory=list)
    last_updated: str = ""
    
    # Missing fields tracking
    missing_fields: list[str] = field(default_factory=list)


class IndiaIncubatorDiscovery:
    """
    Multi-source discovery for Indian incubators.
    
    Expected entities: ~1100-1200
    - Government-backed: ~400 (DST, MeitY, AIM, DBT)
    - Academic: ~300 (IITs, IIMs, IISc, NITs, Central Univs)
    - Private: ~400-500 (Corporate, Independent)
    """
    
    # Government source URLs
    GOV_SOURCES = {
        "startup_india_pdf": "https://www.startupindia.gov.in/content/dam/invest-india/Tenders/Incubator-List.pdf",
        "startup_india_portal": "https://startupindia.gov.in/content/sih/en/startup-scheme/recognized-incubators.html",
        "dst_nidhi": "https://dst.gov.in/nidhi-scheme",
        "meity_tide": "https://meity.gov.in/content/technology-incubation-and-development-entrepreneurs",
        "meity_startup_hub": "https://www.meitystartuphub.in",
        "dbt_bionest": "https://www.birac.nic.in/desc_biotechnology_incubation.php",
        "aim_atl": "https://aim.gov.in/atal-tinkering-labs.php",
        "aim_aic": "https://aim.gov.in/atal-incubation-centres.php",
        "aim_acic": "https://aim.gov.in/atal-community-innovation-centre.php",
        "isba_members": "https://www.isba.in/members.php",
    }
    
    # Major institution patterns
    INSTITUTION_PATTERNS = {
        "iits": [
            "https://www.iitb.ac.in/sine",  # Society for Innovation and Entrepreneurship
            "https://www.iitd.ac.in/incubation",
            "https://www.iitm.ac.in/research/research-centres/rural-technology-and-business-incubator",
            "https://www.iitkgp.ac.in/research/technology-incubation",
            # Add more IITs
        ],
        "iims": [
            "https://www.iimahmedabad.ac.in/entrepreneurship",
            "https://www.iimb.ac.in/entrepreneurship",
            "https://www.iimcal.ac.in/centres/entrepreneurship-centre",
            # Add more IIMs
        ],
        "iisc": [
            "https://www.sociis.io/",  # Society for Innovation and Development
            "https://www.iisc.ac.in/centers/ced/",
        ],
        "state_startups": {
            "karnataka": "https://startup.karnataka.gov.in/incubators",
            "maharashtra": "https://startup.maharashtra.gov.in/incubators",
            "telangana": "https://www.t-hub.co/incubators",
            "tamil_nadu": "https://startup.tn.gov.in/incubators",
            "gujarat": "https:// Gujarat Startup Portal incubators",
        }
    }
    
    def __init__(self):
        self.seeds: list[IncubatorSeed] = []
        self.entities: dict[str, IncubatorEntity] = {}  # key = normalized name
        
    async def discover_all(self, max_concurrent: int = 5) -> list[IncubatorEntity]:
        """
        Main entry: Discover all incubators from all sources.
        
        Returns list of unique IncubatorEntity objects.
        """
        print(f"[IncubatorDiscovery] Starting discovery across {len(self.GOV_SOURCES)} primary sources...")
        
        # Phase 1: Government lists (highest confidence)
        await self._crawl_government_sources()
        
        # Phase 2: Institution pages
        await self._crawl_institutional_sources(max_concurrent)
        
        # Phase 3: Commercial sources
        await self._crawl_commercial_sources()
        
        # Phase 4: Deduplicate
        entities = await self._deduplicate_and_merge()
        
        print(f"[IncubatorDiscovery] Total unique incubators found: {len(entities)}")
        return entities
    
    async def _crawl_government_sources(self):
        """Crawl government directories and lists."""
        print("[IncubatorDiscovery] Phase 1: Government sources...")
        
        for source_name, url in self.GOV_SOURCES.items():
            try:
                if "startup_india" in source_name:
                    await self._parse_startup_india_list(url)
                elif source_name == "aim_aic":
                    await self._parse_aim_incubators(url)
                elif source_name == "isba_members":
                    await self._parse_isba_members(url)
                # Add more specific parsers
                else:
                    # Generic crawl
                    await self._generic_crawl(url, source_name)
            except Exception as e:
                print(f"[IncubatorDiscovery] Error crawling {source_name}: {e}")
    
    async def _parse_startup_india_list(self, url: str):
        """Parse the Startup India recognized incubators list."""
        print(f"[IncubatorDiscovery] Parsing Startup India portal...")
        
        # This would use crawl4ai or Playwright to extract table/list
        # Expected: ~150-200 recognized incubators
        
        # Mock implementation - in real code would crawl HTML/PDF
        sample_data = [
            {"name": "Indian Institute of Technology Bombay - SINE", "location": "Mumbai, Maharashtra", "type": "Academic"},
            {"name": "IIM Ahmedabad - CIIE", "location": "Ahmedabad, Gujarat", "type": "Academic"},
            {"name": "T-Hub", "location": "Hyderabad, Telangana", "type": "Government"},
            {"name": "Kerala Startup Mission", "location": "Kochi, Kerala", "type": "Government"},
        ]
        
        for item in sample_data:
            self.seeds.append(IncubatorSeed(
                name=item["name"],
                source_url=url,
                source_type="gov_list",
                confidence=0.9
            ))
    
    async def _parse_aim_incubators(self, url: str):
        """Parse Atal Incubation Centres list."""
        print(f"[IncubatorDiscovery] Parsing AIM AIC list...")
        # AIM has ~100+ AICs
        # Would crawl aim.gov.in and extract the AIC list
        pass
    
    async def _parse_isba_members(self, url: str):
        """Parse ISBA member list."""
        print(f"[IncubatorDiscovery] Parsing ISBA members...")
        # ISBA has ~80-100 members
        pass
    
    async def _crawl_institutional_sources(self, max_concurrent: int = 5):
        """Crawl IIT, IIM, IISc incubator pages."""
        print("[IncubatorDiscovery] Phase 2: Institutional sources...")
        
        all_urls = []
        for category, urls in self.INSTITUTION_PATTERNS.items():
            if isinstance(urls, list):
                all_urls.extend(urls)
            elif isinstance(urls, dict):
                all_urls.extend(urls.values())
        
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [self._crawl_institution_page(url, semaphore) for url in all_urls]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _crawl_institution_page(self, url: str, semaphore: asyncio.Semaphore):
        """Crawl a specific institution's incubator page."""
        async with semaphore:
            try:
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url)
                    # Extract incubator name, contact, programs
                    # Store in seeds
                    print(f"[IncubatorDiscovery] Crawled: {url}")
            except Exception as e:
                print(f"[IncubatorDiscovery] Failed to crawl {url}: {e}")
    
    async def _crawl_commercial_sources(self):
        """Crawl YourStory, Inc42 for incubator lists."""
        print("[IncubatorDiscovery] Phase 3: Commercial sources...")
        
        # Search patterns for commercial sources
        queries = [
            "top incubators in India",
            "government incubators India",
            "IIT incubators India",
            "T-Hub startup",
            "Kerala Startup Mission",
        ]
        
        # Would use SearXNG or similar to discover these
        pass
    
    async def _deduplicate_and_merge(self) -> list[IncubatorEntity]:
        """
        Deduplicate seeds and create initial entities.
        
        Key matching strategies:
        1. Exact name match (after normalization)
        2. Website domain match
        3. Location + name similarity
        """
        print(f"[IncubatorDiscovery] Phase 4: Deduplicating {len(self.seeds)} seeds...")
        
        entities = {}
        
        for seed in self.seeds:
            normalized_name = self._normalize_name(seed.name)
            
            if normalized_name in entities:
                # Merge sources
                entities[normalized_name].sources.append(seed.source_url)
            else:
                entity = IncubatorEntity(
                    name=seed.name,
                    sources=[seed.source_url],
                )
                entities[normalized_name] = entity
        
        return list(entities.values())
    
    def _normalize_name(self, name: str) -> str:
        """Normalize incubator name for deduplication."""
        # Remove common suffixes
        name = re.sub(r'\s+(incubator|centre|center|hub|foundation|society)\s*$', '', name, flags=re.IGNORECASE)
        # Remove IIT/IIM/institution prefix temporarily
        name = re.sub(r'^(IIT\s+\w+|IIM\s+\w+)\s*[-,]?\s*', '', name, flags=re.IGNORECASE)
        # Normalize whitespace and case
        name = ' '.join(name.lower().split())
        return name
    
    def _extract_from_html(self, html: str, base_url: str) -> list[dict]:
        """Extract incubator data from HTML content."""
        # Use BeautifulSoup or similar to extract structured data
        # Look for: name, contact, location, programs
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ENRICHMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class IncubatorEnricher:
    """
    Iterative enrichment pipeline for incubator entities.
    
    For each entity, attempts to fill missing fields by:
    1. Crawling the official website
    2. Searching for specific missing data
    3. Using LLM extraction from web content
    """
    
    # Priority order for field discovery
    FIELD_PRIORITY = [
        "official_name", "website", "email", "phone",
        "city", "state", "type", "backing",
        "funding_type", "focus_sectors", "programs",
        "established_year", "alumni_count", "team_size"
    ]
    
    async def enrich_entity(self, entity: IncubatorEntity) -> IncubatorEntity:
        """
        Enrich a single incubator entity.
        
        Returns updated entity with data_completeness score.
        """
        missing = self._get_missing_fields(entity)
        entity.missing_fields = missing
        
        if not missing:
            entity.data_completeness = 1.0
            return entity
        
        # Try to get data from website
        if entity.website:
            await self._crawl_website(entity)
        
        # Search for specific missing fields
        for field in missing:
            await self._search_for_field(entity, field)
        
        # Update completeness score
        filled_count = len([f for f in self.FIELD_PRIORITY if getattr(entity, f)])
        entity.data_completeness = filled_count / len(self.FIELD_PRIORITY)
        
        return entity
    
    def _get_missing_fields(self, entity: IncubatorEntity) -> list[str]:
        """Identify which critical fields are missing."""
        missing = []
        for field in self.FIELD_PRIORITY:
            value = getattr(entity, field)
            if not value or (isinstance(value, list) and not value):
                missing.append(field)
        return missing
    
    async def _crawl_website(self, entity: IncubatorEntity):
        """Crawl the incubator's official website for data."""
        if not entity.website:
            return
        
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=entity.website)
                # Extract structured data from website
                # Use LLM or regex patterns
                pass
        except Exception as e:
            print(f"[Enricher] Failed to crawl {entity.website}: {e}")
    
    async def _search_for_field(self, entity: IncubatorEntity, field: str):
        """
        Search specifically for a missing field.
        
        Example: "T-Hub Hyderabad email"
        """
        search_query = f"{entity.name} {entity.city} {field}"
        # Use SearXNG or similar to find this specific info
        pass


# ─────────────────────────────────────────────────────────────────────────────
# DATASET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class IncubatorDatasetManager:
    """
    Manages the comprehensive incubator dataset.
    
    Features:
    - Incremental updates
    - Data versioning
    - Export to CSV/JSON/Excel
    - Completeness reporting
    """
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = output_dir
        self.entities: list[IncubatorEntity] = []
        
    async def build_dataset(self, target_count: int = 1170) -> dict:
        """
        Build the complete dataset.
        
        Returns statistics about the dataset.
        """
        # 1. Discovery
        discovery = IndiaIncubatorDiscovery()
        entities = await discovery.discover_all()
        
        # 2. Enrichment
        enricher = IncubatorEnricher()
        for entity in entities:
            await enricher.enrich_entity(entity)
        
        # 3. Save
        self.entities = entities
        await self._save_dataset()
        
        return {
            "total_entities": len(entities),
            "target_count": target_count,
            "coverage": len(entities) / target_count,
            "avg_completeness": sum(e.data_completeness for e in entities) / len(entities),
            "by_type": self._group_by_type(entities),
            "by_state": self._group_by_state(entities),
        }
    
    def _group_by_type(self, entities: list[IncubatorEntity]) -> dict:
        """Group entities by incubator type."""
        groups = {}
        for e in entities:
            groups[e.type] = groups.get(e.type, 0) + 1
        return groups
    
    def _group_by_state(self, entities: list[IncubatorEntity]) -> dict:
        """Group entities by state."""
        groups = {}
        for e in entities:
            if e.state:
                groups[e.state] = groups.get(e.state, 0) + 1
        return groups
    
    async def _save_dataset(self, format: str = "csv"):
        """Save dataset to disk."""
        import csv
        import os
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save as CSV
        output_path = f"{self.output_dir}/indian_incubators.csv"
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'name', 'official_name', 'website', 'email', 'phone',
                'city', 'state', 'type', 'backing', 'funding_type',
                'focus_sectors', 'programs', 'established_year',
                'alumni_count', 'data_completeness', 'sources'
            ])
            # Write data
            for e in self.entities:
                writer.writerow([
                    e.name, e.official_name, e.website, e.email, e.phone,
                    e.city, e.state, e.type, e.backing, e.funding_type,
                    '|'.join(e.focus_sectors), '|'.join(e.programs), e.established_year,
                    e.alumni_count, e.data_completeness, '|'.join(e.sources)
                ])
        
        print(f"[DatasetManager] Saved {len(self.entities)} entities to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Build the complete Indian incubators dataset."""
    manager = IncubatorDatasetManager()
    
    stats = await manager.build_dataset(target_count=1170)
    
    print("\n" + "="*60)
    print("INDIAN INCUBATORS DATASET - BUILD COMPLETE")
    print("="*60)
    print(f"Total entities discovered: {stats['total_entities']}")
    print(f"Target: {stats['target_count']}")
    print(f"Coverage: {stats['coverage']:.1%}")
    print(f"Average data completeness: {stats['avg_completeness']:.1%}")
    print(f"\nBy type:")
    for t, count in stats['by_type'].items():
        print(f"  {t}: {count}")
    print(f"\nTop states:")
    sorted_states = sorted(stats['by_state'].items(), key=lambda x: x[1], reverse=True)[:10]
    for state, count in sorted_states:
        print(f"  {state}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
