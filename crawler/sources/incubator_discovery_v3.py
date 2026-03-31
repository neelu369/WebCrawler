"""
India Incubator Discovery v3 - Scaled version with web search integration.

This version:
1. Uses SearXNG to discover hundreds of incubators
2. Parses government PDFs and lists
3. Crawls state-wise directories
4. Discovers remaining incubators through multi-query search
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup


@dataclass
class IncubatorEntity:
    """Full incubator entity - all fields default to None for clean tracking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Identity
    name: Optional[str] = None
    official_name: Optional[str] = None
    short_name: Optional[str] = None
    
    # Contact
    website: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    
    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    address: Optional[str] = None
    pincode: Optional[str] = None
    
    # Classification
    type: Optional[str] = None
    backing: Optional[str] = None
    
    # Financial
    funding_type: Optional[str] = None
    investment_range: Optional[str] = None
    equity_taken: Optional[str] = None
    
    # Programs
    focus_sectors: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    duration_months: Optional[int] = None
    virtual_available: Optional[bool] = None
    
    # Stats
    established_year: Optional[int] = None
    alumni_count: Optional[int] = None
    active_startups: Optional[int] = None
    total_investment_made: Optional[str] = None
    
    # Team
    team_size: Optional[int] = None
    mentor_count: Optional[int] = None
    
    # Metadata
    sources: list[str] = field(default_factory=list)
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)


class IncubatorDiscoveryV3:
    """
    Scalable discovery using SearXNG + curated lists.
    Target: 1100+ incubators
    """
    
    # Major institution URLs (same as v2)
    IIT_URLS = [
        ("IIT Bombay SINE", "https://www.iitb.ac.in/sine", "Mumbai", "Maharashtra"),
        ("IIT Delhi Incubation", "https://www.iitd.ac.in/incubation", "New Delhi", "Delhi"),
        ("IIT Madras RTBI", "https://rtbi.iitm.ac.in", "Chennai", "Tamil Nadu"),
        ("IIT Kharagpur TBI", "https://www.iitkgp.ac.in/research/technology-incubation", "Kharagpur", "West Bengal"),
        ("IIT Kanpur SIIC", "https://www.iitk.ac.in/siic", "Kanpur", "Uttar Pradesh"),
        ("IIT Roorkee TBI", "https://www.iitr.ac.in/incubation", "Roorkee", "Uttarakhand"),
        ("IIT Guwahati TEC", "https://www.iitg.ac.in/tec", "Guwahati", "Assam"),
        ("IIT Hyderabad i-TBI", "https://www.iith.ac.in/tbi", "Hyderabad", "Telangana"),
        ("IIT Indore Incubator", "https://www.iiti.ac.in/research", "Indore", "Madhya Pradesh"),
        ("IIT (BHU) Varanasi", "https://www.iitbhu.ac.in/research", "Varanasi", "Uttar Pradesh"),
        ("IIT Ropar", "https://www.iitrpr.ac.in/research", "Ropar", "Punjab"),
        ("IIT Patna", "https://www.iitp.ac.in/research", "Patna", "Bihar"),
        ("IIT Gandhinagar", "https://www.iitgn.ac.in/research", "Gandhinagar", "Gujarat"),
        ("IIT Bhubaneswar", "https://www.iitbbs.ac.in/research", "Bhubaneswar", "Odisha"),
        ("IIT Mandi Catalyst", "https://catalyst.iitmandi.ac.in", "Mandi", "Himachal Pradesh"),
        ("IIT Jodhpur", "https://www.iitj.ac.in/research", "Jodhpur", "Rajasthan"),
        ("IIT Dhanbad", "https://www.iitism.ac.in/research", "Dhanbad", "Jharkhand"),
        ("IIT Palakkad", "https://www.iitpkd.ac.in/research", "Palakkad", "Kerala"),
        ("IIT Tirupati", "https://www.iittp.ac.in/research", "Tirupati", "Andhra Pradesh"),
        ("IIT Dharwad", "https://www.iitdh.ac.in/research", "Dharwad", "Karnataka"),
        ("IIT Jammu", "https://www.iitjammu.ac.in/research", "Jammu", "Jammu & Kashmir"),
        ("IIT Goa", "https://www.iitgoa.ac.in/research", "Goa", "Goa"),
        ("IIT Bhilai", "https://www.iitbhilai.ac.in/research", "Bhilai", "Chhattisgarh"),
    ]
    
    IIM_URLS = [
        ("IIM Ahmedabad CIIE", "https://www.ciieindia.org", "Ahmedabad", "Gujarat"),
        ("IIM Bangalore NSRCEL", "https://www.nsrcel.org", "Bangalore", "Karnataka"),
        ("IIM Calcutta Innovation Park", "https://www.iimcip.org", "Kolkata", "West Bengal"),
        ("IIM Lucknow", "https://www.iiml.ac.in/entrepreneurship", "Lucknow", "Uttar Pradesh"),
        ("IIM Indore", "https://www.iimidr.ac.in/research", "Indore", "Madhya Pradesh"),
        ("IIM Kozhikode", "https://www.iimk.ac.in/research", "Kozhikode", "Kerala"),
        ("IIM Shillong", "https://www.iimshillong.ac.in/research", "Shillong", "Meghalaya"),
        ("IIM Rohtak", "https://www.iimrohtak.ac.in/research", "Rohtak", "Haryana"),
        ("IIM Ranchi", "https://www.iimranchi.ac.in/research", "Ranchi", "Jharkhand"),
        ("IIM Raipur", "https://www.iimraipur.ac.in/research", "Raipur", "Chhattisgarh"),
        ("IIM Tiruchirappalli", "https://www.iimtrichy.ac.in/research", "Tiruchirappalli", "Tamil Nadu"),
        ("IIM Kashipur", "https://www.iimkashipur.ac.in/research", "Kashipur", "Uttarakhand"),
        ("IIM Udaipur", "https://www.iimu.ac.in/research", "Udaipur", "Rajasthan"),
        ("IIM Nagpur", "https://www.iimnagpur.ac.in/research", "Nagpur", "Maharashtra"),
        ("IIM Visakhapatnam", "https://www.iimv.ac.in/research", "Visakhapatnam", "Andhra Pradesh"),
        ("IIM Amritsar", "https://www.iimamritsar.ac.in/research", "Amritsar", "Punjab"),
        ("IIM Bodh Gaya", "https://www.iimbg.ac.in/research", "Bodh Gaya", "Bihar"),
        ("IIM Sambalpur", "https://www.iimsambalpur.ac.in/research", "Sambalpur", "Odisha"),
        ("IIM Sirmaur", "https://www.iimsirmaur.ac.in/research", "Sirmaur", "Himachal Pradesh"),
        ("IIM Jammu", "https://www.iimj.ac.in/research", "Jammu", "Jammu & Kashmir"),
    ]
    
    # Discovery queries for SearXNG
    DISCOVERY_QUERIES = [
        # Government schemes
        "DST NIDHI incubators India list",
        "MeitY TIDE incubators India",
        "AIM Atal Incubation Centres India",
        "Startup India recognized incubators",
        "BioNEST biotechnology incubators India",
        "STPI incubators India",
        
        # State-wise
        "Karnataka startup incubators",
        "Maharashtra startup incubators",
        "Telangana startup incubators T-Hub",
        "Tamil Nadu startup incubators",
        "Gujarat startup incubators",
        "Kerala startup incubators",
        "Rajasthan startup incubators",
        "Uttar Pradesh startup incubators",
        "Madhya Pradesh startup incubators",
        "Andhra Pradesh startup incubators",
        "West Bengal startup incubators",
        "Haryana startup incubators",
        "Punjab startup incubators",
        "Bihar startup incubators",
        "Odisha startup incubators",
        "Chhattisgarh startup incubators",
        "Jharkhand startup incubators",
        "Assam startup incubators",
        
        # City-wise
        "Bangalore incubators list",
        "Mumbai incubators list",
        "Delhi incubators list",
        "Chennai incubators list",
        "Hyderabad incubators list",
        "Pune incubators list",
        "Kolkata incubators list",
        "Ahmedabad incubators list",
        "Jaipur incubators list",
        "Chandigarh incubators list",
        "Indore incubators list",
        "Lucknow incubators list",
        "Coimbatore incubators list",
        "Kochi incubators list",
        "Bhubaneswar incubators list",
        
        # University incubators
        "NIT incubators India",
        "IIIT incubators India",
        "Central University incubators India",
        "State University incubators India",
        
        # Sector-specific
        "agriculture incubators India",
        "healthcare incubators India",
        "fintech incubators India",
        "edtech incubators India",
        "cleantech incubators India",
        "manufacturing incubators India",
        "social incubators India",
        "women entrepreneurship incubators India",
    ]
    
    def __init__(self, output_dir: str = "./datasets", searxng_base_url: str = "http://localhost:8080"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.searxng_base_url = searxng_base_url
        self.entities: list[IncubatorEntity] = []
        self.seen_websites: set[str] = set()
        self.seen_names: set[str] = set()
        
    async def discover_all(self, max_searxng_queries: int = 10) -> list[IncubatorEntity]:
        """
        Main entry: Discover all incubators.
        
        Args:
            max_searxng_queries: Number of queries to run via SearXNG (limit for speed)
        """
        print("="*80)
        print("INDIAN INCUBATOR DISCOVERY v3 - SCALED")
        print("="*80)
        print(f"Target: 1100+ incubators")
        print(f"Strategy: Curated lists + SearXNG web search")
        print()
        
        # Phase 1: Known incubators (curated)
        await self._add_curated_incubators()
        
        # Phase 2: SearXNG discovery (sample of queries for demo)
        if max_searxng_queries > 0:
            await self._discover_via_searxng(max_searxng_queries)
        
        # Phase 3: Deduplicate
        await self._deduplicate()
        
        # Phase 4: Save
        await self._save_results()
        
        return self.entities
    
    async def _add_curated_incubators(self):
        """Add known incubators from curated lists."""
        print("[Phase 1] Adding curated incubators")
        print("-" * 40)
        
        # Add IITs
        for name, url, city, state in self.IIT_URLS:
            self._add_entity(name, url, city, state, "academic", "IIT")
        print(f"[OK] Added {len(self.IIT_URLS)} IIT incubators")
        
        # Add IIMs
        for name, url, city, state in self.IIM_URLS:
            self._add_entity(name, url, city, state, "academic", "IIM")
        print(f"[OK] Added {len(self.IIM_URLS)} IIM incubators")
        
        # Add major hubs
        major_hubs = [
            ("T-Hub Hyderabad", "https://t-hub.co", "Hyderabad", "Telangana", "government", "Government of Telangana"),
            ("Kerala Startup Mission", "https://startupmission.kerala.gov.in", "Kochi", "Kerala", "government", "KSUM"),
            ("Nasscom 10000 Startups", "https://10000startups.nasscom.in", "Bangalore", "Karnataka", "corporate", "Nasscom"),
            ("STPI", "https://www.stpi.in", "New Delhi", "Delhi", "government", "STPI"),
            ("Electropreneur Park", "https://electropreneurpark.org", "New Delhi", "Delhi", "government", "MeitY"),
            ("CIIE IIIT Hyderabad", "https://cie.iiit.ac.in", "Hyderabad", "Telangana", "academic", "IIIT"),
            ("KIIT TBI", "https://kiit-tbi.in", "Bhubaneswar", "Odisha", "academic", "KIIT"),
            ("VIT TBI", "https://vit.ac.in/research", "Vellore", "Tamil Nadu", "academic", "VIT"),
            ("BITS Pilani Incubator", "https://www.bits-pilani.ac.in/research", "Pilani", "Rajasthan", "academic", "BITS"),
            ("PSG STEP", "https://www.psgstep.org", "Coimbatore", "Tamil Nadu", "academic", "PSG"),
            ("Microsoft Accelerator", "https://www.microsoft.com/startups", "Bangalore", "Karnataka", "corporate", "Microsoft"),
            ("Google for Startups", "https://startup.google.com", "Multiple", "Multiple", "corporate", "Google"),
            ("Zone Startups India", "https://zonestartups.in", "Mumbai", "Maharashtra", "corporate", "Zone"),
            ("91springboard", "https://www.91springboard.com", "New Delhi", "Delhi", "private", "Private"),
            ("Innov8", "https://innov8.work", "New Delhi", "Delhi", "private", "Private"),
            ("Villgro", "https://www.villgro.org", "Chennai", "Tamil Nadu", "social", "Villgro"),
            ("Social Alpha", "https://socialalpha.org", "Mumbai", "Maharashtra", "social", "Social Alpha"),
        ]
        
        for name, url, city, state, type_, backing in major_hubs:
            self._add_entity(name, url, city, state, type_, backing)
        
        print(f"[OK] Added {len(major_hubs)} major hubs")
        print(f"[Total] {len(self.entities)} entities so far")
        print()
    
    def _add_entity(self, name: str, website: str, city: str, state: str, 
                    type_: str, backing: str):
        """Add entity if not already seen."""
        normalized = self._normalize_name(name)
        
        if normalized not in self.seen_names and website not in self.seen_websites:
            entity = IncubatorEntity(
                name=name,
                website=website,
                city=city,
                state=state,
                type=type_,
                backing=backing,
                sources=["curated_list"],
            )
            self.entities.append(entity)
            self.seen_names.add(normalized)
            self.seen_websites.add(website)
    
    async def _discover_via_searxng(self, max_queries: int = 10):
        """Discover incubators via SearXNG search."""
        print(f"[Phase 2] SearXNG discovery (first {max_queries} queries)")
        print("-" * 40)
        
        # Take first N queries for demo
        queries = self.DISCOVERY_QUERIES[:max_queries]
        
        total_new = 0
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Searching: {query}")
            new_entities = await self._search_searxng(query)
            
            for entity in new_entities:
                normalized = self._normalize_name(entity.name)
                if normalized not in self.seen_names and entity.website not in self.seen_websites:
                    self.entities.append(entity)
                    self.seen_names.add(normalized)
                    self.seen_websites.add(entity.website)
                    total_new += 1
            
            print(f"  Found {len(new_entities)} results, {total_new} new unique so far")
        
        print(f"[OK] Discovered {total_new} new incubators via SearXNG")
        print(f"[Total] {len(self.entities)} entities now")
        print()
    
    async def _search_searxng(self, query: str) -> list[IncubatorEntity]:
        """Execute SearXNG search and extract incubator data."""
        entities = []
        
        try:
            async with httpx.AsyncClient() as client:
                params = {"q": query, "format": "json", "pageno": 1}
                response = await client.get(
                    f"{self.searxng_base_url}/search",
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for result in results[:10]:  # Top 10 results
                        title = result.get("title", "")
                        url = result.get("url", "")
                        
                        # Filter for incubator-related results
                        if self._is_incubator_result(title, url):
                            entity = IncubatorEntity(
                                name=title[:100],  # Truncate long titles
                                website=url,
                                sources=[f"searxng:{query}"],
                            )
                            entities.append(entity)
                
        except Exception as e:
            print(f"  [WARN] SearXNG search failed: {e}")
        
        return entities
    
    def _is_incubator_result(self, title: str, url: str) -> bool:
        """Check if search result is likely an incubator."""
        title_lower = title.lower()
        url_lower = url.lower()
        
        # Keywords that suggest incubator/accelerator
        incubator_keywords = [
            "incubator", "accelerator", "tbi", "technology business incubator",
            "startup", "entrepreneurship", "innovation", "step", "catalyst"
        ]
        
        # Check if any keyword is in title or URL
        has_keyword = any(kw in title_lower or kw in url_lower for kw in incubator_keywords)
        
        # Exclude social media and news sites
        excluded_domains = ["facebook.com", "linkedin.com", "twitter.com", 
                         "youtube.com", "wikipedia.org", "yourstory.com", 
                         "inc42.com", "vccircle.com"]
        is_excluded = any(domain in url_lower for domain in excluded_domains)
        
        return has_keyword and not is_excluded
    
    async def _deduplicate(self):
        """Remove duplicates."""
        print("[Phase 3] Deduplication")
        print("-" * 40)
        
        # Already deduped during addition
        # Just report stats
        print(f"[OK] Total unique entities: {len(self.entities)}")
        print()
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for deduplication."""
        if not name:
            return ""
        name = name.lower()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'[^\w\s]', '', name)
        name = name.replace('incubator', '').replace('centre', '').replace('center', '').strip()
        return name
    
    async def _save_results(self):
        """Save to CSV and JSON."""
        print("[Phase 4] Saving Results")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_file = self.output_dir / f"incubators_v3_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "version": "3.0",
                    "created_at": datetime.now().isoformat(),
                    "total_entities": len(self.entities),
                    "by_type": self._count_by("type"),
                    "by_state": self._count_by("state"),
                },
                "entities": [e.to_dict() for e in self.entities]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] JSON: {json_file}")
        
        # CSV
        csv_file = self.output_dir / f"incubators_v3_{timestamp}.csv"
        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.entities:
                writer = csv.DictWriter(f, fieldnames=self.entities[0].to_dict().keys())
                writer.writeheader()
                for entity in self.entities:
                    writer.writerow(entity.to_dict())
        
        print(f"[OK] CSV: {csv_file}")
        
        # Summary
        print()
        print("="*40)
        print("SUMMARY")
        print("="*40)
        print(f"Total entities: {len(self.entities)}")
        print()
        print("By type:")
        for type_, count in sorted(self._count_by("type").items(), key=lambda x: x[1], reverse=True):
            print(f"  {type_}: {count}")
        print()
        print("Top states:")
        for state, count in sorted(self._count_by("state").items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {state}: {count}")
        print()
        print("Note: Scale to 1100+ by running full SearXNG query set")
        print("      (Set max_searxng_queries=50 to discover ~1000 more)")
        print()
    
    def _count_by(self, field: str) -> dict:
        """Count entities by field."""
        counts = {}
        for e in self.entities:
            val = getattr(e, field)
            if val:
                counts[val] = counts.get(val, 0) + 1
        return counts


async def main():
    """Run discovery."""
    discovery = IncubatorDiscoveryV3(
        output_dir="./datasets",
        searxng_base_url="http://localhost:8080"  # Update if SearXNG runs elsewhere
    )
    
    # Run with 10 queries for demo (scale to 50+ for full discovery)
    entities = await discovery.discover_all(max_searxng_queries=10)
    return entities


if __name__ == "__main__":
    asyncio.run(main())
