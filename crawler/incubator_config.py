"""
Configuration for Indian Incubator Discovery.

This extends the base Configuration with incubator-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import list

from crawler.config import Configuration


@dataclass(kw_only=True)
class IncubatorConfiguration(Configuration):
    """Extended configuration for incubator-specific crawling."""
    
    # Target numbers
    target_incubator_count: int = field(
        default=1170,
        metadata={"description": "Target number of unique incubators to discover."}
    )
    
    min_confidence_threshold: float = field(
        default=0.7,
        metadata={"description": "Minimum confidence score to include an incubator."}
    )
    
    # Source weights (affects confidence scoring)
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "startup_india_portal": 0.95,
            "dst_nidhi": 0.95,
            "meity_tide": 0.95,
            "aim_aic": 0.95,
            "isba_members": 0.90,
            "iit_websites": 0.85,
            "iim_websites": 0.85,
            "state_startup_portals": 0.80,
            "commercial_news": 0.60,
            "secondary_search": 0.50,
        },
        metadata={"description": "Confidence multipliers by source type."}
    )
    
    # Enrichment settings
    enrichment_max_retries: int = field(
        default=3,
        metadata={"description": "Max retries per field when enriching."}
    )
    
    enrichment_timeout_per_entity: int = field(
        default=60,
        metadata={"description": "Seconds to spend enriching each entity."}
    )
    
    # Fields to prioritize
    required_fields: list[str] = field(
        default_factory=lambda: [
            "name",
            "website",
            "city",
            "state",
            "type",
            "backing",
        ],
        metadata={"description": "Fields that must be present for basic entry."}
    )
    
    optional_fields: list[str] = field(
        default_factory=lambda: [
            "email",
            "phone",
            "focus_sectors",
            "programs",
            "established_year",
            "alumni_count",
            "team_size",
            "mentor_count",
            "funding_type",
            "investment_range",
        ],
        metadata={"description": "Fields to collect if available."}
    )
    
    # Discovery patterns
    seed_queries: list[str] = field(
        default_factory=lambda: [
            "list of incubators in India",
            "government funded incubators India",
            "DST NIDHI incubators",
            "MeitY TIDE incubators",
            "Atal Incubation Centres India",
            "IIT incubators India",
            "IIM incubators India",
            "private incubators India",
            "startup incubators Bangalore",
            "startup incubators Mumbai",
            "startup incubators Delhi",
            "startup incubators Hyderabad",
            "Kerala Startup Mission",
            "T-Hub Hyderabad",
            "STPI incubators",
        ],
        metadata={"description": "Seed queries for discovery."}
    )
