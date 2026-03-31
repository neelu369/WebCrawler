"""
Incubator-specific ranking criteria and weights.

This module defines:
- Custom criteria for ranking incubators
- Dynamic weight adjustment based on user preferences
- Pre-defined ranking profiles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum


class RankingProfile(Enum):
    """Pre-defined ranking profiles."""
    DEFAULT = "default"
    FUNDING = "funding_focused"
    MENTORSHIP = "mentorship_focused"
    SECTOR = "sector_aligned"
    STAGE = "stage_aligned"
    LOCATION = "location_preference"
    EQUITY = "equity_friendly"


@dataclass
class RankingCriterion:
    """Single ranking criterion with weight and direction."""
    name: str
    weight: float
    higher_is_better: bool = True
    description: str = ""
    value_extractor: Optional[Callable] = None
    
    def calculate_score(self, entity: dict) -> float:
        """Calculate normalized score for this criterion."""
        if self.value_extractor:
            value = self.value_extractor(entity)
        else:
            value = entity.get(self.name)
        
        # Handle missing data
        if value is None or value == "":
            return 0.5  # Neutral score for missing data
        
        # Normalize based on type
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Try to extract numeric value
            return self._extract_numeric(value)
        
        return 0.5
    
    def _extract_numeric(self, value: str) -> float:
        """Extract numeric value from string."""
        import re
        
        # Handle special cases
        if "lakh" in value.lower():
            match = re.search(r'(\d+)', value)
            if match:
                return float(match.group(1)) / 100  # Convert to lakhs in crores
        elif "crore" in value.lower():
            match = re.search(r'(\d+)', value)
            if match:
                return float(match.group(1))
        elif "cr" in value.lower():
            match = re.search(r'(\d+)', value)
            if match:
                return float(match.group(1))
        
        # Generic number extraction
        match = re.search(r'(\d+(?:\.\d+)?)', value)
        if match:
            return float(match.group(1))
        
        return 0.5


class IncubatorRankingCriteria:
    """
    Comprehensive ranking criteria for Indian incubators.
    
    Criteria are grouped into categories:
    - Financial metrics (funding, investment, equity)
    - Operational metrics (alumni, team, mentors)
    - Quality metrics (programs, sectors, duration)
    - Geographic/Access metrics (virtual, location)
    """
    
    # ─── Financial Criteria ─────────────────────────────────
    FINANCIAL_CRITERIA = {
        "investment_range": RankingCriterion(
            name="investment_range",
            weight=0.15,
            higher_is_better=True,
            description="Maximum funding provided to startups (in Crores)"
        ),
        "equity_taken": RankingCriterion(
            name="equity_taken",
            weight=0.10,
            higher_is_better=False,  # Lower equity is better
            description="Equity percentage taken (lower is better)"
        ),
        "funding_type": RankingCriterion(
            name="funding_type",
            weight=0.05,
            higher_is_better=True,
            description="Type of funding (grant=best, equity=medium, debt=low)"
        ),
        "total_investment_made": RankingCriterion(
            name="total_investment_made",
            weight=0.10,
            higher_is_better=True,
            description="Total historical investment made by incubator (in Crores)"
        ),
    }
    
    # ─── Operational Criteria ────────────────────────────────
    OPERATIONAL_CRITERIA = {
        "alumni_count": RankingCriterion(
            name="alumni_count",
            weight=0.15,
            higher_is_better=True,
            description="Number of graduated startups"
        ),
        "active_startups": RankingCriterion(
            name="active_startups",
            weight=0.10,
            higher_is_better=True,
            description="Current portfolio size"
        ),
        "team_size": RankingCriterion(
            name="team_size",
            weight=0.05,
            higher_is_better=True,
            description="Incubator management team size"
        ),
        "mentor_count": RankingCriterion(
            name="mentor_count",
            weight=0.10,
            higher_is_better=True,
            description="Number of mentors in network"
        ),
        "established_year": RankingCriterion(
            name="established_year",
            weight=0.05,
            higher_is_better=True,  # Older is better (experience)
            description="Year established (experience)"
        ),
    }
    
    # ─── Program Quality Criteria ───────────────────────────
    QUALITY_CRITERIA = {
        "program_quality": RankingCriterion(
            name="programs",
            weight=0.10,
            higher_is_better=True,
            description="Quality and variety of programs offered"
        ),
        "focus_sectors_count": RankingCriterion(
            name="focus_sectors",
            weight=0.05,
            higher_is_better=True,
            description="Number of focus sectors (diversity)"
        ),
        "duration_months": RankingCriterion(
            name="duration_months",
            weight=0.05,
            higher_is_better=True,
            description="Program duration in months"
        ),
        "virtual_available": RankingCriterion(
            name="virtual_available",
            weight=0.05,
            higher_is_better=True,
            description="Virtual/hybrid program availability"
        ),
    }
    
    # ─── Pre-defined Profiles ────────────────────────────────
    PROFILES = {
        RankingProfile.DEFAULT: {
            "weights": {
                "investment_range": 0.15,
                "equity_taken": 0.10,
                "alumni_count": 0.15,
                "mentor_count": 0.10,
                "active_startups": 0.10,
                "total_investment_made": 0.10,
                "program_quality": 0.10,
                "focus_sectors_count": 0.05,
                "duration_months": 0.05,
                "virtual_available": 0.05,
                "team_size": 0.05,
            },
            "description": "Balanced ranking across all dimensions"
        },
        
        RankingProfile.FUNDING: {
            "weights": {
                "investment_range": 0.30,
                "total_investment_made": 0.25,
                "equity_taken": 0.15,
                "funding_type": 0.10,
                "alumni_count": 0.10,
                "mentor_count": 0.05,
                "active_startups": 0.05,
            },
            "description": "Prioritizes funding and investment capacity"
        },
        
        RankingProfile.MENTORSHIP: {
            "weights": {
                "mentor_count": 0.30,
                "team_size": 0.20,
                "alumni_count": 0.15,
                "program_quality": 0.15,
                "established_year": 0.10,
                "duration_months": 0.10,
            },
            "description": "Prioritizes mentorship and support quality"
        },
        
        RankingProfile.SECTOR: {
            "weights": {
                "focus_sectors_count": 0.20,
                "program_quality": 0.20,
                "alumni_count": 0.15,
                "mentor_count": 0.15,
                "investment_range": 0.15,
                "active_startups": 0.15,
            },
            "description": "Prioritizes sector alignment and diversity"
        },
        
        RankingProfile.STAGE: {
            "weights": {
                "duration_months": 0.25,
                "program_quality": 0.25,
                "alumni_count": 0.15,
                "mentor_count": 0.15,
                "investment_range": 0.10,
                "virtual_available": 0.10,
            },
            "description": "Prioritizes program structure and startup stage fit"
        },
        
        RankingProfile.LOCATION: {
            "weights": {
                "virtual_available": 0.30,
                "active_startups": 0.20,
                "alumni_count": 0.15,
                "program_quality": 0.15,
                "mentor_count": 0.10,
                "investment_range": 0.10,
            },
            "description": "Prioritizes accessibility and location flexibility"
        },
        
        RankingProfile.EQUITY: {
            "weights": {
                "equity_taken": 0.40,
                "funding_type": 0.25,
                "investment_range": 0.15,
                "total_investment_made": 0.10,
                "alumni_count": 0.05,
                "active_startups": 0.05,
            },
            "description": "Prioritizes equity-friendly terms (lower equity, grants)"
        },
    }
    
    @classmethod
    def get_criteria_for_profile(cls, profile: RankingProfile) -> dict[str, RankingCriterion]:
        """Get criteria weights for a specific profile."""
        profile_config = cls.PROFILES.get(profile, cls.PROFILES[RankingProfile.DEFAULT])
        weights = profile_config["weights"]
        
        # Combine all criteria
        all_criteria = {
            **cls.FINANCIAL_CRITERIA,
            **cls.OPERATIONAL_CRITERIA,
            **cls.QUALITY_CRITERIA,
        }
        
        # Update weights based on profile
        result = {}
        for name, weight in weights.items():
            if name in all_criteria:
                criterion = all_criteria[name]
                result[name] = RankingCriterion(
                    name=criterion.name,
                    weight=weight,
                    higher_is_better=criterion.higher_is_better,
                    description=criterion.description,
                    value_extractor=criterion.value_extractor
                )
        
        return result
    
    @classmethod
    def create_custom_weights(cls, user_preferences: dict) -> dict[str, RankingCriterion]:
        """
        Create custom weights based on user preferences.
        
        Args:
            user_preferences: Dict mapping criterion name to weight (0-1)
                             e.g., {"investment_range": 0.4, "equity_taken": 0.3}
        
        Returns:
            Dict of criteria with normalized weights
        """
        all_criteria = {
            **cls.FINANCIAL_CRITERIA,
            **cls.OPERATIONAL_CRITERIA,
            **cls.QUALITY_CRITERIA,
        }
        
        # Start with user preferences
        custom_weights = user_preferences.copy()
        
        # Add default weights for unspecified criteria
        default_weights = cls.PROFILES[RankingProfile.DEFAULT]["weights"]
        for name, criterion in all_criteria.items():
            if name not in custom_weights:
                custom_weights[name] = default_weights.get(name, 0.05)
        
        # Normalize weights to sum to 1.0
        total = sum(custom_weights.values())
        if total > 0:
            custom_weights = {k: v/total for k, v in custom_weights.items()}
        
        # Create criteria objects
        result = {}
        for name, weight in custom_weights.items():
            if name in all_criteria:
                criterion = all_criteria[name]
                result[name] = RankingCriterion(
                    name=criterion.name,
                    weight=weight,
                    higher_is_better=criterion.higher_is_better,
                    description=criterion.description,
                    value_extractor=criterion.value_extractor
                )
        
        return result


# Usage examples
if __name__ == "__main__":
    # Example 1: Get default criteria
    default_criteria = IncubatorRankingCriteria.get_criteria_for_profile(RankingProfile.DEFAULT)
    print("Default Criteria:")
    for name, criterion in default_criteria.items():
        print(f"  {name}: {criterion.weight:.2f} - {criterion.description}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Get funding-focused criteria
    funding_criteria = IncubatorRankingCriteria.get_criteria_for_profile(RankingProfile.FUNDING)
    print("Funding-Focused Criteria:")
    for name, criterion in funding_criteria.items():
        print(f"  {name}: {criterion.weight:.2f} - {criterion.description}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Custom user preferences
    user_prefs = {
        "investment_range": 0.5,  # User cares a lot about funding
        "mentor_count": 0.3,      # User also values mentorship
        "equity_taken": 0.2,      # User wants low equity
    }
    custom_criteria = IncubatorRankingCriteria.create_custom_weights(user_prefs)
    print("Custom Criteria (based on user preferences):")
    for name, criterion in sorted(custom_criteria.items(), key=lambda x: x[1].weight, reverse=True)[:10]:
        print(f"  {name}: {criterion.weight:.2f} - {criterion.description}")
