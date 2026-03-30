"""Shared utilities used across multiple pipeline nodes."""

from __future__ import annotations

import re
from typing import Any


# ── Text cleaning ────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip leftover HTML artifacts and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # HTML entities
    return re.sub(r"\s+", " ", text).strip()


# ── Geographic filtering ────────────────────────────────────

_GEO_FILTERS: dict[str, list[str]] = {
    "india":  ["united states", "usa", "u.s.", "silicon valley", "london", "uk", "singapore", "europe", "australia", "canada"],
    "us":     ["india", "china", "europe", "australia"],
    "uk":     ["india", "usa", "china", "australia"],
    "europe": ["india", "usa", "china", "australia"],
}

_GEO_KEYWORDS: dict[str, list[str]] = {
    "india": ["india", "indian", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai", "pune", "kolkata"],
    "us":    ["us", "usa", "united states", "silicon valley", "san francisco", "new york"],
    "uk":    ["uk", "united kingdom", "london"],
    "europe": ["europe", "european"],
}


def detect_target_region(query: str) -> str | None:
    """Detect the geographic region the query is focused on."""
    q = query.lower()
    for region, keywords in _GEO_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return region
    return None


def geo_filter_entities(
    entities: list[Any],
    user_query: str,
    *,
    location_keys: tuple[str, ...] = ("Located In", "Headquartered In", "Location"),
) -> list[Any]:
    """Remove entities whose location contradicts the query's geographic focus.

    Works with any object that has a `properties` dict and a `name` attribute.
    Returns the original list unchanged if no geographic filter applies, or if
    filtering would remove all entities.
    """
    target_region = detect_target_region(user_query)
    if not target_region:
        return entities

    exclude = set(_GEO_FILTERS.get(target_region, []))
    if not exclude:
        return entities

    filtered, removed = [], []

    for entity in entities:
        props = entity.properties if hasattr(entity, "properties") else {}
        if isinstance(props, dict):
            location = ""
            for key in location_keys:
                val = props.get(key, "")
                if val:
                    location = str(val).lower()
                    break
        else:
            location = ""

        contradicts = (
            any(place in location for place in exclude)
            and target_region not in location
            and bool(location)
        )

        if contradicts:
            name = getattr(entity, "name", getattr(entity, "entity_name", "?"))
            removed.append(name)
        else:
            filtered.append(entity)

    if removed:
        print(f"[GeoFilter] Removed {len(removed)} off-target entities: {removed[:5]}")

    return filtered if filtered else entities
