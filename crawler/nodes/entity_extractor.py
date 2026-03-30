"""Entity Extractor node — extracts triples for Neo4j knowledge graph."""
from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Optional
from crawler.llm import replicate
from langchain_core.runnables import RunnableConfig
from motor.motor_asyncio import AsyncIOMotorClient
from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import GraphEntity, Triple
from crawler.state import State
from crawler.utils import clean_text as _clean_text

_client: AsyncIOMotorClient | None = None

def _get_client():
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        if not uri.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError(f"[EntityExtractor] MONGO_URI invalid: {uri!r}")
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    return _client

_PROMPT = """\
You are a knowledge graph engineer building a ranking dataset. Extract ALL relevant entities and their comparative data.

User query: {query}

Text:
{content}

EXTRACTION RULES:
1. Extract EVERY named entity relevant to this query (aim for all you can find).
2. For EACH entity, extract as many measurable/comparable facts as possible:
   - Numbers, percentages, scores, rates, counts, dates, ranks
   - Difficulty levels, pass rates, cut-offs, application counts, seat counts
   - Any data that could be used to compare entities
3. If the text only lists names without data, still extract the names with entity_type and description.
4. For exams: extract pass rate, number of applicants, difficulty level, conducting body, seats available, cut-off score.
5. For movies: extract IMDB rating, box office, release year, director, genre.
6. For companies: extract funding, valuation, founding year, location, industry.

NEVER extract:
- Website names (IMDb, Wikipedia, etc.)
- Pure category labels

entity_type options: Exam, Film, Company, Person, Organization, University, Technology, Competition, Event, Entity

Return a JSON array. Each element:
{{
  "name": "exact entity name",
  "entity_type": "Exam",
  "description": "1-2 sentences",
  "priority_score": 0.9,
  "triples": [
    {{
      "subject": "entity name",
      "predicate": "HAS_PASS_RATE",
      "object": "0.1%",
      "evidence_snippet": "only 0.1% candidates qualify",
      "confidence": 0.95
    }}
  ]
}}

PREDICATES — use the most specific one:
Exams: HAS_PASS_RATE, HAS_APPLICANTS, HAS_SEATS, HAS_CUTOFF, CONDUCTED_BY, HAS_DIFFICULTY, HELD_IN, REQUIRES_ELIGIBILITY, RANKED_AT
Films: HAS_IMDB_RATING, HAS_BOX_OFFICE, DIRECTED_BY, RELEASED_IN, HAS_GENRE, HAS_RATING, HAS_BUDGET, WON_AWARD
Companies: HAS_FUNDING, HAS_VALUATION, FOUNDED_IN, LOCATED_IN, FOUNDED_BY, SUPPORTS_INDUSTRY
General: HAS_PROPERTY, IS_TYPE_OF, RELATED_TO, RANKED_AT, HAS_SCORE, HAS_RANK

STRICT RULES:
- Return ONLY a raw JSON array, no markdown, no text outside the array.
- Every triple must be a JSON object with all 5 keys — never a string.
- Skip triples where object is null, N/A, unknown, or empty.
- Include entities even with zero triples if they are real named items.
- Aim for maximum coverage — extract every individual item in the text.
"""

_PLACEHOLDER_VALUES = {"not specified","not mentioned","unknown","n/a","not disclosed","not available","not publicly disclosed","none mentioned"}

# Patterns that indicate a list/website/category — not a real entity
_JUNK_NAME_PATTERNS = re.compile(
    r"^("
    # Only block pure list/collection page titles — NOT individual items
    r"list of \w|collection of \w|ranking of \w|"
    # Website/publication names
    r"imdb top|rotten tomatoes|metacritic|box office mojo|afi list|bfi list|"
    r"sight & sound|empire magazine|they shoot pictures|"
    # Pure category labels with no specific name
    r"english movies$|hollywood films$|action genre$|comedy genre$"
    r")",
    re.IGNORECASE,
)
_JUNK_SUBSTRINGS = {
    "list of films", "list of movies", "collection of films",
    "box office mojo", "rotten tomatoes top", "imdb top 250",
}
_JUNK_ENTITY_TYPES = {"website", "publication", "source"}

def _is_junk_entity(name: str, entity_type: str) -> bool:
    """Return True only if this is clearly a list/website — not a real named entity."""
    name_lower = name.lower().strip()
    if entity_type.lower() in _JUNK_ENTITY_TYPES:
        return True
    if _JUNK_NAME_PATTERNS.match(name_lower):
        return True
    if any(junk in name_lower for junk in _JUNK_SUBSTRINGS):
        return True
    return False

async def extract_entities(state: State, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    entity_aggregator: dict[str, GraphEntity] = {}

    for src in state.verified_sources:
        prompt = _PROMPT.format(query=state.user_query, content=_clean_text(src.content)[:4000])
        t0 = time.time()
        try:
            output = replicate.run(configuration.model, input={"prompt": prompt, "max_tokens": 4096, "temperature": 0.1})
            raw_text = "".join(str(c) for c in output)
            tracker.record(node="entity_extractor", model=configuration.model, input_tokens=len(prompt)//4, output_tokens=len(raw_text)//4, latency_s=time.time()-t0)

            cleaned = raw_text.strip()
            # Strip markdown fences
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            # Find the start of the JSON (array or object)
            for start_char in ("[", "{"):
                idx = cleaned.find(start_char)
                if idx != -1:
                    cleaned = cleaned[idx:]
                    break
            entities_data = json.loads(cleaned)
            # Handle {"entities": [...]} wrapper the LLM sometimes emits
            if isinstance(entities_data, dict):
                for key in ("entities", "results", "data", "items"):
                    if isinstance(entities_data.get(key), list):
                        entities_data = entities_data[key]
                        break
                else:
                    # Single entity wrapped in a dict
                    entities_data = [entities_data] if "name" in entities_data else []
            if not isinstance(entities_data, list):
                entities_data = []

            for data in entities_data:
                if not isinstance(data, dict):
                    continue
                name = data.get("name", "Unknown Entity")
                norm_name = name.lower().strip()
                if not norm_name or norm_name == "unknown entity":
                    continue
                entity_type = data.get("entity_type", "Entity")

                # Drop list pages, websites, categories
                if _is_junk_entity(name, entity_type):
                    print(f"[EntityExtractor] Skipping junk entity: {name!r}")
                    continue
                triples = []
                for t in data.get("triples", []):
                    # Guard: LLM sometimes returns a string like "subject, predicate, object, ..."
                    # instead of a proper dict — skip anything that isn't a dict
                    if not isinstance(t, dict):
                        print(f"[EntityExtractor] Skipping non-dict triple: {str(t)[:80]}")
                        continue
                    obj_val = str(t.get("object", "")).strip()
                    if not obj_val or obj_val.lower() in _PLACEHOLDER_VALUES:
                        continue
                    predicate = str(t.get("predicate", "HAS_PROPERTY")).strip() or "HAS_PROPERTY"
                    try:
                        confidence = float(t.get("confidence", 0.8))
                    except (TypeError, ValueError):
                        confidence = 0.8
                    triples.append(Triple(
                        subject=str(t.get("subject", name)).strip() or name,
                        predicate=predicate,
                        object=obj_val,
                        evidence_snippet=str(t.get("evidence_snippet", "")).strip(),
                        source_url=src.url,
                        confidence=confidence,
                    ))

                priority = float(data.get("priority_score", 0.5))
                desc = data.get("description", "")
                if norm_name in entity_aggregator:
                    ex = entity_aggregator[norm_name]
                    if priority > ex.priority_score: ex.priority_score = priority
                    if len(desc) > len(ex.description): ex.description = desc
                    if src.url not in ex.source_url: ex.source_url += f", {src.url}"
                    existing_keys = {(t.predicate, t.object.lower()) for t in ex.triples}
                    for triple in triples:
                        if (triple.predicate, triple.object.lower()) not in existing_keys:
                            ex.triples.append(triple); existing_keys.add((triple.predicate, triple.object.lower()))
                else:
                    entity_aggregator[norm_name] = GraphEntity(name=name, entity_type=entity_type, description=desc, triples=triples, source_url=src.url, priority_score=priority)
        except Exception as exc:
            print(f"[EntityExtractor] Failed for {src.url}: {exc}")

    graph_entities = list(entity_aggregator.values())
    try:
        if graph_entities:
            from datetime import datetime, timezone
            client = _get_client()
            col = client[configuration.mongo_db_name]["graph_entities"]
            now = datetime.now(timezone.utc)
            await col.insert_many([{**ge.model_dump(), "session_id": state.session_id, "created_at": now} for ge in graph_entities])
    except Exception as exc:
        print(f"[EntityExtractor] MongoDB write failed: {exc}")

    print(f"[EntityExtractor] {len(graph_entities)} entities, {sum(len(ge.triples) for ge in graph_entities)} triples")
    return {"graph_entities": graph_entities}