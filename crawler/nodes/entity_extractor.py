"""Entity Extractor node — extracts triples for Neo4j knowledge graph."""
from __future__ import annotations
import json, os, re, time
from typing import Any, Optional
import replicate
from langchain_core.runnables import RunnableConfig
from motor.motor_asyncio import AsyncIOMotorClient
from crawler.config import Configuration
from crawler.cost_tracker import tracker
from crawler.models import GraphEntity, Triple
from crawler.state import State

_client: AsyncIOMotorClient | None = None

def _get_client():
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        if not uri.startswith(("mongodb://", "mongodb+srv://")):
            if uri.startswith(("bolt://", "neo4j://", "bolt+s://", "neo4j+s://")):
                raise ValueError(
                    f"[EntityExtractor] MONGO_URI looks like a Neo4j URI: {uri!r}\n"
                    "Fix .env: MONGO_URI=mongodb://localhost:27017"
                )
            raise ValueError(f"[EntityExtractor] MONGO_URI invalid: {uri!r}")
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    return _client

def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()

_PROMPT = """\
You are a knowledge graph engineer. Extract SPECIFIC INDIVIDUAL ENTITIES that directly answer the user's query.

User query: {query}

Text:
{content}

WHAT TO EXTRACT:
- Extract ONLY specific, individual, named entities that directly answer the query.
- For movie queries: extract individual FILM TITLES only (e.g. "The Godfather", "Inception").
- For startup queries: extract individual COMPANY NAMES only.
- For person queries: extract individual PERSON NAMES only.

WHAT TO NEVER EXTRACT (these are NOT valid entities):
- List titles or collection names (e.g. "Top 100 Movies", "Greatest Films By Decade", "Best Movies of All Time")
- Website names, publication names, or source names (e.g. "IMDb", "Box Office Mojo", "Rotten Tomatoes")
- Category labels (e.g. "English Movies", "Hollywood Films", "Action Genre")
- Any entity that is a ranking list, a webpage, or a category — NOT a specific item

PREDICATES TO USE based on query type:
- Movies: DIRECTED_BY, RELEASED_IN, HAS_IMDB_RATING, HAS_BOX_OFFICE, STARRING, HAS_GENRE, HAS_RUNTIME, HAS_RATING, WON_AWARD, PRODUCED_BY, SET_IN
- Companies: LOCATED_IN, FOUNDED_IN, FOUNDED_BY, HAS_FUNDING, SUPPORTS_INDUSTRY, IS_TYPE_OF
- People: BORN_IN, WORKS_AT, KNOWN_FOR, HAS_NATIONALITY
- General: HAS_PROPERTY, IS_TYPE_OF, RELATED_TO

Return a JSON array where each element has EXACTLY these keys:
  "name"           - the specific entity name (string)
  "entity_type"    - Film, Company, Person, Organization, Technology, or Entity
  "description"    - 1-2 sentence factual summary (string)
  "priority_score" - relevance float 0.0 to 1.0
  "triples"        - array of fact objects, each with EXACTLY:
      "subject"          - same as entity name (string)
      "predicate"        - UPPERCASE_SNAKE_CASE from the list above
      "object"           - the actual fact value (string, NEVER null/unknown/N/A)
      "evidence_snippet" - short quote from the text proving this fact
      "confidence"       - float 0.0 to 1.0

STRICT OUTPUT RULES:
- Return ONLY a raw JSON array. No markdown, no explanation, no extra text.
- Return [] if no valid individual entities are found.
- Every triple must be a JSON object — never a plain string.
- Skip any triple where the object is unknown, N/A, "not specified", or empty.
- An entity with zero good triples should still be included if it is a real specific item.

Example for a movie query:
[{{"name":"The Godfather","entity_type":"Film","description":"1972 crime film directed by Francis Ford Coppola.","priority_score":0.95,"triples":[{{"subject":"The Godfather","predicate":"DIRECTED_BY","object":"Francis Ford Coppola","evidence_snippet":"directed by Francis Ford Coppola","confidence":0.99}},{{"subject":"The Godfather","predicate":"RELEASED_IN","object":"1972","evidence_snippet":"released in 1972","confidence":0.99}},{{"subject":"The Godfather","predicate":"HAS_IMDB_RATING","object":"9.2","evidence_snippet":"IMDb rating of 9.2","confidence":0.95}}]}}]
"""

_PLACEHOLDER_VALUES = {"not specified","not mentioned","unknown","n/a","not disclosed","not available","not publicly disclosed","none mentioned"}

# Patterns that indicate a list/website/category — not a real entity
_JUNK_NAME_PATTERNS = re.compile(
    r"^("
    r"top\s+\d+|greatest\s+\d*|best\s+\d*\s*|most\s+\d*|"
    r"\d+\s+(greatest|best|top|highest)|"
    r"all[- ]time|by\s+(decade|year|genre|director)|"
    r"box\s+office|lifetime\s+gross(es)?|"
    r"imdb\s+top|rotten\s+tomatoes|metacritic|afi\s+|"
    r"greatest\s+(american|english|british|film|movie|series|franchise)|"
    r"most\s+popular|highest\s+rated|hall\s+of\s+fame|"
    r"the\s+21st\s+century|21st\s+century\s+movies|"
    r"top\s+lifetime|empire\s+magazine|sight\s+&\s+sound|"
    r"they\s+shoot\s+pictures|bfi\s+"
    r")",
    re.IGNORECASE,
)
_JUNK_SUBSTRINGS = {
    "list of", "films by", "movies by", "collection of", "series of",
    "ranking of", "greatest films", "greatest movies", "100 films",
    "100 movies", "box office", "all-time", "by decade", "by genre",
    "by year", "top lifetime", "adjusted grosses",
}
_JUNK_ENTITY_TYPES = {"list", "collection", "website", "publication", "source", "category", "ranking", "chart"}

def _is_junk_entity(name: str, entity_type: str) -> bool:
    """Return True if this looks like a list page / website / category rather than a real entity."""
    name_lower = name.lower().strip()
    if entity_type.lower() in _JUNK_ENTITY_TYPES:
        return True
    if _JUNK_NAME_PATTERNS.match(name_lower):
        return True
    if any(junk in name_lower for junk in _JUNK_SUBSTRINGS):
        return True
    # Reject names that are clearly website titles or editorial lists (contain " - " + org name)
    if re.search(r"\s[-–]\s+(afi|bfi|imdb|rotten|metacritic|empire|sight)", name_lower):
        return True
    return False

async def extract_entities(state: State, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    entity_aggregator: dict[str, GraphEntity] = {}

    for src in state.verified_sources:
        prompt = _PROMPT.format(query=state.user_query, content=_clean_text(src.content)[:4000])
        t0 = time.time()
        try:
            output = replicate.run(configuration.model, input={"prompt": prompt, "max_tokens": 2048, "temperature": 0.1})
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
