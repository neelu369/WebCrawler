"""
Neo4j Integration for Incubator Knowledge Graph
================================================

Neo4j plays a critical role in the incubator discovery workflow:

1. ENTITY RELATIONSHIPS - Connect incubators, people, companies, locations
2. KNOWLEDGE PERSISTENCE - Store structured graph data beyond flat CSV
3. INSIGHT GENERATION - Run graph queries for patterns and recommendations
4. DISCOVERY ACCELERATION - Use graph traversal to find related incubators

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class IncubatorGraphEntity:
    """
    Incubator entity represented as graph nodes and relationships.
    
    In Neo4j, this becomes:
    - (i:Incubator) - main incubator node
    - (c:City) - location
    - (s:State) - state/region
    - (t:Type) - incubator type
    - (b:Backing) - backing organization
    - (p:Program) - programs offered
    - (fs:FocusSector) - focus areas
    
    With relationships:
    - (i)-[:LOCATED_IN]->(c)
    - (c)-[:IN_STATE]->(s)
    - (i)-[:TYPE_OF]->(t)
    - (i)-[:BACKED_BY]->(b)
    - (i)-[:OFFERS]->(p)
    - (i)-[:FOCUSES_ON]->(fs)
    """
    
    # Core properties (stored on Incubator node)
    id: str
    name: str
    website: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    established_year: Optional[int] = None
    alumni_count: Optional[int] = None
    team_size: Optional[int] = None
    mentor_count: Optional[int] = None
    data_completeness: float = 0.0
    
    # Related entities (connected via relationships)
    city: Optional[str] = None
    state: Optional[str] = None
    type: Optional[str] = None
    backing: Optional[str] = None
    programs: List[str] = None
    focus_sectors: List[str] = None
    
    # Relationships to other entities
    related_incubators: List[str] = None  # Similar incubators
    partner_organizations: List[str] = None  # Partner orgs
    alumni_startups: List[str] = None  # Graduated companies
    
    def __post_init__(self):
        if self.programs is None:
            self.programs = []
        if self.focus_sectors is None:
            self.focus_sectors = []
        if self.related_incubators is None:
            self.related_incubators = []
        if self.partner_organizations is None:
            self.partner_organizations = []
        if self.alumni_startups is None:
            self.alumni_startups = []


class IncubatorNeo4jSchema:
    """
    Neo4j schema for incubator knowledge graph.
    
    This defines the complete graph structure for:
    - Storing incubator data
    - Querying relationships
    - Finding patterns
    - Generating insights
    """
    
    # =========================================================================
    # NODE TYPES
    # =========================================================================
    
    NODE_INCUBATOR = """
    (i:Incubator {
        id: string,              // Unique identifier
        name: string,            // Incubator name
        website: string,         // Website URL
        email: string,           // Contact email
        phone: string,           // Contact phone
        established_year: int,   // Year founded
        alumni_count: int,       // Number of graduated startups
        team_size: int,          // Incubator team size
        mentor_count: int,       // Mentor network size
        data_completeness: float, // 0.0 to 1.0
        created_at: datetime,    // When added to graph
        updated_at: datetime     // Last update
    })
    """
    
    NODE_CITY = """
    (c:City {
        name: string,            // City name
        state: string,          // State name
        tier: string           // Tier 1/2/3 classification
    })
    """
    
    NODE_STATE = """
    (s:State {
        name: string,           // State name
        region: string,        // North/South/East/West
        startup_friendly: float // Startup ecosystem score
    })
    """
    
    NODE_TYPE = """
    (t:Type {
        name: string  // government, academic, corporate, private, social
    })
    """
    
    NODE_BACKING = """
    (b:Backing {
        name: string,           // Organization name
        type: string           // IIT, IIM, DST, Corporate, etc.
    })
    """
    
    NODE_PROGRAM = """
    (p:Program {
        name: string,          // Program name
        duration_months: int,  // Duration
        virtual: boolean      // Virtual/hybrid availability
    })
    """
    
    NODE_SECTOR = """
    (fs:FocusSector {
        name: string,          // Sector name
        category: string      // Tech/Healthcare/Agriculture/etc
    })
    """
    
    NODE_PERSON = """
    (person:Person {
        name: string,
        role: string,          // Founder, Mentor, Director
        email: string
    })
    """
    
    # =========================================================================
    # RELATIONSHIP TYPES
    # =========================================================================
    
    RELATIONSHIPS = """
    // Location relationships
    (i:Incubator)-[:LOCATED_IN {since: year}]->(c:City)
    (c:City)-[:IN_STATE]->(s:State)
    
    // Classification relationships
    (i:Incubator)-[:TYPE_OF]->(t:Type)
    (i:Incubator)-[:BACKED_BY {grant_amount: float}]->(b:Backing)
    
    // Program relationships
    (i:Incubator)-[:OFFERS {cohort_size: int}]->(p:Program)
    (i:Incubator)-[:FOCUSES_ON]->(fs:FocusSector)
    
    // Entity relationships
    (i:Incubator)-[:SIMILAR_TO {score: float}]->(i2:Incubator)
    (i:Incubator)-[:PARTNERS_WITH]->(i2:Incubator)
    (i:Incubator)-[:MENTORED_BY]->(person:Person)
    (i:Incubator)-[:ALUMNI {graduated: year}]->(startup:Startup)
    
    // Geographic relationships
    (i:Incubator)-[:NEARBY {distance_km: float}]->(i2:Incubator)
    """


class Neo4jIncubatorQueries:
    """
    Cypher queries for incubator knowledge graph.
    
    These queries demonstrate the power of graph databases
    for incubator discovery and analysis.
    """
    
    # =========================================================================
    # DISCOVERY QUERIES
    # =========================================================================
    
    QUERY_FIND_BY_LOCATION = """
    // Find incubators in a specific city
    MATCH (i:Incubator)-[:LOCATED_IN]->(c:City {name: $city_name})
    RETURN i.name, i.website, i.alumni_count, i.data_completeness
    ORDER BY i.alumni_count DESC
    """
    
    QUERY_FIND_BY_TYPE_AND_STATE = """
    // Find government incubators in Karnataka
    MATCH (i:Incubator)-[:TYPE_OF]->(t:Type {name: "government"})
    MATCH (i)-[:LOCATED_IN]->(c:City)-[:IN_STATE]->(s:State {name: $state})
    RETURN i.name, c.name as city, i.alumni_count
    ORDER BY i.alumni_count DESC
    """
    
    QUERY_FIND_BY_SECTOR = """
    // Find incubators focusing on specific sectors
    MATCH (i:Incubator)-[:FOCUSES_ON]->(fs:FocusSector)
    WHERE fs.name IN $sectors
    RETURN i.name, collect(fs.name) as sectors, i.data_completeness
    ORDER BY i.data_completeness DESC
    """
    
    QUERY_NEARBY_INCUBATORS = """
    // Find incubators near a specific location
    MATCH (i:Incubator)-[:LOCATED_IN]->(c:City)
    MATCH (i2:Incubator)-[:LOCATED_IN]->(c2:City)
    WHERE i.name = $incubator_name AND c.state = c2.state AND i <> i2
    RETURN i2.name, c2.name as city, i2.type
    LIMIT 10
    """
    
    # =========================================================================
    # INSIGHT QUERIES
    # =========================================================================
    
    QUERY_TOP_INCUBATORS_BY_ALUMNI = """
    // Top incubators by alumni success
    MATCH (i:Incubator)
    WHERE i.alumni_count > 0
    RETURN i.name, i.city, i.alumni_count, i.type
    ORDER BY i.alumni_count DESC
    LIMIT 20
    """
    
    QUERY_INCUBATOR_ECOSYSTEM_BY_STATE = """
    // Ecosystem strength by state
    MATCH (i:Incubator)-[:LOCATED_IN]->(c:City)-[:IN_STATE]->(s:State)
    RETURN s.name,
           count(i) as incubator_count,
           sum(i.alumni_count) as total_alumni,
           avg(i.data_completeness) as avg_completeness
    ORDER BY incubator_count DESC
    """
    
    QUERY_PROGRAM_DIVERSITY = """
    // Which incubators offer the most diverse programs?
    MATCH (i:Incubator)-[:OFFERS]->(p:Program)
    WITH i, count(p) as program_count
    RETURN i.name, i.city, program_count
    ORDER BY program_count DESC
    """
    
    QUERY_SECTOR_COVERAGE = """
    // Which sectors have the most incubator coverage?
    MATCH (i:Incubator)-[:FOCUSES_ON]->(fs:FocusSector)
    RETURN fs.name, fs.category, count(i) as incubator_count
    ORDER BY incubator_count DESC
    """
    
    QUERY_FIND_SIMILAR = """
    // Find incubators similar to a given one
    MATCH (i:Incubator {name: $incubator_name})-[:FOCUSES_ON]->(fs:FocusSector)
    MATCH (i2:Incubator)-[:FOCUSES_ON]->(fs)
    WHERE i <> i2
    WITH i2, count(fs) as shared_sectors
    RETURN i2.name, i2.city, shared_sectors
    ORDER BY shared_sectors DESC
    LIMIT 10
    """
    
    # =========================================================================
    # RECOMMENDATION QUERIES
    # =========================================================================
    
    QUERY_RECOMMEND_FOR_STARTUP = """
    // Recommend incubators for a startup based on:
    // 1. Sector match
    // 2. Location preference
    // 3. Program type (virtual/physical)
    MATCH (i:Incubator)-[:FOCUSES_ON]->(fs:FocusSector)
    WHERE fs.name IN $startup_sectors
    MATCH (i)-[:LOCATED_IN]->(c:City)-[:IN_STATE]->(s:State)
    WHERE s.name IN $preferred_states
    OPTIONAL MATCH (i)-[:OFFERS]->(p:Program {virtual: $wants_virtual})
    RETURN i.name, i.city, s.name as state,
           i.alumni_count, i.data_completeness,
           count(p) as virtual_programs
    ORDER BY i.data_completeness DESC, i.alumni_count DESC
    LIMIT 10
    """
    
    QUERY_FIND_PARTNERS = """
    // Find potential partner incubators
    // Same sector + different state = complementary
    MATCH (i:Incubator {name: $incubator_name})-[:FOCUSES_ON]->(fs:FocusSector)
    MATCH (i2:Incubator)-[:FOCUSES_ON]->(fs)
    MATCH (i)-[:LOCATED_IN]->(c:City)-[:IN_STATE]->(s:State)
    MATCH (i2)-[:LOCATED_IN]->(c2:City)-[:IN_STATE]->(s2:State)
    WHERE i <> i2 AND s <> s2
    RETURN i2.name, i2.city, s2.name as state, fs.name as shared_sector
    ORDER BY s2.name
    LIMIT 20
    """


class Neo4jIntegrationWorkflow:
    """
    How Neo4j integrates with the hybrid incubator discovery pipeline.
    
    The workflow:
    
    1. DISCOVERY PHASE (CSV/JSON)
       ↓
    2. NEO4J INGESTION
       - Transform flat data to graph
       - Create nodes and relationships
       - Build knowledge graph
       ↓
    3. GRAPH ENRICHMENT
       - Add similarity relationships
       - Link related entities
       - Calculate graph metrics
       ↓
    4. INSIGHT GENERATION
       - Run Cypher queries
       - Find patterns
       - Generate recommendations
       ↓
    5. RANKING (using graph data)
       - Graph-based scoring
       - Relationship strength
       - Network centrality
    """
    
    async def ingest_to_neo4j(self, entities: List[IncubatorGraphEntity]):
        """
        Ingest incubator entities into Neo4j.
        
        This transforms the flat discovery data into a rich
        knowledge graph with relationships.
        """
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        
        async with driver.session() as session:
            for entity in entities:
                # Create incubator node
                await session.run("""
                    MERGE (i:Incubator {id: $id})
                    SET i.name = $name,
                        i.website = $website,
                        i.email = $email,
                        i.phone = $phone,
                        i.established_year = $established_year,
                        i.alumni_count = $alumni_count,
                        i.team_size = $team_size,
                        i.mentor_count = $mentor_count,
                        i.data_completeness = $data_completeness,
                        i.updated_at = datetime()
                """, {
                    "id": entity.id,
                    "name": entity.name,
                    "website": entity.website,
                    "email": entity.email,
                    "phone": entity.phone,
                    "established_year": entity.established_year,
                    "alumni_count": entity.alumni_count,
                    "team_size": entity.team_size,
                    "mentor_count": entity.mentor_count,
                    "data_completeness": entity.data_completeness,
                })
                
                # Create location nodes and relationships
                if entity.city and entity.state:
                    await session.run("""
                        MATCH (i:Incubator {id: $id})
                        MERGE (c:City {name: $city})
                        MERGE (s:State {name: $state})
                        MERGE (c)-[:IN_STATE]->(s)
                        MERGE (i)-[:LOCATED_IN]->(c)
                    """, {
                        "id": entity.id,
                        "city": entity.city,
                        "state": entity.state
                    })
                
                # Create type relationship
                if entity.type:
                    await session.run("""
                        MATCH (i:Incubator {id: $id})
                        MERGE (t:Type {name: $type})
                        MERGE (i)-[:TYPE_OF]->(t)
                    """, {"id": entity.id, "type": entity.type})
                
                # Create backing relationship
                if entity.backing:
                    await session.run("""
                        MATCH (i:Incubator {id: $id})
                        MERGE (b:Backing {name: $backing})
                        MERGE (i)-[:BACKED_BY]->(b)
                    """, {"id": entity.id, "backing": entity.backing})
                
                # Create sector relationships
                for sector in entity.focus_sectors:
                    await session.run("""
                        MATCH (i:Incubator {id: $id})
                        MERGE (fs:FocusSector {name: $sector})
                        MERGE (i)-[:FOCUSES_ON]->(fs)
                    """, {"id": entity.id, "sector": sector})
    
    async def enrich_graph(self):
        """
        Add computed relationships to the graph.
        
        These are derived from the data:
        - Similar incubators (same sector + location)
        - Network centrality scores
        - Ecosystem clusters
        """
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        
        async with driver.session() as session:
            # Create similarity relationships
            await session.run("""
                MATCH (i1:Incubator)-[:FOCUSES_ON]->(fs:FocusSector)
                MATCH (i2:Incubator)-[:FOCUSES_ON]->(fs)
                WHERE i1 <> i2
                WITH i1, i2, count(fs) as shared_sectors
                WHERE shared_sectors >= 2
                MERGE (i1)-[r:SIMILAR_TO]->(i2)
                SET r.score = shared_sectors
            """)
            
            # Create state-level ecosystem relationships
            await session.run("""
                MATCH (i1:Incubator)-[:LOCATED_IN]->(c:City)-[:IN_STATE]->(s:State)
                MATCH (i2:Incubator)-[:LOCATED_IN]->(c2:City)-[:IN_STATE]->(s)
                WHERE i1 <> i2
                MERGE (i1)-[:SAME_STATE_ECOSYSTEM]->(i2)
            """)
    
    async def generate_insights(self) -> Dict:
        """
        Generate insights from the knowledge graph.
        
        Returns structured insights about the ecosystem.
        """
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        
        insights = {}
        
        async with driver.session() as session:
            # Top incubators by alumni
            result = await session.run("""
                MATCH (i:Incubator)
                WHERE i.alumni_count > 0
                RETURN i.name, i.alumni_count
                ORDER BY i.alumni_count DESC
                LIMIT 10
            """)
            insights["top_by_alumni"] = [record.data() async for record in result]
            
            # State ecosystem strength
            result = await session.run("""
                MATCH (i:Incubator)-[:LOCATED_IN]->(c:City)-[:IN_STATE]->(s:State)
                RETURN s.name,
                       count(i) as incubator_count,
                       sum(i.alumni_count) as total_alumni
                ORDER BY incubator_count DESC
            """)
            insights["ecosystem_by_state"] = [record.data() async for record in result]
            
            # Sector coverage
            result = await session.run("""
                MATCH (fs:FocusSector)
                OPTIONAL MATCH (i:Incubator)-[:FOCUSES_ON]->(fs)
                RETURN fs.name, count(i) as incubator_count
                ORDER BY incubator_count DESC
            """)
            insights["sector_coverage"] = [record.data() async for record in result]
        
        return insights


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_neo4j_workflow():
    """
    Example of complete Neo4j workflow.
    """
    # 1. Get entities from discovery
    from crawler.hybrid_incubator_discovery import HybridIncubatorDiscovery
    
    discovery = HybridIncubatorDiscovery()
    entities = await discovery.run_hybrid_pipeline()
    
    # 2. Convert to graph entities
    graph_entities = [
        IncubatorGraphEntity(
            id=e.id,
            name=e.name,
            website=e.website,
            city=e.city,
            state=e.state,
            type=e.type,
            backing=e.backing,
            focus_sectors=e.focus_sectors,
            alumni_count=e.alumni_count,
            data_completeness=e.data_completeness
        )
        for e in entities
    ]
    
    # 3. Ingest to Neo4j
    workflow = Neo4jIntegrationWorkflow()
    await workflow.ingest_to_neo4j(graph_entities)
    
    # 4. Enrich graph
    await workflow.enrich_graph()
    
    # 5. Generate insights
    insights = await workflow.generate_insights()
    
    print("Neo4j Insights:")
    print(f"Top incubators: {insights['top_by_alumni'][:3]}")
    print(f"Strongest state: {insights['ecosystem_by_state'][0]}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_neo4j_workflow())
