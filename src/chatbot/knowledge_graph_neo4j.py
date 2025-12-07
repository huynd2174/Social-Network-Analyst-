"""
Knowledge Graph Module for K-pop Social Network - Neo4j Version

This module connects to Neo4j database and provides graph traversal,
entity lookup, and relationship queries using Cypher.

YÊU CẦU:
- Neo4j server đang chạy
- Dữ liệu đã được import vào Neo4j (dùng merge_and_import_neo4j.py)
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from neo4j import GraphDatabase
import networkx as nx


class KpopKnowledgeGraphNeo4j:
    """
    Knowledge Graph for K-pop entities using Neo4j.
    
    Supports:
    - Entity types: Group, Artist, Song, Album, Company, Genre, Occupation, Instrument
    - Relationship types: MEMBER_OF, SINGS, RELEASED, MANAGED_BY, SUBUNIT_OF, etc.
    - Multi-hop traversal and reasoning using Cypher queries
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = None,
        database: str = None
    ):
        """
        Initialize knowledge graph from Neo4j.
        
        Args:
            uri: Neo4j URI
            user: Username
            password: Password (required)
            database: Database name (None = default)
        """
        if not password:
            raise ValueError("Neo4j password is required")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.uri = uri
        
        # Cache for frequently accessed data
        self._node_cache: Dict[str, Dict] = {}
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)
        self._relationship_index: Dict[str, List[Tuple]] = defaultdict(list)
        
        # Build indices on initialization
        self._build_indices()
        
        print(f"✅ Connected to Neo4j: {uri}")
        print(f"✅ Loaded {len(self._entity_index)} entity types")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def _build_indices(self):
        """Build entity and relationship indices from Neo4j."""
        def get_all_nodes(tx):
            query = """
            MATCH (n)
            RETURN n.id as id, labels(n) as labels, properties(n) as props
            """
            return [dict(r) for r in tx.run(query)]
        
        def get_all_edges(tx):
            query = """
            MATCH (a)-[r]->(b)
            RETURN a.id as source, b.id as target, type(r) as type, properties(r) as props
            """
            return [dict(r) for r in tx.run(query)]
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            nodes = session.execute_read(get_all_nodes)
            edges = session.execute_read(get_all_edges)
        
        # Build entity index
        for node in nodes:
            node_id = node['id']
            labels = node['labels']
            if labels:
                entity_type = labels[0]  # Primary label
                self._entity_index[entity_type].add(node_id)
                self._node_cache[node_id] = {
                    'label': entity_type,
                    'properties': node.get('props', {})
                }
        
        # Build relationship index
        for edge in edges:
            rel_type = edge['type']
            source = edge['source']
            target = edge['target']
            self._relationship_index[rel_type].append((source, target))
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID from Neo4j."""
        if entity_id in self._node_cache:
            return self._node_cache[entity_id]
        
        def get_node(tx, node_id):
            query = """
            MATCH (n {id: $node_id})
            RETURN n.id as id, labels(n) as labels, properties(n) as props
            LIMIT 1
            """
            result = tx.run(query, node_id=node_id)
            record = result.single()
            if record:
                return {
                    'id': record['id'],
                    'label': record['labels'][0] if record['labels'] else 'Unknown',
                    'properties': dict(record['props'])
                }
            return None
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            node_data = session.execute_read(get_node, entity_id)
            if node_data:
                self._node_cache[entity_id] = node_data
            return node_data
    
    def get_entity_type(self, entity_id: str) -> Optional[str]:
        """Get entity type (label) from Neo4j."""
        entity = self.get_entity(entity_id)
        return entity.get('label') if entity else None
    
    def get_neighbors(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Tuple[str, str, Dict]]:
        """
        Get neighbors of an entity using Cypher.
        
        Args:
            entity_id: Entity ID
            relationship_type: Filter by relationship type (None = all)
            direction: "outgoing", "incoming", or "both"
        
        Returns:
            List of (source, target, relationship_data) tuples
        """
        def get_neighbors_query(tx, node_id, rel_type, dir):
            if dir == "outgoing":
                pattern = "(a {id: $node_id})-[r]->(b)"
            elif dir == "incoming":
                pattern = "(a)<-[r]-(b {id: $node_id})"
            else:  # both
                pattern = "(a {id: $node_id})-[r]-(b)"
            
            rel_filter = f" AND type(r) = '{rel_type}'" if rel_type else ""
            
            query = f"""
            MATCH {pattern}
            WHERE a.id = $node_id {rel_filter}
            RETURN a.id as source, b.id as target, type(r) as type, properties(r) as props
            """
            return [dict(r) for r in tx.run(query, node_id=node_id)]
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            results = session.execute_read(get_neighbors_query, entity_id, relationship_type, direction)
            return [
                (r['source'], r['target'], {'type': r['type'], **r.get('props', {})})
                for r in results
            ]
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Find shortest path between two entities using Cypher.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_hops: Maximum number of hops
            relationship_types: Filter by relationship types
        
        Returns:
            List of (source, target, relationship_type) tuples or None
        """
        def find_path_query(tx, src, tgt, max_h, rel_types):
            rel_filter = ""
            if rel_types:
                rel_list_str = ','.join([f"'{rt}'" for rt in rel_types])
                rel_filter = f" AND type(r) IN [{rel_list_str}]"
            
            query = f"""
            MATCH path = shortestPath((a {{id: $src}})-[*1..{max_h}]-(b {{id: $tgt}}))
            WHERE ALL(r in relationships(path) {rel_filter.replace('AND', '') if rel_filter else 'true'})
            RETURN path
            LIMIT 1
            """
            result = tx.run(query, src=src, tgt=tgt)
            record = result.single()
            if record:
                path = record['path']
                # Extract path as list of tuples
                nodes = [node['id'] for node in path.nodes]
                relationships = [rel.type for rel in path.relationships]
                path_tuples = []
                for i in range(len(nodes) - 1):
                    path_tuples.append((nodes[i], nodes[i+1], relationships[i]))
                return path_tuples
            return None
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            return session.execute_read(find_path_query, source_id, target_id, max_hops, relationship_types)
    
    def get_entities_by_type(self, entity_type: str) -> List[str]:
        """Get all entities of a specific type."""
        return list(self._entity_index.get(entity_type, set()))
    
    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics from Neo4j."""
        def get_stats(tx):
            node_query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            """
            edge_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            """
            
            node_results = [dict(r) for r in tx.run(node_query)]
            edge_results = [dict(r) for r in tx.run(edge_query)]
            
            return {
                'nodes': node_results,
                'edges': edge_results
            }
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            stats = session.execute_read(get_stats)
        
        entity_types = {r['label']: r['count'] for r in stats['nodes']}
        relationship_types = {r['type']: r['count'] for r in stats['edges']}
        
        total_nodes = sum(entity_types.values())
        total_edges = sum(relationship_types.values())
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'entity_types': entity_types,
            'relationship_types': relationship_types
        }
    
    # Compatibility methods (same interface as NetworkX version)
    def get_group_members(self, group_name: str) -> List[str]:
        """Get all members of a K-pop group."""
        members = []
        neighbors = self.get_neighbors(group_name, "MEMBER_OF", "incoming")
        for source, target, data in neighbors:
            if data.get('type') == 'MEMBER_OF':
                source_type = self.get_entity_type(source)
                if source_type == 'Artist':
                    members.append(source)
        return members
    
    def get_group_songs(self, group_name: str) -> List[str]:
        """Get all songs by a group."""
        songs = []
        neighbors = self.get_neighbors(group_name, "SINGS", "outgoing")
        for source, target, data in neighbors:
            if data.get('type') == 'SINGS':
                songs.append(target)
        return songs
    
    def get_company_of_group(self, group_name: str) -> Optional[str]:
        """Get company managing a group."""
        neighbors = self.get_neighbors(group_name, "MANAGED_BY", "outgoing")
        for source, target, data in neighbors:
            if data.get('type') == 'MANAGED_BY':
                return target
        return None
    
    def is_member_of(self, artist_name: str, group_name: str) -> bool:
        """Check if artist is member of group."""
        def check_membership(tx, artist, group):
            query = """
            MATCH (a {id: $artist})-[r:MEMBER_OF]->(g {id: $group})
            RETURN r
            LIMIT 1
            """
            result = tx.run(query, artist=artist, group=group)
            return result.single() is not None
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            return session.execute_read(check_membership, artist_name, group_name)
    
    def get_artist_groups(self, artist_name: str) -> List[str]:
        """Get all groups an artist belongs to."""
        groups = []
        neighbors = self.get_neighbors(artist_name, "MEMBER_OF", "outgoing")
        for source, target, data in neighbors:
            if data.get('type') == 'MEMBER_OF':
                groups.append(target)
        return groups
    
    def get_song_artists(self, song_name: str) -> List[str]:
        """Get all artists who sing a song."""
        artists = []
        neighbors = self.get_neighbors(song_name, "SINGS", "incoming")
        for source, target, data in neighbors:
            if data.get('type') == 'SINGS':
                artists.append(source)
        return artists

