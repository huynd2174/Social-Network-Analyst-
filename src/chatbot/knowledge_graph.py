"""
Knowledge Graph Module for K-pop Social Network

This module builds and manages the knowledge graph from the merged K-pop data.
It provides graph traversal, entity lookup, and relationship queries.
"""

import json
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import os


class KpopKnowledgeGraph:
    """
    Knowledge Graph for K-pop entities.
    
    Supports:
    - Entity types: Group, Artist, Song, Album, Company, Genre, Occupation, Instrument
    - Relationship types: MEMBER_OF, SINGS, RELEASED, MANAGED_BY, SUBUNIT_OF, etc.
    - Multi-hop traversal and reasoning
    """
    
    def __init__(self, data_path: str = "data/korean_artists_graph_bfs.json"):
        """Initialize knowledge graph from merged data."""
        self.data_path = data_path
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entities
        self.relationship_index: Dict[str, List[Tuple]] = defaultdict(list)  # type -> (src, tgt)
        
        # Load and build graph
        self._load_data()
        self._build_graph()
        self._build_indices()
        
    def _load_data(self):
        """Load merged K-pop data from JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.metadata = data.get('metadata', {})
        self.nodes = data.get('nodes', {})
        self.edges = data.get('edges', [])
        
        print(f"✅ Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
        
    def _build_graph(self):
        """Build NetworkX graph from nodes and edges."""
        # Add nodes
        for node_id, node_data in self.nodes.items():
            self.graph.add_node(
                node_id,
                label=node_data.get('label', 'Unknown'),
                title=node_data.get('title', node_id),
                infobox=node_data.get('infobox', {}),
                url=node_data.get('url', ''),
                depth=node_data.get('depth', 0)
            )
            
        # Add edges
        for edge in self.edges:
            source = edge.get('source')
            target = edge.get('target')
            rel_type = edge.get('type', 'RELATED')
            
            if source and target and source in self.graph and target in self.graph:
                self.graph.add_edge(
                    source, 
                    target,
                    type=rel_type,
                    confidence=edge.get('confidence', 1.0),
                    method=edge.get('method', 'unknown')
                )
                
        print(f"✅ Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def _build_indices(self):
        """Build lookup indices for fast retrieval."""
        # Entity type index
        for node_id, data in self.graph.nodes(data=True):
            label = data.get('label', 'Unknown')
            self.entity_index[label].add(node_id)
            
        # Relationship type index
        for src, tgt, data in self.graph.edges(data=True):
            rel_type = data.get('type', 'RELATED')
            self.relationship_index[rel_type].append((src, tgt))
            
        print(f"✅ Built indices for {len(self.entity_index)} entity types and {len(self.relationship_index)} relationship types")
        
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID."""
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None
        
    def get_entity_type(self, entity_id: str) -> Optional[str]:
        """Get entity type by ID."""
        if entity_id in self.graph:
            return self.graph.nodes[entity_id].get('label')
        return None
        
    def get_entities_by_type(self, entity_type: str) -> Set[str]:
        """Get all entities of a specific type."""
        return self.entity_index.get(entity_type, set())
        
    def get_neighbors(self, entity_id: str, direction: str = 'both') -> List[Tuple[str, str, str]]:
        """
        Get neighbors of an entity.
        
        Args:
            entity_id: Entity to get neighbors for
            direction: 'out', 'in', or 'both'
            
        Returns:
            List of (neighbor_id, relationship_type, direction)
        """
        neighbors = []
        
        if direction in ['out', 'both']:
            for _, target, data in self.graph.out_edges(entity_id, data=True):
                neighbors.append((target, data.get('type', 'RELATED'), 'out'))
                
        if direction in ['in', 'both']:
            for source, _, data in self.graph.in_edges(entity_id, data=True):
                neighbors.append((source, data.get('type', 'RELATED'), 'in'))
                
        return neighbors
        
    def get_relationships(self, entity_id: str) -> List[Dict]:
        """Get all relationships for an entity."""
        relationships = []
        
        # Outgoing relationships
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            relationships.append({
                'source': entity_id,
                'target': target,
                'type': data.get('type', 'RELATED'),
                'direction': 'outgoing'
            })
            
        # Incoming relationships
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            relationships.append({
                'source': source,
                'target': entity_id,
                'type': data.get('type', 'RELATED'),
                'direction': 'incoming'
            })
            
        return relationships
        
    def find_path(self, source: str, target: str, max_hops: int = 5) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        try:
            path = nx.shortest_path(self.graph, source, target)
            if len(path) - 1 <= max_hops:
                return path
        except nx.NetworkXNoPath:
            pass
        return None
        
    def find_all_paths(self, source: str, target: str, max_hops: int = 3) -> List[List[str]]:
        """Find all simple paths between two entities (up to max_hops)."""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_hops))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
            
    def get_path_details(self, path: List[str]) -> List[Dict]:
        """Get detailed information about a path."""
        details = []
        for i, node in enumerate(path):
            node_data = self.get_entity(node)
            step = {
                'hop': i,
                'entity': node,
                'type': node_data.get('label') if node_data else 'Unknown'
            }
            
            if i < len(path) - 1:
                # Get edge to next node
                edge_data = self.graph.get_edge_data(node, path[i + 1])
                if edge_data:
                    step['relationship_to_next'] = edge_data.get('type', 'RELATED')
                    
            details.append(step)
        return details
        
    def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Search entities by name/title."""
        query_lower = query.lower()
        results = []
        
        for node_id, data in self.graph.nodes(data=True):
            # Filter by type if specified
            if entity_type and data.get('label') != entity_type:
                continue
                
            # Check if query matches
            title = data.get('title', node_id).lower()
            if query_lower in title or query_lower in node_id.lower():
                results.append({
                    'id': node_id,
                    'type': data.get('label'),
                    'title': data.get('title', node_id),
                    'score': 1.0 if query_lower == title else 0.8
                })
                
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
        
    def get_subgraph(self, entity_ids: List[str], include_neighbors: bool = True) -> nx.DiGraph:
        """Extract a subgraph containing specified entities."""
        nodes_to_include = set(entity_ids)
        
        if include_neighbors:
            for entity_id in entity_ids:
                for neighbor, _, _ in self.get_neighbors(entity_id):
                    nodes_to_include.add(neighbor)
                    
        return self.graph.subgraph(nodes_to_include).copy()
        
    def get_group_members(self, group_name: str) -> List[str]:
        """
        Get all members of a K-pop group.
        
        Strategy:
        1. First try to get from infobox (most accurate)
        2. Fallback to MEMBER_OF edges (filtered by type and confidence)
        """
        # Try to get from infobox first (most accurate)
        group_data = self.get_entity(group_name)
        if group_data and group_data.get('infobox'):
            infobox = group_data['infobox']
            members_str = infobox.get('Thành viên', '')
            if members_str and members_str.strip():
                # Parse members from infobox (format: "Jin, Suga, J-Hope, ...")
                members_list = [m.strip() for m in members_str.split(',') if m.strip()]
                if members_list:
                    # Try to match with actual entities in graph
                    matched_members = []
                    for member_name in members_list:
                        # Try exact match first
                        if member_name in self.graph:
                            if self.get_entity_type(member_name) == 'Artist':
                                matched_members.append(member_name)
                        else:
                            # Try fuzzy match (remove suffixes like "(ca sĩ)", "(rapper)")
                            base_name = member_name.split('(')[0].strip()
                            for node_id in self.graph.nodes():
                                if base_name.lower() in node_id.lower() or node_id.lower() in base_name.lower():
                                    if self.get_entity_type(node_id) == 'Artist':
                                        matched_members.append(node_id)
                                        break
                    if matched_members:
                        return matched_members
        
        # Fallback: Get from MEMBER_OF edges (with strict filtering)
        members = []
        for source, _, data in self.graph.in_edges(group_name, data=True):
            if data.get('type') == 'MEMBER_OF':
                # Only include if source is actually an Artist
                source_type = self.get_entity_type(source)
                if source_type == 'Artist':
                    # Check confidence if available
                    confidence = data.get('confidence', 1.0)
                    if confidence >= 0.7:  # Stricter threshold
                        # Exclude obvious non-members (check if name looks like a member name)
                        # Members usually don't have suffixes like "(Album)", "(Song)", etc.
                        if '(' not in source or any(kw in source.lower() for kw in ['rapper', 'ca sĩ', 'singer']):
                            members.append(source)
        return members
        
    def get_artist_groups(self, artist_name: str) -> List[str]:
        """Get all groups an artist belongs to."""
        groups = []
        for _, target, data in self.graph.out_edges(artist_name, data=True):
            if data.get('type') == 'MEMBER_OF':
                groups.append(target)
        return groups
        
    def get_group_songs(self, group_name: str) -> List[str]:
        """Get all songs by a group."""
        songs = []
        # Check in_edges: Song → SINGS → Group
        for source, _, data in self.graph.in_edges(group_name, data=True):
            if data.get('type') == 'SINGS':
                songs.append(source)
        # Also check out_edges: Group → SINGS → Song (if direction is reversed)
        for _, target, data in self.graph.out_edges(group_name, data=True):
            if data.get('type') == 'SINGS':
                songs.append(target)
        return list(set(songs))  # Remove duplicates
    
    def get_song_groups(self, song_name: str) -> List[str]:
        """Get all groups that performed a song."""
        groups = []
        # Check out_edges: Song → SINGS → Group (if Song is source)
        for _, target, data in self.graph.out_edges(song_name, data=True):
            if data.get('type') == 'SINGS':
                target_type = self.get_entity_type(target)
                if target_type == 'Group':
                    groups.append(target)
        # Check in_edges: Group → SINGS → Song (if Group is source)
        for source, _, data in self.graph.in_edges(song_name, data=True):
            if data.get('type') == 'SINGS':
                source_type = self.get_entity_type(source)
                if source_type == 'Group':
                    groups.append(source)
        return list(set(groups))  # Remove duplicates
    
    def get_song_artists(self, song_name: str) -> List[str]:
        """Get all artists that performed a song."""
        artists = []
        # Check out_edges: Song → SINGS → Artist (if Song is source)
        for _, target, data in self.graph.out_edges(song_name, data=True):
            if data.get('type') == 'SINGS':
                target_type = self.get_entity_type(target)
                if target_type == 'Artist':
                    artists.append(target)
        # Check in_edges: Artist → SINGS → Song (if Artist is source)
        for source, _, data in self.graph.in_edges(song_name, data=True):
            if data.get('type') == 'SINGS':
                source_type = self.get_entity_type(source)
                if source_type == 'Artist':
                    artists.append(source)
        return list(set(artists))  # Remove duplicates
        
    def get_group_company(self, group_name: str) -> Optional[str]:
        """Get the company managing a group (returns first one for backward compatibility)."""
        companies = self.get_group_companies(group_name)
        return companies[0] if companies else None
    
    def get_group_companies(self, group_name: str) -> List[str]:
        """Get ALL companies managing a group (a group can have multiple companies)."""
        companies = []
        for _, target, data in self.graph.out_edges(group_name, data=True):
            if data.get('type') == 'MANAGED_BY':
                companies.append(target)
        return companies
    
    def get_artist_companies(self, artist_name: str) -> List[str]:
        """Get ALL companies managing an artist directly (Artist → Company)."""
        companies = []
        for _, target, data in self.graph.out_edges(artist_name, data=True):
            if data.get('type') == 'MANAGED_BY':
                # Check if target is a Company
                if self.get_entity_type(target) == 'Company':
                    companies.append(target)
        return companies
        
    def get_company_groups(self, company_name: str) -> List[str]:
        """Get all groups under a company."""
        groups = []
        for source, _, data in self.graph.in_edges(company_name, data=True):
            if data.get('type') == 'MANAGED_BY':
                if self.get_entity_type(source) == 'Group':
                    groups.append(source)
        return groups
        
    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_types': {k: len(v) for k, v in self.entity_index.items()},
            'relationship_types': {k: len(v) for k, v in self.relationship_index.items()},
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'density': nx.density(self.graph)
        }
        
    def export_triples(self) -> List[Tuple[str, str, str]]:
        """Export graph as RDF-like triples (subject, predicate, object)."""
        triples = []
        for src, tgt, data in self.graph.edges(data=True):
            triples.append((src, data.get('type', 'RELATED'), tgt))
        return triples
        
    def get_entity_context(self, entity_id: str, max_depth: int = 2) -> Dict:
        """
        Get comprehensive context for an entity (for RAG).
        
        Returns entity info, relationships, and connected entities.
        """
        if entity_id not in self.graph:
            return {}
            
        entity_data = self.get_entity(entity_id)
        relationships = self.get_relationships(entity_id)
        
        # Get 2-hop neighborhood
        connected_entities = {}
        visited = {entity_id}
        current_level = [entity_id]
        
        for depth in range(max_depth):
            next_level = []
            for node in current_level:
                for neighbor, rel_type, direction in self.get_neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        neighbor_data = self.get_entity(neighbor)
                        connected_entities[neighbor] = {
                            'type': neighbor_data.get('label'),
                            'depth': depth + 1,
                            'relationship': rel_type
                        }
            current_level = next_level
            
        return {
            'entity': entity_data,
            'relationships': relationships,
            'connected_entities': connected_entities
        }
        

def main():
    """Test the knowledge graph."""
    kg = KpopKnowledgeGraph()
    
    print("\n📊 Graph Statistics:")
    stats = kg.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("\n🔍 Search for 'BTS':")
    results = kg.search_entities("BTS", entity_type="Group")
    for r in results:
        print(f"  {r['id']} ({r['type']})")
        
    print("\n👥 BTS Members:")
    members = kg.get_group_members("BTS")
    for m in members[:5]:
        print(f"  - {m}")
        
    print("\n🏢 BTS Company:")
    company = kg.get_group_company("BTS")
    print(f"  {company}")
    

if __name__ == "__main__":
    main()

