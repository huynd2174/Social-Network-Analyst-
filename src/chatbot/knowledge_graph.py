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
        self.original_to_cleaned: Dict[str, str] = {}  # Mapping from original ID to cleaned ID
        self.cleaned_to_original: Dict[str, str] = {}  # Mapping from cleaned ID to original ID (for reverse lookup if needed)
        
        # Load and build graph
        self._load_data()
        self._build_graph()
        self._build_indices()
    
    def _clean_entity_id(self, entity_id: str) -> str:
        """
        Clean entity ID by removing common prefixes like Genre_, Company_, etc.
        
        Args:
            entity_id: Original entity ID
            
        Returns:
            Cleaned entity ID without prefix
        """
        # List of prefixes to remove
        prefixes = [
            'Genre_',
            'Company_',
            'Album_',
            'Song_',
            'Artist_',
            'Group_',
            'Occupation_',
            'Instrument_'
        ]
        
        cleaned_id = entity_id
        for prefix in prefixes:
            if cleaned_id.startswith(prefix):
                cleaned_id = cleaned_id[len(prefix):]
                break  # Only remove one prefix
        
        return cleaned_id
    
    def _resolve_entity_id(self, entity_id: str) -> Optional[str]:
        """
        Resolve entity ID to cleaned ID used in graph.
        
        Args:
            entity_id: Entity ID (can be original or cleaned)
            
        Returns:
            Cleaned entity ID if exists in graph, None otherwise
        """
        # Try direct lookup first (might be cleaned ID)
        if entity_id in self.graph:
            return entity_id
        
        # Try mapping from original to cleaned
        cleaned_id = self.original_to_cleaned.get(entity_id)
        if cleaned_id and cleaned_id in self.graph:
            return cleaned_id
        
        # Try cleaning the ID
        cleaned_id = self._clean_entity_id(entity_id)
        if cleaned_id in self.graph:
            return cleaned_id
        
        return None
        
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
        # Add nodes with cleaned IDs
        for original_node_id, node_data in self.nodes.items():
            cleaned_node_id = self._clean_entity_id(original_node_id)
            
            # Store mapping
            self.original_to_cleaned[original_node_id] = cleaned_node_id
            # Handle case where multiple original IDs map to same cleaned ID (shouldn't happen but be safe)
            if cleaned_node_id not in self.cleaned_to_original:
                self.cleaned_to_original[cleaned_node_id] = original_node_id
            
            # Clean title if it contains prefix
            title = node_data.get('title', original_node_id)
            cleaned_title = self._clean_entity_id(title) if title == original_node_id else title
            
            self.graph.add_node(
                cleaned_node_id,
                label=node_data.get('label', 'Unknown'),
                title=cleaned_title,
                infobox=node_data.get('infobox', {}),
                url=node_data.get('url', ''),
                depth=node_data.get('depth', 0),
                original_id=original_node_id  # Keep original ID for reference if needed
            )
            
        # Add edges with cleaned IDs
        # Track edges to handle multiple relationship types between same nodes
        edge_relations = {}  # (source, target) -> list of relationship types
        
        for edge in self.edges:
            original_source = edge.get('source')
            original_target = edge.get('target')
            rel_type = edge.get('type', 'RELATED')
            
            if original_source and original_target:
                # Map original IDs to cleaned IDs
                cleaned_source = self.original_to_cleaned.get(original_source, self._clean_entity_id(original_source))
                cleaned_target = self.original_to_cleaned.get(original_target, self._clean_entity_id(original_target))
                
                # Only add edge if both nodes exist in graph
                if cleaned_source in self.graph and cleaned_target in self.graph:
                    edge_key = (cleaned_source, cleaned_target)
                    if edge_key not in edge_relations:
                        edge_relations[edge_key] = []
                    edge_relations[edge_key].append({
                        'type': rel_type,
                        'confidence': edge.get('confidence', 1.0),
                        'method': edge.get('method', 'unknown')
                    })
        
        # Add edges with all relationship types
        for (source, target), relations in edge_relations.items():
            if len(relations) == 1:
                # Single relationship type - keep backward compatible format
                self.graph.add_edge(
                    source,
                    target,
                    type=relations[0]['type'],
                    confidence=relations[0]['confidence'],
                    method=relations[0]['method']
                )
            else:
                # Multiple relationship types - store as list
                self.graph.add_edge(
                    source,
                    target,
                    type=relations[0]['type'],  # Keep first as primary for backward compatibility
                    types=[r['type'] for r in relations],  # Store all types
                    confidence=relations[0]['confidence'],
                    method=relations[0]['method']
                )
                
        print(f"✅ Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges (with cleaned IDs)")
        
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
        """Get entity by ID (handles both original and cleaned IDs)."""
        # Try direct lookup first (might be cleaned ID)
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        
        # Try mapping from original to cleaned
        cleaned_id = self.original_to_cleaned.get(entity_id)
        if cleaned_id and cleaned_id in self.graph:
            return dict(self.graph.nodes[cleaned_id])
        
        # Try cleaning the ID
        cleaned_id = self._clean_entity_id(entity_id)
        if cleaned_id in self.graph:
            return dict(self.graph.nodes[cleaned_id])
        
        return None
        
    def _has_relationship_type(self, edge_data: Dict, rel_type: str) -> bool:
        """
        Check if edge has a specific relationship type.
        Handles both single 'type' and multiple 'types' list.
        
        Args:
            edge_data: Edge data dictionary from graph
            rel_type: Relationship type to check (e.g., 'SINGS', 'WROTE', 'MEMBER_OF')
            
        Returns:
            True if edge has the relationship type, False otherwise
        """
        # Check if 'types' list exists (multiple relationship types)
        if 'types' in edge_data and isinstance(edge_data['types'], list):
            return rel_type in edge_data['types']
        # Fallback to single 'type'
        return edge_data.get('type') == rel_type
    
    def get_entity_type(self, entity_id: str) -> Optional[str]:
        """Get entity type by ID (handles both original and cleaned IDs)."""
        # Try direct lookup first (might be cleaned ID)
        if entity_id in self.graph:
            return self.graph.nodes[entity_id].get('label')
        
        # Try mapping from original to cleaned
        cleaned_id = self.original_to_cleaned.get(entity_id)
        if cleaned_id and cleaned_id in self.graph:
            return self.graph.nodes[cleaned_id].get('label')
        
        # Try cleaning the ID
        cleaned_id = self._clean_entity_id(entity_id)
        if cleaned_id in self.graph:
            return self.graph.nodes[cleaned_id].get('label')
        
        return None
    
    def extract_year_from_infobox(self, entity_id: str, year_type: str = 'activity') -> Optional[str]:
        """
        Extract year information from entity infobox.
        
        Args:
            entity_id: Entity ID
            year_type: Type of year to extract:
                - 'activity': Năm hoạt động (for groups/artists)
                - 'release': Phát hành (for songs/albums)
                - 'founding': Năm thành lập (for companies)
        
        Returns:
            Year string (e.g., "2013–nay", "2021", "2016–2023") or None
        """
        entity = self.get_entity(entity_id)
        if not entity:
            return None
        
        infobox = entity.get('infobox', {})
        if not infobox:
            return None
        
        # Map year_type to infobox keys
        key_mapping = {
            'activity': 'Năm hoạt động',
            'release': 'Phát hành',
            'founding': 'Năm thành lập'
        }
        
        key = key_mapping.get(year_type, 'Năm hoạt động')
        year_str = infobox.get(key)
        
        if not year_str:
            return None
        
        # Extract year from various formats:
        # - "2013–nay" -> "2013–nay"
        # - "10 tháng 9 năm 2021 (2021-09-10)" -> "2021"
        # - "2016–2023" -> "2016–2023"
        import re
        # Try to extract year from date format like "10 tháng 9 năm 2021"
        year_match = re.search(r'(\d{4})', year_str)
        if year_match:
            # If it's a date format, return just the year
            if 'tháng' in year_str or 'năm' in year_str:
                return year_match.group(1)
            # Otherwise return the full range
            return year_str
        
        return year_str
        
    def get_entities_by_type(self, entity_type: str) -> Set[str]:
        """Get all entities of a specific type."""
        return self.entity_index.get(entity_type, set())
        
    def get_neighbors(self, entity_id: str, direction: str = 'both') -> List[Tuple[str, str, str]]:
        """
        Get neighbors of an entity.
        
        Args:
            entity_id: Entity to get neighbors for (handles both original and cleaned IDs)
            direction: 'out', 'in', or 'both'
            
        Returns:
            List of (neighbor_id, relationship_type, direction)
            Note: If edge has multiple relationship types, returns one tuple per type
        """
        # Resolve entity_id to cleaned ID
        cleaned_entity_id = entity_id
        if entity_id not in self.graph:
            cleaned_entity_id = self.original_to_cleaned.get(entity_id, self._clean_entity_id(entity_id))
        
        if cleaned_entity_id not in self.graph:
            return []
        
        neighbors = []
        
        if direction in ['out', 'both']:
            for _, target, data in self.graph.out_edges(cleaned_entity_id, data=True):
                # Get all relationship types (single or multiple)
                rel_types = data.get('types', [data.get('type', 'RELATED')])
                if not isinstance(rel_types, list):
                    rel_types = [rel_types]
                # Return one tuple per relationship type
                for rel_type in rel_types:
                    neighbors.append((target, rel_type, 'out'))
                
        if direction in ['in', 'both']:
            for source, _, data in self.graph.in_edges(cleaned_entity_id, data=True):
                # Get all relationship types (single or multiple)
                rel_types = data.get('types', [data.get('type', 'RELATED')])
                if not isinstance(rel_types, list):
                    rel_types = [rel_types]
                # Return one tuple per relationship type
                for rel_type in rel_types:
                    neighbors.append((source, rel_type, 'in'))
                
        return neighbors
        
    def get_relationships(self, entity_id: str) -> List[Dict]:
        """
        Get all relationships for an entity.
        
        Args:
            entity_id: Entity ID (handles both original and cleaned IDs)
        
        Returns:
            List of relationship dicts with 'source', 'target', 'type', 'types', etc.
            If edge has multiple relationship types, returns one dict per type
        """
        # Resolve entity_id to cleaned ID
        cleaned_entity_id = entity_id
        if entity_id not in self.graph:
            cleaned_entity_id = self.original_to_cleaned.get(entity_id, self._clean_entity_id(entity_id))
        
        if cleaned_entity_id not in self.graph:
            return []
        
        relationships = []
        
        # Outgoing relationships
        for _, target, data in self.graph.out_edges(cleaned_entity_id, data=True):
            # Get all relationship types (single or multiple)
            rel_types = data.get('types', [data.get('type', 'RELATED')])
            if not isinstance(rel_types, list):
                rel_types = [rel_types]
            # Return one relationship dict per type
            for rel_type in rel_types:
                relationships.append({
                    'source': cleaned_entity_id,
                    'target': target,
                    'type': rel_type,
                    'types': data.get('types', [rel_type]),  # Include all types
                    'direction': 'outgoing',
                    'confidence': data.get('confidence', 1.0),
                    'method': data.get('method', 'unknown')
                })
            
        # Incoming relationships
        for source, _, data in self.graph.in_edges(cleaned_entity_id, data=True):
            # Get all relationship types (single or multiple)
            rel_types = data.get('types', [data.get('type', 'RELATED')])
            if not isinstance(rel_types, list):
                rel_types = [rel_types]
            # Return one relationship dict per type
            for rel_type in rel_types:
                relationships.append({
                    'source': source,
                    'target': cleaned_entity_id,
                    'type': rel_type,
                    'types': data.get('types', [rel_type]),  # Include all types
                    'direction': 'incoming',
                    'confidence': data.get('confidence', 1.0),
                    'method': data.get('method', 'unknown')
                })
            
        return relationships
        
    def find_path(self, source: str, target: str, max_hops: int = 5) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        # Clean IDs if they contain prefixes
        cleaned_source = self._clean_entity_id(source)
        cleaned_target = self._clean_entity_id(target)
        # Try mapping from original to cleaned if available
        cleaned_source = self.original_to_cleaned.get(source, cleaned_source)
        cleaned_target = self.original_to_cleaned.get(target, cleaned_target)
        
        try:
            path = nx.shortest_path(self.graph, cleaned_source, cleaned_target)
            if len(path) - 1 <= max_hops:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None
        
    def find_all_paths(self, source: str, target: str, max_hops: int = 3) -> List[List[str]]:
        """Find all simple paths between two entities (up to max_hops)."""
        # Clean IDs if they contain prefixes
        cleaned_source = self._clean_entity_id(source)
        cleaned_target = self._clean_entity_id(target)
        # Try mapping from original to cleaned if available
        cleaned_source = self.original_to_cleaned.get(source, cleaned_source)
        cleaned_target = self.original_to_cleaned.get(target, cleaned_target)
        
        try:
            paths = list(nx.all_simple_paths(self.graph, cleaned_source, cleaned_target, cutoff=max_hops))
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
        # Clean query to remove prefixes for better matching
        cleaned_query = self._clean_entity_id(query)
        query_lower = query.lower()
        cleaned_query_lower = cleaned_query.lower()
        results = []
        
        for node_id, data in self.graph.nodes(data=True):
            # Filter by type if specified
            if entity_type and data.get('label') != entity_type:
                continue
                
            # Check if query matches (try both original query and cleaned query)
            title = data.get('title', node_id).lower()
            node_id_lower = node_id.lower()
            
            # Match with cleaned IDs (nodes in graph now use cleaned IDs)
            if cleaned_query_lower in title or cleaned_query_lower in node_id_lower:
                score = 1.0 if (cleaned_query_lower == title or cleaned_query_lower == node_id_lower) else 0.8
                results.append({
                    'id': node_id,  # Return cleaned ID (as stored in graph)
                    'type': data.get('label'),
                    'title': data.get('title', node_id),
                    'score': score
                })
            # Also try matching with original query (in case user searches with prefix)
            elif query_lower in title or query_lower in node_id_lower:
                score = 0.7  # Lower score for prefix matches
                results.append({
                    'id': node_id,
                    'type': data.get('label'),
                    'title': data.get('title', node_id),
                    'score': score
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
        # Clean group_name if it contains prefix
        cleaned_group_name = self._clean_entity_id(group_name)
        if group_name not in self.graph:
            cleaned_group_name = self.original_to_cleaned.get(group_name, cleaned_group_name)
        group_name_to_use = cleaned_group_name if cleaned_group_name in self.graph else group_name
        
        # Try to get from infobox first (most accurate)
        group_data = self.get_entity(group_name_to_use)
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
        if group_name_to_use in self.graph:
            for source, _, data in self.graph.in_edges(group_name_to_use, data=True):
                if self._has_relationship_type(data, 'MEMBER_OF'):
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
            if self._has_relationship_type(data, 'MEMBER_OF'):
                groups.append(target)
        return groups
        
    def get_group_songs(self, group_name: str) -> List[str]:
        """Get all songs by a group."""
        songs = []
        # Check in_edges: Song → SINGS → Group
        for source, _, data in self.graph.in_edges(group_name, data=True):
            if self._has_relationship_type(data, 'SINGS'):
                songs.append(source)
        # Also check out_edges: Group → SINGS → Song (if direction is reversed)
        for _, target, data in self.graph.out_edges(group_name, data=True):
            if self._has_relationship_type(data, 'SINGS'):
                songs.append(target)
        return list(set(songs))  # Remove duplicates
    
    def get_song_groups(self, song_name: str) -> List[str]:
        """Get all groups that performed a song."""
        groups = []
        # Check out_edges: Song → SINGS → Group (if Song is source)
        for _, target, data in self.graph.out_edges(song_name, data=True):
            if self._has_relationship_type(data, 'SINGS'):
                target_type = self.get_entity_type(target)
                if target_type == 'Group':
                    groups.append(target)
        # Check in_edges: Group → SINGS → Song (if Group is source)
        for source, _, data in self.graph.in_edges(song_name, data=True):
            if self._has_relationship_type(data, 'SINGS'):
                source_type = self.get_entity_type(source)
                if source_type == 'Group':
                    groups.append(source)
        return list(set(groups))  # Remove duplicates
    
    def get_song_artists(self, song_name: str) -> List[str]:
        """Get all artists that performed a song."""
        artists = []
        # Check out_edges: Song → SINGS → Artist (if Song is source)
        for _, target, data in self.graph.out_edges(song_name, data=True):
            if self._has_relationship_type(data, 'SINGS'):
                target_type = self.get_entity_type(target)
                if target_type == 'Artist':
                    artists.append(target)
        # Check in_edges: Artist → SINGS → Song (if Artist is source)
        for source, _, data in self.graph.in_edges(song_name, data=True):
            if self._has_relationship_type(data, 'SINGS'):
                source_type = self.get_entity_type(source)
                if source_type == 'Artist':
                    artists.append(source)
        return list(set(artists))  # Remove duplicates
    
    def get_album_groups(self, album_name: str) -> List[str]:
        """Get all groups that released an album."""
        groups = []
        # Check in_edges: Group → RELEASED → Album (if Group is source)
        for source, _, data in self.graph.in_edges(album_name, data=True):
            if self._has_relationship_type(data, 'RELEASED'):
                source_type = self.get_entity_type(source)
                if source_type == 'Group':
                    groups.append(source)
        # Check out_edges: Album → RELEASED → Group (if Album is source - less common)
        for _, target, data in self.graph.out_edges(album_name, data=True):
            if self._has_relationship_type(data, 'RELEASED'):
                target_type = self.get_entity_type(target)
                if target_type == 'Group':
                    groups.append(target)
        return list(set(groups))  # Remove duplicates
    
    def get_album_artists(self, album_name: str) -> List[str]:
        """Get all artists that released an album."""
        artists = []
        # Check in_edges: Artist → RELEASED → Album (if Artist is source)
        for source, _, data in self.graph.in_edges(album_name, data=True):
            if self._has_relationship_type(data, 'RELEASED'):
                source_type = self.get_entity_type(source)
                if source_type == 'Artist':
                    artists.append(source)
        # Check out_edges: Album → RELEASED → Artist (if Album is source - less common)
        for _, target, data in self.graph.out_edges(album_name, data=True):
            if self._has_relationship_type(data, 'RELEASED'):
                target_type = self.get_entity_type(target)
                if target_type == 'Artist':
                    artists.append(target)
        return list(set(artists))  # Remove duplicates
        
    def get_group_company(self, group_name: str) -> Optional[str]:
        """Get the company managing a group (returns first one for backward compatibility)."""
        companies = self.get_group_companies(group_name)
        return companies[0] if companies else None
    
    def get_group_companies(self, group_name: str) -> List[str]:
        """Get ALL companies managing a group (a group can have multiple companies)."""
        companies = []
        for _, target, data in self.graph.out_edges(group_name, data=True):
            if self._has_relationship_type(data, 'MANAGED_BY'):
                companies.append(target)
        return companies
    
    def get_artist_companies(self, artist_name: str) -> List[str]:
        """Get ALL companies managing an artist directly (Artist → Company)."""
        companies = []
        for _, target, data in self.graph.out_edges(artist_name, data=True):
            if self._has_relationship_type(data, 'MANAGED_BY'):
                # Check if target is a Company
                if self.get_entity_type(target) == 'Company':
                    companies.append(target)
        return companies
        
    def get_company_groups(self, company_name: str) -> List[str]:
        """Get all groups under a company."""
        groups = []
        for source, _, data in self.graph.in_edges(company_name, data=True):
            if self._has_relationship_type(data, 'MANAGED_BY'):
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

