"""
Multi-hop Reasoning Engine for K-pop Knowledge Graph

This module implements multi-hop reasoning over the knowledge graph,
enabling the chatbot to answer complex questions that require traversing
multiple relationships.

Multi-hop reasoning patterns:
- 1-hop: Direct relationship (e.g., "Who are BTS members?")
- 2-hop: One intermediate entity (e.g., "What company manages BTS members?")
- 3-hop: Two intermediate entities (e.g., "What songs did artists from HYBE companies sing?")

Reasoning strategies:
- Chain reasoning: A → B → C
- Aggregation: A → {B1, B2, ...} → aggregate
- Comparison: A vs B based on common properties
- Set operations: intersection, union, difference
"""

import json
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from .knowledge_graph import KpopKnowledgeGraph


class ReasoningType(Enum):
    """Types of multi-hop reasoning."""
    CHAIN = "chain"              # A → B → C
    AGGREGATION = "aggregation"  # A → {B1, B2, ...} → count/list
    COMPARISON = "comparison"    # Compare A and B
    INTERSECTION = "intersection"  # Common entities
    UNION = "union"              # All related entities
    FILTER = "filter"            # Filter by condition


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    hop_number: int
    operation: str
    source_entities: List[str]
    relationship: str
    target_entities: List[str]
    explanation: str
    

@dataclass
class ReasoningResult:
    """Result of multi-hop reasoning."""
    query: str
    reasoning_type: ReasoningType
    steps: List[ReasoningStep]
    answer_entities: List[str]
    answer_text: str
    confidence: float
    explanation: str


class MultiHopReasoner:
    """
    Multi-hop reasoning engine for the K-pop knowledge graph.
    
    Supports complex queries requiring graph traversal across
    multiple relationships and entity types.
    """
    
    def __init__(self, knowledge_graph: Optional[KpopKnowledgeGraph] = None):
        """Initialize with knowledge graph."""
        self.kg = knowledge_graph or KpopKnowledgeGraph()
        
        # Query templates for different reasoning patterns
        self._init_query_templates()
        
    def _init_query_templates(self):
        """Initialize query pattern templates."""
        self.query_templates = {
            # 1-hop patterns
            'group_members': {
                'pattern': r'(thành viên|members?)\s+(của|of)\s+(.+)',
                'hops': 1,
                'chain': ['Group', 'MEMBER_OF', 'Artist'],
                'direction': 'incoming'
            },
            'artist_group': {
                'pattern': r'(.+)\s+(thuộc|là thành viên|belongs to)\s+(.+)',
                'hops': 1,
                'chain': ['Artist', 'MEMBER_OF', 'Group'],
                'direction': 'outgoing'
            },
            'group_company': {
                'pattern': r'(công ty|company)\s+(quản lý|của|manages?)\s+(.+)',
                'hops': 1,
                'chain': ['Group', 'MANAGED_BY', 'Company'],
                'direction': 'outgoing'
            },
            
            # 2-hop patterns
            'company_artists': {
                'pattern': r'(nghệ sĩ|artists?)\s+(của|thuộc)\s+(công ty|company)\s+(.+)',
                'hops': 2,
                'chain': ['Company', 'MANAGED_BY', 'Group', 'MEMBER_OF', 'Artist'],
                'direction': 'incoming'
            },
            'artist_company': {
                'pattern': r'(.+)\s+(thuộc công ty|under company)',
                'hops': 2,
                'chain': ['Artist', 'MEMBER_OF', 'Group', 'MANAGED_BY', 'Company'],
                'direction': 'outgoing'
            },
            'same_company': {
                'pattern': r'(.+)\s+(và|and)\s+(.+)\s+(cùng công ty|same company)',
                'hops': 2,
                'chain': ['Group', 'MANAGED_BY', 'Company'],
                'type': 'comparison'
            },
            
            # 3-hop patterns
            'company_songs': {
                'pattern': r'(bài hát|songs?)\s+(của|by)\s+(công ty|company)\s+(.+)',
                'hops': 3,
                'chain': ['Company', 'MANAGED_BY', 'Group', 'SINGS', 'Song'],
                'direction': 'incoming'
            },
        }
        
    def reason(
        self,
        query: str,
        start_entities: List[str],
        max_hops: int = 3
    ) -> ReasoningResult:
        """
        Perform multi-hop reasoning for a query.
        
        Args:
            query: Natural language query
            start_entities: Starting entities for reasoning
            max_hops: Maximum reasoning hops
            
        Returns:
            ReasoningResult with answer and explanation
        """
        query_lower = query.lower()
        
        # ============================================
        # SPECIALIZED QUERY HANDLERS - ưu tiên cao nhất
        # ============================================
        
        # 1a. Câu hỏi Yes/No về membership: "Jungkook có phải thành viên BTS không?" hoặc "BTS có thành viên V không?"
        if any(kw in query_lower for kw in ['có phải', 'phải', 'là thành viên', 'is a member', 'belongs to', 'có thành viên']):
            # Pattern 1: "X có thành viên Y không?" (group has member)
            if 'có thành viên' in query_lower or 'has member' in query_lower:
                # Find group entity
                group_entity = None
                for entity in start_entities:
                    if self.kg.get_entity_type(entity) == 'Group':
                        group_entity = entity
                        break
                
                if group_entity:
                    # Get all members of the group
                    members = self.kg.get_group_members(group_entity)
                    
                    # Try to find artist name in query
                    # Extract potential artist names from query
                    query_words = query_lower.split()
                    potential_artist = None
                    
                    # Check if any entity is an Artist
                    for entity in start_entities:
                        if self.kg.get_entity_type(entity) == 'Artist':
                            potential_artist = entity
                            break
                    
                    # If found artist entity, check membership
                    if potential_artist:
                        if potential_artist in members:
                            return ReasoningResult(
                                query=query,
                                reasoning_type=ReasoningType.CHAIN,
                                steps=[ReasoningStep(
                                    hop_number=1,
                                    operation='check_membership',
                                    source_entities=[potential_artist],
                                    relationship='MEMBER_OF',
                                    target_entities=[group_entity],
                                    explanation=f"Kiểm tra {potential_artist} có phải thành viên {group_entity}"
                                )],
                                answer_entities=[group_entity],
                                answer_text=f"Có, {potential_artist} là thành viên của {group_entity}",
                                confidence=1.0,
                                explanation=f"1-hop: {potential_artist} → MEMBER_OF → {group_entity}"
                            )
                        else:
                            return ReasoningResult(
                                query=query,
                                reasoning_type=ReasoningType.CHAIN,
                                steps=[ReasoningStep(
                                    hop_number=1,
                                    operation='check_membership',
                                    source_entities=[potential_artist],
                                    relationship='MEMBER_OF',
                                    target_entities=[],
                                    explanation=f"Kiểm tra {potential_artist} có phải thành viên {group_entity}"
                                )],
                                answer_entities=[],
                                answer_text=f"Không, {potential_artist} không phải là thành viên của {group_entity}",
                                confidence=1.0,
                                explanation=f"1-hop: {potential_artist} không có quan hệ MEMBER_OF với {group_entity}"
                            )
                    
                    # Try fuzzy match with member names
                    for member in members:
                        # Check if any word from query matches member name
                        member_words = member.lower().split()
                        for word in query_words:
                            if len(word) > 2 and word in member_words:
                                return ReasoningResult(
                                    query=query,
                                    reasoning_type=ReasoningType.CHAIN,
                                    steps=[ReasoningStep(
                                        hop_number=1,
                                        operation='check_membership',
                                        source_entities=[member],
                                        relationship='MEMBER_OF',
                                        target_entities=[group_entity],
                                        explanation=f"Kiểm tra {member} có phải thành viên {group_entity}"
                                    )],
                                    answer_entities=[group_entity],
                                    answer_text=f"Có, {member} là thành viên của {group_entity}",
                                    confidence=1.0,
                                    explanation=f"1-hop: {member} → MEMBER_OF → {group_entity}"
                                )
            
            # Pattern 2: "X có phải thành viên của Y không?" (artist is member of group)
            # Extract artist and group names from query
            artist_entity = None
            group_entity = None
            
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Artist':
                    artist_entity = entity
                elif entity_type == 'Group':
                    group_entity = entity
            
            if artist_entity and group_entity:
                groups = self.kg.get_artist_groups(artist_entity)
                if group_entity in groups:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[ReasoningStep(
                            hop_number=1,
                            operation='check_membership',
                            source_entities=[artist_entity],
                            relationship='MEMBER_OF',
                            target_entities=[group_entity],
                            explanation=f"Kiểm tra {artist_entity} có phải thành viên {group_entity}"
                        )],
                        answer_entities=[group_entity],
                        answer_text=f"Có, {artist_entity} là thành viên của {group_entity}",
                        confidence=1.0,
                        explanation=f"1-hop: {artist_entity} → MEMBER_OF → {group_entity}"
                    )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[ReasoningStep(
                            hop_number=1,
                            operation='check_membership',
                            source_entities=[artist_entity],
                            relationship='MEMBER_OF',
                            target_entities=[],
                            explanation=f"Kiểm tra {artist_entity} có phải thành viên {group_entity}"
                        )],
                        answer_entities=[],
                        answer_text=f"Không, {artist_entity} không phải thành viên của {group_entity}",
                        confidence=1.0,
                        explanation=f"1-hop: {artist_entity} không có quan hệ MEMBER_OF với {group_entity}"
                    )
        
        # 1b. Câu hỏi về thành viên: "BTS có bao nhiêu thành viên", "Thành viên của BTS", "Ai là thành viên"
        # Pattern: "Ai là thành viên", "Who are members", "thành viên của X", "members of X"
        is_list_members_question = any(kw in query_lower for kw in [
            'ai là thành viên', 'who are', 'thành viên của', 'members of', 
            'thành viên nhóm', 'thành viên ban nhạc', 'có những thành viên'
        ]) and 'có phải' not in query_lower and 'không' not in query_lower
        
        if is_list_members_question:
            # Tìm group entity từ start_entities hoặc extract từ query
            group_entity = None
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    group_entity = entity
                    break
            
            # Nếu không tìm được từ start_entities, extract từ query
            if not group_entity:
                extracted = self._extract_entities_from_query(query)
                for e in extracted:
                    if self.kg.get_entity_type(e) == 'Group':
                        group_entity = e
                        break
            
            if group_entity:
                return self.get_group_members(group_entity)
        
        # Fallback: Câu hỏi về thành viên (không phải Yes/No)
        if any(kw in query_lower for kw in ['thành viên', 'members', 'member']) and 'có phải' not in query_lower and 'không' not in query_lower:
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_group_members(entity)
        
        # 1c. Câu hỏi về nhóm nhạc của artist: "Lisa thuộc nhóm nhạc nào", "Nhóm nào có Lisa"
        if any(kw in query_lower for kw in ['thuộc nhóm', 'thuộc nhóm nhạc', 'nhóm nào', 'nhóm nhạc nào', 'belongs to group', 'group of']):
            # Tìm artist entity từ start_entities
            artist_entity = None
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Artist':
                    artist_entity = entity
                    break
            
            # Nếu không tìm được từ start_entities, extract từ query
            if not artist_entity:
                extracted = self._extract_entities_from_query(query)
                for e in extracted:
                    if self.kg.get_entity_type(e) == 'Artist':
                        artist_entity = e
                        break
            
            if artist_entity:
                return self.get_artist_groups(artist_entity)
                    
        # 2. Câu hỏi về công ty: "Công ty nào quản lý BTS", "BLACKPINK thuộc công ty nào"
        if any(kw in query_lower for kw in ['công ty', 'company', 'label', 'hãng', 'quản lý']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_company_of_group(entity)
                elif entity_type == 'Artist':
                    return self.get_artist_company(entity)
                    
        # 3. Câu hỏi so sánh công ty: "BTS và SEVENTEEN có cùng công ty không"
        if any(kw in query_lower for kw in ['cùng công ty', 'same company', 'cùng hãng', 'cùng label', 'cùng hãng đĩa']):
            # Nếu không có đủ 2 entities, tự extract từ query (case-insensitive)
            if len(start_entities) < 2:
                # Tự extract entities từ query
                extracted = self._extract_entities_from_query(query)
                if len(extracted) >= 2:
                    return self.check_same_company(extracted[0], extracted[1])
                elif len(extracted) == 1 and len(start_entities) == 1:
                    # Có 1 từ query, 1 từ start_entities
                    return self.check_same_company(start_entities[0], extracted[0])
            elif len(start_entities) >= 2:
                return self.check_same_company(start_entities[0], start_entities[1])
        
        # 3.5. Câu hỏi so sánh nhóm nhạc: "Lisa và Jennie có cùng nhóm nhạc không"
        if any(kw in query_lower for kw in [
            'cùng nhóm', 'cùng nhóm nhạc', 'cùng một nhóm', 'cùng một nhóm nhạc',
            'same group', 'cùng ban nhạc', 'chung nhóm', 'chung nhóm nhạc'
        ]):
            # ✅ Ưu tiên dùng start_entities nếu đã có đủ 2
            if len(start_entities) >= 2:
                return self.check_same_group(start_entities[0], start_entities[1])
            # Nếu không có đủ 2 entities, tự extract từ query
            elif len(start_entities) < 2:
                # Tự extract entities từ query (case-insensitive)
                extracted = self._extract_entities_from_query(query)
                if len(extracted) >= 2:
                    return self.check_same_group(extracted[0], extracted[1])
                elif len(extracted) == 1 and len(start_entities) == 1:
                    # Có 1 từ query, 1 từ start_entities
                    return self.check_same_group(start_entities[0], extracted[0])
                elif len(extracted) == 1:
                    # Chỉ có 1 entity → không đủ để so sánh
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.COMPARISON,
                        steps=[],
                        answer_entities=[],
                        answer_text="Không tìm đủ thông tin để so sánh. Vui lòng cung cấp tên đầy đủ của cả hai nghệ sĩ.",
                        confidence=0.0,
                        explanation="Cần ít nhất 2 entities để so sánh cùng nhóm"
                    )
                
        # 4. Câu hỏi về nhóm cùng công ty: "Các nhóm cùng công ty với BTS"
        if any(kw in query_lower for kw in ['nhóm cùng công ty', 'groups same company', 'labelmates', 'cùng công ty với']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_labelmates(entity)
        
        # 5. Câu hỏi về hợp tác/collaboration: "Nhóm nhạc đã hợp tác với BTS"
        if any(kw in query_lower for kw in ['hợp tác', 'collaborat', 'partnership', 'đã làm việc với', 'đã cộng tác']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_collaborating_groups(entity)
        
        # 6. Câu hỏi về bài hát: "BTS hát bài nào?", "Bài hát của BTS"
        if any(kw in query_lower for kw in ['bài hát', 'song', 'track', 'ca khúc', 'hát bài']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_group_songs(entity)
                elif entity_type == 'Artist':
                    return self.get_artist_songs(entity)
        
        # 7. Câu hỏi về album: "Album của BTS", "BTS phát hành album nào"
        if any(kw in query_lower for kw in ['album', 'đĩa nhạc', 'ep', 'lp']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_group_albums(entity)
                elif entity_type == 'Artist':
                    return self.get_artist_albums(entity)
        
        # 8. Câu hỏi về thể loại/genre: "Thể loại nhạc của BTS"
        if any(kw in query_lower for kw in ['thể loại', 'genre', 'dòng nhạc']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type in ['Group', 'Artist', 'Song']:
                    return self.get_entity_genres(entity)
        
        # 9. Câu hỏi về nhạc cụ: "Jungkook chơi nhạc cụ gì"
        if any(kw in query_lower for kw in ['nhạc cụ', 'instrument', 'chơi']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Artist':
                    return self.get_artist_instruments(entity)
        
        # 10. Câu hỏi về nghề nghiệp: "Vai trò của Jungkook trong BTS"
        if any(kw in query_lower for kw in ['nghề nghiệp', 'occupation', 'vai trò', 'công việc']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Artist':
                    return self.get_artist_occupations(entity)
        
        # 11. Câu hỏi về nhóm con/subunit: "Nhóm con của SM Entertainment"
        if any(kw in query_lower for kw in ['nhóm con', 'subunit', 'sub-unit']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_subunits(entity)
        
        # 12. Câu hỏi về sáng tác: "Ai viết bài hát này"
        if any(kw in query_lower for kw in ['viết', 'wrote', 'sáng tác', 'compose', 'tác giả']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Song':
                    return self.get_song_writers(entity)
        
        # 13. Câu hỏi về sản xuất bài hát: "Ai sản xuất bài hát này"
        if any(kw in query_lower for kw in ['sản xuất bài hát', 'produced song', 'nhà sản xuất bài hát']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Song':
                    return self.get_song_producers(entity)
        
        # 14. Câu hỏi về sản xuất album: "Ai sản xuất album này"
        if any(kw in query_lower for kw in ['sản xuất album', 'produced album', 'nhà sản xuất album']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Album':
                    return self.get_album_producers(entity)
        
        # 15. Câu hỏi về bài hát trong album: "Album này có bài hát nào"
        if any(kw in query_lower for kw in ['bài hát trong album', 'songs in album', 'track trong']):
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Album':
                    return self.get_album_songs(entity)
                    
        # ============================================
        # FALLBACK - reasoning type detection
        # ============================================
        reasoning_type = self._detect_reasoning_type(query)
        
        # Execute appropriate reasoning strategy
        if reasoning_type == ReasoningType.CHAIN:
            return self._chain_reasoning(query, start_entities, max_hops)
        elif reasoning_type == ReasoningType.AGGREGATION:
            return self._aggregation_reasoning(query, start_entities, max_hops)
        elif reasoning_type == ReasoningType.COMPARISON:
            return self._comparison_reasoning(query, start_entities)
        elif reasoning_type == ReasoningType.INTERSECTION:
            return self._intersection_reasoning(query, start_entities)
        else:
            return self._chain_reasoning(query, start_entities, max_hops)
            
    def _detect_reasoning_type(self, query: str) -> ReasoningType:
        """Detect the type of reasoning needed for a query."""
        query_lower = query.lower()
        
        # Comparison indicators
        if any(kw in query_lower for kw in ['so sánh', 'compare', 'và', 'and', 'vs', 'cùng', 'same', 'giống']):
            if any(kw in query_lower for kw in ['chung', 'common', 'giống']):
                return ReasoningType.INTERSECTION
            return ReasoningType.COMPARISON
            
        # Aggregation indicators
        if any(kw in query_lower for kw in ['bao nhiêu', 'how many', 'tất cả', 'all', 'liệt kê', 'list']):
            return ReasoningType.AGGREGATION
            
        # Default to chain reasoning
        return ReasoningType.CHAIN
        
    def _chain_reasoning(
        self,
        query: str,
        start_entities: List[str],
        max_hops: int
    ) -> ReasoningResult:
        """
        Perform chain reasoning: A → B → C.
        
        Traverses the graph following relationships to find target entities.
        Cải thiện: Sử dụng path finding để suy luận chính xác hơn.
        """
        if not start_entities:
            return ReasoningResult(
                query=query,
                reasoning_type=ReasoningType.CHAIN,
                steps=[],
                answer_entities=[],
                answer_text="Không tìm thấy entities để bắt đầu suy luận",
                confidence=0.0,
                explanation="No starting entities"
            )
        
        steps = []
        current_entities = start_entities
        visited = set(start_entities)
        all_paths = []
        
        # Detect target entity type from query (if any)
        target_type = self._detect_target_type(query)
        target_relationship = self._detect_target_relationship(query)
        
        for hop in range(max_hops):
            step_result = {
                'hop': hop + 1,
                'from': current_entities,
                'to': [],
                'relationships': []
            }
            
            next_entities = set()
            relationship_types = set()
            
            for entity in current_entities[:10]:  # Limit for performance
                neighbors = self.kg.get_neighbors(entity)
                
                for neighbor, rel_type, direction in neighbors:
                    # Skip if already visited (avoid cycles)
                    if neighbor in visited:
                        continue
                    
                    # Filter by target relationship if specified
                    if target_relationship and rel_type != target_relationship:
                        continue
                    
                    # Filter by target type if specified
                    if target_type:
                        neighbor_type = self.kg.get_entity_type(neighbor)
                        if neighbor_type != target_type:
                            continue
                    
                    next_entities.add(neighbor)
                    visited.add(neighbor)
                    relationship_types.add(rel_type)
                    
                    step_result['to'].append(neighbor)
                    step_result['relationships'].append({
                        'from': entity,
                        'rel': rel_type,
                        'to': neighbor,
                        'direction': direction
                    })
                    
            if not next_entities:
                break
                
            reasoning_step = ReasoningStep(
                hop_number=hop + 1,
                operation='traverse',
                source_entities=list(current_entities),
                relationship=', '.join(sorted(relationship_types)) if relationship_types else 'various',
                target_entities=list(next_entities)[:20],
                explanation=f"Hop {hop + 1}: Từ {len(current_entities)} entities, tìm thấy {len(next_entities)} entities liên quan qua quan hệ {', '.join(sorted(relationship_types)[:3])}"
            )
            steps.append(reasoning_step)
            
            current_entities = list(next_entities)
            
        # If we have multiple start entities, try to find paths between them
        if len(start_entities) >= 2:
            for i in range(len(start_entities) - 1):
                for j in range(i + 1, len(start_entities)):
                    source = start_entities[i]
                    target = start_entities[j]
                    paths = self.kg.find_all_paths(source, target, max_hops=max_hops)
                    if paths:
                        all_paths.extend(paths[:3])  # Limit paths
                        # Add path as reasoning step
                        for path in paths[:1]:  # Use shortest path
                            path_details = self.kg.get_path_details(path)
                            if path_details:
                                rel_types = [p.get('type', '') for p in path_details if p.get('type')]
                                steps.append(ReasoningStep(
                                    hop_number=len(path) - 1,
                                    operation='path_finding',
                                    source_entities=[source],
                                    relationship=' → '.join(rel_types) if rel_types else 'path',
                                    target_entities=[target],
                                    explanation=f"Tìm thấy path {len(path)-1}-hop: {' → '.join(path[:5])}"
                                ))
        
        # Generate answer
        answer_entities = current_entities[:20]
        answer_text = self._generate_answer_text(query, steps, answer_entities)
        
        return ReasoningResult(
            query=query,
            reasoning_type=ReasoningType.CHAIN,
            steps=steps,
            answer_entities=answer_entities,
            answer_text=answer_text,
            confidence=self._calculate_confidence(steps),
            explanation=self._generate_explanation(steps)
        )
    
    def _detect_target_type(self, query: str) -> Optional[str]:
        """Detect target entity type from query - Hỗ trợ TẤT CẢ entity types."""
        query_lower = query.lower()
        
        # Entity types trong knowledge graph: Album, Artist, Company, Genre, Group, Instrument, Occupation, Song
        
        if any(kw in query_lower for kw in ['nhóm', 'group', 'ban nhạc']):
            return 'Group'
        elif any(kw in query_lower for kw in ['nghệ sĩ', 'artist', 'ca sĩ', 'singer']):
            return 'Artist'
        elif any(kw in query_lower for kw in ['công ty', 'company', 'label', 'hãng']):
            return 'Company'
        elif any(kw in query_lower for kw in ['bài hát', 'song', 'track', 'ca khúc']):
            return 'Song'
        elif any(kw in query_lower for kw in ['album', 'đĩa nhạc', 'ep', 'lp']):
            return 'Album'
        elif any(kw in query_lower for kw in ['thể loại', 'genre', 'dòng nhạc']):
            return 'Genre'
        elif any(kw in query_lower for kw in ['nhạc cụ', 'instrument', 'dụng cụ']):
            return 'Instrument'
        elif any(kw in query_lower for kw in ['nghề nghiệp', 'occupation', 'công việc', 'vai trò']):
            return 'Occupation'
        
        return None
    
    def _detect_target_relationship(self, query: str) -> Optional[str]:
        """Detect target relationship type from query - Hỗ trợ TẤT CẢ relationship types."""
        query_lower = query.lower()
        
        # Relationship types trong knowledge graph:
        # CONTAINS, HAS_OCCUPATION, IS_GENRE, MANAGED_BY, MEMBER_OF, PLAYS, 
        # PRODUCED_ALBUM, PRODUCED_SONG, RELEASED, SINGS, SUBUNIT_OF, WROTE
        
        if any(kw in query_lower for kw in ['thành viên', 'member', 'member of', 'thuộc nhóm']):
            return 'MEMBER_OF'
        elif any(kw in query_lower for kw in ['quản lý', 'manages', 'managed by', 'thuộc công ty']):
            return 'MANAGED_BY'
        elif any(kw in query_lower for kw in ['hát', 'sings', 'performs', 'thể hiện', 'ca sĩ']):
            return 'SINGS'
        elif any(kw in query_lower for kw in ['phát hành', 'released', 'ra mắt', 'release']):
            return 'RELEASED'
        elif any(kw in query_lower for kw in ['sản xuất bài hát', 'produced song', 'sản xuất ca khúc']):
            return 'PRODUCED_SONG'
        elif any(kw in query_lower for kw in ['sản xuất album', 'produced album', 'sản xuất đĩa']):
            return 'PRODUCED_ALBUM'
        elif any(kw in query_lower for kw in ['chứa', 'contains', 'bao gồm', 'gồm']):
            return 'CONTAINS'
        elif any(kw in query_lower for kw in ['nhóm con', 'subunit', 'sub-unit', 'sub unit']):
            return 'SUBUNIT_OF'
        elif any(kw in query_lower for kw in ['chơi', 'plays', 'sử dụng nhạc cụ']):
            return 'PLAYS'
        elif any(kw in query_lower for kw in ['viết', 'wrote', 'sáng tác', 'compose']):
            return 'WROTE'
        elif any(kw in query_lower for kw in ['thuộc thể loại', 'is genre', 'dòng nhạc']):
            return 'IS_GENRE'
        elif any(kw in query_lower for kw in ['nghề nghiệp', 'has occupation', 'làm nghề', 'vai trò']):
            return 'HAS_OCCUPATION'
        
        return None
        
    def _aggregation_reasoning(
        self,
        query: str,
        start_entities: List[str],
        max_hops: int
    ) -> ReasoningResult:
        """
        Perform aggregation reasoning: A → {B1, B2, ...} → count/list.
        
        Collects all related entities and aggregates results.
        """
        steps = []
        all_entities = set()
        
        for entity in start_entities:
            # Collect all entities within max_hops
            visited = {entity}
            current_level = [entity]
            
            for hop in range(max_hops):
                next_level = []
                for node in current_level:
                    neighbors = self.kg.get_neighbors(node)
                    for neighbor, rel_type, _ in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_level.append(neighbor)
                            all_entities.add(neighbor)
                            
                if hop == 0:
                    step = ReasoningStep(
                        hop_number=hop + 1,
                        operation='collect',
                        source_entities=[entity],
                        relationship='various',
                        target_entities=next_level[:20],
                        explanation=f"Thu thập {len(next_level)} entities từ {entity}"
                    )
                    steps.append(step)
                    
                current_level = next_level
                
        # Aggregate by type
        type_counts = defaultdict(list)
        for entity in all_entities:
            entity_type = self.kg.get_entity_type(entity)
            if entity_type:
                type_counts[entity_type].append(entity)
                
        # Generate aggregated answer
        answer_entities = list(all_entities)[:50]
        
        aggregation_summary = []
        for entity_type, entities in type_counts.items():
            aggregation_summary.append(f"{entity_type}: {len(entities)}")
            
        answer_text = f"Tìm thấy {len(all_entities)} entities. " + ", ".join(aggregation_summary)
        
        return ReasoningResult(
            query=query,
            reasoning_type=ReasoningType.AGGREGATION,
            steps=steps,
            answer_entities=answer_entities,
            answer_text=answer_text,
            confidence=0.85,
            explanation=f"Aggregated {len(all_entities)} entities from {len(start_entities)} starting points"
        )
        
    def _comparison_reasoning(
        self,
        query: str,
        entities: List[str]
    ) -> ReasoningResult:
        """
        Compare two or more entities.
        
        Finds common and different properties/relationships.
        """
        steps = []
        
        if len(entities) < 2:
            return ReasoningResult(
                query=query,
                reasoning_type=ReasoningType.COMPARISON,
                steps=[],
                answer_entities=entities,
                answer_text="Cần ít nhất 2 entities để so sánh",
                confidence=0.0,
                explanation="Insufficient entities for comparison"
            )
            
        entity1, entity2 = entities[0], entities[1]
        
        # Get relationships for both entities
        rels1 = self.kg.get_relationships(entity1)
        rels2 = self.kg.get_relationships(entity2)
        
        # Find common relationships
        rels1_set = {(r['type'], r['target']) for r in rels1}
        rels2_set = {(r['type'], r['target']) for r in rels2}
        
        common_targets = rels1_set.intersection(rels2_set)
        only_in_1 = rels1_set - rels2_set
        only_in_2 = rels2_set - rels1_set
        
        # Check specific comparisons
        comparisons = {
            'common': list(common_targets),
            'only_entity1': list(only_in_1),
            'only_entity2': list(only_in_2)
        }
        
        # Check if same company
        company1 = self.kg.get_group_company(entity1)
        company2 = self.kg.get_group_company(entity2)
        same_company = company1 and company2 and company1 == company2
        
        step = ReasoningStep(
            hop_number=1,
            operation='compare',
            source_entities=[entity1, entity2],
            relationship='comparison',
            target_entities=list(set(t for _, t in common_targets)),
            explanation=f"So sánh {entity1} và {entity2}"
        )
        steps.append(step)
        
        # Generate answer
        if 'cùng công ty' in query.lower() or 'same company' in query.lower():
            if same_company:
                answer_text = f"Có, {entity1} và {entity2} cùng thuộc công ty {company1}"
            else:
                answer_text = f"Không, {entity1} thuộc {company1}, còn {entity2} thuộc {company2}"
        else:
            answer_text = f"{entity1} và {entity2} có {len(common_targets)} điểm chung"
            
        return ReasoningResult(
            query=query,
            reasoning_type=ReasoningType.COMPARISON,
            steps=steps,
            answer_entities=[entity1, entity2],
            answer_text=answer_text,
            confidence=0.9,
            explanation=f"Compared {entity1} and {entity2}"
        )
        
    def _intersection_reasoning(
        self,
        query: str,
        entities: List[str]
    ) -> ReasoningResult:
        """
        Find intersection/common entities.
        """
        steps = []
        
        if len(entities) < 2:
            return ReasoningResult(
                query=query,
                reasoning_type=ReasoningType.INTERSECTION,
                steps=[],
                answer_entities=entities,
                answer_text="Cần ít nhất 2 entities",
                confidence=0.0,
                explanation="Insufficient entities"
            )
            
        # Get neighbors for all entities
        neighbor_sets = []
        for entity in entities[:5]:
            neighbors = {n for n, _, _ in self.kg.get_neighbors(entity)}
            neighbor_sets.append(neighbors)
            
        # Find intersection
        if neighbor_sets:
            common = neighbor_sets[0]
            for ns in neighbor_sets[1:]:
                common = common.intersection(ns)
                
            step = ReasoningStep(
                hop_number=1,
                operation='intersection',
                source_entities=entities,
                relationship='common',
                target_entities=list(common)[:20],
                explanation=f"Tìm điểm chung giữa {', '.join(entities)}"
            )
            steps.append(step)
            
            answer_text = f"Điểm chung: {', '.join(list(common)[:10])}" if common else "Không tìm thấy điểm chung"
        else:
            common = set()
            answer_text = "Không tìm thấy điểm chung"
            
        return ReasoningResult(
            query=query,
            reasoning_type=ReasoningType.INTERSECTION,
            steps=steps,
            answer_entities=list(common)[:20],
            answer_text=answer_text,
            confidence=0.8,
            explanation=f"Found {len(common)} common entities"
        )
        
    def _calculate_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate confidence score based on reasoning steps."""
        if not steps:
            return 0.0
            
        # Base confidence
        confidence = 1.0
        
        # Reduce for each hop
        for step in steps:
            hop_factor = 0.9 if step.hop_number <= 2 else 0.8
            confidence *= hop_factor
            
        return min(confidence, 1.0)
        
    def _generate_answer_text(
        self,
        query: str,
        steps: List[ReasoningStep],
        answer_entities: List[str]
    ) -> str:
        """Generate natural language answer."""
        if not answer_entities:
            return "Không tìm thấy thông tin liên quan."
            
        # Get types of answer entities
        entity_types = defaultdict(list)
        for entity in answer_entities[:20]:
            entity_type = self.kg.get_entity_type(entity) or 'Unknown'
            entity_types[entity_type].append(entity)
            
        # Format answer based on entity types
        parts = []
        for entity_type, entities in entity_types.items():
            if len(entities) <= 5:
                parts.append(f"{entity_type}: {', '.join(entities)}")
            else:
                parts.append(f"{entity_type}: {', '.join(entities[:5])} và {len(entities) - 5} khác")
                
        return " | ".join(parts)
        
    def _generate_explanation(self, steps: List[ReasoningStep]) -> str:
        """Generate explanation of reasoning process."""
        if not steps:
            return "Không thực hiện suy luận"
            
        explanations = []
        for step in steps:
            explanations.append(
                f"Bước {step.hop_number}: {step.explanation} "
                f"(tìm thấy {len(step.target_entities)} entities)"
            )
            
        return " → ".join(explanations)
        
    # =========== Specialized Multi-hop Queries ===========
    
    def get_group_members(self, group_name: str) -> ReasoningResult:
        """Get all members of a group (1-hop)."""
        members = self.kg.get_group_members(group_name)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_members',
            source_entities=[group_name],
            relationship='MEMBER_OF',
            target_entities=members,
            explanation=f"Lấy thành viên của {group_name}"
        )
        
        return ReasoningResult(
            query=f"Thành viên của {group_name}",
            reasoning_type=ReasoningType.CHAIN,
            steps=[step],
            answer_entities=members,
            answer_text=f"{group_name} có {len(members)} thành viên: {', '.join(members)}",
            confidence=1.0,
            explanation=f"1-hop: {group_name} → MEMBER_OF → Artists"
        )
    
    def get_artist_groups(self, artist_name: str) -> ReasoningResult:
        """Get all groups that an artist belongs to (1-hop)."""
        groups = self.kg.get_artist_groups(artist_name)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_groups',
            source_entities=[artist_name],
            relationship='MEMBER_OF',
            target_entities=groups,
            explanation=f"Lấy nhóm nhạc của {artist_name}"
        )
        
        if groups:
            groups_str = ', '.join(groups)
            return ReasoningResult(
                query=f"Nhóm nhạc của {artist_name}",
                reasoning_type=ReasoningType.CHAIN,
                steps=[step],
                answer_entities=groups,
                answer_text=f"{artist_name} thuộc nhóm nhạc: {groups_str}",
                confidence=1.0,
                explanation=f"1-hop: {artist_name} → MEMBER_OF → {groups_str}"
            )
        else:
            return ReasoningResult(
                query=f"Nhóm nhạc của {artist_name}",
                reasoning_type=ReasoningType.CHAIN,
                steps=[step],
                answer_entities=[],
                answer_text=f"{artist_name} không thuộc nhóm nhạc nào",
                confidence=1.0,
                explanation=f"1-hop: {artist_name} không có quan hệ MEMBER_OF với nhóm nhạc nào"
            )
        
    def get_company_of_group(self, group_name: str) -> ReasoningResult:
        """Get company managing a group (1-hop)."""
        company = self.kg.get_group_company(group_name)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_company',
            source_entities=[group_name],
            relationship='MANAGED_BY',
            target_entities=[company] if company else [],
            explanation=f"Lấy công ty quản lý {group_name}"
        )
        
        return ReasoningResult(
            query=f"Công ty quản lý {group_name}",
            reasoning_type=ReasoningType.CHAIN,
            steps=[step],
            answer_entities=[company] if company else [],
            answer_text=f"{group_name} thuộc công ty {company}" if company else f"Không tìm thấy công ty của {group_name}",
            confidence=1.0 if company else 0.0,
            explanation=f"1-hop: {group_name} → MANAGED_BY → Company"
        )
        
    def get_artist_company(self, artist_name: str) -> ReasoningResult:
        """Get company of an artist (2-hop: Artist → Group → Company)."""
        steps = []
        
        # Step 1: Get groups
        groups = self.kg.get_artist_groups(artist_name)
        steps.append(ReasoningStep(
            hop_number=1,
            operation='get_groups',
            source_entities=[artist_name],
            relationship='MEMBER_OF',
            target_entities=groups,
            explanation=f"Lấy nhóm nhạc của {artist_name}"
        ))
        
        # Step 2: Get companies of those groups
        companies = []
        for group in groups:
            company = self.kg.get_group_company(group)
            if company:
                companies.append(company)
                
        steps.append(ReasoningStep(
            hop_number=2,
            operation='get_companies',
            source_entities=groups,
            relationship='MANAGED_BY',
            target_entities=list(set(companies)),
            explanation=f"Lấy công ty quản lý các nhóm"
        ))
        
        unique_companies = list(set(companies))
        
        return ReasoningResult(
            query=f"Công ty của {artist_name}",
            reasoning_type=ReasoningType.CHAIN,
            steps=steps,
            answer_entities=unique_companies,
            answer_text=f"{artist_name} thuộc công ty: {', '.join(unique_companies)}" if unique_companies else f"Không tìm thấy công ty của {artist_name}",
            confidence=0.9 if unique_companies else 0.0,
            explanation=f"2-hop: {artist_name} → MEMBER_OF → Group → MANAGED_BY → Company"
        )
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract entity names from query (case-insensitive).
        Tìm tất cả artists/groups có thể có trong query.
        """
        entities = []
        query_lower = query.lower()
        
        # Lấy tất cả artists và groups từ KG
        all_artists = [node for node, data in self.kg.graph.nodes(data=True) 
                      if data.get('label') == 'Artist']
        all_groups = [node for node, data in self.kg.graph.nodes(data=True) 
                     if data.get('label') == 'Group']
        
        # Tìm tất cả artists trong query (case-insensitive)
        # QUAN TRỌNG: Xử lý node có đuôi như "Lisa (ca sĩ)"
        query_words_list = query_lower.split()  # List để giữ thứ tự
        
        for artist in all_artists:
            artist_lower = artist.lower()
            # Extract base name (không có đuôi)
            base_name = self._normalize_entity_name(artist)
            base_name_lower = base_name.lower()
            
            # Tạo variants để match với nhiều format: "g-dragon", "g dragon", "gdragon"
            base_name_variants = [
                base_name_lower,
                base_name_lower.replace('-', ' '),  # "g-dragon" → "g dragon"
                base_name_lower.replace('-', ''),    # "g-dragon" → "gdragon"
                base_name_lower.replace(' ', ''),    # "black pink" → "blackpink"
            ]
            
            # Check nếu base name hoặc variants là một từ trong query
            # Ví dụ: query "lisa có cùng nhóm" → word "lisa" match với base_name "lisa"
            if any(variant in query_words_list for variant in base_name_variants):
                if artist not in entities:
                    entities.append(artist)
                    continue
            
            # Check nếu base name hoặc variants xuất hiện trong query text
            if any(variant in query_lower for variant in base_name_variants if len(variant) >= 3):
                if artist not in entities:
                    entities.append(artist)
                    continue
            
            # Check từng word trong query với base name và variants
            for word in query_words_list:
                if len(word) < 3:  # Skip short words
                    continue
                # Exact match với base name hoặc variants
                if word in base_name_variants or word == base_name_lower:
                    if artist not in entities:
                        entities.append(artist)
                        break
                # Partial match: word là một phần của base name hoặc ngược lại
                elif (word in base_name_lower and len(word) >= 3) or (base_name_lower in word and len(base_name_lower) >= 3):
                    if artist not in entities:
                        entities.append(artist)
                        break
                # Xử lý tên có dấu gạch ngang: "g-dragon" match với "g" và "dragon"
                elif '-' in base_name_lower:
                    base_parts = base_name_lower.split('-')
                    if word in base_parts and len(word) >= 3:
                        # Check xem có part khác cũng trong query không
                        other_parts = [p for p in base_parts if p != word]
                        if any(p in query_lower for p in other_parts):
                            if artist not in entities:
                                entities.append(artist)
                                break
        
        # Tìm tất cả groups trong query (case-insensitive)
        for group in all_groups:
            group_lower = group.lower()
            if group_lower in query_lower:
                query_words = set(query_lower.split())
                group_words = set(group_lower.split())
                if group_words.intersection(query_words) or group_lower in query_lower:
                    if group not in entities:
                        entities.append(group)
        
        # Nếu chưa đủ, try fuzzy match với từng word (bao gồm base name)
        if len(entities) < 2:
            words = query_lower.split()
            for word in words:
                if len(word) < 3:
                    continue
                # Try exact match với full name hoặc base name
                for node in self.kg.graph.nodes():
                    node_lower = node.lower()
                    base_name = self._normalize_entity_name(node).lower()
                    
                    # Exact match với full name hoặc base name
                    if node_lower == word or base_name == word:
                        if node not in entities:
                            node_data = self.kg.get_entity(node)
                            if node_data and node_data.get('label') in ['Artist', 'Group']:
                                entities.append(node)
                        break
                    # Partial match
                    elif (word in base_name and len(word) >= 3) or (base_name in word and len(base_name) >= 3):
                        if node not in entities:
                            node_data = self.kg.get_entity(node)
                            if node_data and node_data.get('label') in ['Artist', 'Group']:
                                entities.append(node)
                        break
        
        return entities[:10]  # Return max 10
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """
        Normalize entity name bằng cách remove suffixes trong parentheses.
        
        Ví dụ:
        - "Lisa (ca sĩ)" → "Lisa"
        - "BLACKPINK (nhóm nhạc)" → "BLACKPINK"
        
        Args:
            entity_name: Entity name có thể có đuôi
            
        Returns:
            Base name (không có đuôi)
        """
        import re
        # Remove suffixes trong parentheses
        normalized = re.sub(r'\s*\([^)]+\)\s*$', '', entity_name)
        return normalized.strip()
        
    def check_same_group(self, entity1: str, entity2: str) -> ReasoningResult:
        """Check if two artists are in the same group (1-hop comparison)."""
        steps = []
        
        # Get groups for entity1
        if self.kg.get_entity_type(entity1) == 'Artist':
            groups1 = self.kg.get_artist_groups(entity1)
        else:
            # If entity1 is a Group, check if entity2 is a member
            groups1 = [entity1] if self.kg.get_entity_type(entity1) == 'Group' else []
            
        # Get groups for entity2
        if self.kg.get_entity_type(entity2) == 'Artist':
            groups2 = self.kg.get_artist_groups(entity2)
        else:
            # If entity2 is a Group, check if entity1 is a member
            groups2 = [entity2] if self.kg.get_entity_type(entity2) == 'Group' else []
        
        # If one is a group and the other is an artist, check membership
        if self.kg.get_entity_type(entity1) == 'Group' and self.kg.get_entity_type(entity2) == 'Artist':
            members = self.kg.get_group_members(entity1)
            if entity2 in members:
                return ReasoningResult(
                    query=f"{entity2} có cùng nhóm nhạc với {entity1} không?",
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[ReasoningStep(
                        hop_number=1,
                        operation='check_membership',
                        source_entities=[entity2],
                        relationship='MEMBER_OF',
                        target_entities=[entity1],
                        explanation=f"{entity2} là thành viên của {entity1}"
                    )],
                    answer_entities=[entity1],
                    answer_text=f"Có, {entity2} là thành viên của {entity1}",
                    confidence=1.0,
                    explanation=f"1-hop: {entity2} → MEMBER_OF → {entity1}"
                )
            else:
                return ReasoningResult(
                    query=f"{entity2} có cùng nhóm nhạc với {entity1} không?",
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[ReasoningStep(
                        hop_number=1,
                        operation='check_membership',
                        source_entities=[entity2],
                        relationship='MEMBER_OF',
                        target_entities=[],
                        explanation=f"{entity2} không phải thành viên của {entity1}"
                    )],
                    answer_entities=[],
                    answer_text=f"Không, {entity2} không phải thành viên của {entity1}",
                    confidence=1.0,
                    explanation=f"1-hop: {entity2} không có quan hệ MEMBER_OF với {entity1}"
                )
        
        if self.kg.get_entity_type(entity2) == 'Group' and self.kg.get_entity_type(entity1) == 'Artist':
            members = self.kg.get_group_members(entity2)
            if entity1 in members:
                return ReasoningResult(
                    query=f"{entity1} có cùng nhóm nhạc với {entity2} không?",
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[ReasoningStep(
                        hop_number=1,
                        operation='check_membership',
                        source_entities=[entity1],
                        relationship='MEMBER_OF',
                        target_entities=[entity2],
                        explanation=f"{entity1} là thành viên của {entity2}"
                    )],
                    answer_entities=[entity2],
                    answer_text=f"Có, {entity1} là thành viên của {entity2}",
                    confidence=1.0,
                    explanation=f"1-hop: {entity1} → MEMBER_OF → {entity2}"
                )
            else:
                return ReasoningResult(
                    query=f"{entity1} có cùng nhóm nhạc với {entity2} không?",
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[ReasoningStep(
                        hop_number=1,
                        operation='check_membership',
                        source_entities=[entity1],
                        relationship='MEMBER_OF',
                        target_entities=[],
                        explanation=f"{entity1} không phải thành viên của {entity2}"
                    )],
                    answer_entities=[],
                    answer_text=f"Không, {entity1} không phải thành viên của {entity2}",
                    confidence=1.0,
                    explanation=f"1-hop: {entity1} không có quan hệ MEMBER_OF với {entity2}"
                )
        
        steps.append(ReasoningStep(
            hop_number=1,
            operation='get_groups',
            source_entities=[entity1],
            relationship='MEMBER_OF',
            target_entities=groups1,
            explanation=f"Lấy nhóm nhạc của {entity1}"
        ))
        
        steps.append(ReasoningStep(
            hop_number=1,
            operation='get_groups',
            source_entities=[entity2],
            relationship='MEMBER_OF',
            target_entities=groups2,
            explanation=f"Lấy nhóm nhạc của {entity2}"
        ))
        
        # Compare groups
        common_groups = set(groups1).intersection(set(groups2))
        
        if common_groups:
            group_list = list(common_groups)
            return ReasoningResult(
                query=f"{entity1} và {entity2} có cùng nhóm nhạc không?",
                reasoning_type=ReasoningType.COMPARISON,
                steps=steps,
                answer_entities=group_list,
                answer_text=f"Có, {entity1} và {entity2} đều là thành viên của {', '.join(group_list)}",
                confidence=1.0,
                explanation=f"1-hop: Cả {entity1} và {entity2} đều có quan hệ MEMBER_OF với {', '.join(group_list)}"
            )
        else:
            return ReasoningResult(
                query=f"{entity1} và {entity2} có cùng nhóm nhạc không?",
                reasoning_type=ReasoningType.COMPARISON,
                steps=steps,
                answer_entities=[],
                answer_text=f"Không, {entity1} và {entity2} không cùng nhóm nhạc. {entity1} thuộc {', '.join(groups1) if groups1 else 'không có nhóm'}, còn {entity2} thuộc {', '.join(groups2) if groups2 else 'không có nhóm'}",
                confidence=1.0,
                explanation=f"1-hop: {entity1} thuộc {', '.join(groups1) if groups1 else 'không có nhóm'}, {entity2} thuộc {', '.join(groups2) if groups2 else 'không có nhóm'}"
            )
        
    def check_same_company(self, entity1: str, entity2: str) -> ReasoningResult:
        """Check if two groups/artists are under same company (2-hop comparison)."""
        steps = []
        
        # Get ALL companies for entity1 (một entity có thể có nhiều companies)
        if self.kg.get_entity_type(entity1) == 'Artist':
            groups1 = self.kg.get_artist_groups(entity1)
            companies1 = []
            for g in groups1:
                # Lấy TẤT CẢ companies của group này
                group_companies = self.kg.get_group_companies(g)
                companies1.extend(group_companies)
            companies1 = list(set(companies1))  # Remove duplicates
        else:
            # Group có thể có nhiều companies
            companies1 = self.kg.get_group_companies(entity1)
            
        # Get ALL companies for entity2 (một entity có thể có nhiều companies)
        if self.kg.get_entity_type(entity2) == 'Artist':
            groups2 = self.kg.get_artist_groups(entity2)
            companies2 = []
            for g in groups2:
                # Lấy TẤT CẢ companies của group này
                group_companies = self.kg.get_group_companies(g)
                companies2.extend(group_companies)
            companies2 = list(set(companies2))  # Remove duplicates
        else:
            # Group có thể có nhiều companies
            companies2 = self.kg.get_group_companies(entity2)
            
        steps.append(ReasoningStep(
            hop_number=1,
            operation='get_companies',
            source_entities=[entity1],
            relationship='MANAGED_BY',
            target_entities=companies1,
            explanation=f"Lấy công ty của {entity1}"
        ))
        
        steps.append(ReasoningStep(
            hop_number=2,
            operation='get_companies',
            source_entities=[entity2],
            relationship='MANAGED_BY',
            target_entities=companies2,
            explanation=f"Lấy công ty của {entity2}"
        ))
        
        # Compare
        common_companies = set(companies1).intersection(set(companies2))
        
        if common_companies:
            answer_text = f"Có, {entity1} và {entity2} cùng thuộc công ty: {', '.join(common_companies)}"
        else:
            answer_text = f"Không, {entity1} ({', '.join(companies1) if companies1 else 'không rõ'}) và {entity2} ({', '.join(companies2) if companies2 else 'không rõ'}) khác công ty"
            
        return ReasoningResult(
            query=f"{entity1} và {entity2} có cùng công ty không?",
            reasoning_type=ReasoningType.COMPARISON,
            steps=steps,
            answer_entities=list(common_companies),
            answer_text=answer_text,
            confidence=1.0,
            explanation=f"2-hop comparison: Find companies of both entities and compare"
        )
        
    def get_labelmates(self, artist_or_group: str) -> ReasoningResult:
        """Get all labelmates (same company) of an artist/group (3-hop)."""
        steps = []
        
        # Step 1: Get ALL companies (một entity có thể có nhiều companies)
        if self.kg.get_entity_type(artist_or_group) == 'Artist':
            groups = self.kg.get_artist_groups(artist_or_group)
            companies = []
            for g in groups:
                # Lấy TẤT CẢ companies của group này
                group_companies = self.kg.get_group_companies(g)
                companies.extend(group_companies)
            companies = list(set(companies))  # Remove duplicates
        else:
            # Group có thể có nhiều companies
            companies = self.kg.get_group_companies(artist_or_group)
            
        steps.append(ReasoningStep(
            hop_number=1,
            operation='get_company',
            source_entities=[artist_or_group],
            relationship='MANAGED_BY',
            target_entities=companies,
            explanation=f"Lấy công ty của {artist_or_group}"
        ))
        
        # Step 2: Get all groups under those companies
        all_groups = set()
        for company in companies:
            company_groups = self.kg.get_company_groups(company)
            all_groups.update(company_groups)
            
        # Remove the original group
        all_groups.discard(artist_or_group)
        
        steps.append(ReasoningStep(
            hop_number=2,
            operation='get_labelmates',
            source_entities=companies,
            relationship='MANAGED_BY',
            target_entities=list(all_groups),
            explanation=f"Lấy các nhóm cùng công ty"
        ))
        
        return ReasoningResult(
            query=f"Các nhóm cùng công ty với {artist_or_group}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=steps,
            answer_entities=list(all_groups),
            answer_text=f"Có {len(all_groups)} nhóm cùng công ty: {', '.join(list(all_groups)[:10])}{'...' if len(all_groups) > 10 else ''}",
            confidence=0.95,
            explanation=f"3-hop: {artist_or_group} → Company → Other Groups"
        )
    
    def get_collaborating_groups(self, group_name: str) -> ReasoningResult:
        """
        Get groups that have collaborated with the given group.
        
        Strategy: Find groups that share songs (collaboration songs).
        In the graph: Song → SINGS → Group (song is sung by group)
        """
        steps = []
        collaborating_groups = set()
        
        # Step 1: Get all songs by this group
        # In graph: Song has in_edge to Group with type='SINGS'
        songs = self.kg.get_group_songs(group_name)
        steps.append(ReasoningStep(
            hop_number=1,
            operation='get_songs',
            source_entities=[group_name],
            relationship='SINGS',
            target_entities=songs[:10],  # Limit for display
            explanation=f"Lấy các bài hát của {group_name}"
        ))
        
        # Step 2: For each song, find other groups that also sing it (collaboration)
        # Check all in_edges to the song with type='SINGS'
        for song in songs[:100]:  # Limit to avoid too many iterations
            # Get all groups/artists that also sing this song
            # In graph: song has in_edges from other groups with type='SINGS'
            for source, _, data in self.kg.graph.in_edges(song, data=True):
                if data.get('type') == 'SINGS':
                    source_type = self.kg.get_entity_type(source)
                    # If it's a Group and not the original group, it's a collaborator
                    if source_type == 'Group' and source != group_name:
                        collaborating_groups.add(source)
                    # If it's an Artist, get their groups
                    elif source_type == 'Artist':
                        artist_groups = self.kg.get_artist_groups(source)
                        for ag in artist_groups:
                            if ag != group_name:
                                collaborating_groups.add(ag)
        
        # Step 3: Also check for direct collaboration relationships
        relationships = self.kg.get_relationships(group_name)
        for rel in relationships:
            if rel['type'] in ['COLLABORATES_WITH', 'FEATURED', 'FEATURES']:
                target = rel['target']
                target_type = self.kg.get_entity_type(target)
                if target_type == 'Group':
                    collaborating_groups.add(target)
        
        steps.append(ReasoningStep(
            hop_number=2,
            operation='find_collaborators',
            source_entities=songs[:10],
            relationship='COLLABORATION',
            target_entities=list(collaborating_groups)[:10],
            explanation=f"Tìm các nhóm hợp tác qua bài hát chung"
        ))
        
        if collaborating_groups:
            groups_list = list(collaborating_groups)
            answer_text = f"Có {len(groups_list)} nhóm nhạc đã hợp tác với {group_name}: {', '.join(groups_list[:10])}"
            if len(groups_list) > 10:
                answer_text += f" và {len(groups_list) - 10} nhóm khác"
        else:
            answer_text = f"Không tìm thấy thông tin về các nhóm nhạc đã hợp tác với {group_name} trong đồ thị tri thức."
        
        return ReasoningResult(
            query=f"Các nhóm nhạc đã hợp tác với {group_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=steps,
            answer_entities=list(collaborating_groups),
            answer_text=answer_text,
            confidence=0.8 if collaborating_groups else 0.0,
            explanation=f"2-hop: {group_name} → Songs → Other Groups"
        )
    
    def get_group_songs(self, group_name: str) -> ReasoningResult:
        """Get all songs by a group (1-hop: Group → SINGS → Song)."""
        songs = self.kg.get_group_songs(group_name)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_songs',
            source_entities=[group_name],
            relationship='SINGS',
            target_entities=songs,
            explanation=f"Lấy các bài hát của {group_name}"
        )
        
        return ReasoningResult(
            query=f"Bài hát của {group_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=songs,
            answer_text=f"{group_name} có {len(songs)} bài hát: {', '.join(songs[:10])}{'...' if len(songs) > 10 else ''}",
            confidence=1.0 if songs else 0.0,
            explanation=f"1-hop: {group_name} → SINGS → Song"
        )
    
    def get_artist_songs(self, artist_name: str) -> ReasoningResult:
        """Get all songs by an artist (1-hop: Artist → SINGS → Song)."""
        songs = []
        for _, target, data in self.kg.graph.out_edges(artist_name, data=True):
            if data.get('type') == 'SINGS':
                songs.append(target)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_songs',
            source_entities=[artist_name],
            relationship='SINGS',
            target_entities=songs,
            explanation=f"Lấy các bài hát của {artist_name}"
        )
        
        return ReasoningResult(
            query=f"Bài hát của {artist_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=songs,
            answer_text=f"{artist_name} có {len(songs)} bài hát: {', '.join(songs[:10])}{'...' if len(songs) > 10 else ''}",
            confidence=1.0 if songs else 0.0,
            explanation=f"1-hop: {artist_name} → SINGS → Song"
        )
    
    def get_group_albums(self, group_name: str) -> ReasoningResult:
        """Get all albums by a group (1-hop: Group → RELEASED → Album)."""
        albums = []
        for _, target, data in self.kg.graph.out_edges(group_name, data=True):
            if data.get('type') == 'RELEASED':
                if self.kg.get_entity_type(target) == 'Album':
                    albums.append(target)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_albums',
            source_entities=[group_name],
            relationship='RELEASED',
            target_entities=albums,
            explanation=f"Lấy các album của {group_name}"
        )
        
        return ReasoningResult(
            query=f"Album của {group_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=albums,
            answer_text=f"{group_name} có {len(albums)} album: {', '.join(albums[:10])}{'...' if len(albums) > 10 else ''}",
            confidence=1.0 if albums else 0.0,
            explanation=f"1-hop: {group_name} → RELEASED → Album"
        )
    
    def get_artist_albums(self, artist_name: str) -> ReasoningResult:
        """Get all albums by an artist (2-hop: Artist → Group → RELEASED → Album)."""
        steps = []
        
        # Step 1: Get groups
        groups = self.kg.get_artist_groups(artist_name)
        steps.append(ReasoningStep(
            hop_number=1,
            operation='get_groups',
            source_entities=[artist_name],
            relationship='MEMBER_OF',
            target_entities=groups,
            explanation=f"Lấy nhóm của {artist_name}"
        ))
        
        # Step 2: Get albums from groups
        albums = []
        for group in groups:
            for _, target, data in self.kg.graph.out_edges(group, data=True):
                if data.get('type') == 'RELEASED':
                    if self.kg.get_entity_type(target) == 'Album':
                        albums.append(target)
        
        steps.append(ReasoningStep(
            hop_number=2,
            operation='get_albums',
            source_entities=groups,
            relationship='RELEASED',
            target_entities=albums,
            explanation=f"Lấy album từ các nhóm"
        ))
        
        unique_albums = list(set(albums))
        
        return ReasoningResult(
            query=f"Album của {artist_name}",
            reasoning_type=ReasoningType.CHAIN,
            steps=steps,
            answer_entities=unique_albums,
            answer_text=f"{artist_name} có {len(unique_albums)} album: {', '.join(unique_albums[:10])}{'...' if len(unique_albums) > 10 else ''}",
            confidence=0.9 if unique_albums else 0.0,
            explanation=f"2-hop: {artist_name} → MEMBER_OF → Group → RELEASED → Album"
        )
    
    def get_entity_genres(self, entity_name: str) -> ReasoningResult:
        """Get genres of an entity (1-hop: Entity → IS_GENRE → Genre)."""
        genres = []
        for _, target, data in self.kg.graph.out_edges(entity_name, data=True):
            if data.get('type') == 'IS_GENRE':
                if self.kg.get_entity_type(target) == 'Genre':
                    genres.append(target)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_genres',
            source_entities=[entity_name],
            relationship='IS_GENRE',
            target_entities=genres,
            explanation=f"Lấy thể loại của {entity_name}"
        )
        
        return ReasoningResult(
            query=f"Thể loại của {entity_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=genres,
            answer_text=f"{entity_name} thuộc thể loại: {', '.join(genres) if genres else 'Không xác định'}",
            confidence=1.0 if genres else 0.0,
            explanation=f"1-hop: {entity_name} → IS_GENRE → Genre"
        )
    
    def get_artist_instruments(self, artist_name: str) -> ReasoningResult:
        """Get instruments played by an artist (1-hop: Artist → PLAYS → Instrument)."""
        instruments = []
        for _, target, data in self.kg.graph.out_edges(artist_name, data=True):
            if data.get('type') == 'PLAYS':
                if self.kg.get_entity_type(target) == 'Instrument':
                    instruments.append(target)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_instruments',
            source_entities=[artist_name],
            relationship='PLAYS',
            target_entities=instruments,
            explanation=f"Lấy nhạc cụ của {artist_name}"
        )
        
        return ReasoningResult(
            query=f"Nhạc cụ của {artist_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=instruments,
            answer_text=f"{artist_name} chơi: {', '.join(instruments) if instruments else 'Không có thông tin'}",
            confidence=1.0 if instruments else 0.0,
            explanation=f"1-hop: {artist_name} → PLAYS → Instrument"
        )
    
    def get_artist_occupations(self, artist_name: str) -> ReasoningResult:
        """Get occupations of an artist (1-hop: Artist → HAS_OCCUPATION → Occupation)."""
        occupations = []
        for _, target, data in self.kg.graph.out_edges(artist_name, data=True):
            if data.get('type') == 'HAS_OCCUPATION':
                if self.kg.get_entity_type(target) == 'Occupation':
                    occupations.append(target)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_occupations',
            source_entities=[artist_name],
            relationship='HAS_OCCUPATION',
            target_entities=occupations,
            explanation=f"Lấy nghề nghiệp của {artist_name}"
        )
        
        return ReasoningResult(
            query=f"Nghề nghiệp của {artist_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=occupations,
            answer_text=f"{artist_name} có vai trò: {', '.join(occupations) if occupations else 'Không có thông tin'}",
            confidence=1.0 if occupations else 0.0,
            explanation=f"1-hop: {artist_name} → HAS_OCCUPATION → Occupation"
        )
    
    def get_subunits(self, group_name: str) -> ReasoningResult:
        """Get subunits of a group (1-hop: Group → SUBUNIT_OF → Subunit)."""
        subunits = []
        for _, target, data in self.kg.graph.out_edges(group_name, data=True):
            if data.get('type') == 'SUBUNIT_OF':
                if self.kg.get_entity_type(target) == 'Group':
                    subunits.append(target)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_subunits',
            source_entities=[group_name],
            relationship='SUBUNIT_OF',
            target_entities=subunits,
            explanation=f"Lấy nhóm con của {group_name}"
        )
        
        return ReasoningResult(
            query=f"Nhóm con của {group_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=subunits,
            answer_text=f"{group_name} có {len(subunits)} nhóm con: {', '.join(subunits) if subunits else 'Không có'}",
            confidence=1.0 if subunits else 0.0,
            explanation=f"1-hop: {group_name} → SUBUNIT_OF → Subunit"
        )
    
    def get_song_writers(self, song_name: str) -> ReasoningResult:
        """Get writers of a song (1-hop: Song ← WROTE ← Artist)."""
        writers = []
        for source, _, data in self.kg.graph.in_edges(song_name, data=True):
            if data.get('type') == 'WROTE':
                if self.kg.get_entity_type(source) == 'Artist':
                    writers.append(source)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_writers',
            source_entities=[song_name],
            relationship='WROTE',
            target_entities=writers,
            explanation=f"Lấy tác giả của {song_name}"
        )
        
        return ReasoningResult(
            query=f"Tác giả của {song_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=writers,
            answer_text=f"{song_name} được viết bởi: {', '.join(writers) if writers else 'Không có thông tin'}",
            confidence=1.0 if writers else 0.0,
            explanation=f"1-hop: Song ← WROTE ← Artist"
        )
    
    def get_album_songs(self, album_name: str) -> ReasoningResult:
        """Get songs in an album (1-hop: Album ← CONTAINS ← Song)."""
        songs = []
        for source, _, data in self.kg.graph.in_edges(album_name, data=True):
            if data.get('type') == 'CONTAINS':
                if self.kg.get_entity_type(source) == 'Song':
                    songs.append(source)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_songs',
            source_entities=[album_name],
            relationship='CONTAINS',
            target_entities=songs,
            explanation=f"Lấy các bài hát trong album {album_name}"
        )
        
        return ReasoningResult(
            query=f"Bài hát trong album {album_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=songs,
            answer_text=f"Album {album_name} có {len(songs)} bài hát: {', '.join(songs[:10])}{'...' if len(songs) > 10 else ''}",
            confidence=1.0 if songs else 0.0,
            explanation=f"1-hop: Album ← CONTAINS ← Song"
        )
    
    def get_song_producers(self, song_name: str) -> ReasoningResult:
        """Get producers of a song (1-hop: Song ← PRODUCED_SONG ← Artist/Group)."""
        producers = []
        for source, _, data in self.kg.graph.in_edges(song_name, data=True):
            if data.get('type') == 'PRODUCED_SONG':
                producers.append(source)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_producers',
            source_entities=[song_name],
            relationship='PRODUCED_SONG',
            target_entities=producers,
            explanation=f"Lấy nhà sản xuất của {song_name}"
        )
        
        return ReasoningResult(
            query=f"Nhà sản xuất của {song_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=producers,
            answer_text=f"{song_name} được sản xuất bởi: {', '.join(producers) if producers else 'Không có thông tin'}",
            confidence=1.0 if producers else 0.0,
            explanation=f"1-hop: Song ← PRODUCED_SONG ← Producer"
        )
    
    def get_album_producers(self, album_name: str) -> ReasoningResult:
        """Get producers of an album (1-hop: Album ← PRODUCED_ALBUM ← Artist/Group)."""
        producers = []
        for source, _, data in self.kg.graph.in_edges(album_name, data=True):
            if data.get('type') == 'PRODUCED_ALBUM':
                producers.append(source)
        
        step = ReasoningStep(
            hop_number=1,
            operation='get_producers',
            source_entities=[album_name],
            relationship='PRODUCED_ALBUM',
            target_entities=producers,
            explanation=f"Lấy nhà sản xuất của album {album_name}"
        )
        
        return ReasoningResult(
            query=f"Nhà sản xuất của album {album_name}",
            reasoning_type=ReasoningType.AGGREGATION,
            steps=[step],
            answer_entities=producers,
            answer_text=f"Album {album_name} được sản xuất bởi: {', '.join(producers) if producers else 'Không có thông tin'}",
            confidence=1.0 if producers else 0.0,
            explanation=f"1-hop: Album ← PRODUCED_ALBUM ← Producer"
        )


def main():
    """Test multi-hop reasoning."""
    print("🔄 Initializing Multi-hop Reasoner...")
    reasoner = MultiHopReasoner()
    
    # Test cases
    print("\n" + "="*60)
    print("📊 Test 1: Get BTS members (1-hop)")
    result = reasoner.get_group_members("BTS")
    print(f"Answer: {result.answer_text}")
    print(f"Confidence: {result.confidence}")
    print(f"Explanation: {result.explanation}")
    
    print("\n" + "="*60)
    print("📊 Test 2: Get BTS company (1-hop)")
    result = reasoner.get_company_of_group("BTS")
    print(f"Answer: {result.answer_text}")
    print(f"Explanation: {result.explanation}")
    
    print("\n" + "="*60)
    print("📊 Test 3: Check if BTS and SEVENTEEN are under same company (2-hop)")
    result = reasoner.check_same_company("BTS", "SEVENTEEN")
    print(f"Answer: {result.answer_text}")
    print(f"Explanation: {result.explanation}")
    
    print("\n" + "="*60)
    print("📊 Test 4: Get BTS labelmates (3-hop)")
    result = reasoner.get_labelmates("BTS")
    print(f"Answer: {result.answer_text}")
    print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    main()

