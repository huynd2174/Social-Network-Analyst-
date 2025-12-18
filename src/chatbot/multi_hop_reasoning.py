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
    
    def __init__(self, knowledge_graph: Optional[KpopKnowledgeGraph] = None, graph_rag: Optional['GraphRAG'] = None):
        """Initialize with knowledge graph and optional GraphRAG for LLM entity extraction."""
        self.kg = knowledge_graph or KpopKnowledgeGraph()
        self.graph_rag = graph_rag  # GraphRAG instance để dùng LLM extract entities khi thiếu
        
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
        # SPECIALIZED QUERY HANDLERS - ƯU TIÊN MULTI-HOP TRƯỚC SINGLE-HOP
        # ============================================
        # QUAN TRỌNG: Ưu tiên các câu hỏi multi-hop (cần nhiều bước suy luận) 
        # trước các câu hỏi single-hop (chỉ cần 1 bước)
        
        # ============================================
        # MULTI-HOP QUESTIONS (ƯU TIÊN CAO NHẤT)
        # ============================================
        
        # -1. Câu hỏi về năm hoạt động/phát hành/sáng tác/ra mắt/debut (1-hop, 2-hop, 3-hop)
        # Pattern: "năm hoạt động của X", "năm phát hành của X", "năm sáng tác của X", "X ra mắt vào năm nào", "X debut vào năm nào"
        is_year_question = (
            ('năm' in query_lower) and
            ('hoạt động' in query_lower or 'phát hành' in query_lower or 'thành lập' in query_lower or 'sáng tác' in query_lower or 'ra mắt' in query_lower or 'debut' in query_lower) or
            # Pattern "X debut vào năm nào" (không cần từ "năm" ở đầu)
            ('debut' in query_lower and 'năm' in query_lower) or
            ('debut' in query_lower and any(kw in query_lower for kw in ['vào năm', 'năm nào']))
        )
        
        if is_year_question:
            # Determine year type
            year_type = 'activity'  # default
            extract_first_year = False  # Flag để extract năm đầu tiên từ range
            if 'phát hành' in query_lower or 'sáng tác' in query_lower:
                year_type = 'release'
            elif 'thành lập' in query_lower:
                year_type = 'founding'
            elif 'ra mắt' in query_lower or 'debut' in query_lower:
                year_type = 'activity'  # "ra mắt"/"debut" = năm hoạt động
                extract_first_year = True  # "ra mắt"/"debut" cần lấy năm đầu tiên (ví dụ: "2016–nay" -> "2016")
            
            # 3-hop: năm hoạt động của nhóm nhạc có ca sĩ đã thể hiện ca khúc X (Song → Artist → Group → Year)
            is_song_artist_group_year_question = (
                ('bài hát' in query_lower or 'ca khúc' in query_lower) and
                ('ca sĩ' in query_lower or 'nghệ sĩ' in query_lower) and
                ('nhóm nhạc' in query_lower or 'nhóm' in query_lower) and
                ('thể hiện' in query_lower or 'trình bày' in query_lower or 'có' in query_lower)
            )
            
            if is_song_artist_group_year_question:
                # Extract song entity - ưu tiên dùng helper để extract chính xác tên bài hát
                song_entity = self._extract_song_name_from_query(query)
                
                # Nếu helper không tìm được, dùng _extract_entities_robust
                if not song_entity:
                    all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
                    for entity in all_entities:
                        if self.kg.get_entity_type(entity) == 'Song':
                            song_entity = entity
                            break
                
                if not song_entity:
                    # Không tìm thấy song entity
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                        confidence=0.0,
                        explanation=f"Không extract được song entity từ query: {query}"
                    )
                
                # Dùng tên hiển thị sạch (loại hậu tố như "(bài hát của Rosé)")
                song_display = self._normalize_entity_name(song_entity)
                
                # Step 1: Get artists
                artists = self.kg.get_song_artists(song_entity)
                if not artists:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[ReasoningStep(
                            hop_number=1,
                            operation='get_artists_from_song',
                            source_entities=[song_entity],
                            relationship='SINGS',
                            target_entities=[],
                            explanation=f"Không tìm thấy ca sĩ nào đã thể hiện {song_display}"
                        )],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về ca sĩ đã thể hiện bài hát {song_display} trong đồ thị tri thức.",
                        confidence=0.0,
                        explanation=f"1-hop: {song_display} không có quan hệ SINGS với Artist nào"
                    )
                
                steps = []
                steps.append(ReasoningStep(
                    hop_number=1,
                    operation='get_artists_from_song',
                    source_entities=[song_entity],
                    relationship='SINGS',
                    target_entities=artists,
                    explanation=f"Lấy các ca sĩ đã thể hiện {song_display}"
                ))
                
                # Step 2: Get groups
                groups = []
                for artist in artists:
                    artist_groups = self.kg.get_artist_groups(artist)
                    groups.extend(artist_groups)
                    if artist_groups:
                        steps.append(ReasoningStep(
                            hop_number=2,
                            operation='get_groups_from_artist',
                            source_entities=[artist],
                            relationship='MEMBER_OF',
                            target_entities=artist_groups,
                            explanation=f"Lấy các nhóm nhạc của {artist}"
                        ))
                
                groups = list(set(groups))
                if not groups:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=steps,
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về nhóm nhạc của các ca sĩ đã thể hiện bài hát {song_display} trong đồ thị tri thức.",
                        confidence=0.0,
                        explanation=f"2-hop: Các ca sĩ {', '.join(artists[:3])} không có quan hệ MEMBER_OF với nhóm nhạc nào"
                    )
                
                # Step 3: Get year from groups
                years = []
                for group in groups:
                    year = self.kg.extract_year_from_infobox(group, year_type)
                    if year:
                        years.append((group, year))
                
                if not years:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=steps,
                        answer_entities=groups,
                        answer_text=f"Không tìm thấy thông tin về năm hoạt động của các nhóm nhạc ({', '.join(groups[:3])}) đã thể hiện bài hát {song_display} trong đồ thị tri thức.",
                        confidence=0.3,
                        explanation=f"3-hop: Tìm thấy nhóm nhạc {', '.join(groups[:3])} nhưng không có thông tin năm hoạt động trong infobox"
                    )
                
                # Format answer - chuẩn hóa format dựa trên query
                if 'ra mắt' in query_lower or 'debut' in query_lower:
                    # Format cho câu hỏi về ra mắt/debut
                    if len(years) == 1:
                        group_display = self._normalize_entity_name(years[0][0])
                        year_str = years[0][1]
                        # Extract năm đầu tiên nếu là range
                        if '–' in year_str or '-' in year_str:
                            year_str = year_str.split('–')[0].split('-')[0]
                        answer_text = f"{group_display} ra mắt vào năm {year_str}"
                    else:
                        # Nhiều nhóm → format: "Nhóm X ra mắt vào năm Y; Nhóm Z ra mắt vào năm W"
                        group_year_pairs = []
                        for group, year in years:
                            group_display = self._normalize_entity_name(group)
                            year_str = year.split('–')[0].split('-')[0] if ('–' in year or '-' in year) else year
                            group_year_pairs.append(f"{group_display} ra mắt vào năm {year_str}")
                        answer_text = '; '.join(group_year_pairs)
                else:
                    # Format năm thành ngôn ngữ tự nhiên và ngăn cách rõ ràng
                    if len(years) == 1:
                        year_formatted = self._format_year_natural(years[0][1])
                        answer_text = f"Năm hoạt động của nhóm nhạc có ca sĩ đã thể hiện bài hát {song_display} là: {year_formatted}"
                    else:
                        # Nhiều nhóm → format: "Nhóm X: từ Y đến nay; Nhóm Z: từ W đến nay"
                        group_year_pairs = []
                        for group, year in years:
                            group_display = self._normalize_entity_name(group)
                            year_formatted = self._format_year_natural(year)
                            group_year_pairs.append(f"{group_display}: {year_formatted}")
                        answer_text = f"Năm hoạt động của nhóm nhạc có ca sĩ đã thể hiện bài hát {song_display} là: {'; '.join(group_year_pairs)}"
                
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=steps,
                    answer_entities=[group for group, _ in years],
                    answer_text=answer_text,
                    confidence=0.95,
                    explanation=f"3-hop: {song_display} → SINGS → {', '.join(artists[:3])} → MEMBER_OF → {', '.join(groups[:3])} → Year"
                )
            
            # 2-hop: năm hoạt động của nhóm nhạc đã thể hiện ca khúc X (Song → Group → Year)
            is_song_group_year_question = (
                ('bài hát' in query_lower or 'ca khúc' in query_lower) and
                ('nhóm nhạc' in query_lower or 'nhóm' in query_lower) and
                ('thể hiện' in query_lower or 'trình bày' in query_lower or 'đã' in query_lower) and
                not ('ca sĩ' in query_lower or 'nghệ sĩ' in query_lower)  # Not 3-hop
            )
            
            if is_song_group_year_question:
                # Extract song entity - ưu tiên dùng helper để extract chính xác tên bài hát
                song_entity = self._extract_song_name_from_query(query)
                
                # Nếu helper không tìm được, dùng _extract_entities_robust
                if not song_entity:
                    all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
                    for entity in all_entities:
                        if self.kg.get_entity_type(entity) == 'Song':
                            song_entity = entity
                            break
                
                if not song_entity:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                        confidence=0.0,
                        explanation=f"Không extract được song entity từ query: {query}"
                    )
                
                # Dùng tên hiển thị sạch cho bài hát
                song_display = self._normalize_entity_name(song_entity)
                
                # Step 1: Get groups
                groups = self.kg.get_song_groups(song_entity)
                if not groups:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[ReasoningStep(
                            hop_number=1,
                            operation='get_groups_from_song',
                            source_entities=[song_entity],
                            relationship='SINGS',
                            target_entities=[],
                            explanation=f"Không tìm thấy nhóm nhạc nào đã thể hiện {song_display}"
                        )],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về nhóm nhạc đã thể hiện bài hát {song_display} trong đồ thị tri thức.",
                        confidence=0.0,
                        explanation=f"1-hop: {song_display} không có quan hệ SINGS với Group nào"
                    )
                
                steps = []
                steps.append(ReasoningStep(
                    hop_number=1,
                    operation='get_groups_from_song',
                    source_entities=[song_entity],
                    relationship='SINGS',
                    target_entities=groups,
                    explanation=f"Lấy các nhóm nhạc đã thể hiện {song_display}"
                ))
                
                # Step 2: Get year from groups
                years = []
                for group in groups:
                    year = self.kg.extract_year_from_infobox(group, year_type)
                    if year:
                        years.append((group, year))
                
                if not years:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=steps,
                        answer_entities=groups,
                        answer_text=f"Không tìm thấy thông tin về năm hoạt động của các nhóm nhạc ({', '.join(groups[:3])}) đã thể hiện bài hát {song_display} trong đồ thị tri thức.",
                        confidence=0.3,
                        explanation=f"2-hop: Tìm thấy nhóm nhạc {', '.join(groups[:3])} nhưng không có thông tin năm hoạt động trong infobox"
                    )
                
                # Format năm thành ngôn ngữ tự nhiên và ngăn cách rõ ràng
                if len(years) == 1:
                    year_formatted = self._format_year_natural(years[0][1])
                    answer_text = f"Năm hoạt động của nhóm nhạc đã thể hiện ca khúc {song_display} là: {year_formatted}"
                else:
                    # Nhiều nhóm → format: "Nhóm X: năm Y; Nhóm Z: năm W"
                    group_year_pairs = []
                    for group, year in years:
                        group_display = self._normalize_entity_name(group)
                        year_formatted = self._format_year_natural(year)
                        group_year_pairs.append(f"{group_display}: {year_formatted}")
                    answer_text = f"Năm hoạt động của nhóm nhạc đã thể hiện ca khúc {song_display} là: {'; '.join(group_year_pairs)}"
                
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=steps,
                    answer_entities=[group for group, _ in years],
                    answer_text=answer_text,
                    confidence=0.95,
                    explanation=f"2-hop: {song_display} → SINGS → {', '.join(groups[:3])} → Year"
                )
            
            # 2-hop: năm hoạt động của ca sĩ thể hiện bài hát X (Song → Artist → Year)
            is_song_artist_year_question = (
                ('bài hát' in query_lower or 'ca khúc' in query_lower) and
                ('ca sĩ' in query_lower or 'nghệ sĩ' in query_lower) and
                ('thể hiện' in query_lower or 'trình bày' in query_lower or 'hát' in query_lower) and
                not ('nhóm nhạc' in query_lower or 'nhóm' in query_lower)  # Không phải 3-hop (song-artist-group-year)
            )
            
            if is_song_artist_year_question:
                # Extract song entity - ưu tiên dùng helper để extract chính xác tên bài hát
                song_entity = self._extract_song_name_from_query(query)
                
                # Nếu helper không tìm được, dùng _extract_entities_robust
                if not song_entity:
                    all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
                    for entity in all_entities:
                        if self.kg.get_entity_type(entity) == 'Song':
                            song_entity = entity
                            break
                
                if not song_entity:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                        confidence=0.0,
                        explanation=f"Không extract được song entity từ query: {query}"
                    )
                
                # Dùng tên hiển thị sạch cho bài hát
                song_display = self._normalize_entity_name(song_entity)
                
                # Step 1: Get artists
                artists = self.kg.get_song_artists(song_entity)
                if not artists:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[ReasoningStep(
                            hop_number=1,
                            operation='get_artists_from_song',
                            source_entities=[song_entity],
                            relationship='SINGS',
                            target_entities=[],
                            explanation=f"Không tìm thấy ca sĩ nào đã thể hiện {song_display}"
                        )],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về ca sĩ đã thể hiện bài hát {song_display} trong đồ thị tri thức.",
                        confidence=0.0,
                        explanation=f"1-hop: {song_display} không có quan hệ SINGS với Artist nào"
                    )
                
                steps = []
                steps.append(ReasoningStep(
                    hop_number=1,
                    operation='get_artists_from_song',
                    source_entities=[song_entity],
                    relationship='SINGS',
                    target_entities=artists,
                    explanation=f"Lấy các ca sĩ đã thể hiện {song_display}"
                ))
                
                # Step 2: Get year from artists
                years = []
                for artist in artists:
                    year = self.kg.extract_year_from_infobox(artist, year_type)
                    if year:
                        years.append((artist, year))
                
                if not years:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=steps,
                        answer_entities=artists,
                        answer_text=f"Không tìm thấy thông tin về năm hoạt động của ca sĩ ({', '.join(artists[:3])}) đã thể hiện bài hát {song_display} trong đồ thị tri thức.",
                        confidence=0.3,
                        explanation=f"2-hop: Tìm thấy ca sĩ {', '.join(artists[:3])} nhưng không có thông tin năm hoạt động trong infobox"
                    )
                
                # Format năm thành ngôn ngữ tự nhiên và ngăn cách rõ ràng
                if len(years) == 1:
                    year_formatted = self._format_year_natural(years[0][1])
                    answer_text = f"Năm hoạt động của ca sĩ thể hiện bài hát {song_display} là: {year_formatted}"
                else:
                    # Nhiều ca sĩ → format: "Ca sĩ X: năm Y; Ca sĩ Z: năm W"
                    artist_year_pairs = []
                    for artist, year in years:
                        artist_display = self._normalize_entity_name(artist)
                        year_formatted = self._format_year_natural(year)
                        artist_year_pairs.append(f"{artist_display}: {year_formatted}")
                    answer_text = f"Năm hoạt động của ca sĩ thể hiện bài hát {song_display} là: {'; '.join(artist_year_pairs)}"
                
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=steps,
                    answer_entities=[artist for artist, _ in years],
                    answer_text=answer_text,
                    confidence=0.95,
                    explanation=f"2-hop: {song_display} → SINGS → {', '.join(artists[:3])} → Year"
                )
            
            # 1-hop: năm hoạt động/phát hành của X (direct)
            # Determine expected types based on year_type
            if year_type == 'release':
                # For release year, expect Song or Album
                expected_types = ['Song', 'Album']
            elif year_type == 'activity':
                # For activity year, expect Group or Artist
                expected_types = ['Group', 'Artist']
            elif year_type == 'founding':
                # For founding year, expect Company
                expected_types = ['Company']
            else:
                expected_types = ['Group', 'Artist', 'Song', 'Album', 'Company']
            
            # Extract entity
            all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=expected_types)
            if all_entities:
                entity = all_entities[0]
                # Nếu là câu hỏi "ra mắt", extract năm đầu tiên từ range
                year = self.kg.extract_year_from_infobox(entity, year_type, extract_first_year=extract_first_year)
                if year:
                    # Format year thành ngôn ngữ tự nhiên (trừ khi là debut/ra mắt)
                    if 'ra mắt' in query_lower or 'debut' in query_lower:
                        # Với "ra mắt"/"debut", chỉ lấy năm đầu tiên, không format "từ...đến"
                        year_formatted = year.split('–')[0].split('-')[0] if ('–' in year or '-' in year) else year
                    else:
                        # Format năm thành ngôn ngữ tự nhiên (ví dụ: "2016–nay" → "từ 2016 đến nay")
                        year_formatted = self._format_year_natural(year)
                    
                    # Format answer text based on year_type
                    entity_display = self._normalize_entity_name(entity)
                    if year_type == 'release':
                        answer_text = f"Năm phát hành của {entity_display} là: {year_formatted}"
                    elif year_type == 'activity':
                        if 'ra mắt' in query_lower or 'debut' in query_lower:
                            answer_text = f"{entity_display} ra mắt vào năm {year_formatted}"
                        else:
                            answer_text = f"Năm hoạt động của {entity_display} là: {year_formatted}"
                    elif year_type == 'founding':
                        answer_text = f"Năm thành lập của {entity_display} là: {year_formatted}"
                    else:
                        answer_text = f"Năm của {entity_display} là: {year_formatted}"
                    
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[ReasoningStep(
                            hop_number=1,
                            operation='get_year_from_entity',
                            source_entities=[entity],
                            relationship='INFOBOX',
                            target_entities=[entity],
                            explanation=f"Lấy năm từ thông tin của {entity}"
                        )],
                        answer_entities=[entity],
                        answer_text=answer_text,
                        confidence=0.95,
                        explanation=f"1-hop: {entity} → Year ({year_type}) from infobox"
                    )
        
        # 0. Câu hỏi về công ty/thể loại của nhóm nhạc đã thể hiện ca khúc X (Song → Group → Company/Genre)
        # QUAN TRỌNG: Check song-group questions TRƯỚC same_company để tránh conflict
        # Pattern: "công ty của nhóm nhạc đã thể hiện ca khúc X"
        is_song_group_company_question = (
            ('bài hát' in query_lower or 'ca khúc' in query_lower or 'song' in query_lower) and
            ('nhóm nhạc' in query_lower or 'nhóm' in query_lower or 'group' in query_lower) and
            ('thể hiện' in query_lower or 'trình bày' in query_lower or 'đã' in query_lower) and
            ('công ty' in query_lower or 'company' in query_lower or 'label' in query_lower or 'hãng' in query_lower)
        )
        
        if is_song_group_company_question:
            # Extract song entity từ query
            all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
            song_entity = None
            for entity in all_entities:
                if self.kg.get_entity_type(entity) == 'Song':
                    song_entity = entity
                    break
            
            if song_entity:
                # Bước 1: Lấy groups thể hiện bài hát này
                groups = self.kg.get_song_groups(song_entity)
                if groups:
                    # Bước 2: Lấy công ty của group đầu tiên (hoặc tất cả)
                    companies = []
                    steps = []
                    steps.append(ReasoningStep(
                        hop_number=1,
                        operation='get_groups_from_song',
                        source_entities=[song_entity],
                        relationship='SINGS',
                        target_entities=groups,
                        explanation=f"Lấy các nhóm nhạc đã thể hiện {song_entity}"
                    ))
                    
                    for group in groups:
                        group_companies = self.kg.get_group_companies(group)
                        companies.extend(group_companies)
                        if group_companies:
                            steps.append(ReasoningStep(
                                hop_number=2,
                                operation='get_company_from_group',
                                source_entities=[group],
                                relationship='MANAGED_BY',
                                target_entities=group_companies,
                                explanation=f"Lấy công ty quản lý {group}"
                            ))
                    
                    companies = list(set(companies))  # Remove duplicates
                    if companies:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=companies,
                            answer_text=f"Công ty quản lý nhóm nhạc đã thể hiện ca khúc {song_entity} là: {', '.join(companies)}",
                            confidence=0.95,
                            explanation=f"2-hop: {song_entity} → SINGS → {', '.join(groups)} → MANAGED_BY → {', '.join(companies)}"
                        )
                    else:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=[],
                            answer_text=f"Không tìm thấy công ty quản lý nhóm nhạc đã thể hiện ca khúc {song_entity}",
                            confidence=0.5,
                            explanation=f"Tìm thấy nhóm nhạc {', '.join(groups)} đã thể hiện {song_entity} nhưng không có thông tin về công ty"
                        )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy nhóm nhạc nào đã thể hiện ca khúc {song_entity}",
                        confidence=0.3,
                        explanation=f"Không tìm thấy nhóm nhạc nào có quan hệ SINGS với {song_entity}"
                    )
            else:
                # Không tìm thấy song entity
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                    confidence=0.0,
                    explanation=f"Không extract được song entity từ query: {query}"
                )
        
        # 1. Câu hỏi so sánh công ty (2-hop): "BTS và SEVENTEEN có cùng công ty không"
        # QUAN TRỌNG: Check same_company SAU song-group questions để tránh conflict
        if any(kw in query_lower for kw in ['cùng công ty', 'same company', 'cùng hãng', 'cùng label', 'cùng hãng đĩa', 'cùng công ty quản lý']):
            # QUAN TRỌNG: Phải có đủ 2 entities để so sánh
            # Dùng _extract_entities_robust để đảm bảo extract đủ 2 entities với LLM fallback
            all_entities = self._extract_entities_robust(query, start_entities, min_count=2, expected_types=['Artist', 'Group'])
            
            if len(all_entities) >= 2:
                # Có đủ 2 entities → so sánh
                return self.check_same_company(all_entities[0], all_entities[1])
            elif len(all_entities) == 1:
                # Chỉ có 1 entity → không đủ để so sánh, trả về lỗi rõ ràng
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm đủ thông tin để so sánh. Chỉ tìm thấy {all_entities[0]}. Vui lòng cung cấp tên đầy đủ của cả hai nghệ sĩ/nhóm nhạc.",
                    confidence=0.0,
                    explanation=f"Chỉ extract được 1 entity: {all_entities[0]}, cần ít nhất 2 entities để so sánh công ty"
                )
            else:
                # Không tìm được entity nào
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[],
                    answer_entities=[],
                    answer_text="Không tìm thấy thông tin về các nghệ sĩ/nhóm nhạc trong câu hỏi. Vui lòng kiểm tra lại tên.",
                    confidence=0.0,
                    explanation="Không extract được entities từ query"
                )
        
        # 2. Câu hỏi so sánh nhóm nhạc (2-hop): "Lisa và Jennie có cùng nhóm nhạc không"
        if any(kw in query_lower for kw in [
            'cùng nhóm', 'cùng nhóm nhạc', 'cùng một nhóm', 'cùng một nhóm nhạc',
            'same group', 'cùng ban nhạc', 'chung nhóm', 'chung nhóm nhạc'
        ]):
            # Extract entities với LLM fallback
            all_entities = self._extract_entities_robust(query, start_entities, min_count=2, expected_types=['Artist', 'Group'])
            
            if len(all_entities) >= 2:
                return self.check_same_group(all_entities[0], all_entities[1])
            elif len(all_entities) == 1:
                # Chỉ có 1 entity → không đủ để so sánh
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm đủ thông tin để so sánh. Chỉ tìm thấy {all_entities[0]}. Vui lòng cung cấp tên đầy đủ của cả hai nghệ sĩ.",
                    confidence=0.0,
                    explanation=f"Chỉ extract được 1 entity: {all_entities[0]}, cần ít nhất 2 entities để so sánh cùng nhóm"
                )
            else:
                # Không tìm được entity nào
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.COMPARISON,
                    steps=[],
                    answer_entities=[],
                    answer_text="Không tìm thấy thông tin về các nghệ sĩ trong câu hỏi. Vui lòng kiểm tra lại tên.",
                    confidence=0.0,
                    explanation="Không extract được entities từ query"
                )
        
        # 2b. Câu hỏi về thể loại của nhóm nhạc có ca sĩ thể hiện bài hát X (Song → Artist → Group → Genre) - 3-hop
        # Pattern: "thể loại của nhóm nhạc có ca sĩ thể hiện bài hát X"
        is_song_artist_group_genre_question = (
            ('bài hát' in query_lower or 'ca khúc' in query_lower or 'song' in query_lower) and
            ('ca sĩ' in query_lower or 'nghệ sĩ' in query_lower or 'artist' in query_lower) and
            ('nhóm nhạc' in query_lower or 'nhóm' in query_lower or 'group' in query_lower) and
            ('thể hiện' in query_lower or 'trình bày' in query_lower or 'có' in query_lower) and
            ('thể loại' in query_lower or 'genre' in query_lower or 'dòng nhạc' in query_lower)
        )
        
        if is_song_artist_group_genre_question:
            # Extract song entity - ưu tiên dùng helper để extract chính xác tên bài hát
            song_entity = self._extract_song_name_from_query(query)
            
            # Nếu helper không tìm được, dùng _extract_entities_robust
            if not song_entity:
                all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
                for entity in all_entities:
                    if self.kg.get_entity_type(entity) == 'Song':
                        song_entity = entity
                        break
            
            if song_entity:
                # Normalize song name để hiển thị sạch (bỏ hậu tố)
                song_display = self._normalize_entity_name(song_entity)
                # Bước 1: Lấy artists thể hiện bài hát này (Song → Artist)
                artists = self.kg.get_song_artists(song_entity)
                if artists:
                    steps = []
                    steps.append(ReasoningStep(
                        hop_number=1,
                        operation='get_artists_from_song',
                        source_entities=[song_entity],
                        relationship='SINGS',
                        target_entities=artists,
                        explanation=f"Lấy các ca sĩ đã thể hiện {song_entity}"
                    ))
                    
                    # Bước 2: Lấy groups của các artists (Artist → Group)
                    groups = []
                    for artist in artists:
                        artist_groups = self.kg.get_artist_groups(artist)
                        groups.extend(artist_groups)
                        if artist_groups:
                            steps.append(ReasoningStep(
                                hop_number=2,
                                operation='get_groups_from_artist',
                                source_entities=[artist],
                                relationship='MEMBER_OF',
                                target_entities=artist_groups,
                                explanation=f"Lấy các nhóm nhạc của {artist}"
                            ))
                    
                    groups = list(set(groups))  # Remove duplicates
                    if groups:
                        # Bước 3: Lấy thể loại của các groups (Group → Genre)
                        genres = []
                        for group in groups:
                            group_genres = []
                            for _, target, data in self.kg.graph.out_edges(group, data=True):
                                if data.get('type') == 'IS_GENRE' and self.kg.get_entity_type(target) == 'Genre':
                                    group_genres.append(target)
                            genres.extend(group_genres)
                            if group_genres:
                                steps.append(ReasoningStep(
                                    hop_number=3,
                                    operation='get_genre_from_group',
                                    source_entities=[group],
                                    relationship='IS_GENRE',
                                    target_entities=group_genres,
                                    explanation=f"Lấy thể loại của {group}"
                                ))
                        
                        genres = list(set(genres))  # Remove duplicates
                        if genres:
                            # Clean prefix "Genre_" để output tự nhiên hơn
                            genres_clean = [g[6:] if g.startswith("Genre_") else g for g in genres]
                            return ReasoningResult(
                                query=query,
                                reasoning_type=ReasoningType.CHAIN,
                                steps=steps,
                                answer_entities=genres,
                                answer_text=f"Thể loại của nhóm nhạc có ca sĩ thể hiện bài hát {song_display} là: {', '.join(genres_clean)}",
                                confidence=0.95,
                                explanation=f"3-hop: {song_entity} → SINGS → {', '.join(artists[:3])} → MEMBER_OF → {', '.join(groups[:3])} → IS_GENRE → {', '.join(genres)}"
                            )
                        else:
                            return ReasoningResult(
                                query=query,
                                reasoning_type=ReasoningType.CHAIN,
                                steps=steps,
                                answer_entities=[],
                                answer_text=f"Không tìm thấy thể loại của nhóm nhạc có ca sĩ thể hiện bài hát {song_display}",
                                confidence=0.5,
                                explanation=f"Tìm thấy nhóm nhạc {', '.join(groups[:3])} nhưng không có thông tin về thể loại"
                            )
                    else:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=[],
                            answer_text=f"Không tìm thấy nhóm nhạc nào của ca sĩ đã thể hiện bài hát {song_display}",
                            confidence=0.4,
                            explanation=f"Tìm thấy ca sĩ {', '.join(artists[:3])} nhưng không tìm thấy nhóm nhạc"
                        )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy ca sĩ nào đã thể hiện bài hát {song_display}",
                        confidence=0.3,
                        explanation=f"Không tìm thấy ca sĩ nào có quan hệ SINGS với {song_entity}"
                    )
            else:
                # Không tìm thấy song entity
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                    confidence=0.0,
                    explanation=f"Không extract được song entity từ query: {query}"
                )
        
        # 2c1. Câu hỏi về nghề nghiệp của ca sĩ đã thể hiện bài hát X (Song → Artist → Occupation) - 2-hop
        # Pattern: "Nghề nghiệp của ca sĩ đã thể hiện bài hát X là gì?", "occupation của nghệ sĩ đã thể hiện ca khúc X"
        is_song_artist_occupation_question = (
            ('bài hát' in query_lower or 'ca khúc' in query_lower or 'song' in query_lower) and
            ('ca sĩ' in query_lower or 'nghệ sĩ' in query_lower or 'artist' in query_lower) and
            ('thể hiện' in query_lower or 'trình bày' in query_lower or 'hát' in query_lower) and
            ('nghề nghiệp' in query_lower or 'occupation' in query_lower or 'vai trò' in query_lower) and
            not ('nhóm nhạc' in query_lower or 'nhóm' in query_lower)  # Không phải song-artist-group-occupation
        )
        
        if is_song_artist_occupation_question:
            # Extract song entity - ưu tiên dùng helper để extract chính xác tên bài hát
            song_entity = self._extract_song_name_from_query(query)
            
            # Nếu helper không tìm được, dùng _extract_entities_robust
            if not song_entity:
                all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
                for entity in all_entities:
                    if self.kg.get_entity_type(entity) == 'Song':
                        song_entity = entity
                        break
            
            if song_entity:
                # Normalize song name để hiển thị sạch (bỏ hậu tố)
                song_display = self._normalize_entity_name(song_entity)
                
                # Bước 1: Lấy artists thể hiện bài hát này (Song → Artist)
                artists = self.kg.get_song_artists(song_entity)
                if artists:
                    steps = []
                    steps.append(ReasoningStep(
                        hop_number=1,
                        operation='get_artists_from_song',
                        source_entities=[song_entity],
                        relationship='SINGS',
                        target_entities=artists,
                        explanation=f"Lấy các ca sĩ đã thể hiện {song_entity}"
                    ))
                    
                    # Bước 2: Lấy nghề nghiệp (occupation) của các artists (Artist → Occupation)
                    occupations = []
                    for artist in artists:
                        # Lấy occupations của artist (dùng HAS_OCCUPATION relationship)
                        artist_occupations = []
                        for _, target, data in self.kg.graph.out_edges(artist, data=True):
                            if data.get('type') == 'HAS_OCCUPATION' and self.kg.get_entity_type(target) == 'Occupation':
                                artist_occupations.append(target)
                        occupations.extend(artist_occupations)
                        if artist_occupations:
                            steps.append(ReasoningStep(
                                hop_number=2,
                                operation='get_occupation_from_artist',
                                source_entities=[artist],
                                relationship='HAS_OCCUPATION',
                                target_entities=artist_occupations,
                                explanation=f"Lấy nghề nghiệp của {artist}"
                            ))
                    
                    occupations = list(set(occupations))  # Remove duplicates
                    if occupations:
                        # Clean prefix "Occupation_" nếu có
                        occupations_clean = [occ.replace("Occupation_", "") if occ.startswith("Occupation_") else occ for occ in occupations]
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=occupations,
                            answer_text=f"Nghề nghiệp của ca sĩ đã thể hiện bài hát {song_display} là: {', '.join(occupations_clean)}",
                            confidence=0.95,
                            explanation=f"2-hop: {song_entity} → SINGS → {', '.join(artists[:3])} → HAS_OCCUPATION → {', '.join(occupations)}"
                        )
                    else:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=[],
                            answer_text=f"Không tìm thấy nghề nghiệp của ca sĩ đã thể hiện bài hát {song_display}",
                            confidence=0.5,
                            explanation=f"Tìm thấy ca sĩ {', '.join(artists[:3])} nhưng không có thông tin về nghề nghiệp"
                        )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy ca sĩ nào đã thể hiện bài hát {song_display}",
                        confidence=0.3,
                        explanation=f"Không tìm thấy ca sĩ nào có quan hệ SINGS với {song_entity}"
                    )
            else:
                # Không tìm thấy song entity
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                    confidence=0.0,
                    explanation=f"Không extract được song entity từ query: {query}"
                )
        
        # 2b1. Câu hỏi về bài hát phát hành trong năm cụ thể: "X có những bài hát nào phát hành năm Y"
        # Ví dụ: "BTS có những bài hát nào phát hành năm 2019"
        import re
        year_pattern = r'\b(19|20)\d{2}\b'  # Pattern để tìm năm (1900-2099)
        year_match = re.search(year_pattern, query)
        
        is_songs_by_year_question = (
            ('bài hát' in query_lower or 'ca khúc' in query_lower) and
            ('phát hành' in query_lower or 'ra mắt' in query_lower) and
            'năm' in query_lower and
            year_match is not None  # Có chứa năm
        )
        
        if is_songs_by_year_question and start_entities:
            target_year = year_match.group(0)
            # Extract entity (Group hoặc Artist) từ start_entities
            group_or_artist = None
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type in ['Group', 'Artist']:
                    group_or_artist = entity
                    break
            
            if group_or_artist:
                # Lấy tất cả bài hát của entity này
                all_songs = []
                if self.kg.get_entity_type(group_or_artist) == 'Group':
                    all_songs = self.kg.get_group_songs(group_or_artist)
                elif self.kg.get_entity_type(group_or_artist) == 'Artist':
                    # Lấy bài hát từ artist (SINGS relationship)
                    for _, target, data in self.kg.graph.out_edges(group_or_artist, data=True):
                        if data.get('type') == 'SINGS' and self.kg.get_entity_type(target) == 'Song':
                            all_songs.append(target)
                
                # Filter bài hát theo năm phát hành
                songs_in_year = []
                steps = []
                steps.append(ReasoningStep(
                    hop_number=1,
                    operation='get_songs_from_entity',
                    source_entities=[group_or_artist],
                    relationship='SINGS',
                    target_entities=all_songs[:10],  # Hiển thị tối đa 10 trong step
                    explanation=f"Lấy tất cả bài hát của {group_or_artist}"
                ))
                
                for song in all_songs:
                    # Lấy năm phát hành từ infobox
                    year_str = self.kg.extract_year_from_infobox(song, 'release')
                    if year_str:
                        # Extract năm từ các format khác nhau: "2019", "10 tháng 9 năm 2019", "2019–2020"
                        # Pattern để extract toàn bộ năm (4 chữ số)
                        song_years = re.findall(r'\b(19\d{2}|20\d{2})\b', year_str)
                        if song_years:
                            # Kiểm tra xem năm target có trong danh sách năm của bài hát không
                            for song_year in song_years:
                                if song_year == target_year:
                                    songs_in_year.append(song)
                                    break
                
                if songs_in_year:
                    # Normalize tên bài hát để hiển thị sạch
                    songs_display = [self._normalize_entity_name(song) for song in songs_in_year]
                    entity_display = self._normalize_entity_name(group_or_artist)
                    
                    steps.append(ReasoningStep(
                        hop_number=2,
                        operation='filter_songs_by_year',
                        source_entities=songs_in_year,
                        relationship='HAS_YEAR',
                        target_entities=[target_year],
                        explanation=f"Lọc bài hát phát hành năm {target_year}"
                    ))
                    
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=steps,
                        answer_entities=songs_in_year,
                        answer_text=f"{entity_display} có những bài hát sau phát hành năm {target_year}: {', '.join(songs_display)}",
                        confidence=0.9,
                        explanation=f"2-hop: {group_or_artist} → SINGS → {len(all_songs)} bài hát → Filter năm {target_year} → {len(songs_in_year)} bài hát"
                    )
                else:
                    entity_display = self._normalize_entity_name(group_or_artist)
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=steps,
                        answer_entities=[],
                        answer_text=f"Không tìm thấy bài hát nào của {entity_display} phát hành năm {target_year} trong đồ thị tri thức.",
                        confidence=0.5,
                        explanation=f"Tìm thấy {len(all_songs)} bài hát của {group_or_artist} nhưng không có bài nào phát hành năm {target_year}"
                    )
        
        # 2c. Câu hỏi về thể loại của nhóm nhạc đã thể hiện ca khúc X (Song → Group → Genre) - 2-hop
        # Pattern: "thể loại của nhóm nhạc đã thể hiện ca khúc X", "genre của nhóm nhạc đã thể hiện ca khúc X"
        # QUAN TRỌNG: Check sau song_artist_group_genre_question để tránh conflict
        is_song_group_genre_question = (
            ('bài hát' in query_lower or 'ca khúc' in query_lower or 'song' in query_lower) and
            ('nhóm nhạc' in query_lower or 'nhóm' in query_lower or 'group' in query_lower) and
            ('thể hiện' in query_lower or 'trình bày' in query_lower or 'đã' in query_lower) and
            ('thể loại' in query_lower or 'genre' in query_lower or 'dòng nhạc' in query_lower) and
            not ('ca sĩ' in query_lower or 'nghệ sĩ' in query_lower or 'artist' in query_lower)  # Không phải song-artist-group-genre
        )
        
        if is_song_group_genre_question:
            # Extract song entity từ query
            all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
            song_entity = None
            for entity in all_entities:
                if self.kg.get_entity_type(entity) == 'Song':
                    song_entity = entity
                    break
            
            if song_entity:
                # Bước 1: Lấy groups thể hiện bài hát này
                groups = self.kg.get_song_groups(song_entity)
                if groups:
                    # Bước 2: Lấy thể loại của group đầu tiên (hoặc tất cả)
                    genres = []
                    steps = []
                    steps.append(ReasoningStep(
                        hop_number=1,
                        operation='get_groups_from_song',
                        source_entities=[song_entity],
                        relationship='SINGS',
                        target_entities=groups,
                        explanation=f"Lấy các nhóm nhạc đã thể hiện {song_entity}"
                    ))
                    
                    for group in groups:
                        # Lấy genres của group (dùng IS_GENRE relationship)
                        group_genres = []
                        for _, target, data in self.kg.graph.out_edges(group, data=True):
                            if data.get('type') == 'IS_GENRE' and self.kg.get_entity_type(target) == 'Genre':
                                group_genres.append(target)
                        genres.extend(group_genres)
                        if group_genres:
                            steps.append(ReasoningStep(
                                hop_number=2,
                                operation='get_genre_from_group',
                                source_entities=[group],
                                relationship='IS_GENRE',
                                target_entities=group_genres,
                                explanation=f"Lấy thể loại của {group}"
                            ))
                    
                    genres = list(set(genres))  # Remove duplicates
                    if genres:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=genres,
                            answer_text=f"Thể loại của nhóm nhạc đã thể hiện ca khúc {song_entity} là: {', '.join([g[6:] if g.startswith('Genre_') else g for g in genres])}",
                            confidence=0.95,
                            explanation=f"2-hop: {song_entity} → SINGS → {', '.join(groups)} → IS_GENRE → {', '.join([g[6:] if g.startswith('Genre_') else g for g in genres])}"
                        )
                    else:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=[],
                            answer_text=f"Không tìm thấy thể loại của nhóm nhạc đã thể hiện ca khúc {song_entity}",
                            confidence=0.5,
                            explanation=f"Tìm thấy nhóm nhạc {', '.join(groups)} đã thể hiện {song_entity} nhưng không có thông tin về thể loại"
                        )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy nhóm nhạc nào đã thể hiện ca khúc {song_entity}",
                        confidence=0.3,
                        explanation=f"Không tìm thấy nhóm nhạc nào có quan hệ SINGS với {song_entity}"
                    )
            else:
                # Không tìm thấy song entity
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                    confidence=0.0,
                    explanation=f"Không extract được song entity từ query: {query}"
                )
        
        # 2d. Câu hỏi về thể loại của nhóm nhạc đã ra mắt album X (Album → Group → Genre)
        # Pattern: "thể loại của nhóm nhạc đã ra mắt album X", "genre của nhóm nhạc đã ra mắt album X"
        is_album_group_genre_question = (
            ('album' in query_lower) and
            ('nhóm nhạc' in query_lower or 'nhóm' in query_lower or 'group' in query_lower) and
            ('ra mắt' in query_lower or 'phát hành' in query_lower or 'đã' in query_lower) and
            ('thể loại' in query_lower or 'genre' in query_lower or 'dòng nhạc' in query_lower)
        )
        
        if is_album_group_genre_question:
            # Extract album entity - ưu tiên dùng helper để extract chính xác tên album
            album_entity = self._extract_album_name_from_query(query)
            
            # Nếu helper không tìm được, dùng _extract_entities_robust
            if not album_entity:
                all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Album'])
                for entity in all_entities:
                    if self.kg.get_entity_type(entity) == 'Album':
                        album_entity = entity
                        break
            
            if album_entity:
                # Bước 1: Lấy groups ra mắt album này
                groups = self.kg.get_album_groups(album_entity)
                if groups:
                    # Bước 2: Lấy thể loại của group đầu tiên (hoặc tất cả)
                    genres = []
                    steps = []
                    steps.append(ReasoningStep(
                        hop_number=1,
                        operation='get_groups_from_album',
                        source_entities=[album_entity],
                        relationship='RELEASED',
                        target_entities=groups,
                        explanation=f"Lấy các nhóm nhạc đã ra mắt {album_entity}"
                    ))
                    
                    for group in groups:
                        # Lấy genres của group (dùng IS_GENRE relationship)
                        group_genres = []
                        for _, target, data in self.kg.graph.out_edges(group, data=True):
                            if data.get('type') == 'IS_GENRE' and self.kg.get_entity_type(target) == 'Genre':
                                group_genres.append(target)
                        genres.extend(group_genres)
                        if group_genres:
                            steps.append(ReasoningStep(
                                hop_number=2,
                                operation='get_genre_from_group',
                                source_entities=[group],
                                relationship='IS_GENRE',
                                target_entities=group_genres,
                                explanation=f"Lấy thể loại của {group}"
                            ))
                    
                    genres = list(set(genres))  # Remove duplicates
                    if genres:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=genres,
                            answer_text=f"Thể loại của nhóm nhạc đã ra mắt album {album_entity} là: {', '.join([g[6:] if g.startswith('Genre_') else g for g in genres])}",
                            confidence=0.95,
                            explanation=f"2-hop: {album_entity} → RELEASED → {', '.join(groups)} → IS_GENRE → {', '.join([g[6:] if g.startswith('Genre_') else g for g in genres])}"
                        )
                    else:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=[],
                            answer_text=f"Không tìm thấy thể loại của nhóm nhạc đã ra mắt album {album_entity}",
                            confidence=0.5,
                            explanation=f"Tìm thấy nhóm nhạc {', '.join(groups)} đã ra mắt {album_entity} nhưng không có thông tin về thể loại"
                        )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy nhóm nhạc nào đã ra mắt album {album_entity}",
                        confidence=0.3,
                        explanation=f"Không tìm thấy nhóm nhạc nào có quan hệ RELEASED với {album_entity}"
                    )
            else:
                # Không tìm thấy album entity
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm thấy thông tin về album trong câu hỏi. Vui lòng cung cấp tên album cụ thể.",
                    confidence=0.0,
                    explanation=f"Không extract được album entity từ query: {query}"
                )
        
        # 2e. Câu hỏi về nghề nghiệp của ca sĩ đã ra mắt album X (Album → Artist → Occupation)
        # Pattern: "Nghề nghiệp khác của ca sĩ đã ra mắt album X", "occupation của ca sĩ đã ra mắt album X"
        is_album_artist_occupation_question = (
            ('album' in query_lower) and
            ('ca sĩ' in query_lower or 'nghệ sĩ' in query_lower or 'artist' in query_lower) and
            ('ra mắt' in query_lower or 'phát hành' in query_lower or 'đã' in query_lower) and
            ('nghề nghiệp' in query_lower or 'occupation' in query_lower or 'vai trò' in query_lower)
        )
        
        if is_album_artist_occupation_question:
            # Extract album entity - ưu tiên dùng helper để extract chính xác tên album
            album_entity = self._extract_album_name_from_query(query)
            
            # Nếu helper không tìm được, dùng _extract_entities_robust
            if not album_entity:
                all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Album'])
                for entity in all_entities:
                    if self.kg.get_entity_type(entity) == 'Album':
                        album_entity = entity
                        break
            
            if album_entity:
                # Bước 1: Lấy artists ra mắt album này
                artists = self.kg.get_album_artists(album_entity)
                if artists:
                    # Bước 2: Lấy nghề nghiệp (occupation) của artist đầu tiên (hoặc tất cả)
                    occupations = []
                    steps = []
                    steps.append(ReasoningStep(
                        hop_number=1,
                        operation='get_artists_from_album',
                        source_entities=[album_entity],
                        relationship='RELEASED',
                        target_entities=artists,
                        explanation=f"Lấy các ca sĩ đã ra mắt {album_entity}"
                    ))
                    
                    for artist in artists:
                        # Lấy occupations của artist (dùng HAS_OCCUPATION relationship)
                        artist_occupations = []
                        for _, target, data in self.kg.graph.out_edges(artist, data=True):
                            if data.get('type') == 'HAS_OCCUPATION' and self.kg.get_entity_type(target) == 'Occupation':
                                artist_occupations.append(target)
                        occupations.extend(artist_occupations)
                        if artist_occupations:
                            steps.append(ReasoningStep(
                                hop_number=2,
                                operation='get_occupation_from_artist',
                                source_entities=[artist],
                                relationship='HAS_OCCUPATION',
                                target_entities=artist_occupations,
                                explanation=f"Lấy nghề nghiệp của {artist}"
                            ))
                    
                    occupations = list(set(occupations))  # Remove duplicates
                    if occupations:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=occupations,
                            answer_text=f"Nghề nghiệp của ca sĩ đã ra mắt album {album_entity} là: {', '.join(occupations)}",
                            confidence=0.95,
                            explanation=f"2-hop: {album_entity} → RELEASED → {', '.join(artists)} → HAS_OCCUPATION → {', '.join(occupations)}"
                        )
                    else:
                        return ReasoningResult(
                            query=query,
                            reasoning_type=ReasoningType.CHAIN,
                            steps=steps,
                            answer_entities=[],
                            answer_text=f"Không tìm thấy nghề nghiệp của ca sĩ đã ra mắt album {album_entity}",
                            confidence=0.5,
                            explanation=f"Tìm thấy ca sĩ {', '.join(artists)} đã ra mắt {album_entity} nhưng không có thông tin về nghề nghiệp"
                        )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy ca sĩ nào đã ra mắt album {album_entity}",
                        confidence=0.3,
                        explanation=f"Không tìm thấy ca sĩ nào có quan hệ RELEASED với {album_entity}"
                    )
            else:
                # Không tìm thấy album entity
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm thấy thông tin về album trong câu hỏi. Vui lòng cung cấp tên album cụ thể.",
                    confidence=0.0,
                    explanation=f"Không extract được album entity từ query: {query}"
                )
        
        # 3. Câu hỏi 3-hop: Song → Artist → Group → Company hoặc các chuỗi 3-hop khác
        # Pattern: "qua... rồi...", "thông qua... sau đó...", "(3-hop)", "bài hát... công ty..."
        is_three_hop_question = (
            ('qua' in query_lower and 'rồi' in query_lower) or
            ('thông qua' in query_lower and 'sau đó' in query_lower) or
            '(3-hop)' in query_lower or
            ('bài hát' in query_lower and ('công ty' in query_lower or 'label' in query_lower)) or
            ('qua' in query_lower and 'nhóm' in query_lower and 'công ty' in query_lower) or
            ('album' in query_lower and 'bài hát' in query_lower and 'nhóm' in query_lower)
        )
        
        if is_three_hop_question:
            # Extract entities với LLM fallback để đảm bảo có đủ entities cho 3-hop chain
            all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=[])
            
            if len(all_entities) >= 1:
                # Sử dụng chain reasoning với max_hops=3 để tìm đường đi 3-hop
                return self._chain_reasoning(query, all_entities, max_hops=3)
            else:
                # Nếu không extract được entity, vẫn thử chain reasoning với start_entities
                if start_entities:
                    return self._chain_reasoning(query, start_entities, max_hops=3)
        
        # ============================================
        # SINGLE-HOP QUESTIONS (ƯU TIÊN THẤP HƠN)
        # ============================================
        
        # 3a. Câu hỏi Yes/No về membership: "Jungkook có phải thành viên BTS không?" hoặc "BTS có thành viên V không?" hoặc "IU có thuộc blackpink không?"
        # QUAN TRỌNG: Check membership questions TRƯỚC comparison để tránh nhầm lẫn
        if any(kw in query_lower for kw in ['có phải', 'phải', 'là thành viên', 'is a member', 'belongs to', 'có thành viên', 'có thuộc']):
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
            
            # Pattern 2: "X có phải thành viên của Y không?" hoặc "X có thuộc Y không?" (artist is member of group)
            # Extract artist and group names từ query với LLM fallback
            # QUAN TRỌNG: Với "có thuộc", có thể chỉ có 1 entity (artist), cần extract group từ query
            min_count = 1 if 'có thuộc' in query_lower else 2
            all_entities = self._extract_entities_robust(query, start_entities, min_count=min_count, expected_types=['Artist', 'Group'])
            
            artist_entity = None
            group_entity = None
            
            for entity in all_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Artist' and not artist_entity:
                    artist_entity = entity
                elif entity_type == 'Group' and not group_entity:
                    group_entity = entity
            
            # Nếu chỉ có artist và không có group, thử extract group từ query
            if artist_entity and not group_entity:
                # Thử extract group name từ query (sau "có thuộc" hoặc "thuộc")
                group_extracted = self._extract_group_name_from_query(query)
                if group_extracted:
                    group_entity = group_extracted
                else:
                    # Fallback: extract tất cả entities và tìm group
                    all_entities_fallback = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Group'])
                    for entity in all_entities_fallback:
                        if self.kg.get_entity_type(entity) == 'Group':
                            group_entity = entity
                            break
            
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
            # QUAN TRỌNG: Nếu không extract được đầy đủ artist và group, KHÔNG trả lời mà để LLM xử lý
            # (không return gì ở đây, sẽ fallback sang LLM)
        
        # 1b. Câu hỏi về thành viên: "BTS có bao nhiêu thành viên", "Thành viên của BTS", "Ai là thành viên"
        # Pattern: "Ai là thành viên", "Who are members", "thành viên của X", "members of X", "có bao nhiêu/mấy thành viên"
        is_list_members_question = any(kw in query_lower for kw in [
            'ai là thành viên', 'who are', 'thành viên của', 'members of', 
            'thành viên nhóm', 'thành viên ban nhạc', 'có những thành viên', 'có các thành viên',
            'bao nhiêu thành viên', 'mấy thành viên', 'có mấy thành viên'
        ]) and 'có phải' not in query_lower and 'không' not in query_lower
        
        if is_list_members_question:
            # Tìm group entity từ start_entities hoặc extract từ query
            group_entity = None
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    group_entity = entity
                    break
            
            # Nếu không tìm được từ start_entities, extract từ query với robust method
            if not group_entity:
                all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Group'])
                for e in all_entities:
                    if self.kg.get_entity_type(e) == 'Group':
                        group_entity = e
                        break
            
            if group_entity:
                return self.get_group_members(group_entity)
        
        # Fallback: Câu hỏi về thành viên (không phải Yes/No)
        if any(kw in query_lower for kw in ['thành viên', 'members', 'member']) and 'có phải' not in query_lower and 'không' not in query_lower:
            # Extract group entity với LLM fallback
            all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Group'])
            
            for entity in all_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_group_members(entity)
        
        # 1c. Câu hỏi về nhóm nhạc của artist: "Lisa thuộc nhóm nhạc nào", "Nhóm nào có Lisa"
        if any(kw in query_lower for kw in ['thuộc nhóm', 'thuộc nhóm nhạc', 'nhóm nào', 'nhóm nhạc nào', 'belongs to group', 'group of']):
            # Extract artist entity với LLM fallback
            all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Artist'])
            
            artist_entity = None
            for entity in all_entities:
                if self.kg.get_entity_type(entity) == 'Artist':
                    artist_entity = entity
                    break
            
            if artist_entity:
                return self.get_artist_groups(artist_entity)
        
        # 4. Câu hỏi về công ty (2-hop single entity): "Công ty nào quản lý BTS", "BLACKPINK thuộc công ty nào"
        # QUAN TRỌNG: Chỉ xử lý nếu KHÔNG phải same_company question (để tránh conflict)
        # Đặt SAU same_company check để ưu tiên same_company
        is_company_question = any(kw in query_lower for kw in ['công ty', 'company', 'label', 'hãng', 'quản lý'])
        is_same_company_check = any(kw in query_lower for kw in ['cùng công ty', 'same company', 'cùng hãng', 'cùng label', 'cùng hãng đĩa', 'cùng công ty quản lý'])
        
        if is_company_question and not is_same_company_check:
            # Extract entities với LLM fallback
            all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Group', 'Artist'])
            
            for entity in all_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Group':
                    return self.get_company_of_group(entity)
                elif entity_type == 'Artist':
                    return self.get_artist_company(entity)
                
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
        
        # 15. Câu hỏi về bài hát trong album: "Album này có bài hát nào", "kể tên bài hát có trong album X"
        is_songs_in_album_question = (
            (any(kw in query_lower for kw in ['bài hát trong album', 'songs in album', 'track trong', 'bài hát có trong album']) or
             ('kể tên' in query_lower and 'bài hát' in query_lower)) and
            'album nào' not in query_lower  # Không phải "bài hát trong album nào"
        )
        
        if is_songs_in_album_question:
            # Tìm album entity từ start_entities hoặc extract từ query
            album_entity = None
            for entity in start_entities:
                entity_type = self.kg.get_entity_type(entity)
                if entity_type == 'Album':
                    album_entity = entity
                    break
            
            # Nếu không tìm được từ start_entities, thử extract từ query
            if not album_entity:
                album_entity = self._extract_album_name_from_query(query)
                if not album_entity:
                    # Fallback: extract tất cả entities và tìm album
                    all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Album'])
                    for entity in all_entities:
                        if self.kg.get_entity_type(entity) == 'Album':
                            album_entity = entity
                            break
            
            if album_entity:
                return self.get_album_songs(album_entity)
        
        # 16. Câu hỏi về album chứa bài hát: "Bài hát X thuộc album nào?", "Bài hát X nằm trong album nào?", "X thuộc album nào?"
        is_song_in_which_album_question = (
            ('nằm trong album nào' in query_lower or 'thuộc album nào' in query_lower or 
             'trong album nào' in query_lower or 'ở album nào' in query_lower) and
            (('bài hát' in query_lower or 'ca khúc' in query_lower or 'bài' in query_lower) or
             # Nếu không có từ khóa "bài hát", kiểm tra xem có extract được song entity không
             any(self.kg.get_entity_type(e) == 'Song' for e in start_entities) if start_entities else False)
        )
        
        if is_song_in_which_album_question:
            # Extract song entity - ưu tiên dùng helper để extract chính xác tên bài hát
            song_entity = self._extract_song_name_from_query(query)
            
            # Nếu helper không tìm được, dùng _extract_entities_robust
            if not song_entity:
                all_entities = self._extract_entities_robust(query, start_entities, min_count=1, expected_types=['Song'])
                for entity in all_entities:
                    if self.kg.get_entity_type(entity) == 'Song':
                        song_entity = entity
                        break
            
            if song_entity:
                # Normalize song name để hiển thị sạch
                song_display = self._normalize_entity_name(song_entity)
                
                # Lấy albums chứa bài hát này
                albums = self.kg.get_song_albums(song_entity)
                
                if albums:
                    # Ưu tiên album đầu tiên (hoặc có thể lấy tất cả)
                    album = albums[0]
                    album_display = self._normalize_entity_name(album)
                    
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[ReasoningStep(
                            hop_number=1,
                            operation='get_album_from_song',
                            source_entities=[song_entity],
                            relationship='CONTAINS',
                            target_entities=[album],
                            explanation=f"Tìm album chứa bài hát {song_entity}"
                        )],
                        answer_entities=[album],
                        answer_text=f"Bài hát '{song_display}' nằm trong album: {album_display}",
                        confidence=0.95,
                        explanation=f"1-hop: {song_entity} ← CONTAINS ← {album}"
                    )
                else:
                    return ReasoningResult(
                        query=query,
                        reasoning_type=ReasoningType.CHAIN,
                        steps=[],
                        answer_entities=[],
                        answer_text=f"Không tìm thấy thông tin về album chứa bài hát '{song_display}' trong đồ thị tri thức.",
                        confidence=0.3,
                        explanation=f"Không tìm thấy album nào có quan hệ CONTAINS với {song_entity}"
                    )
            else:
                # Không tìm thấy song entity
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text=f"Không tìm thấy thông tin về bài hát trong câu hỏi. Vui lòng cung cấp tên bài hát cụ thể.",
                    confidence=0.0,
                    explanation=f"Không extract được song entity từ query: {query}"
                )
                    
        # ============================================
        # FALLBACK - LLM-based understanding khi không có pattern khớp
        # ============================================
        # Nếu không có pattern nào khớp, dùng LLM để hiểu query và đề xuất reasoning path
        if self.graph_rag and self.graph_rag.llm_for_understanding:
            # Sử dụng LLM để hiểu query: intent, entities, relations, multi-hop depth
            llm_understanding = self._understand_query_with_llm_for_reasoning(query, start_entities)
            
            if llm_understanding:
                # LLM đã hiểu được query, thực hiện reasoning dựa trên thông tin từ LLM
                return self._execute_reasoning_from_llm_understanding(query, llm_understanding, max_hops)
        
        # Fallback tiếp: pattern-based reasoning type detection (nếu LLM không available)
        reasoning_type = self._detect_reasoning_type(query)
        
        # Execute appropriate reasoning strategy
        if reasoning_type == ReasoningType.CHAIN:
            return self._chain_reasoning(query, start_entities, max_hops)
        elif reasoning_type == ReasoningType.AGGREGATION:
            return self._aggregation_reasoning(query, start_entities, max_hops)
        elif reasoning_type == ReasoningType.COMPARISON:
            # QUAN TRỌNG: Chỉ so sánh khi có đủ 2 entities, nếu không thì để LLM xử lý
            if len(start_entities) >= 2:
                comparison_result = self._comparison_reasoning(query, start_entities)
                # Nếu comparison_result là None (không đủ entities), không return, để LLM xử lý
                if comparison_result:
                    return comparison_result
            # Nếu không đủ entities hoặc comparison_result là None, không trả lời comparison mà để LLM xử lý
            # (không return gì, sẽ tiếp tục fallback)
        elif reasoning_type == ReasoningType.INTERSECTION:
            return self._intersection_reasoning(query, start_entities)
        else:
            # Final fallback: chain reasoning với entities hiện có
            if start_entities:
                return self._chain_reasoning(query, start_entities, max_hops)
            else:
                return ReasoningResult(
                    query=query,
                    reasoning_type=ReasoningType.CHAIN,
                    steps=[],
                    answer_entities=[],
                    answer_text="Không thể hiểu câu hỏi và không tìm thấy entities để suy luận. Vui lòng cung cấp thêm thông tin.",
                    confidence=0.0,
                    explanation="No pattern matched and no entities found"
                )
            
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
    
    def _understand_query_with_llm_for_reasoning(self, query: str, start_entities: List[str]) -> Optional[Dict]:
        """
        Sử dụng LLM để hiểu query và đề xuất reasoning strategy khi không có pattern khớp.
        
        Args:
            query: User query
            start_entities: Entities đã có sẵn
            
        Returns:
            Dict chứa thông tin từ LLM understanding: {
                'intent': str,
                'entities': List[str],
                'relations': List[str],
                'multi_hop_depth': int,
                'target_type': Optional[str],
                'target_relationship': Optional[str]
            }
        """
        if not self.graph_rag or not self.graph_rag.llm_for_understanding:
            return None
        
        try:
            # Sử dụng method có sẵn từ GraphRAG để extract entities và metadata
            llm_entities = self.graph_rag._extract_entities_with_llm(query)
            
            if not llm_entities:
                return None
            
            # Extract metadata từ LLM entities (đã được thêm vào trong _extract_entities_with_llm)
            # Lấy từ entity đầu tiên (thường tất cả đều có cùng metadata)
            first_entity = llm_entities[0] if llm_entities else {}
            intent = first_entity.get('intent', 'multi_hop')
            relations = first_entity.get('relations', [])
            multi_hop_depth = first_entity.get('multi_hop_depth', 2)
            
            # Collect all entity IDs from LLM
            llm_entity_ids = []
            for llm_e in llm_entities:
                entity_id = llm_e.get('text', '')
                if entity_id:
                    # Validate entity exists in KG
                    entity_data = self.kg.get_entity(entity_id)
                    if entity_data:
                        llm_entity_ids.append(entity_id)
            
            # Combine với start_entities
            all_entities = list(set(start_entities + llm_entity_ids))
            
            # Detect target type và relationship từ query
            target_type = self._detect_target_type_from_intent(intent, query.lower())
            target_relationship = relations[0] if relations else None
            
            return {
                'intent': intent,
                'entities': all_entities,
                'relations': relations,
                'multi_hop_depth': min(multi_hop_depth, 3),  # Limit to max 3 hops
                'target_type': target_type,
                'target_relationship': target_relationship
            }
        except Exception as e:
            # Nếu LLM fail, return None để fallback về pattern-based
            return None
    
    def _detect_target_type_from_intent(self, intent: str, query_lower: str) -> Optional[str]:
        """Detect target entity type từ intent và query."""
        # Map intent và keywords to entity types
        if 'company' in intent or 'công ty' in query_lower or 'hãng' in query_lower:
            return 'Company'
        elif 'group' in intent or 'nhóm' in query_lower:
            return 'Group'
        elif 'artist' in intent or 'nghệ sĩ' in query_lower or 'ca sĩ' in query_lower:
            return 'Artist'
        elif 'song' in intent or 'bài hát' in query_lower or 'ca khúc' in query_lower:
            return 'Song'
        elif 'album' in intent:
            return 'Album'
        elif 'genre' in intent or 'thể loại' in query_lower:
            return 'Genre'
        elif 'occupation' in intent or 'nghề nghiệp' in query_lower:
            return 'Occupation'
        return None
    
    def _execute_reasoning_from_llm_understanding(
        self,
        query: str,
        llm_understanding: Dict,
        max_hops: int
    ) -> ReasoningResult:
        """
        Thực hiện reasoning dựa trên thông tin từ LLM understanding.
        
        Args:
            query: Original query
            llm_understanding: Dict từ _understand_query_with_llm_for_reasoning
            max_hops: Maximum hops
            
        Returns:
            ReasoningResult
        """
        entities = llm_understanding.get('entities', [])
        intent = llm_understanding.get('intent', 'multi_hop')
        multi_hop_depth = llm_understanding.get('multi_hop_depth', 2)
        target_type = llm_understanding.get('target_type')
        target_relationship = llm_understanding.get('target_relationship')
        
        if not entities:
            return ReasoningResult(
                query=query,
                reasoning_type=ReasoningType.CHAIN,
                steps=[],
                answer_entities=[],
                answer_text="Không thể trích xuất entities từ câu hỏi.",
                confidence=0.0,
                explanation="No entities extracted by LLM"
            )
        
        # Điều chỉnh max_hops dựa trên LLM understanding
        effective_max_hops = min(multi_hop_depth, max_hops)
        
        # Xác định reasoning type từ intent
        if 'comparison' in intent or 'same' in intent:
            if len(entities) >= 2:
                return self._comparison_reasoning(query, entities)
            else:
                # Fallback to chain reasoning
                return self._chain_reasoning_with_targets(query, entities, effective_max_hops, target_type, target_relationship)
        elif 'intersection' in intent or 'common' in intent:
            if len(entities) >= 2:
                return self._intersection_reasoning(query, entities)
            else:
                return self._chain_reasoning_with_targets(query, entities, effective_max_hops, target_type, target_relationship)
        elif 'aggregation' in intent or 'list' in intent:
            return self._aggregation_reasoning(query, entities, effective_max_hops)
        else:
            # Default: chain reasoning với target hints từ LLM
            return self._chain_reasoning_with_targets(query, entities, effective_max_hops, target_type, target_relationship)
    
    def _chain_reasoning_with_targets(
        self,
        query: str,
        start_entities: List[str],
        max_hops: int,
        target_type: Optional[str] = None,
        target_relationship: Optional[str] = None
    ) -> ReasoningResult:
        """
        Chain reasoning với target type và relationship hints từ LLM.
        Tương tự _chain_reasoning nhưng có thêm hints để filter chính xác hơn.
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
        
        for hop in range(max_hops):
            next_entities = set()
            relationship_types = set()
            
            for entity in current_entities[:10]:  # Limit for performance
                neighbors = self.kg.get_neighbors(entity)
                
                for neighbor, rel_type, direction in neighbors:
                    if neighbor in visited:
                        continue
                    
                    # Filter by target relationship nếu có
                    if target_relationship and rel_type != target_relationship:
                        continue
                    
                    # Filter by target type nếu có
                    if target_type:
                        neighbor_type = self.kg.get_entity_type(neighbor)
                        if neighbor_type != target_type:
                            continue
                    
                    next_entities.add(neighbor)
                    visited.add(neighbor)
                    relationship_types.add(rel_type)
            
            if not next_entities:
                break
            
            reasoning_step = ReasoningStep(
                hop_number=hop + 1,
                operation='traverse',
                source_entities=list(current_entities),
                relationship=', '.join(sorted(relationship_types)) if relationship_types else 'various',
                target_entities=list(next_entities)[:20],
                explanation=f"Hop {hop + 1}: Từ {len(current_entities)} entities, tìm thấy {len(next_entities)} entities liên quan"
            )
            steps.append(reasoning_step)
            current_entities = list(next_entities)
        
        # Nếu có target type, ưu tiên entities có type đó
        final_entities = list(current_entities)
        if target_type:
            filtered = [e for e in final_entities if self.kg.get_entity_type(e) == target_type]
            if filtered:
                final_entities = filtered
        
        return ReasoningResult(
            query=query,
            reasoning_type=ReasoningType.CHAIN,
            steps=steps,
            answer_entities=final_entities[:20],
            answer_text=f"Tìm thấy {len(final_entities)} entities liên quan: {', '.join(final_entities[:10])}{'...' if len(final_entities) > 10 else ''}",
            confidence=0.8 if final_entities else 0.3,
            explanation=f"LLM-guided reasoning: {len(steps)} hops, target: {target_type or 'any'}, relationship: {target_relationship or 'any'}"
        )
        
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
        
        # QUAN TRỌNG: Kiểm tra đủ entities và đúng loại trước khi so sánh
        if len(entities) < 2:
            # Nếu không đủ entities, return None để LLM xử lý (không trả lời sai)
            return None
            
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
        
        # Helper function để clean company name (loại bỏ prefix "Company_")
        def _clean_company_name(name):
            if not name:
                return "không rõ"
            # Loại bỏ prefix "Company_" nếu có
            if name.startswith("Company_"):
                return name[8:]  # Bỏ "Company_"
            return name
        
        # Clean company names cho output tự nhiên
        company1_clean = _clean_company_name(company1)
        company2_clean = _clean_company_name(company2)
        
        # Generate answer
        if 'cùng công ty' in query.lower() or 'same company' in query.lower() or 'cùng hãng' in query.lower():
            if same_company:
                answer_text = f"Có, {entity1} và {entity2} cùng thuộc công ty {company1_clean}"
            else:
                answer_text = f"Không, {entity1} ({company1_clean}) và {entity2} ({company2_clean}) khác công ty"
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
        
        # Normalize entity names for display
        group_display = self._normalize_entity_name(group_name)
        members_display = [self._normalize_entity_name(m) for m in members]
        
        return ReasoningResult(
            query=f"Thành viên của {group_name}",
            reasoning_type=ReasoningType.CHAIN,
            steps=[step],
            answer_entities=members,
            answer_text=f"{group_display} có {len(members)} thành viên: {', '.join(members_display)}",
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
        """Get company of an artist (1-hop: Artist → Company, or 2-hop: Artist → Group → Company)."""
        steps = []
        companies = []
        
        # Step 1: Check direct relationship Artist → Company (1-hop)
        direct_companies = self.kg.get_artist_companies(artist_name)
        if direct_companies:
            companies.extend(direct_companies)
            steps.append(ReasoningStep(
                hop_number=1,
                operation='get_direct_companies',
                source_entities=[artist_name],
                relationship='MANAGED_BY',
                target_entities=direct_companies,
                explanation=f"Lấy công ty quản lý trực tiếp của {artist_name}"
            ))
        
        # Step 2: Get companies through groups (2-hop: Artist → Group → Company)
        groups = self.kg.get_artist_groups(artist_name)
        if groups:
            steps.append(ReasoningStep(
                hop_number=1 if not direct_companies else 2,
                operation='get_groups',
                source_entities=[artist_name],
                relationship='MEMBER_OF',
                target_entities=groups,
                explanation=f"Lấy nhóm nhạc của {artist_name}"
            ))
            
            # Get companies of those groups
            group_companies = []
            for group in groups:
                group_company_list = self.kg.get_group_companies(group)
                group_companies.extend(group_company_list)
            
            if group_companies:
                companies.extend(group_companies)
                steps.append(ReasoningStep(
                    hop_number=2 if not direct_companies else 3,
                    operation='get_companies',
                    source_entities=groups,
                    relationship='MANAGED_BY',
                    target_entities=list(set(group_companies)),
                    explanation=f"Lấy công ty quản lý các nhóm"
                ))
        
        unique_companies = list(set(companies))
        hops = 1 if direct_companies else (2 if groups else 0)
        
        return ReasoningResult(
            query=f"Công ty của {artist_name}",
            reasoning_type=ReasoningType.CHAIN,
            steps=steps,
            answer_entities=unique_companies,
            answer_text=f"{artist_name} thuộc công ty: {', '.join(unique_companies)}" if unique_companies else f"Không tìm thấy công ty của {artist_name}",
            confidence=0.9 if unique_companies else 0.0,
            explanation=f"{hops}-hop: {artist_name} → MANAGED_BY → Company" if direct_companies else f"{hops}-hop: {artist_name} → MEMBER_OF → Group → MANAGED_BY → Company"
        )
    
    def _extract_entities_with_llm_fallback(self, query: str, existing_entities: List[str]) -> List[str]:
        """
        Dùng LLM để extract thêm entities khi rule-based không đủ.
        Đảm bảo không trùng với existing_entities.
        
        Args:
            query: User query
            existing_entities: List entities đã extract được (không trùng)
            
        Returns:
            List entities mới từ LLM (không trùng với existing_entities)
        """
        if not self.graph_rag or not self.graph_rag.llm_for_understanding:
            return []
        
        try:
            # Gọi LLM để extract entities
            llm_entities = self.graph_rag._extract_entities_with_llm(query)
            new_entities = []
            existing_lower = {e.lower() for e in existing_entities}
            
            for llm_e in llm_entities:
                entity_id = llm_e.get('text', '')
                if entity_id and entity_id.lower() not in existing_lower:
                    # Validate với KG
                    entity_data = self.kg.get_entity(entity_id)
                    if entity_data:
                        new_entities.append(entity_id)
                        existing_lower.add(entity_id.lower())  # Tránh trùng
            
            return new_entities
        except Exception as e:
            # Nếu LLM fail, return empty list
            return []
    
    def _extract_entities_robust(self, query: str, start_entities: List[str], min_count: int = 1, expected_types: Optional[List[str]] = None) -> List[str]:
        """
        Extract entities từ query một cách robust với LLM fallback.
        Áp dụng cho TẤT CẢ các usecase.
        
        Args:
            query: User query
            start_entities: Entities đã có sẵn
            min_count: Số lượng entities tối thiểu cần extract
            expected_types: Loại entities mong đợi (Artist, Group, Company, etc.) - None = tất cả
            
        Returns:
            List entities đã extract (không trùng, đã validate với KG)
        """
        # 1. Bắt đầu với start_entities
        all_entities = list(start_entities)
        # Track normalized names để tránh duplicate (ví dụ: "Rosé" và "Rosé (ca sĩ)" → chỉ giữ 1)
        normalized_seen = set()
        for e in start_entities:
            normalized = self._normalize_entity_name(e).lower()
            normalized_seen.add(normalized)
        
        # 2. Rule-based extraction từ query
        extracted = self._extract_entities_from_query(query, expected_types)
        # QUAN TRỌNG: Sắp xếp extracted entities theo độ dài tên (dài trước) để ưu tiên full names
        # Ví dụ: "Yoo Jeong-yeon" sẽ được thêm trước "Yoo"
        extracted_sorted = sorted(extracted, key=lambda x: len(self._normalize_entity_name(x)), reverse=True)
        # Combine với start_entities (không trùng, normalize để tránh "Rosé" và "Rosé (ca sĩ)")
        for e in extracted_sorted:
            normalized = self._normalize_entity_name(e).lower()
            # Check duplicate bằng normalized name
            if normalized not in normalized_seen:
                # QUAN TRỌNG: Nếu đã có entity với tên ngắn hơn (single word) và entity mới là full name (multi-word),
                # thì loại bỏ entity ngắn hơn và thêm entity dài hơn
                # Ví dụ: nếu đã có "Yoo" và bây giờ có "Yoo Jeong-yeon", loại bỏ "Yoo" và thêm "Yoo Jeong-yeon"
                normalized_words = normalized.split()
                if len(normalized_words) >= 2:  # Entity mới là multi-word
                    # Tìm và loại bỏ các entity ngắn hơn có cùng từ đầu
                    to_remove = []
                    for existing_e in all_entities:
                        existing_normalized = self._normalize_entity_name(existing_e).lower()
                        existing_words = existing_normalized.split()
                        # Nếu existing entity là single word và là một phần của entity mới
                        if len(existing_words) == 1 and existing_words[0] in normalized_words:
                            to_remove.append(existing_e)
                    # Loại bỏ các entity ngắn hơn
                    for e_to_remove in to_remove:
                        all_entities.remove(e_to_remove)
                        removed_normalized = self._normalize_entity_name(e_to_remove).lower()
                        normalized_seen.discard(removed_normalized)
                
                # Filter theo expected_types nếu có
                if expected_types:
                    entity_type = self.kg.get_entity_type(e)
                    if entity_type not in expected_types:
                        continue
                all_entities.append(e)
                normalized_seen.add(normalized)
        
        # 3. Nếu vẫn thiếu, dùng LLM để vá
        if len(all_entities) < min_count:
            llm_entities = self._extract_entities_with_llm_fallback(query, all_entities)
            for e in llm_entities:
                normalized = self._normalize_entity_name(e).lower()
                # Check duplicate bằng normalized name
                if normalized not in normalized_seen:
                    # Filter theo expected_types nếu có
                    if expected_types:
                        entity_type = self.kg.get_entity_type(e)
                        if entity_type not in expected_types:
                            continue
                    all_entities.append(e)
                    normalized_seen.add(normalized)
        
        return all_entities
    
    def _extract_song_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract song name from query for song-related questions.
        Returns the song name if found in Knowledge Graph, None otherwise.
        """
        import re
        
        # Pattern để extract tên bài hát từ query - ưu tiên patterns cụ thể trước
        patterns = [
            # Pattern với quotes - ưu tiên cao nhất
            r'bài hát\s+["\']([^"\']+)["\']',  # Bài hát "Name"
            r'ca khúc\s+["\']([^"\']+)["\']',  # Ca khúc "Name"
            # Pattern "bài hát X thuộc album nào" - ưu tiên cao để extract chính xác
            r'bài hát\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:thuộc|nằm trong|trong|ở)\s+album)',  # Bài hát Name thuộc/nằm trong album
            r'ca khúc\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:thuộc|nằm trong|trong|ở)\s+album)',  # Ca khúc Name thuộc/nằm trong album
            # Pattern với "có ca sĩ thể hiện bài hát X" hoặc "nhóm nhạc có ca sĩ thể hiện bài hát X"
            r'có\s+ca sĩ\s+thể hiện\s+(?:bài hát|ca khúc)\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+|$)',  # có ca sĩ thể hiện bài hát Name
            r'nhóm nhạc\s+có\s+ca sĩ\s+thể hiện\s+(?:bài hát|ca khúc)\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+|$)',  # nhóm nhạc có ca sĩ thể hiện bài hát Name
            # Pattern với "thể hiện bài hát"
            r'thể hiện\s+(?:bài hát|ca khúc)\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+|$)',  # thể hiện bài hát Name
            # Pattern "thể loại của nhóm nhạc có ca sĩ thể hiện ca khúc X"
            r'thể loại\s+của\s+nhóm nhạc\s+có\s+ca sĩ\s+thể hiện\s+(?:bài hát|ca khúc)\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+|$)',  # thể loại của nhóm nhạc có ca sĩ thể hiện ca khúc Name
            # Pattern với từ khóa cụ thể sau tên bài hát
            r'bài hát\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+là\s+|$)',  # Bài hát Name là hoặc end
            r'ca khúc\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+là\s+|$)',  # Ca khúc Name là hoặc end
            # Pattern tổng quát hơn - tìm từ sau "bài hát" hoặc "ca khúc" đến cuối câu hoặc đến từ khóa
            r'(?:bài hát|ca khúc)\s+([A-Z][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:của|do|thuộc|là)|$)',  # Bài hát Name (của/do/thuộc/là) hoặc end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                song_name = match.group(1).strip().rstrip('.,;:!?').strip()
                if not song_name:
                    continue
                    
                # Thử tìm trong KG với các biến thể
                variants = [
                    song_name,
                    song_name + " (bài hát)",
                    song_name.replace(":", " -"),
                    song_name.replace(" - ", ": "),
                ]
                for variant in variants:
                    entity_data = self.kg.get_entity(variant)
                    if entity_data and entity_data.get('label') == 'Song':
                        return variant
                
                # Thử tìm case-insensitive với normalize
                song_name_lower = self._normalize_entity_name(song_name).lower()
                for node, data in self.kg.graph.nodes(data=True):
                    if data.get('label') == 'Song':
                        base_name = self._normalize_entity_name(node)
                        base_name_lower = base_name.lower()
                        # Exact match hoặc starts with
                        if base_name_lower == song_name_lower or base_name_lower.startswith(song_name_lower):
                            # Ưu tiên exact match
                            if base_name_lower == song_name_lower:
                                return node
                            # Nếu không có exact match, giữ lại để kiểm tra tiếp
                            elif not any(self._normalize_entity_name(n).lower() == song_name_lower 
                                       for n, d in self.kg.graph.nodes(data=True) if d.get('label') == 'Song'):
                                return node
        
        return None
    
    def _extract_album_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract album name from query for album-related questions.
        Returns the album name if found in Knowledge Graph, None otherwise.
        Verifies entity type is Album to avoid confusion with other types.
        """
        import re
        
        # Pattern để extract tên album từ query
        patterns = [
            # Pattern với quotes
            r'album\s+["\']([^"\']+)["\']',  # Album "Name"
            # Pattern với từ khóa cụ thể
            r'album\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+thuộc|\s+của|\s+do|\s+là|$)',  # Album Name thuộc/của/do/là
            r'album\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+ra mắt|\s+phát hành|$)',  # Album Name ra mắt/phát hành
            # Pattern "nhóm nhạc đã ra mắt album X"
            r'(?:nhóm nhạc|nhóm)\s+(?:đã\s+)?(?:ra mắt|phát hành)\s+album\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+|$)',  # nhóm nhạc đã ra mắt album Name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                album_name = match.group(1).strip().rstrip('.,;:!?').strip()
                if not album_name:
                    continue
                    
                # Thử tìm trong KG với các biến thể, VERIFY TYPE là Album
                variants = [
                    album_name,
                    album_name + " (album)",
                    album_name.replace(":", " -"),
                    album_name.replace(" - ", ": "),
                ]
                for variant in variants:
                    entity_data = self.kg.get_entity(variant)
                    if entity_data and entity_data.get('label') == 'Album':  # VERIFY: Chỉ trả về nếu là Album
                        return variant
                
                # Thử tìm case-insensitive với normalize, VERIFY TYPE
                album_name_lower = self._normalize_entity_name(album_name).lower()
                for node, data in self.kg.graph.nodes(data=True):
                    if data.get('label') == 'Album':  # CHỈ tìm trong Album nodes
                        base_name = self._normalize_entity_name(node)
                        base_name_lower = base_name.lower()
                        if base_name_lower == album_name_lower or base_name_lower.startswith(album_name_lower):
                            if base_name_lower == album_name_lower:
                                return node
                            elif not any(self._normalize_entity_name(n).lower() == album_name_lower 
                                       for n, d in self.kg.graph.nodes(data=True) if d.get('label') == 'Album'):
                                return node
        
        return None
    
    def _extract_artist_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract artist name from query for artist-related questions.
        Returns the artist name if found in Knowledge Graph, None otherwise.
        Verifies entity type is Artist to avoid confusion with Group or other types.
        """
        import re
        query_lower = query.lower()
        
        # Pattern để extract tên ca sĩ/nghệ sĩ từ query
        patterns = [
            # Pattern với "ca sĩ X", "nghệ sĩ X"
            r'ca sĩ\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:thuộc|hát|thể hiện)|$)',  # ca sĩ Name thuộc/hát/thể hiện
            r'nghệ sĩ\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:thuộc|hát|thể hiện)|$)',  # nghệ sĩ Name thuộc/hát/thể hiện
            # Pattern "ai hát bài X" - cần context để extract artist
            # Pattern "ca sĩ thể hiện bài hát X" - extract từ context, không phải từ pattern này
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                artist_name = match.group(1).strip().rstrip('.,;:!?').strip()
                if not artist_name:
                    continue
                    
                # Thử tìm trong KG, VERIFY TYPE là Artist
                variants = [
                    artist_name,
                    artist_name + " (ca sĩ)",
                    artist_name + " (nghệ sĩ)",
                ]
                for variant in variants:
                    entity_data = self.kg.get_entity(variant)
                    if entity_data and entity_data.get('label') == 'Artist':  # VERIFY: Chỉ trả về nếu là Artist
                        return variant
                
                # Thử tìm case-insensitive với normalize, VERIFY TYPE
                artist_name_lower = self._normalize_entity_name(artist_name).lower()
                for node, data in self.kg.graph.nodes(data=True):
                    if data.get('label') == 'Artist':  # CHỈ tìm trong Artist nodes
                        base_name = self._normalize_entity_name(node)
                        base_name_lower = base_name.lower()
                        if base_name_lower == artist_name_lower:
                            return node
        
        return None
    
    def _extract_group_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract group name from query for group-related questions.
        Returns the group name if found in Knowledge Graph, None otherwise.
        Verifies entity type is Group to avoid confusion with Artist or Company.
        """
        import re
        query_lower = query.lower()
        
        # Pattern để extract tên nhóm nhạc từ query
        patterns = [
            # Pattern với "nhóm nhạc X", "nhóm X"
            r'nhóm nhạc\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:thuộc|của|do|là)|$)',  # nhóm nhạc Name thuộc/của/do/là
            r'nhóm\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:thuộc|của|do|là|có)|$)',  # nhóm Name thuộc/của/do/là/có
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                group_name = match.group(1).strip().rstrip('.,;:!?').strip()
                if not group_name:
                    continue
                    
                # Thử tìm trong KG, VERIFY TYPE là Group
                variants = [
                    group_name,
                    group_name + " (nhóm nhạc)",
                ]
                for variant in variants:
                    entity_data = self.kg.get_entity(variant)
                    if entity_data and entity_data.get('label') == 'Group':  # VERIFY: Chỉ trả về nếu là Group
                        return variant
                
                # Thử tìm case-insensitive với normalize, VERIFY TYPE
                group_name_lower = self._normalize_entity_name(group_name).lower()
                for node, data in self.kg.graph.nodes(data=True):
                    if data.get('label') == 'Group':  # CHỈ tìm trong Group nodes
                        base_name = self._normalize_entity_name(node)
                        base_name_lower = base_name.lower()
                        if base_name_lower == group_name_lower:
                            return node
        
        return None
    
    def _extract_company_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract company name from query for company-related questions.
        Returns the company name if found in Knowledge Graph, None otherwise.
        Verifies entity type is Company to avoid confusion with Group or other types.
        """
        import re
        query_lower = query.lower()
        
        # Pattern để extract tên công ty từ query
        patterns = [
            # Pattern với "công ty X", "hãng đĩa X", "label X"
            r'công ty\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+(?:quản lý|sở hữu|của)|$)',  # công ty Name quản lý/sở hữu/của
            r'hãng đĩa\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+|$)',  # hãng đĩa Name
            r'label\s+([A-Za-z0-9][A-Za-z0-9\s\-:\.]+?)(?:\s+|$)',  # label Name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip().rstrip('.,;:!?').strip()
                if not company_name:
                    continue
                    
                # Thử tìm trong KG, VERIFY TYPE là Company
                variants = [
                    company_name,
                    "Company_" + company_name,
                    company_name.replace("Company_", ""),
                ]
                for variant in variants:
                    entity_data = self.kg.get_entity(variant)
                    if entity_data and entity_data.get('label') == 'Company':  # VERIFY: Chỉ trả về nếu là Company
                        return variant
                
                # Thử tìm case-insensitive với normalize, VERIFY TYPE
                company_name_lower = self._normalize_entity_name(company_name).lower()
                for node, data in self.kg.graph.nodes(data=True):
                    if data.get('label') == 'Company':  # CHỈ tìm trong Company nodes
                        base_name = self._normalize_entity_name(node)
                        base_name_lower = base_name.lower()
                        if base_name_lower == company_name_lower or company_name_lower in base_name_lower:
                            return node
        
        return None
    
    def _extract_entities_from_query(self, query: str, expected_types: Optional[List[str]] = None) -> List[str]:
        """
        Extract entity names from query (case-insensitive).
        Tìm tất cả artists/groups/songs có thể có trong query.
        
        Args:
            query: Query string
            expected_types: Loại entities mong đợi (None = tất cả)
        """
        entities = []
        query_lower = query.lower()
        
        # Lấy tất cả artists và groups từ KG
        all_artists = [node for node, data in self.kg.graph.nodes(data=True) 
                      if data.get('label') == 'Artist']
        all_groups = [node for node, data in self.kg.graph.nodes(data=True) 
                     if data.get('label') == 'Group']
        
        # Thêm songs, albums, genres, companies nếu cần
        all_songs = []
        all_albums = []
        all_genres = []
        all_companies = []
        
        if not expected_types or 'Song' in expected_types:
            all_songs = [node for node, data in self.kg.graph.nodes(data=True) 
                        if data.get('label') == 'Song']
        if not expected_types or 'Album' in expected_types:
            all_albums = [node for node, data in self.kg.graph.nodes(data=True) 
                         if data.get('label') == 'Album']
        if not expected_types or 'Genre' in expected_types:
            all_genres = [node for node, data in self.kg.graph.nodes(data=True) 
                         if data.get('label') == 'Genre']
        if not expected_types or 'Company' in expected_types:
            all_companies = [node for node, data in self.kg.graph.nodes(data=True) 
                            if data.get('label') == 'Company']
        
        # Tìm tất cả artists trong query (case-insensitive)
        # QUAN TRỌNG: Xử lý node có đuôi như "Lisa (ca sĩ)"
        query_words_list = query_lower.split()  # List để giữ thứ tự
        
        # Tạo n-grams từ query (2-4 words) để bắt tên phức tạp như "Cho Seung-youn", "Jang Won Young", "jang won-young"
        # QUAN TRỌNG: Xử lý tokens có dash trong đó (như "seung-youn", "won-young")
        expanded_words = []
        for word in query_words_list:
            expanded_words.append(word)  # Giữ nguyên: "seung-youn"
            if '-' in word:
                # Tách token có dash thành parts
                parts = word.split('-')
                expanded_words.extend(parts)  # "seung-youn" → ["seung-youn", "seung", "youn"]
                # Thêm variant với space: "seung youn"
                expanded_words.append(" ".join(parts))
        
        query_ngrams = []
        for n in [2, 3, 4]:  # Tăng lên 4 để bắt tên dài như "Jang Won Young"
            # Tạo n-grams từ cả query_words_list và expanded_words
            for word_list in [query_words_list, expanded_words]:
                for i in range(len(word_list) - n + 1):
                    ngram = " ".join(word_list[i:i+n])
                    query_ngrams.append(ngram)  # Original: "jang won-young", "jang won young"
                    # Thêm variant không có space: "jangwonyoung"
                    query_ngrams.append(ngram.replace(" ", ""))
                    # Thêm variant với dash: "jang-won-young"
                    query_ngrams.append(ngram.replace(" ", "-"))
                    # QUAN TRỌNG: Nếu ngram có dấu gạch ngang, tạo thêm variant với space
                    if '-' in ngram:
                        query_ngrams.append(ngram.replace("-", " "))  # "jang won-young" → "jang won young", "won-young" → "won young"
                        query_ngrams.append(ngram.replace("-", ""))   # "won-young" → "wonyoung"
        
        # Loại bỏ trùng lặp
        query_ngrams = list(dict.fromkeys(query_ngrams))
        
        # Track normalized names để tránh duplicate (ví dụ: "Rosé" và "Rosé (ca sĩ)" → chỉ giữ 1)
        normalized_seen = set()
        # Track các từ đã được match trong tên đầy đủ để tránh match single word khi đã có match đầy đủ
        # Ví dụ: nếu đã match "Yoo Jeong-yeon", thì không match "Yoo" nữa
        words_in_matched_full_names = set()
        
        # QUAN TRỌNG: Sắp xếp artists theo độ dài tên (dài trước) để ưu tiên match tên đầy đủ trước
        # Ví dụ: "Yoo Jeong-yeon" sẽ được duyệt trước "Yoo" để match đúng
        all_artists_sorted = sorted(all_artists, key=lambda x: len(self._normalize_entity_name(x)), reverse=True)
        
        for artist in all_artists_sorted:
            artist_lower = artist.lower()
            # Extract base name (không có đuôi)
            base_name = self._normalize_entity_name(artist)
            base_name_lower = base_name.lower()
            
            # Check duplicate bằng normalized name TRƯỚC khi match
            if base_name_lower in normalized_seen:
                continue  # Đã có entity với cùng normalized name
            
            # QUAN TRỌNG: Định nghĩa base_name_word_count TRƯỚC khi dùng
            base_name_word_count = len(base_name_lower.split())
            
            # Tạo variants để match với nhiều format: "g-dragon", "g dragon", "gdragon", "go won", "go-won", "gowon"
            base_name_variants = [
                base_name_lower,  # Original
                base_name_lower.replace('-', ' '),  # "g-dragon" → "g dragon", "go-won" → "go won"
                base_name_lower.replace('-', ''),    # "g-dragon" → "gdragon", "go-won" → "gowon"
                base_name_lower.replace(' ', ''),    # "black pink" → "blackpink", "go won" → "gowon"
                base_name_lower.replace(' ', '-'),   # "go won" → "go-won", "jang won young" → "jang-won-young"
            ]
            # Loại bỏ trùng lặp
            base_name_variants = list(dict.fromkeys(base_name_variants))
            
            # QUAN TRỌNG: Ưu tiên match đầy đủ tên (n-gram) TRƯỚC khi match single word
            # Đảo thứ tự: Method 2 (n-gram) trước, Method 1 (single word) sau
            
            # Method 2: Check n-gram matching (2-4 words) để bắt tên phức tạp như "Cho Seung-youn", "Yoo Jeong-yeon"
            # QUAN TRỌNG: Duyệt tất cả n-grams trước để match tên đầy đủ, sau đó mới check single word
            matched_in_ngram = False
            for ngram in query_ngrams:
                if len(ngram) < 3:
                    continue
                # QUAN TRỌNG: Nếu base_name có nhiều từ, chỉ match với n-gram có ít nhất 2 từ
                # Tránh match "Yoo Jeong-yeon" với n-gram "yoo" (single word)
                if base_name_word_count >= 2:
                    ngram_word_count = len(ngram.split())
                    if ngram_word_count < 2:
                        continue  # Skip single word n-grams cho multi-word names
                for variant in base_name_variants:
                    # Exact match (ưu tiên cao nhất)
                    if variant == ngram:
                        if base_name_lower not in normalized_seen:
                            entities.append(artist)
                            normalized_seen.add(base_name_lower)
                            # Track các từ trong tên đầy đủ đã match để tránh match single word sau
                            # QUAN TRỌNG: Normalize (thay dash bằng space) trước khi split để tách đúng các từ
                            if base_name_word_count >= 2:
                                normalized_name = base_name_lower.replace('-', ' ').replace('  ', ' ').strip()
                                words_in_matched_full_names.update(normalized_name.split())
                            matched_in_ngram = True
                            break
                    # QUAN TRỌNG: Xử lý tên có dash trước khi check substring
                    # Normalize cả 2 về cùng format để so sánh chính xác hơn
                    elif '-' in variant or '-' in ngram:
                        # Normalize cả 2 về cùng format (space) để so sánh
                        variant_normalized = variant.replace('-', ' ').replace('  ', ' ').strip()
                        ngram_normalized = ngram.replace('-', ' ').replace('  ', ' ').strip()
                        # Exact match sau khi normalize
                        if variant_normalized == ngram_normalized:
                            if base_name_lower not in normalized_seen:
                                entities.append(artist)
                                normalized_seen.add(base_name_lower)
                                # Track các từ trong tên đầy đủ đã match
                                if base_name_word_count >= 2:
                                    words_in_matched_full_names.update(variant_normalized.split())
                                matched_in_ngram = True
                                break
                        # So sánh parts: nếu có ít nhất 2 parts giống nhau → match
                        variant_parts = set(variant_normalized.split())
                        ngram_parts = set(ngram_normalized.split())
                        if len(variant_parts.intersection(ngram_parts)) >= 2:
                            if base_name_lower not in normalized_seen:
                                entities.append(artist)
                                normalized_seen.add(base_name_lower)
                                # Track các từ trong tên đầy đủ đã match
                                if base_name_word_count >= 2:
                                    words_in_matched_full_names.update(variant_normalized.split())
                                matched_in_ngram = True
                                break
                    # Substring match (variant trong ngram hoặc ngược lại) - chỉ khi không có dash
                    elif variant in ngram or ngram in variant:
                        # QUAN TRỌNG: Chỉ match nếu base_name có nhiều từ VÀ n-gram cũng có nhiều từ
                        # Tránh match "Yoo" (single word) trong n-gram matching
                        variant_words = variant.split()
                        ngram_words = ngram.split()
                        # Chỉ match nếu cả 2 đều có ít nhất 2 từ (ưu tiên match đầy đủ tên)
                        if len(variant_words) >= 2 and len(ngram_words) >= 2:
                            # Check xem có ít nhất 2 từ trùng nhau không
                            variant_set = set(variant_words)
                            ngram_set = set(ngram_words)
                            if len(variant_set.intersection(ngram_set)) >= 2:
                                if base_name_lower not in normalized_seen:
                                    entities.append(artist)
                                    normalized_seen.add(base_name_lower)
                                    # Track các từ trong tên đầy đủ đã match
                                    # QUAN TRỌNG: Normalize (thay dash bằng space) trước khi split để tách đúng các từ
                                    if base_name_word_count >= 2:
                                        normalized_name = base_name_lower.replace('-', ' ').replace('  ', ' ').strip()
                                        words_in_matched_full_names.update(normalized_name.split())
                                    matched_in_ngram = True
                                    break
                        # Nếu base_name chỉ có 1 từ VÀ n-gram cũng chỉ có 1 từ, có thể match (nhưng ưu tiên thấp)
                        # Nhưng chỉ match nếu chưa có từ nào trong words_in_matched_full_names
                        elif len(variant_words) == 1 and len(ngram_words) == 1:
                            # Chỉ match single word nếu từ đó chưa được match trong tên đầy đủ nào
                            if base_name_lower not in words_in_matched_full_names:
                                if base_name_lower not in normalized_seen:
                                    entities.append(artist)
                                    normalized_seen.add(base_name_lower)
                                    matched_in_ngram = True
                                    break
                if matched_in_ngram:
                    break
            # QUAN TRỌNG: Nếu đã match trong n-gram, skip tất cả các method khác
            if matched_in_ngram:
                continue
            
            if base_name_lower in normalized_seen:
                continue
            
            # Method 1: Check nếu base name hoặc variants là một từ trong query
            # QUAN TRỌNG: Chỉ chạy nếu base_name chỉ có 1 từ (tránh match "Yoo" với "Yoo Jeong-yeon")
            # VÀ từ đó chưa được match trong tên đầy đủ nào (tránh match "Yoo" khi đã match "Yoo Jeong-yeon")
            # Ví dụ: query "lisa có cùng nhóm" → word "lisa" match với base_name "lisa"
            base_name_word_count = len(base_name_lower.split())
            if base_name_word_count == 1:
                # Check xem từ này đã được match trong tên đầy đủ nào chưa
                if base_name_lower in words_in_matched_full_names:
                    continue  # Đã được match trong tên đầy đủ, không match single word nữa
                
                if any(variant in query_words_list for variant in base_name_variants):
                    entities.append(artist)
                    normalized_seen.add(base_name_lower)
                    continue
            
            if base_name_lower in normalized_seen:
                continue
            
            # Method 3: Check substring match (cho tên phức tạp)
            if len(base_name_lower) >= 4:
                for variant in base_name_variants:
                    if len(variant) >= 4 and variant in query_lower:
                        # Verify: phải có ít nhất 2 từ trong variant xuất hiện trong query
                        variant_words = variant.split()
                        if len(variant_words) >= 2:
                            matched_words = sum(1 for w in variant_words if len(w) >= 3 and w in query_lower)
                            if matched_words >= 2:
                                if base_name_lower not in normalized_seen:
                                    entities.append(artist)
                                    normalized_seen.add(base_name_lower)
                                    break
                        elif len(variant_words) == 1 and variant in query_lower:
                            if variant in query_words_list or any(variant in w for w in query_words_list if len(w) >= len(variant)):
                                if base_name_lower not in normalized_seen:
                                    entities.append(artist)
                                    normalized_seen.add(base_name_lower)
                                    break
                if base_name_lower in normalized_seen:
                    continue
            
            # Method 4: Check nếu base name hoặc variants xuất hiện trong query text
            if any(variant in query_lower for variant in base_name_variants if len(variant) >= 3):
                if base_name_lower not in normalized_seen:
                    entities.append(artist)
                    normalized_seen.add(base_name_lower)
                    continue
            
            # Method 5: Check từng word trong query với base name và variants
            # QUAN TRỌNG: Chỉ match single word nếu base_name chỉ có 1 từ (tránh match "Punch" với "Punch (ca sĩ)" khi query có "Rocket Punch")
            # Nếu base_name có nhiều từ, chỉ match nếu tất cả các từ đều xuất hiện trong query
            base_name_word_count = len(base_name_lower.split())
            
            for word in query_words_list:
                if len(word) < 3:  # Skip short words
                    continue
                
                # QUAN TRỌNG: Check xem từ này đã được match trong tên đầy đủ nào chưa
                if word in words_in_matched_full_names:
                    continue  # Đã được match trong tên đầy đủ, không match single word nữa
                
                # Chỉ match single word nếu base_name cũng chỉ có 1 từ (tránh match sai)
                if base_name_word_count == 1:
                    # Exact match với base name hoặc variants
                    if word in base_name_variants or word == base_name_lower:
                        if base_name_lower not in normalized_seen:
                            entities.append(artist)
                            normalized_seen.add(base_name_lower)
                            break
                    # Partial match: word là một phần của base name hoặc ngược lại (chỉ cho single word names)
                    elif (word in base_name_lower and len(word) >= 3) or (base_name_lower in word and len(base_name_lower) >= 3):
                        if base_name_lower not in normalized_seen:
                            entities.append(artist)
                            normalized_seen.add(base_name_lower)
                            break
                # Nếu base_name có nhiều từ, chỉ match nếu tất cả các từ đều xuất hiện trong query
                elif base_name_word_count > 1:
                    base_words = set(base_name_lower.split())
                    query_words_set = set(query_words_list)
                    # Nếu tất cả các từ trong base_name đều có trong query → match
                    if base_words.issubset(query_words_set):
                        if base_name_lower not in normalized_seen:
                            entities.append(artist)
                            normalized_seen.add(base_name_lower)
                            break
                
                # Xử lý tên có dấu gạch ngang: "g-dragon" match với "g" và "dragon"
                # Chỉ match nếu có đủ các parts trong query
                if '-' in base_name_lower:
                    # QUAN TRỌNG: Check xem từ này đã được match trong tên đầy đủ nào chưa
                    if word in words_in_matched_full_names:
                        continue  # Đã được match trong tên đầy đủ, không match single word nữa
                    
                    base_parts = base_name_lower.split('-')
                    if word in base_parts and len(word) >= 3:
                        # Check xem có part khác cũng trong query không (phải có ít nhất 2 parts)
                        other_parts = [p for p in base_parts if p != word]
                        if len(other_parts) > 0 and any(p in query_lower for p in other_parts):
                            # Verify: phải có ít nhất 2 parts trong query để match
                            matched_parts = sum(1 for p in base_parts if p in query_lower)
                            if matched_parts >= 2 and base_name_lower not in normalized_seen:
                                entities.append(artist)
                                normalized_seen.add(base_name_lower)
                                break
        
        # Tìm tất cả groups trong query (case-insensitive)
        # QUAN TRỌNG: Ưu tiên match groups trước artists để tránh match sai (ví dụ: "Rocket Punch" group vs "Punch" artist)
        for group in all_groups:
            group_lower = group.lower()
            base_name = self._normalize_entity_name(group).lower()
            
            # Check duplicate bằng normalized name
            if base_name in normalized_seen:
                continue
            
            # Tạo variants cho group name
            group_variants = [
                base_name,
                base_name.replace('-', ' '),
                base_name.replace('-', ''),
                base_name.replace(' ', ''),
                base_name.replace(' ', '-'),
            ]
            group_variants = list(dict.fromkeys(group_variants))
            
            # Method 1: Check n-gram matching (ưu tiên match đầy đủ tên trước)
            for ngram in query_ngrams:
                if len(ngram) < 3:
                    continue
                for variant in group_variants:
                    # Exact match hoặc substring match
                    if variant == ngram or variant in ngram or ngram in variant:
                        entities.append(group)
                        normalized_seen.add(base_name)
                        break
                if base_name in normalized_seen:
                    break
            
            if base_name in normalized_seen:
                continue
            
            # Method 2: Check nếu tất cả các từ trong group name đều có trong query
            group_words = set(base_name.split())
            query_words_set = set(query_words_list)
            if group_words.issubset(query_words_set) and len(group_words) >= 2:
                entities.append(group)
                normalized_seen.add(base_name)
                continue
            
            # Method 3: Check substring match (fallback - chỉ cho single word groups)
            if len(base_name.split()) == 1:
                if base_name in query_words_list or any(variant in query_words_list for variant in group_variants):
                    entities.append(group)
                    normalized_seen.add(base_name)
                    continue
            
            # Method 4: Check substring match (fallback - yêu cầu ít nhất 2 từ trùng)
            if group_lower in query_lower:
                query_words = set(query_lower.split())
                group_words_set = set(group_lower.split())
                # Chỉ match nếu có ít nhất 2 từ trùng nhau (tránh match "Punch" với "Rocket Punch")
                if len(group_words_set.intersection(query_words)) >= 2 or (len(group_words_set) == 1 and group_lower in query_lower):
                    entities.append(group)
                    normalized_seen.add(base_name)
        
        # Tìm songs nếu cần (khi expected_types có 'Song') - Áp dụng logic tương tự như artists
        if all_songs and (not expected_types or 'Song' in expected_types):
            # Sắp xếp songs theo độ dài tên (dài trước) để ưu tiên match tên đầy đủ
            all_songs_sorted = sorted(all_songs, key=lambda x: len(self._normalize_entity_name(x)), reverse=True)
            
            for song in all_songs_sorted:
                song_lower = song.lower()
                # Extract base name (không có đuôi)
                base_name = self._normalize_entity_name(song)
                base_name_lower = base_name.lower()
                
                # Check duplicate bằng normalized name TRƯỚC khi match
                if base_name_lower in normalized_seen:
                    continue  # Đã có entity với cùng normalized name
                
                # Định nghĩa base_name_word_count
                base_name_word_count = len(base_name_lower.split())
                
                # Tạo variants để match với nhiều format
                song_variants = [
                    base_name_lower,  # Original
                    base_name_lower.replace('-', ' '),  # "kill-this-love" → "kill this love"
                    base_name_lower.replace('-', ''),    # "kill-this-love" → "killthislove"
                    base_name_lower.replace(' ', ''),    # "kill this love" → "killthislove"
                    base_name_lower.replace(' ', '-'),   # "kill this love" → "kill-this-love"
                ]
                song_variants = list(dict.fromkeys(song_variants))
                
                # Method 2: Check n-gram matching (2-4 words) để bắt tên phức tạp như "Kill This Love"
                # Ưu tiên match đầy đủ tên TRƯỚC khi match single word
                matched_in_ngram = False
                for ngram in query_ngrams:
                    if len(ngram) < 3:
                        continue
                    # Nếu base_name có nhiều từ, chỉ match với n-gram có ít nhất 2 từ
                    if base_name_word_count >= 2:
                        ngram_word_count = len(ngram.split())
                        if ngram_word_count < 2:
                            continue  # Skip single word n-grams cho multi-word names
                    for variant in song_variants:
                        # Exact match (ưu tiên cao nhất)
                        if variant == ngram:
                            if base_name_lower not in normalized_seen:
                                entities.append(song)
                                normalized_seen.add(base_name_lower)
                                matched_in_ngram = True
                                break
                        # Xử lý tên có dash trước khi check substring
                        elif '-' in variant or '-' in ngram:
                            # Normalize cả 2 về cùng format (space) để so sánh
                            variant_normalized = variant.replace('-', ' ').replace('  ', ' ').strip()
                            ngram_normalized = ngram.replace('-', ' ').replace('  ', ' ').strip()
                            # Exact match sau khi normalize
                            if variant_normalized == ngram_normalized:
                                if base_name_lower not in normalized_seen:
                                    entities.append(song)
                                    normalized_seen.add(base_name_lower)
                                    matched_in_ngram = True
                                    break
                            # So sánh parts: nếu có ít nhất 2 parts giống nhau → match
                            variant_parts = set(variant_normalized.split())
                            ngram_parts = set(ngram_normalized.split())
                            if len(variant_parts.intersection(ngram_parts)) >= 2:
                                if base_name_lower not in normalized_seen:
                                    entities.append(song)
                                    normalized_seen.add(base_name_lower)
                                    matched_in_ngram = True
                                    break
                        # Substring match (variant trong ngram hoặc ngược lại) - chỉ khi không có dash
                        elif variant in ngram or ngram in variant:
                            # Chỉ match nếu cả 2 đều có ít nhất 2 từ (ưu tiên match đầy đủ tên)
                            variant_words = variant.split()
                            ngram_words = ngram.split()
                            if len(variant_words) >= 2 and len(ngram_words) >= 2:
                                # Check xem có ít nhất 2 từ trùng nhau không
                                variant_set = set(variant_words)
                                ngram_set = set(ngram_words)
                                if len(variant_set.intersection(ngram_set)) >= 2:
                                    if base_name_lower not in normalized_seen:
                                        entities.append(song)
                                        normalized_seen.add(base_name_lower)
                                        matched_in_ngram = True
                                        break
                # Nếu đã match trong n-gram, skip các method khác
                if matched_in_ngram:
                    continue
                
                if base_name_lower in normalized_seen:
                    continue
                
                # Method 3: Check substring match (cho tên phức tạp)
                if len(base_name_lower) >= 4:
                    for variant in song_variants:
                        if len(variant) >= 4 and variant in query_lower:
                            # Verify: phải có ít nhất 2 từ trong variant xuất hiện trong query
                            variant_words = variant.split()
                            if len(variant_words) >= 2:
                                matched_words = sum(1 for w in variant_words if len(w) >= 3 and w in query_lower)
                                if matched_words >= 2:
                                    if base_name_lower not in normalized_seen:
                                        entities.append(song)
                                        normalized_seen.add(base_name_lower)
                                        break
                            elif len(variant_words) == 1 and variant in query_lower:
                                if variant in query_words_list or any(variant in w for w in query_words_list if len(w) >= len(variant)):
                                    if base_name_lower not in normalized_seen:
                                        entities.append(song)
                                        normalized_seen.add(base_name_lower)
                                        break
                    if base_name_lower in normalized_seen:
                        continue
                
                # Method 4: Check nếu tất cả các từ trong song name đều có trong query
                song_words = set(base_name_lower.split())
                query_words_set = set(query_words_list)
                if song_words.issubset(query_words_set) and len(song_words) >= 2:
                    if base_name_lower not in normalized_seen:
                        entities.append(song)
                        normalized_seen.add(base_name_lower)
                        continue
        
        # Tìm albums nếu cần (khi expected_types có 'Album') - Áp dụng logic tương tự như songs
        if all_albums and (not expected_types or 'Album' in expected_types):
            # Sắp xếp albums theo độ dài tên (dài trước) để ưu tiên match tên đầy đủ
            all_albums_sorted = sorted(all_albums, key=lambda x: len(self._normalize_entity_name(x)), reverse=True)
            
            for album in all_albums_sorted:
                album_lower = album.lower()
                # Extract base name (không có đuôi)
                base_name = self._normalize_entity_name(album)
                base_name_lower = base_name.lower()
                
                # Check duplicate bằng normalized name TRƯỚC khi match
                if base_name_lower in normalized_seen:
                    continue  # Đã có entity với cùng normalized name
                
                # Định nghĩa base_name_word_count
                base_name_word_count = len(base_name_lower.split())
                
                # Tạo variants để match với nhiều format
                album_variants = [
                    base_name_lower,  # Original
                    base_name_lower.replace('-', ' '),  # "born-pink" → "born pink"
                    base_name_lower.replace('-', ''),    # "born-pink" → "bornpink"
                    base_name_lower.replace(' ', ''),    # "born pink" → "bornpink"
                    base_name_lower.replace(' ', '-'),   # "born pink" → "born-pink"
                ]
                album_variants = list(dict.fromkeys(album_variants))
                
                # Method 2: Check n-gram matching (2-4 words) để bắt tên phức tạp như "Born Pink", "A Flower Bookmark"
                # Ưu tiên match đầy đủ tên TRƯỚC khi match single word
                matched_in_ngram = False
                for ngram in query_ngrams:
                    if len(ngram) < 3:
                        continue
                    # Nếu base_name có nhiều từ, chỉ match với n-gram có ít nhất 2 từ
                    if base_name_word_count >= 2:
                        ngram_word_count = len(ngram.split())
                        if ngram_word_count < 2:
                            continue  # Skip single word n-grams cho multi-word names
                    for variant in album_variants:
                        # Exact match (ưu tiên cao nhất)
                        if variant == ngram:
                            if base_name_lower not in normalized_seen:
                                entities.append(album)
                                normalized_seen.add(base_name_lower)
                                matched_in_ngram = True
                                break
                        # Xử lý tên có dash trước khi check substring
                        elif '-' in variant or '-' in ngram:
                            # Normalize cả 2 về cùng format (space) để so sánh
                            variant_normalized = variant.replace('-', ' ').replace('  ', ' ').strip()
                            ngram_normalized = ngram.replace('-', ' ').replace('  ', ' ').strip()
                            # Exact match sau khi normalize
                            if variant_normalized == ngram_normalized:
                                if base_name_lower not in normalized_seen:
                                    entities.append(album)
                                    normalized_seen.add(base_name_lower)
                                    matched_in_ngram = True
                                    break
                            # So sánh parts: nếu có ít nhất 2 parts giống nhau → match
                            variant_parts = set(variant_normalized.split())
                            ngram_parts = set(ngram_normalized.split())
                            if len(variant_parts.intersection(ngram_parts)) >= 2:
                                if base_name_lower not in normalized_seen:
                                    entities.append(album)
                                    normalized_seen.add(base_name_lower)
                                    matched_in_ngram = True
                                    break
                        # Substring match (variant trong ngram hoặc ngược lại) - chỉ khi không có dash
                        elif variant in ngram or ngram in variant:
                            # Chỉ match nếu cả 2 đều có ít nhất 2 từ (ưu tiên match đầy đủ tên)
                            variant_words = variant.split()
                            ngram_words = ngram.split()
                            if len(variant_words) >= 2 and len(ngram_words) >= 2:
                                # Check xem có ít nhất 2 từ trùng nhau không
                                variant_set = set(variant_words)
                                ngram_set = set(ngram_words)
                                if len(variant_set.intersection(ngram_set)) >= 2:
                                    if base_name_lower not in normalized_seen:
                                        entities.append(album)
                                        normalized_seen.add(base_name_lower)
                                        matched_in_ngram = True
                                        break
                # Nếu đã match trong n-gram, skip các method khác
                if matched_in_ngram:
                    continue
                
                if base_name_lower in normalized_seen:
                    continue
                
                # Method 3: Check substring match (cho tên phức tạp)
                if len(base_name_lower) >= 4:
                    for variant in album_variants:
                        if len(variant) >= 4 and variant in query_lower:
                            # Verify: phải có ít nhất 2 từ trong variant xuất hiện trong query
                            variant_words = variant.split()
                            if len(variant_words) >= 2:
                                matched_words = sum(1 for w in variant_words if len(w) >= 3 and w in query_lower)
                                if matched_words >= 2:
                                    if base_name_lower not in normalized_seen:
                                        entities.append(album)
                                        normalized_seen.add(base_name_lower)
                                        break
                            elif len(variant_words) == 1 and variant in query_lower:
                                if variant in query_words_list or any(variant in w for w in query_words_list if len(w) >= len(variant)):
                                    if base_name_lower not in normalized_seen:
                                        entities.append(album)
                                        normalized_seen.add(base_name_lower)
                                        break
                    if base_name_lower in normalized_seen:
                        continue
                
                # Method 4: Check nếu tất cả các từ trong album name đều có trong query
                album_words = set(base_name_lower.split())
                query_words_set = set(query_words_list)
                if album_words.issubset(query_words_set) and len(album_words) >= 2:
                    if base_name_lower not in normalized_seen:
                        entities.append(album)
                        normalized_seen.add(base_name_lower)
                        continue
        
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
                    
                    # Check duplicate bằng normalized name
                    if base_name in normalized_seen:
                        continue
                    
                    # Filter theo expected_types nếu có
                    if expected_types:
                        node_data = self.kg.get_entity(node)
                        if not node_data or node_data.get('label') not in expected_types:
                            continue
                    
                    # Exact match với full name hoặc base name
                    if node_lower == word or base_name == word:
                        node_data = self.kg.get_entity(node)
                        if node_data and (not expected_types or node_data.get('label') in expected_types):
                            entities.append(node)
                            normalized_seen.add(base_name)
                        break
                    # Partial match
                    elif (word in base_name and len(word) >= 3) or (base_name in word and len(base_name) >= 3):
                        node_data = self.kg.get_entity(node)
                        if node_data and (not expected_types or node_data.get('label') in expected_types):
                            entities.append(node)
                            normalized_seen.add(base_name)
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
    
    def _format_year_natural(self, year_str: str) -> str:
        """
        Format year string thành ngôn ngữ tự nhiên hơn.
        
        Ví dụ:
        - "2016–nay" → "từ 2016 đến nay"
        - "2013–2023" → "từ 2013 đến 2023"
        - "2019" → "2019"
        - "20122016–nay" → "từ 2016 đến nay" (xử lý nhiều năm dính liền)
        - "2012–2016–nay" → "từ 2016 đến nay" (lấy năm cuối cùng trước "nay")
        
        Args:
            year_str: Year string từ infobox (có thể là range)
            
        Returns:
            Formatted year string
        """
        if not year_str:
            return year_str
        
        import re
        
        # Xử lý trường hợp có nhiều ranges với text trong ngoặc
        # Ví dụ: "2009–2017 (MBK)2018–nay (tự do)" → "từ 2009 đến 2017 (MBK), từ 2018 đến nay (tự do)"
        # Tìm tất cả các ranges có dạng: YYYY–YYYY hoặc YYYY–nay, cùng với text trong ngoặc (nếu có)
        # Pattern để tìm range và text trong ngoặc ngay sau đó
        range_pattern = r'(\d{4})\s*[–-]\s*(\d{4}|nay)\s*(\([^)]+\))?'
        ranges_found = re.findall(range_pattern, year_str, re.IGNORECASE)
        
        if len(ranges_found) > 1:
            # Có nhiều ranges → format tất cả và ngăn cách bằng dấu phẩy
            formatted_ranges = []
            for start_year, end_year, note in ranges_found:
                note_text = f" {note.strip()}" if note else ""
                if end_year.lower() == 'nay':
                    formatted = f"từ {start_year} đến nay{note_text}"
                else:
                    formatted = f"từ {start_year} đến {end_year}{note_text}"
                formatted_ranges.append(formatted)
            return ", ".join(formatted_ranges)
        elif len(ranges_found) == 1:
            # Chỉ có 1 range
            start_year, end_year, note = ranges_found[0]
            note_text = f" {note.strip()}" if note else ""
            if end_year.lower() == 'nay':
                return f"từ {start_year} đến nay{note_text}"
            else:
                return f"từ {start_year} đến {end_year}{note_text}"
        
        # Xử lý trường hợp có nhiều năm dính liền (ví dụ: "20122016–nay" hoặc "2017)2018")
        # Tách các năm 4 chữ số
        year_pattern = r'\d{4}'
        years_found = re.findall(year_pattern, year_str)
        
        # Nếu có nhiều năm 4 chữ số, kiểm tra xem có dính liền không
        # Ví dụ: "20122016–nay" → ["2012", "2016"] - cần tách phần đầu
        # Ví dụ: "2017)2018" → ["2017", "2018"] - năm dính liền với ký tự đặc biệt
        if len(years_found) > 1:
            # Tìm vị trí của năm đầu tiên và năm thứ hai
            first_year_pos = year_str.find(years_found[0])
            second_year_pos = year_str.find(years_found[1])
            
            # Kiểm tra xem có năm dính liền không (khoảng cách <= 5 ký tự, có thể có ký tự đặc biệt)
            # Ví dụ: "20122016" (4 ký tự), "2017)2018" (5 ký tự)
            distance = second_year_pos - first_year_pos
            if 4 <= distance <= 5:
                # Có nhiều năm dính liền hoặc gần nhau, lấy năm cuối cùng
                end_year = years_found[-1]
                # Kiểm tra xem có "nay" hoặc dấu nối không
                if 'nay' in year_str.lower():
                    return f"từ {end_year} đến nay"
                elif '–' in year_str or '-' in year_str:
                    # Có dấu nối sau các năm dính liền, lấy phần sau dấu nối
                    separator = '–' if '–' in year_str else '-'
                    after_sep = year_str.split(separator, 1)[1].strip()
                    if after_sep.lower() == 'nay':
                        return f"từ {end_year} đến nay"
                    else:
                        # Lấy năm cuối cùng từ phần sau dấu nối
                        years_after = re.findall(year_pattern, after_sep)
                        if years_after:
                            return f"từ {end_year} đến {years_after[-1]}"
                        else:
                            return f"từ {end_year} đến {after_sep}"
                else:
                    return f"từ {years_found[0]} đến {end_year}"
        
        # Xử lý range với "–" (en dash) hoặc "-" (hyphen)
        if '–' in year_str:
            # Xử lý trường hợp có nhiều dấu nối: "2012–2016–nay"
            parts = year_str.split('–')
            # Loại bỏ các phần rỗng
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 2:
                # Lấy phần đầu và phần cuối
                start = parts[0]
                end = parts[-1]
                
                if end.lower() == 'nay':
                    return f"từ {start} đến nay"
                elif re.match(r'\d{4}', end):
                    return f"từ {start} đến {end}"
                else:
                    return f"từ {start} đến {end}"
            elif len(parts) == 1:
                return parts[0]
            else:
                return year_str
                
        elif '-' in year_str:
            # Xử lý với hyphen, chỉ xử lý khi có đúng 2 phần
            parts = year_str.split('-')
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) == 2:
                start = parts[0]
                end = parts[1]
                if end.lower() == 'nay':
                    return f"từ {start} đến nay"
                elif re.match(r'\d{4}', end):
                    return f"từ {start} đến {end}"
                else:
                    return f"từ {start} đến {end}"
            elif len(parts) == 1:
                return parts[0]
            else:
                # Nhiều hơn 2 phần, lấy đầu và cuối
                start = parts[0]
                end = parts[-1]
                if end.lower() == 'nay':
                    return f"từ {start} đến nay"
                else:
                    return f"từ {start} đến {end}"
        
        return year_str
        
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
        """Check if two groups/artists are under same company (1-hop or 2-hop comparison)."""
        steps = []
        
        # Get ALL companies for entity1 (một entity có thể có nhiều companies)
        if self.kg.get_entity_type(entity1) == 'Artist':
            # Check direct relationship first (1-hop)
            companies1 = self.kg.get_artist_companies(entity1)
            # Then check through groups (2-hop)
            groups1 = self.kg.get_artist_groups(entity1)
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
            # Check direct relationship first (1-hop)
            companies2 = self.kg.get_artist_companies(entity2)
            # Then check through groups (2-hop)
            groups2 = self.kg.get_artist_groups(entity2)
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
        
        # Helper function để clean company name (loại bỏ prefix "Company_")
        def _clean_company_name(name):
            if not name:
                return "không rõ"
            if name.startswith("Company_"):
                return name[8:]
            return name
        
        # Clean company names cho output tự nhiên
        companies1_clean = [_clean_company_name(c) for c in companies1]
        companies2_clean = [_clean_company_name(c) for c in companies2]
        common_clean = [_clean_company_name(c) for c in common_companies]
        
        if common_companies:
            answer_text = f"Có, {entity1} và {entity2} cùng thuộc công ty: {', '.join(common_clean)}"
        else:
            answer_text = f"Không, {entity1} ({', '.join(companies1_clean) if companies1_clean else 'không rõ'}) và {entity2} ({', '.join(companies2_clean) if companies2_clean else 'không rõ'}) khác công ty"
            
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
            answer_text=f"{entity_name} thuộc thể loại: {', '.join([g[6:] if g.startswith('Genre_') else g for g in genres]) if genres else 'Không xác định'}",
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

