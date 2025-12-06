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
        # Detect reasoning type from query
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
        """
        steps = []
        current_entities = start_entities
        all_paths = []
        
        for hop in range(max_hops):
            step_result = {
                'hop': hop + 1,
                'from': current_entities,
                'to': [],
                'relationships': []
            }
            
            next_entities = set()
            for entity in current_entities[:10]:  # Limit for performance
                neighbors = self.kg.get_neighbors(entity)
                for neighbor, rel_type, direction in neighbors:
                    next_entities.add(neighbor)
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
                source_entities=current_entities,
                relationship=', '.join(set(r['rel'] for r in step_result['relationships'])),
                target_entities=list(next_entities)[:20],
                explanation=f"Hop {hop + 1}: Tìm thấy {len(next_entities)} entities liên quan"
            )
            steps.append(reasoning_step)
            
            current_entities = list(next_entities)
            
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
        
    def check_same_company(self, entity1: str, entity2: str) -> ReasoningResult:
        """Check if two groups/artists are under same company (2-hop comparison)."""
        steps = []
        
        # Get company for entity1
        if self.kg.get_entity_type(entity1) == 'Artist':
            groups1 = self.kg.get_artist_groups(entity1)
            companies1 = [self.kg.get_group_company(g) for g in groups1]
            companies1 = [c for c in companies1 if c]
        else:
            companies1 = [self.kg.get_group_company(entity1)]
            companies1 = [c for c in companies1 if c]
            
        # Get company for entity2
        if self.kg.get_entity_type(entity2) == 'Artist':
            groups2 = self.kg.get_artist_groups(entity2)
            companies2 = [self.kg.get_group_company(g) for g in groups2]
            companies2 = [c for c in companies2 if c]
        else:
            companies2 = [self.kg.get_group_company(entity2)]
            companies2 = [c for c in companies2 if c]
            
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
        
        # Step 1: Get company
        if self.kg.get_entity_type(artist_or_group) == 'Artist':
            groups = self.kg.get_artist_groups(artist_or_group)
            companies = [self.kg.get_group_company(g) for g in groups]
            companies = [c for c in companies if c]
        else:
            companies = [self.kg.get_group_company(artist_or_group)]
            companies = [c for c in companies if c]
            
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

