#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Evaluation Dataset Generator for K-pop Chatbot

Dựa trên script mẫu để tạo dataset đánh giá với:
- True/False questions
- Yes/No questions  
- Multiple Choice questions

Tối thiểu 2000 câu hỏi đánh giá multi-hop reasoning.
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from .knowledge_graph import KpopKnowledgeGraph


@dataclass
class EvaluationQuestion:
    """An evaluation question."""
    id: str
    question: str
    question_type: str  # 'true_false', 'yes_no', 'multiple_choice'
    answer: str  # 'Đúng'/'Sai', 'Có'/'Không', or 'A'/'B'/'C'/'D'
    choices: List[str]  # Empty for true_false/yes_no
    hops: int  # Number of reasoning hops required
    entities: List[str]  # Entities involved
    relationships: List[str]  # Relationships involved
    explanation: str  # Explanation of the answer
    difficulty: str  # 'easy', 'medium', 'hard'
    category: str  # 'membership', 'company', 'song', etc.
    path: Optional[Dict] = None  # Two-hop path information


class EnhancedEvaluationGenerator:
    """
    Enhanced evaluation dataset generator based on two-hop path finding.
    
    Tạo dataset từ:
    1. Two-hop paths (như script mẫu)
    2. One-hop paths (đơn giản hóa)
    3. Three-hop paths (mở rộng)
    """
    
    def __init__(self, knowledge_graph: Optional[KpopKnowledgeGraph] = None):
        """Initialize with knowledge graph."""
        self.kg = knowledge_graph or KpopKnowledgeGraph()
        self.question_counter = 0
        
        # Cache graph structure
        self._build_graph_cache()
        
    def _build_graph_cache(self):
        """Build graph cache from knowledge graph."""
        print("🔄 Building graph cache...")
        
        self.nodes = {}
        self.edges = []
        
        # Get all nodes
        for node_id in self.kg.graph.nodes():
            node_data = self.kg.get_entity(node_id)
            if node_data:
                self.nodes[node_id] = {
                    'id': node_id,
                    'label': node_data.get('label', 'Unknown'),
                    'properties': node_data
                }
        
        # Get all edges
        for source, target, edge_data in self.kg.graph.edges(data=True):
            self.edges.append({
                'source': source,
                'target': target,
                'type': edge_data.get('type', 'RELATED_TO'),
                'text': edge_data.get('text', ''),
                'properties': edge_data
            })
        
        print(f"✓ Cached {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def _next_id(self) -> str:
        """Generate next question ID."""
        self.question_counter += 1
        return f"Q{self.question_counter:05d}"
    
    def find_two_hop_paths(self) -> List[Dict[str, Any]]:
        """
        Tìm tất cả các đường đi two-hop (2 bước, 3 nodes).
        Dựa trên script mẫu.
        """
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        
        # Build forward graph
        for edge in self.edges:
            source = edge['source']
            target = edge['target']
            if source and target:
                graph[source].append(edge)
        
        # Build reverse graph
        for edge in self.edges:
            source = edge['source']
            target = edge['target']
            if source and target:
                reverse_graph[target].append({
                    **edge,
                    'source': target,
                    'target': source,
                    'type': edge.get('type', 'RELATED_TO')
                })
        
        two_hop_paths = []
        node_set = set(self.nodes.keys())
        
        # Duyệt qua tất cả các node
        for start_node in node_set:
            if start_node not in graph:
                continue
            
            # Hướng thuận
            for edge1 in graph[start_node]:
                middle_node = edge1['target']
                
                if middle_node not in node_set:
                    continue
                
                # Tìm các node ở bước 2 - hướng thuận
                for edge2 in graph.get(middle_node, []):
                    end_node = edge2['target']
                    
                    if end_node not in node_set or start_node == end_node:
                        continue
                    
                    path = {
                        'start_node': start_node,
                        'middle_node': middle_node,
                        'end_node': end_node,
                        'edge1': edge1,
                        'edge2': edge2,
                        'start_label': self.nodes[start_node].get('label', 'Unknown'),
                        'middle_label': self.nodes[middle_node].get('label', 'Unknown'),
                        'end_label': self.nodes[end_node].get('label', 'Unknown')
                    }
                    two_hop_paths.append(path)
                
                # Tìm các node ở bước 2 - hướng ngược
                for edge2 in reverse_graph.get(middle_node, []):
                    end_node = edge2['target']
                    
                    if end_node not in node_set or start_node == end_node:
                        continue
                    
                    path = {
                        'start_node': start_node,
                        'middle_node': middle_node,
                        'end_node': end_node,
                        'edge1': edge1,
                        'edge2': edge2,
                        'start_label': self.nodes[start_node].get('label', 'Unknown'),
                        'middle_label': self.nodes[middle_node].get('label', 'Unknown'),
                        'end_label': self.nodes[end_node].get('label', 'Unknown')
                    }
                    two_hop_paths.append(path)
        
        return two_hop_paths
    
    def score_path(self, path: Dict[str, Any]) -> int:
        """
        Đánh giá độ thú vị của path (cao hơn = thú vị hơn).
        Dựa trên script mẫu.
        """
        edge1_type = path['edge1']['type']
        edge2_type = path['edge2']['type']
        start_label = path['start_label']
        middle_label = path['middle_label']
        end_label = path['end_label']
        
        score = 0
        
        # Các quan hệ rất thú vị (điểm cao)
        if (edge1_type, edge2_type) == ('MEMBER_OF', 'MEMBER_OF') and start_label == 'Artist' and end_label == 'Artist':
            score += 100  # Nghệ sĩ cùng nhóm
        elif (edge1_type, edge2_type) == ('MANAGED_BY', 'MANAGED_BY') and start_label == 'Group' and end_label == 'Group':
            score += 90  # Nhóm cùng công ty
        elif (edge1_type, edge2_type) == ('IS_GENRE', 'IS_GENRE') and start_label == 'Group' and end_label == 'Group':
            score += 85  # Nhóm cùng thể loại
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'SINGS') and start_label == 'Artist' and end_label == 'Song':
            score += 80  # Nghệ sĩ trình bày bài hát qua nhóm
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'RELEASED') and start_label == 'Artist' and end_label == 'Album':
            score += 75  # Nghệ sĩ phát hành album qua nhóm
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'IS_GENRE') and start_label == 'Artist' and end_label == 'Genre':
            score += 70  # Nghệ sĩ liên quan đến thể loại qua nhóm
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'MANAGED_BY') and start_label == 'Artist' and end_label == 'Company':
            score += 65  # Nghệ sĩ liên quan đến công ty qua nhóm
        elif (edge1_type, edge2_type) == ('HAS_OCCUPATION', 'HAS_OCCUPATION') and start_label == 'Artist' and end_label == 'Artist':
            score += 60  # Nghệ sĩ cùng nghề nghiệp
        elif (edge1_type, edge2_type) == ('RELEASED', 'CONTAINS') and start_label == 'Group' and end_label == 'Song':
            score += 55  # Bài hát trong album của nhóm
        
        # Trừ điểm cho các quan hệ không thú vị
        if edge1_type == 'PRODUCED_ALBUM':
            score -= 30
        if start_label == end_label and start_label in ['Genre', 'Company', 'Occupation']:
            score -= 20
        
        return score
    
    def clean_name(self, name: str) -> str:
        """Loại bỏ prefix để hiển thị đẹp hơn."""
        for prefix in ['Genre_', 'Company_', 'Occupation_']:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    
    def generate_question_from_path(
        self,
        path: Dict[str, Any],
        question_type: str = 'yes_no'
    ) -> Optional[EvaluationQuestion]:
        """
        Tạo câu hỏi từ một two-hop path.
        Dựa trên script mẫu với nhiều biến thể.
        """
        start = path['start_node']
        middle = path['middle_node']
        end = path['end_node']
        
        edge1_type = path['edge1']['type']
        edge2_type = path['edge2']['type']
        
        start_label = path['start_label']
        middle_label = path['middle_label']
        end_label = path['end_label']
        
        clean_middle = self.clean_name(middle)
        clean_end = self.clean_name(end)
        
        # Mapping các pattern quan hệ → câu hỏi
        question_templates = {
            # Nghệ sĩ cùng nhóm
            ('MEMBER_OF', 'MEMBER_OF'): {
                'yes_no': [
                    f"{start} và {end} có cùng nhóm nhạc không?",
                    f"{start} và {end} có phải thành viên của cùng một nhóm nhạc không?",
                    f"{start} và {end} có thuộc cùng nhóm nhạc {clean_middle} không?",
                ],
                'true_false': [
                    f"{start} và {end} đều là thành viên của nhóm nhạc {clean_middle}.",
                    f"{start} và {end} thuộc cùng một nhóm nhạc.",
                ],
                'multiple_choice': [
                    f"Ai là thành viên khác của nhóm nhạc {clean_middle} cùng với {start}?",
                    f"Nghệ sĩ nào khác cũng là thành viên của nhóm nhạc {clean_middle} giống như {start}?",
                ]
            },
            
            # Nhóm cùng công ty
            ('MANAGED_BY', 'MANAGED_BY'): {
                'yes_no': [
                    f"{start} và {end} có cùng công ty quản lý không?",
                    f"{start} và {end} có thuộc cùng công ty {clean_middle} không?",
                ],
                'true_false': [
                    f"{start} và {end} đều được quản lý bởi công ty {clean_middle}.",
                    f"{start} và {end} thuộc cùng một công ty quản lý.",
                ],
                'multiple_choice': [
                    f"Nhóm nhạc nào khác cũng được quản lý bởi công ty {clean_middle} giống như {start}?",
                    f"Cùng được quản lý bởi {clean_middle}, nhóm nhạc nào khác ngoài {start}?",
                ]
            },
            
            # Nghệ sĩ trình bày bài hát qua nhóm
            ('MEMBER_OF', 'SINGS'): {
                'yes_no': [
                    f"Nghệ sĩ {start} có trình bày bài hát {clean_end} thông qua nhóm nhạc {clean_middle} không?",
                ],
                'true_false': [
                    f"Nghệ sĩ {start} trình bày bài hát {clean_end} thông qua nhóm nhạc {clean_middle}.",
                ],
                'multiple_choice': [
                    f"Bài hát nào mà nghệ sĩ {start} trình bày thông qua nhóm nhạc {clean_middle}?",
                ]
            },
            
            # Nghệ sĩ liên quan đến công ty qua nhóm
            ('MEMBER_OF', 'MANAGED_BY'): {
                'yes_no': [
                    f"Nghệ sĩ {start} có được quản lý bởi công ty {clean_end} thông qua nhóm nhạc {clean_middle} không?",
                ],
                'true_false': [
                    f"Nghệ sĩ {start} được quản lý bởi công ty {clean_end} thông qua nhóm nhạc {clean_middle}.",
                ],
                'multiple_choice': [
                    f"Công ty nào quản lý nghệ sĩ {start} thông qua nhóm nhạc {clean_middle}?",
                ]
            },
        }
        
        key = (edge1_type, edge2_type)
        
        # Tạo câu hỏi dựa trên type
        if key in question_templates and question_type in question_templates[key]:
            templates = question_templates[key][question_type]
            path_hash = hash((start, middle, end))
            question_text = templates[abs(path_hash) % len(templates)]
        else:
            # Fallback: tạo câu hỏi tổng quát
            if question_type == 'yes_no':
                question_text = f"{start} và {clean_end} có liên quan thông qua {clean_middle} không?"
            elif question_type == 'true_false':
                question_text = f"{start} liên quan đến {clean_end} thông qua {clean_middle}."
            else:
                question_text = f"Ai/cái gì liên quan đến {start} thông qua {clean_middle}?"
        
        # Tạo answer và explanation
        if question_type == 'yes_no':
            answer = "Có"
            explanation = f"Có, {start} → [{edge1_type}] → {clean_middle} → [{edge2_type}] → {clean_end}."
        elif question_type == 'true_false':
            answer = "Đúng"
            explanation = f"Đúng, {start} → [{edge1_type}] → {clean_middle} → [{edge2_type}] → {clean_end}."
        else:  # multiple_choice
            answer = "A"  # Sẽ được set sau
            explanation = f"{clean_end} liên quan đến {start} thông qua {clean_middle}."
        
        # Tạo choices cho multiple choice
        choices = []
        if question_type == 'multiple_choice':
            # Lấy các entities cùng loại với end để tạo distractors
            same_type_entities = [
                node_id for node_id, data in self.nodes.items()
                if data.get('label') == end_label and node_id != end
            ]
            # Chọn 3 distractors ngẫu nhiên
            distractors = random.sample(same_type_entities, min(3, len(same_type_entities)))
            choices = [clean_end] + [self.clean_name(d) for d in distractors]
            random.shuffle(choices)
            # Tìm vị trí của đáp án đúng
            answer = chr(65 + choices.index(clean_end))  # A, B, C, hoặc D
        
        return EvaluationQuestion(
            id=self._next_id(),
            question=question_text,
            question_type=question_type,
            answer=answer,
            choices=choices,
            hops=2,
            entities=[start, middle, end],
            relationships=[edge1_type, edge2_type],
            explanation=explanation,
            difficulty="medium",
            category=self._get_category(edge1_type, edge2_type),
            path={
                'start': {'id': start, 'label': start_label},
                'middle': {'id': middle, 'label': middle_label},
                'end': {'id': end, 'label': end_label}
            }
        )
    
    def _get_category(self, edge1_type: str, edge2_type: str) -> str:
        """Xác định category dựa trên edge types."""
        if edge1_type == 'MEMBER_OF' and edge2_type == 'MEMBER_OF':
            return 'same_group'
        elif edge1_type == 'MANAGED_BY' and edge2_type == 'MANAGED_BY':
            return 'same_company'
        elif edge1_type == 'MEMBER_OF' and edge2_type == 'SINGS':
            return 'artist_song'
        elif edge1_type == 'MEMBER_OF' and edge2_type == 'MANAGED_BY':
            return 'artist_company'
        else:
            return 'general'
    
    def generate_dataset(
        self,
        target_count: int = 2000,
        tf_ratio: float = 0.4,
        yn_ratio: float = 0.3,
        mc_ratio: float = 0.3
    ) -> List[EvaluationQuestion]:
        """
        Generate comprehensive evaluation dataset.
        
        Args:
            target_count: Target number of questions (default 2000)
            tf_ratio: Ratio of True/False questions (default 0.4)
            yn_ratio: Ratio of Yes/No questions (default 0.3)
            mc_ratio: Ratio of Multiple Choice questions (default 0.3)
        """
        print(f"🔄 Generating evaluation dataset (target: {target_count} questions)...")
        
        # Find all two-hop paths
        print("  🔍 Finding two-hop paths...")
        all_paths = self.find_two_hop_paths()
        print(f"  ✓ Found {len(all_paths)} two-hop paths")
        
        # Filter and score paths
        print("  📊 Filtering and scoring paths...")
        filtered_paths = []
        seen_combinations = set()
        
        for path in all_paths:
            edge1_type = path['edge1']['type']
            edge2_type = path['edge2']['type']
            
            # Skip uninteresting relations
            if edge1_type == 'PRODUCED_ALBUM':
                continue
            
            # Avoid duplicates
            combo = (path['start_node'], path['end_node'], edge1_type, edge2_type)
            if combo in seen_combinations:
                continue
            seen_combinations.add(combo)
            
            filtered_paths.append(path)
        
        # Sort by score
        filtered_paths.sort(key=self.score_path, reverse=True)
        print(f"  ✓ Filtered to {len(filtered_paths)} unique paths")
        
        # Generate questions
        print("  📝 Generating questions...")
        questions = []
        
        tf_count = int(target_count * tf_ratio)
        yn_count = int(target_count * yn_ratio)
        mc_count = int(target_count * mc_ratio)
        
        seen_paths_tf = set()
        seen_paths_yn = set()
        seen_paths_mc = set()
        
        # Generate True/False questions
        for path in filtered_paths:
            if len(questions) >= target_count:
                break
            
            path_key = (path['start_node'], path['end_node'])
            if path_key in seen_paths_tf:
                continue
            
            question = self.generate_question_from_path(path, 'true_false')
            if question:
                questions.append(question)
                seen_paths_tf.add(path_key)
                if len([q for q in questions if q.question_type == 'true_false']) >= tf_count:
                    break
        
        # Generate Yes/No questions
        for path in filtered_paths:
            if len(questions) >= target_count:
                break
            
            path_key = (path['start_node'], path['end_node'])
            if path_key in seen_paths_yn:
                continue
            
            question = self.generate_question_from_path(path, 'yes_no')
            if question:
                questions.append(question)
                seen_paths_yn.add(path_key)
                if len([q for q in questions if q.question_type == 'yes_no']) >= yn_count:
                    break
        
        # Generate Multiple Choice questions
        for path in filtered_paths:
            if len(questions) >= target_count:
                break
            
            path_key = (path['start_node'], path['end_node'])
            if path_key in seen_paths_mc:
                continue
            
            question = self.generate_question_from_path(path, 'multiple_choice')
            if question:
                questions.append(question)
                seen_paths_mc.add(path_key)
                if len([q for q in questions if q.question_type == 'multiple_choice']) >= mc_count:
                    break
        
        print(f"  ✓ Generated {len(questions)} questions")
        print(f"    - True/False: {len([q for q in questions if q.question_type == 'true_false'])}")
        print(f"    - Yes/No: {len([q for q in questions if q.question_type == 'yes_no'])}")
        print(f"    - Multiple Choice: {len([q for q in questions if q.question_type == 'multiple_choice'])}")
        
        return questions
    
    def save_dataset(
        self,
        questions: List[EvaluationQuestion],
        output_file: str = "data/evaluation_dataset_enhanced.json"
    ):
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        data = {
            'metadata': {
                'total_questions': len(questions),
                'true_false': len([q for q in questions if q.question_type == 'true_false']),
                'yes_no': len([q for q in questions if q.question_type == 'yes_no']),
                'multiple_choice': len([q for q in questions if q.question_type == 'multiple_choice']),
                'generated_at': str(datetime.now())
            },
            'questions': [asdict(q) for q in questions]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved dataset to {output_file}")


def main():
    """Main function to generate evaluation dataset."""
    from datetime import datetime
    
    print("="*80)
    print("ENHANCED EVALUATION DATASET GENERATOR")
    print("="*80)
    
    generator = EnhancedEvaluationGenerator()
    
    # Generate dataset
    questions = generator.generate_dataset(target_count=2000)
    
    # Save to file
    generator.save_dataset(questions)
    
    print("\n" + "="*80)
    print("✅ Dataset generation completed!")
    print("="*80)


if __name__ == '__main__':
    main()

