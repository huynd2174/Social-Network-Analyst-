"""
Enhanced Two-Hop Evaluation Dataset Generator

Dựa trên script mẫu để tạo 2000+ câu hỏi đánh giá từ two-hop paths.
Tạo các câu hỏi True/False, Yes/No, Multiple Choice đa dạng.
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

from .knowledge_graph import KpopKnowledgeGraph
from .evaluation import EvaluationQuestion


class TwoHopPathGenerator:
    """Generate two-hop paths from knowledge graph."""
    
    def __init__(self, knowledge_graph: Optional[KpopKnowledgeGraph] = None):
        """Initialize with knowledge graph."""
        self.kg = knowledge_graph or KpopKnowledgeGraph()
        self.graph = self.kg.graph
        
    def find_all_two_hop_paths(self) -> List[Dict[str, Any]]:
        """
        Tìm tất cả các đường đi two-hop (2 bước, 3 nodes).
        
        Returns:
            List of paths với structure:
            {
                'start_node': str,
                'middle_node': str,
                'end_node': str,
                'edge1': dict,
                'edge2': dict,
                'start_label': str,
                'middle_label': str,
                'end_label': str
            }
        """
        two_hop_paths = []
        node_set = set(self.graph.nodes())
        
        # Build reverse graph
        reverse_edges = defaultdict(list)
        for source, target, data in self.graph.edges(data=True):
            reverse_edges[target].append({
                'source': target,
                'target': source,
                'type': data.get('type', 'RELATED_TO'),
                'data': data
            })
        
        # Duyệt qua tất cả nodes
        for start_node in node_set:
            # Tìm neighbors (one-hop)
            for edge1_data in self.graph.out_edges(start_node, data=True):
                middle_node = edge1_data[1]
                
                if middle_node not in node_set:
                    continue
                
                edge1 = {
                    'source': start_node,
                    'target': middle_node,
                    'type': edge1_data[2].get('type', 'RELATED_TO'),
                    'data': edge1_data[2]
                }
                
                # Tìm two-hop neighbors (forward)
                for edge2_data in self.graph.out_edges(middle_node, data=True):
                    end_node = edge2_data[1]
                    
                    if end_node not in node_set or start_node == end_node:
                        continue
                    
                    edge2 = {
                        'source': middle_node,
                        'target': end_node,
                        'type': edge2_data[2].get('type', 'RELATED_TO'),
                        'data': edge2_data[2]
                    }
                    
                    path = {
                        'start_node': start_node,
                        'middle_node': middle_node,
                        'end_node': end_node,
                        'edge1': edge1,
                        'edge2': edge2,
                        'start_label': self.kg.get_entity_type(start_node) or 'Unknown',
                        'middle_label': self.kg.get_entity_type(middle_node) or 'Unknown',
                        'end_label': self.kg.get_entity_type(end_node) or 'Unknown'
                    }
                    two_hop_paths.append(path)
                
                # Tìm two-hop neighbors (reverse)
                for edge2 in reverse_edges.get(middle_node, []):
                    end_node = edge2['target']
                    
                    if end_node not in node_set or start_node == end_node:
                        continue
                    
                    path = {
                        'start_node': start_node,
                        'middle_node': middle_node,
                        'end_node': end_node,
                        'edge1': edge1,
                        'edge2': edge2,
                        'start_label': self.kg.get_entity_type(start_node) or 'Unknown',
                        'middle_label': self.kg.get_entity_type(middle_node) or 'Unknown',
                        'end_label': self.kg.get_entity_type(end_node) or 'Unknown'
                    }
                    two_hop_paths.append(path)
        
        return two_hop_paths
    
    def score_path(self, path: Dict[str, Any]) -> int:
        """Đánh giá độ thú vị của path (cao hơn = thú vị hơn)."""
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
        elif (edge1_type, edge2_type) == ('SINGS', 'SINGS') and start_label == 'Song' and end_label == 'Song':
            score += 50  # Bài hát cùng nhóm trình bày
        
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


class TwoHopQuestionGenerator:
    """Generate questions from two-hop paths."""
    
    def __init__(self, path_generator: TwoHopPathGenerator):
        """Initialize with path generator."""
        self.path_gen = path_generator
        self.kg = path_generator.kg
        self.question_counter = 0
        
    def _next_id(self) -> str:
        """Generate next question ID."""
        self.question_counter += 1
        return f"Q{self.question_counter:05d}"
    
    def generate_true_false_from_path(self, path: Dict[str, Any]) -> Optional[EvaluationQuestion]:
        """Generate True/False question from a two-hop path."""
        start = path['start_node']
        middle = path['middle_node']
        end = path['end_node']
        edge1_type = path['edge1']['type']
        edge2_type = path['edge2']['type']
        start_label = path['start_label']
        middle_label = path['middle_label']
        end_label = path['end_label']
        
        clean_middle = self.path_gen.clean_name(middle)
        clean_end = self.path_gen.clean_name(end)
        
        # Tạo câu hỏi True/False dựa trên loại quan hệ
        question = None
        answer = "Đúng"
        explanation = f"{start} → [{edge1_type}] → {clean_middle} → [{edge2_type}] → {clean_end}"
        
        if (edge1_type, edge2_type) == ('MEMBER_OF', 'MEMBER_OF') and start_label == 'Artist' and end_label == 'Artist':
            question = f"{start} và {end} đều là thành viên của nhóm nhạc {clean_middle}."
        elif (edge1_type, edge2_type) == ('MANAGED_BY', 'MANAGED_BY') and start_label == 'Group' and end_label == 'Group':
            question = f"{start} và {end} đều được quản lý bởi công ty {clean_middle}."
        elif (edge1_type, edge2_type) == ('IS_GENRE', 'IS_GENRE') and start_label == 'Group' and end_label == 'Group':
            question = f"{start} và {end} đều thuộc thể loại nhạc {clean_middle}."
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'SINGS') and start_label == 'Artist' and end_label == 'Song':
            question = f"Nghệ sĩ {start} đã trình bày bài hát {clean_end} thông qua nhóm nhạc {clean_middle}."
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'RELEASED') and start_label == 'Artist' and end_label == 'Album':
            question = f"Nghệ sĩ {start} đã phát hành album {clean_end} thông qua nhóm nhạc {clean_middle}."
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'MANAGED_BY') and start_label == 'Artist' and end_label == 'Company':
            question = f"Nghệ sĩ {start} được quản lý bởi công ty {clean_end} thông qua nhóm nhạc {clean_middle}."
        elif (edge1_type, edge2_type) == ('RELEASED', 'CONTAINS') and start_label == 'Group' and end_label == 'Song':
            question = f"Nhóm nhạc {start} đã phát hành album {clean_middle} chứa bài hát {clean_end}."
        
        if not question:
            return None
        
        return EvaluationQuestion(
            id=self._next_id(),
            question=question,
            question_type="true_false",
            answer=answer,
            choices=[],
            hops=2,
            entities=[start, middle, end],
            relationships=[edge1_type, edge2_type],
            explanation=explanation,
            difficulty="medium",
            category=f"{edge1_type}_{edge2_type}"
        )
    
    def generate_yes_no_from_path(self, path: Dict[str, Any]) -> Optional[EvaluationQuestion]:
        """Generate Yes/No question from a two-hop path."""
        start = path['start_node']
        middle = path['middle_node']
        end = path['end_node']
        edge1_type = path['edge1']['type']
        edge2_type = path['edge2']['type']
        start_label = path['start_label']
        end_label = path['end_label']
        
        clean_middle = self.path_gen.clean_name(middle)
        clean_end = self.path_gen.clean_name(end)
        
        question = None
        answer = "Có"
        explanation = f"{start} → [{edge1_type}] → {clean_middle} → [{edge2_type}] → {clean_end}"
        
        if (edge1_type, edge2_type) == ('MEMBER_OF', 'MEMBER_OF') and start_label == 'Artist' and end_label == 'Artist':
            question = f"{start} và {end} có cùng nhóm nhạc {clean_middle} không?"
        elif (edge1_type, edge2_type) == ('MANAGED_BY', 'MANAGED_BY') and start_label == 'Group' and end_label == 'Group':
            question = f"{start} và {end} có cùng công ty quản lý {clean_middle} không?"
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'SINGS') and start_label == 'Artist' and end_label == 'Song':
            question = f"Nghệ sĩ {start} có trình bày bài hát {clean_end} qua nhóm nhạc {clean_middle} không?"
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'MANAGED_BY') and start_label == 'Artist' and end_label == 'Company':
            question = f"Nghệ sĩ {start} có được quản lý bởi công ty {clean_end} qua nhóm nhạc {clean_middle} không?"
        
        if not question:
            return None
        
        return EvaluationQuestion(
            id=self._next_id(),
            question=question,
            question_type="yes_no",
            answer=answer,
            choices=[],
            hops=2,
            entities=[start, middle, end],
            relationships=[edge1_type, edge2_type],
            explanation=explanation,
            difficulty="medium",
            category=f"{edge1_type}_{edge2_type}"
        )
    
    def generate_multiple_choice_from_path(self, path: Dict[str, Any]) -> Optional[EvaluationQuestion]:
        """Generate Multiple Choice question from a two-hop path."""
        start = path['start_node']
        middle = path['middle_node']
        end = path['end_node']
        edge1_type = path['edge1']['type']
        edge2_type = path['edge2']['type']
        start_label = path['start_label']
        end_label = path['end_label']
        
        clean_middle = self.path_gen.clean_name(middle)
        clean_end = self.path_gen.clean_name(end)
        
        question = None
        correct_answer = None
        choices = []
        explanation = f"{start} → [{edge1_type}] → {clean_middle} → [{edge2_type}] → {clean_end}"
        
        # Tìm các entities cùng loại để làm distractors
        all_entities = list(self.kg.graph.nodes())
        same_type_entities = [
            e for e in all_entities 
            if self.kg.get_entity_type(e) == end_label and e != end
        ]
        
        if len(same_type_entities) < 3:
            return None  # Không đủ distractors
        
        distractors = random.sample(same_type_entities, min(3, len(same_type_entities)))
        choices = [clean_end] + [self.path_gen.clean_name(d) for d in distractors]
        random.shuffle(choices)
        correct_idx = choices.index(clean_end)
        correct_answer = ['A', 'B', 'C', 'D'][correct_idx]
        
        if (edge1_type, edge2_type) == ('MEMBER_OF', 'MEMBER_OF') and start_label == 'Artist' and end_label == 'Artist':
            question = f"Ai là thành viên khác của nhóm nhạc {clean_middle} cùng với {start}?"
        elif (edge1_type, edge2_type) == ('MANAGED_BY', 'MANAGED_BY') and start_label == 'Group' and end_label == 'Group':
            question = f"Nhóm nhạc nào khác cũng được quản lý bởi công ty {clean_middle} giống như {start}?"
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'SINGS') and start_label == 'Artist' and end_label == 'Song':
            question = f"Bài hát nào mà nghệ sĩ {start} trình bày thông qua nhóm nhạc {clean_middle}?"
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'MANAGED_BY') and start_label == 'Artist' and end_label == 'Company':
            question = f"Công ty nào quản lý nghệ sĩ {start} thông qua nhóm nhạc {clean_middle}?"
        
        if not question:
            return None
        
        return EvaluationQuestion(
            id=self._next_id(),
            question=question,
            question_type="multiple_choice",
            answer=correct_answer,
            choices=choices,
            hops=2,
            entities=[start, middle, end],
            relationships=[edge1_type, edge2_type],
            explanation=explanation,
            difficulty="medium",
            category=f"{edge1_type}_{edge2_type}"
        )
    
    def generate_questions_from_paths(
        self,
        paths: List[Dict[str, Any]],
        target_count: int = 2000,
        tf_ratio: float = 0.4,
        yn_ratio: float = 0.3,
        mc_ratio: float = 0.3
    ) -> List[EvaluationQuestion]:
        """
        Generate questions from two-hop paths.
        
        Args:
            paths: List of two-hop paths
            target_count: Target number of questions
            tf_ratio: Ratio of True/False questions
            yn_ratio: Ratio of Yes/No questions
            mc_ratio: Ratio of Multiple Choice questions
        """
        questions = []
        seen_paths = set()
        
        # Sắp xếp paths theo điểm số
        paths_sorted = sorted(paths, key=self.path_gen.score_path, reverse=True)
        
        tf_count = int(target_count * tf_ratio)
        yn_count = int(target_count * yn_ratio)
        mc_count = target_count - tf_count - yn_count
        
        # Generate True/False questions
        print(f"  📝 Generating {tf_count} True/False questions...")
        for path in paths_sorted:
            if len(questions) >= tf_count:
                break
            
            path_key = (path['start_node'], path['middle_node'], path['end_node'])
            if path_key in seen_paths:
                continue
            
            q = self.generate_true_false_from_path(path)
            if q:
                questions.append(q)
                seen_paths.add(path_key)
        
        # Generate Yes/No questions
        print(f"  📝 Generating {yn_count} Yes/No questions...")
        seen_paths_yn = set()
        for path in paths_sorted:
            if len([q for q in questions if q.question_type == 'yes_no']) >= yn_count:
                break
            
            path_key = (path['start_node'], path['middle_node'], path['end_node'])
            if path_key in seen_paths_yn:
                continue
            
            q = self.generate_yes_no_from_path(path)
            if q:
                questions.append(q)
                seen_paths_yn.add(path_key)
        
        # Generate Multiple Choice questions
        print(f"  📝 Generating {mc_count} Multiple Choice questions...")
        seen_paths_mc = set()
        for path in paths_sorted:
            if len([q for q in questions if q.question_type == 'multiple_choice']) >= mc_count:
                break
            
            path_key = (path['start_node'], path['middle_node'], path['end_node'])
            if path_key in seen_paths_mc:
                continue
            
            q = self.generate_multiple_choice_from_path(path)
            if q:
                questions.append(q)
                seen_paths_mc.add(path_key)
        
        return questions


def generate_two_hop_evaluation_dataset(
    output_file: str = "data/evaluation_dataset_two_hop.json",
    target_count: int = 2000,
    knowledge_graph: Optional[KpopKnowledgeGraph] = None
) -> List[EvaluationQuestion]:
    """
    Generate comprehensive two-hop evaluation dataset.
    
    Args:
        output_file: Output JSON file path
        target_count: Target number of questions
        knowledge_graph: Optional knowledge graph instance
    
    Returns:
        List of evaluation questions
    """
    print("🔄 Generating Two-Hop Evaluation Dataset...")
    
    # Initialize generators
    path_gen = TwoHopPathGenerator(knowledge_graph)
    question_gen = TwoHopQuestionGenerator(path_gen)
    
    # Find all two-hop paths
    print("  🔍 Finding all two-hop paths...")
    all_paths = path_gen.find_all_two_hop_paths()
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
    
    print(f"  ✓ Filtered to {len(filtered_paths)} unique paths")
    
    # Generate questions
    print("  📝 Generating questions...")
    questions = question_gen.generate_questions_from_paths(
        filtered_paths,
        target_count=target_count,
        tf_ratio=0.4,  # 40% True/False
        yn_ratio=0.3,  # 30% Yes/No
        mc_ratio=0.3   # 30% Multiple Choice
    )
    
    print(f"  ✓ Generated {len(questions)} questions")
    
    # Save to file
    print(f"  💾 Saving to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(q) for q in questions], f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Saved {len(questions)} questions to {output_file}")
    
    # Print statistics
    print("\n📊 Statistics:")
    print(f"  Total questions: {len(questions)}")
    print(f"  True/False: {len([q for q in questions if q.question_type == 'true_false'])}")
    print(f"  Yes/No: {len([q for q in questions if q.question_type == 'yes_no'])}")
    print(f"  Multiple Choice: {len([q for q in questions if q.question_type == 'multiple_choice'])}")
    print(f"  All are 2-hop questions")
    
    return questions


if __name__ == '__main__':
    import os
    questions = generate_two_hop_evaluation_dataset(target_count=2000)
    print(f"\n✅ Generated {len(questions)} evaluation questions!")

