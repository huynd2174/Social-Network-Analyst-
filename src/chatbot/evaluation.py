"""
Evaluation Dataset Generator for K-pop Chatbot

This module generates a comprehensive evaluation dataset with:
- True/False questions
- Yes/No questions
- Multiple choice questions

All questions require multi-hop reasoning over the knowledge graph.

Target: 2000+ evaluation questions
- 700+ 1-hop questions
- 700+ 2-hop questions  
- 600+ 3-hop questions
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
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


class EvaluationDatasetGenerator:
    """
    Generates evaluation dataset for K-pop chatbot.
    
    Question types:
    1. True/False: Statement is true or false
    2. Yes/No: Question answerable with yes or no
    3. Multiple Choice: Select correct answer from options
    
    Hop levels:
    - 1-hop: Direct relationship (Artist-Group, Group-Company)
    - 2-hop: One intermediate (Artist-Group-Company)
    - 3-hop: Two intermediates (Song-Artist-Group-Company)
    """
    
    def __init__(self, knowledge_graph: Optional[KpopKnowledgeGraph] = None):
        """Initialize with knowledge graph."""
        self.kg = knowledge_graph or KpopKnowledgeGraph()
        
        # Cache frequently used data
        self._cache_data()
        
        # Question ID counter
        self.question_counter = 0
        
    def _cache_data(self):
        """Cache frequently used entities and relationships."""
        print("🔄 Caching knowledge graph data...")
        
        # Groups with members
        self.groups_with_members = {}
        for group in self.kg.get_entities_by_type("Group"):
            members = self.kg.get_group_members(group)
            if len(members) >= 2:  # Only groups with 2+ members
                self.groups_with_members[group] = members
                
        # Groups with companies
        self.groups_with_companies = {}
        for group in self.kg.get_entities_by_type("Group"):
            company = self.kg.get_group_company(group)
            if company:
                self.groups_with_companies[group] = company
                
        # Companies with groups
        self.companies_with_groups = defaultdict(list)
        for group, company in self.groups_with_companies.items():
            self.companies_with_groups[company].append(group)
            
        # Filter companies with multiple groups
        self.companies_with_groups = {
            c: g for c, g in self.companies_with_groups.items()
            if len(g) >= 2
        }
        
        # Songs by groups/artists
        self.entity_songs = defaultdict(list)
        for edge in self.kg.edges:
            if edge.get('type') == 'SINGS':
                song = edge.get('source')
                artist = edge.get('target')
                if song and artist:
                    self.entity_songs[artist].append(song)
                    
        print(f"✅ Cached: {len(self.groups_with_members)} groups, "
              f"{len(self.companies_with_groups)} companies, "
              f"{len(self.entity_songs)} entities with songs")
              
    def _next_id(self) -> str:
        """Generate next question ID."""
        self.question_counter += 1
        return f"Q{self.question_counter:05d}"
        
    # =========== 1-HOP QUESTIONS ===========
    
    def generate_1hop_membership_tf(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 1-hop True/False questions about group membership."""
        questions = []
        groups = list(self.groups_with_members.keys())
        
        for _ in range(count):
            if len(groups) < 2:
                break
                
            group = random.choice(groups)
            members = self.groups_with_members[group]
            
            if random.random() > 0.5 and len(members) > 0:
                # True statement
                member = random.choice(members)
                question = f"{member} là thành viên của {group}."
                answer = "Đúng"
                explanation = f"{member} thực sự là thành viên của nhóm nhạc {group}."
            else:
                # False statement - pick member from different group
                other_groups = [g for g in groups if g != group and self.groups_with_members.get(g)]
                if other_groups:
                    other_group = random.choice(other_groups)
                    other_member = random.choice(self.groups_with_members[other_group])
                    question = f"{other_member} là thành viên của {group}."
                    answer = "Sai"
                    explanation = f"{other_member} không phải thành viên của {group}, mà là thành viên của {other_group}."
                else:
                    continue
                    
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="true_false",
                answer=answer,
                choices=[],
                hops=1,
                entities=[group],
                relationships=["MEMBER_OF"],
                explanation=explanation,
                difficulty="easy",
                category="membership"
            ))
            
        return questions
        
    def generate_1hop_membership_yn(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 1-hop Yes/No questions about group membership."""
        questions = []
        groups = list(self.groups_with_members.keys())
        
        for _ in range(count):
            if len(groups) < 2:
                break
                
            group = random.choice(groups)
            members = self.groups_with_members[group]
            
            if random.random() > 0.5 and len(members) > 0:
                member = random.choice(members)
                question = f"{member} có phải là thành viên của {group} không?"
                answer = "Có"
                explanation = f"Có, {member} là thành viên của {group}."
            else:
                other_groups = [g for g in groups if g != group and self.groups_with_members.get(g)]
                if other_groups:
                    other_group = random.choice(other_groups)
                    other_member = random.choice(self.groups_with_members[other_group])
                    question = f"{other_member} có phải là thành viên của {group} không?"
                    answer = "Không"
                    explanation = f"Không, {other_member} thuộc {other_group}, không phải {group}."
                else:
                    continue
                    
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="yes_no",
                answer=answer,
                choices=[],
                hops=1,
                entities=[group],
                relationships=["MEMBER_OF"],
                explanation=explanation,
                difficulty="easy",
                category="membership"
            ))
            
        return questions
        
    def generate_1hop_membership_mc(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 1-hop multiple choice about group membership."""
        questions = []
        groups = list(self.groups_with_members.keys())
        
        for _ in range(count):
            if len(groups) < 4:
                break
                
            # Select a member and their group
            group = random.choice(groups)
            members = self.groups_with_members.get(group, [])
            if not members:
                continue
                
            member = random.choice(members)
            
            # Create wrong choices (other groups)
            other_groups = [g for g in groups if g != group]
            if len(other_groups) < 3:
                continue
                
            wrong_choices = random.sample(other_groups, 3)
            
            # Shuffle choices
            all_choices = [group] + wrong_choices
            random.shuffle(all_choices)
            correct_index = all_choices.index(group)
            
            question = f"{member} là thành viên của nhóm nhạc nào?"
            
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="multiple_choice",
                answer=chr(65 + correct_index),
                choices=all_choices,
                hops=1,
                entities=[member, group],
                relationships=["MEMBER_OF"],
                explanation=f"{member} là thành viên của {group}.",
                difficulty="easy",
                category="membership"
            ))
            
        return questions
        
    def generate_1hop_company_tf(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 1-hop True/False about group-company relationship."""
        questions = []
        groups = list(self.groups_with_companies.keys())
        
        for _ in range(count):
            if len(groups) < 2:
                break
                
            group = random.choice(groups)
            company = self.groups_with_companies[group]
            
            if random.random() > 0.5:
                # True
                question = f"{group} thuộc công ty {company}."
                answer = "Đúng"
                explanation = f"{group} thực sự được quản lý bởi {company}."
            else:
                # False - different company
                other_companies = [c for c in self.companies_with_groups.keys() if c != company]
                if other_companies:
                    wrong_company = random.choice(other_companies)
                    question = f"{group} thuộc công ty {wrong_company}."
                    answer = "Sai"
                    explanation = f"{group} thuộc {company}, không phải {wrong_company}."
                else:
                    continue
                    
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="true_false",
                answer=answer,
                choices=[],
                hops=1,
                entities=[group, company],
                relationships=["MANAGED_BY"],
                explanation=explanation,
                difficulty="easy",
                category="company"
            ))
            
        return questions
        
    def generate_1hop_company_mc(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 1-hop multiple choice about group company."""
        questions = []
        groups = list(self.groups_with_companies.keys())
        companies = list(self.companies_with_groups.keys())
        
        for _ in range(count):
            if len(companies) < 4:
                break
                
            group = random.choice(groups)
            correct_company = self.groups_with_companies.get(group)
            if not correct_company:
                continue
                
            wrong_companies = [c for c in companies if c != correct_company]
            if len(wrong_companies) < 3:
                continue
                
            wrong_choices = random.sample(wrong_companies, 3)
            all_choices = [correct_company] + wrong_choices
            random.shuffle(all_choices)
            correct_index = all_choices.index(correct_company)
            
            question = f"Công ty nào quản lý {group}?"
            
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="multiple_choice",
                answer=chr(65 + correct_index),
                choices=all_choices,
                hops=1,
                entities=[group, correct_company],
                relationships=["MANAGED_BY"],
                explanation=f"{group} được quản lý bởi {correct_company}.",
                difficulty="easy",
                category="company"
            ))
            
        return questions
        
    def generate_1hop_member_count(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate questions about member count."""
        questions = []
        groups = list(self.groups_with_members.keys())
        
        for _ in range(count):
            group = random.choice(groups)
            members = self.groups_with_members[group]
            correct_count = len(members)
            
            if correct_count < 2:
                continue
                
            # Multiple choice with different counts
            wrong_counts = []
            for offset in [-2, -1, 1, 2]:
                if correct_count + offset > 0:
                    wrong_counts.append(correct_count + offset)
                    
            if len(wrong_counts) < 3:
                continue
                
            all_counts = [correct_count] + random.sample(wrong_counts, 3)
            all_choices = [str(c) for c in all_counts]
            random.shuffle(all_choices)
            correct_index = all_choices.index(str(correct_count))
            
            question = f"{group} có bao nhiêu thành viên?"
            
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="multiple_choice",
                answer=chr(65 + correct_index),
                choices=all_choices,
                hops=1,
                entities=[group],
                relationships=["MEMBER_OF"],
                explanation=f"{group} có {correct_count} thành viên.",
                difficulty="medium",
                category="membership"
            ))
            
        return questions
        
    # =========== 2-HOP QUESTIONS ===========
    
    def generate_2hop_artist_company_tf(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 2-hop True/False: Artist → Group → Company."""
        questions = []
        
        for _ in range(count):
            # Find artist with known group and company
            groups = list(self.groups_with_companies.keys())
            groups_with_both = [g for g in groups if g in self.groups_with_members and self.groups_with_members[g]]
            
            if not groups_with_both:
                break
                
            group = random.choice(groups_with_both)
            members = self.groups_with_members[group]
            company = self.groups_with_companies[group]
            member = random.choice(members)
            
            if random.random() > 0.5:
                # True
                question = f"{member} thuộc công ty {company}."
                answer = "Đúng"
                explanation = f"{member} là thành viên của {group}, và {group} thuộc {company}."
            else:
                # False
                other_companies = [c for c in self.companies_with_groups.keys() if c != company]
                if other_companies:
                    wrong_company = random.choice(other_companies)
                    question = f"{member} thuộc công ty {wrong_company}."
                    answer = "Sai"
                    explanation = f"{member} thuộc {group} → {company}, không phải {wrong_company}."
                else:
                    continue
                    
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="true_false",
                answer=answer,
                choices=[],
                hops=2,
                entities=[member, group, company],
                relationships=["MEMBER_OF", "MANAGED_BY"],
                explanation=explanation,
                difficulty="medium",
                category="artist_company"
            ))
            
        return questions
        
    def generate_2hop_same_company_yn(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 2-hop Yes/No: Do two groups share same company?"""
        questions = []
        companies = list(self.companies_with_groups.keys())
        
        for _ in range(count):
            if random.random() > 0.5:
                # Same company - Yes
                company = random.choice(companies)
                groups = self.companies_with_groups[company]
                if len(groups) < 2:
                    continue
                group1, group2 = random.sample(groups, 2)
                question = f"{group1} và {group2} có cùng công ty quản lý không?"
                answer = "Có"
                explanation = f"Có, cả {group1} và {group2} đều thuộc {company}."
            else:
                # Different company - No
                if len(companies) < 2:
                    continue
                company1, company2 = random.sample(companies, 2)
                group1 = random.choice(self.companies_with_groups[company1])
                group2 = random.choice(self.companies_with_groups[company2])
                question = f"{group1} và {group2} có cùng công ty quản lý không?"
                answer = "Không"
                explanation = f"Không, {group1} thuộc {company1}, còn {group2} thuộc {company2}."
                
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="yes_no",
                answer=answer,
                choices=[],
                hops=2,
                entities=[group1, group2],
                relationships=["MANAGED_BY"],
                explanation=explanation,
                difficulty="medium",
                category="same_company"
            ))
            
        return questions
        
    def generate_2hop_labelmates_mc(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 2-hop MC: Which group is labelmate of X?"""
        questions = []
        companies = list(self.companies_with_groups.keys())
        
        for _ in range(count):
            company = random.choice(companies)
            groups = self.companies_with_groups[company]
            
            if len(groups) < 2:
                continue
                
            group1, correct_labelmate = random.sample(groups, 2)
            
            # Wrong choices from different companies
            other_groups = []
            for c, g_list in self.companies_with_groups.items():
                if c != company:
                    other_groups.extend(g_list)
                    
            if len(other_groups) < 3:
                continue
                
            wrong_choices = random.sample(other_groups, 3)
            all_choices = [correct_labelmate] + wrong_choices
            random.shuffle(all_choices)
            correct_index = all_choices.index(correct_labelmate)
            
            question = f"Nhóm nào cùng công ty với {group1}?"
            
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="multiple_choice",
                answer=chr(65 + correct_index),
                choices=all_choices,
                hops=2,
                entities=[group1, correct_labelmate, company],
                relationships=["MANAGED_BY"],
                explanation=f"{group1} và {correct_labelmate} cùng thuộc {company}.",
                difficulty="medium",
                category="labelmates"
            ))
            
        return questions
        
    def generate_2hop_same_group_yn(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 2-hop Yes/No: Are two artists in the same group?"""
        questions = []
        groups = list(self.groups_with_members.keys())
        
        for _ in range(count):
            if random.random() > 0.5:
                # Same group - Yes
                group = random.choice(groups)
                members = self.groups_with_members[group]
                if len(members) < 2:
                    continue
                member1, member2 = random.sample(members, 2)
                question = f"{member1} và {member2} có cùng nhóm nhạc không?"
                answer = "Có"
                explanation = f"Có, cả {member1} và {member2} đều là thành viên của {group}."
            else:
                # Different group - No
                groups_with_members = [g for g in groups if len(self.groups_with_members.get(g, [])) >= 1]
                if len(groups_with_members) < 2:
                    continue
                group1, group2 = random.sample(groups_with_members, 2)
                member1 = random.choice(self.groups_with_members[group1])
                member2 = random.choice(self.groups_with_members[group2])
                question = f"{member1} và {member2} có cùng nhóm nhạc không?"
                answer = "Không"
                explanation = f"Không, {member1} thuộc {group1}, còn {member2} thuộc {group2}."
                
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="yes_no",
                answer=answer,
                choices=[],
                hops=2,
                entities=[member1, member2],
                relationships=["MEMBER_OF"],
                explanation=explanation,
                difficulty="medium",
                category="same_group"
            ))
            
        return questions
        
    # =========== 3-HOP QUESTIONS ===========
    
    def generate_3hop_artist_labelmate_tf(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 3-hop TF: Are two artists labelmates? (Artist→Group→Company←Group←Artist)"""
        questions = []
        companies = list(self.companies_with_groups.keys())
        
        for _ in range(count):
            if random.random() > 0.5:
                # Same company - True
                company = random.choice(companies)
                groups = self.companies_with_groups[company]
                groups_with_members = [g for g in groups if self.groups_with_members.get(g)]
                
                if len(groups_with_members) < 2:
                    continue
                    
                group1, group2 = random.sample(groups_with_members, 2)
                member1 = random.choice(self.groups_with_members[group1])
                member2 = random.choice(self.groups_with_members[group2])
                
                question = f"{member1} và {member2} thuộc cùng công ty quản lý."
                answer = "Đúng"
                explanation = f"{member1} ({group1}) và {member2} ({group2}) đều thuộc {company}."
            else:
                # Different company - False
                if len(companies) < 2:
                    continue
                    
                company1, company2 = random.sample(companies, 2)
                groups1 = [g for g in self.companies_with_groups[company1] if self.groups_with_members.get(g)]
                groups2 = [g for g in self.companies_with_groups[company2] if self.groups_with_members.get(g)]
                
                if not groups1 or not groups2:
                    continue
                    
                group1 = random.choice(groups1)
                group2 = random.choice(groups2)
                member1 = random.choice(self.groups_with_members[group1])
                member2 = random.choice(self.groups_with_members[group2])
                
                question = f"{member1} và {member2} thuộc cùng công ty quản lý."
                answer = "Sai"
                explanation = f"{member1} ({group1}) thuộc {company1}, còn {member2} ({group2}) thuộc {company2}."
                
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="true_false",
                answer=answer,
                choices=[],
                hops=3,
                entities=[member1, member2],
                relationships=["MEMBER_OF", "MANAGED_BY"],
                explanation=explanation,
                difficulty="hard",
                category="artist_labelmate"
            ))
            
        return questions
        
    def generate_3hop_company_of_artist_mc(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 3-hop MC: Which company does artist X belong to?"""
        questions = []
        companies = list(self.companies_with_groups.keys())
        
        for _ in range(count):
            # Find artist with known path to company
            company = random.choice(companies)
            groups = self.companies_with_groups[company]
            groups_with_members = [g for g in groups if self.groups_with_members.get(g)]
            
            if not groups_with_members:
                continue
                
            group = random.choice(groups_with_members)
            member = random.choice(self.groups_with_members[group])
            
            # Wrong choices
            wrong_companies = [c for c in companies if c != company]
            if len(wrong_companies) < 3:
                continue
                
            wrong_choices = random.sample(wrong_companies, 3)
            all_choices = [company] + wrong_choices
            random.shuffle(all_choices)
            correct_index = all_choices.index(company)
            
            question = f"{member} thuộc công ty nào?"
            
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="multiple_choice",
                answer=chr(65 + correct_index),
                choices=all_choices,
                hops=3,
                entities=[member, group, company],
                relationships=["MEMBER_OF", "MANAGED_BY"],
                explanation=f"{member} → {group} → {company}.",
                difficulty="hard",
                category="artist_company"
            ))
            
        return questions
        
    # =========== MAIN GENERATION ===========
    
    def generate_full_dataset(
        self,
        target_count: int = 2000,
        output_path: str = "data/evaluation_dataset.json"
    ) -> Dict:
        """
        Generate full evaluation dataset.
        
        Args:
            target_count: Target number of questions (minimum 2000)
            output_path: Path to save dataset
            
        Returns:
            Dataset statistics
        """
        print("🔄 Generating evaluation dataset...")
        
        all_questions = []
        
        # 1-hop questions (700+)
        print("  📝 Generating 1-hop questions...")
        all_questions.extend(self.generate_1hop_membership_tf(100))
        all_questions.extend(self.generate_1hop_membership_yn(100))
        all_questions.extend(self.generate_1hop_membership_mc(100))
        all_questions.extend(self.generate_1hop_company_tf(100))
        all_questions.extend(self.generate_1hop_company_mc(100))
        all_questions.extend(self.generate_1hop_member_count(100))
        
        # Additional 1-hop to reach ~700
        all_questions.extend(self.generate_1hop_membership_tf(50))
        all_questions.extend(self.generate_1hop_membership_yn(50))
        
        # 2-hop questions (700+)
        print("  📝 Generating 2-hop questions...")
        all_questions.extend(self.generate_2hop_artist_company_tf(150))
        all_questions.extend(self.generate_2hop_same_company_yn(150))
        all_questions.extend(self.generate_2hop_labelmates_mc(150))
        all_questions.extend(self.generate_2hop_same_group_yn(150))
        
        # Additional 2-hop to reach ~700
        all_questions.extend(self.generate_2hop_artist_company_tf(100))
        
        # 3-hop questions (600+)
        print("  📝 Generating 3-hop questions...")
        all_questions.extend(self.generate_3hop_artist_labelmate_tf(200))
        all_questions.extend(self.generate_3hop_company_of_artist_mc(200))
        
        # Additional 3-hop
        all_questions.extend(self.generate_3hop_artist_labelmate_tf(200))
        
        # Shuffle
        random.shuffle(all_questions)
        
        # Calculate statistics
        hop_counts = defaultdict(int)
        type_counts = defaultdict(int)
        category_counts = defaultdict(int)
        difficulty_counts = defaultdict(int)
        
        for q in all_questions:
            hop_counts[q.hops] += 1
            type_counts[q.question_type] += 1
            category_counts[q.category] += 1
            difficulty_counts[q.difficulty] += 1
            
        stats = {
            "total_questions": len(all_questions),
            "by_hops": dict(hop_counts),
            "by_type": dict(type_counts),
            "by_category": dict(category_counts),
            "by_difficulty": dict(difficulty_counts),
            "generated_at": datetime.now().isoformat()
        }
        
        # Save dataset
        dataset = {
            "metadata": stats,
            "questions": [asdict(q) for q in all_questions]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Generated {len(all_questions)} questions")
        print(f"   By hops: {dict(hop_counts)}")
        print(f"   By type: {dict(type_counts)}")
        print(f"   Saved to: {output_path}")
        
        return stats
        
    def load_dataset(self, path: str = "data/evaluation_dataset.json") -> List[Dict]:
        """Load evaluation dataset from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['questions']


def main():
    """Generate evaluation dataset."""
    generator = EvaluationDatasetGenerator()
    stats = generator.generate_full_dataset(target_count=2000)
    
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

