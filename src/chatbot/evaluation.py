"""
Evaluation Dataset Generator for K-pop Chatbot

This module generates a comprehensive evaluation dataset with:
- True/False questions
- Yes/No questions
- Multiple choice questions

All questions require multi-hop reasoning (≥2 hops) over the knowledge graph.

Target: 2000+ evaluation questions (ưu tiên 2-hop và 3-hop)
- ~60% 2-hop
- ~40% 3-hop

Optional: Can use ChatGPT/OpenAI API to generate additional questions.
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
import os
import sys

# Allow running as script: add project root and src to path
CURR_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURR_DIR, "..", ".."))
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from .knowledge_graph import KpopKnowledgeGraph
except ImportError:
    from knowledge_graph import KpopKnowledgeGraph

# Optional: ChatGPT support
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
                
        # Artists with companies (direct relationship)
        self.artists_with_companies = {}
        for artist in self.kg.get_entities_by_type("Artist"):
            companies = self.kg.get_artist_companies(artist)
            if companies:
                # Store first company for backward compatibility
                self.artists_with_companies[artist] = companies[0]
        
        # Companies with groups
        self.companies_with_groups = defaultdict(list)
        for group, company in self.groups_with_companies.items():
            self.companies_with_groups[company].append(group)
            
        # Companies with artists (direct)
        self.companies_with_artists = defaultdict(list)
        for artist, company in self.artists_with_companies.items():
            self.companies_with_artists[company].append(artist)
            
        # Filter companies with multiple groups or artists
        self.companies_with_groups = {
            c: g for c, g in self.companies_with_groups.items()
            if len(g) >= 2
        }
        self.companies_with_artists = {
            c: a for c, a in self.companies_with_artists.items()
            if len(a) >= 2
        }
        
        # Songs by groups/artists
        self.entity_songs = defaultdict(list)
        # First, get songs directly linked to artists/groups via SINGS edges
        for edge in self.kg.edges:
            if edge.get('type') == 'SINGS':
                song = edge.get('source')
                entity = edge.get('target')  # Can be Artist or Group
                if song and entity:
                    self.entity_songs[entity].append(song)
        
        # Also get songs for artists through their groups
        # This is important because many songs are linked to Groups, not Artists
        for artist in self.kg.get_entities_by_type("Artist"):
            artist_groups = self.kg.get_artist_groups(artist)
            for group in artist_groups:
                group_songs = self.kg.get_group_songs(group)
                for song in group_songs:
                    if song not in self.entity_songs[artist]:
                        self.entity_songs[artist].append(song)
                    
        print(f"✅ Cached: {len(self.groups_with_members)} groups, "
              f"{len(self.companies_with_groups)} companies with groups, "
              f"{len(self.companies_with_artists)} companies with artists, "
              f"{len(self.entity_songs)} entities with songs")
              
    def _next_id(self) -> str:
        """Generate next question ID."""
        self.question_counter += 1
        return f"Q{self.question_counter:05d}"
        
    # =========== 2-HOP QUESTIONS ===========
    
    def generate_2hop_artist_company_tf(self, count: int = 100) -> List[EvaluationQuestion]:
        """Generate 2-hop True/False: Artist → Group → Company or Artist → Company (direct)."""
        questions = []
        
        # Collect all artists with companies (through groups or direct)
        artists_with_company_info = []
        
        # Artists through groups (2-hop)
        for group in self.groups_with_companies.keys():
            if group in self.groups_with_members:
                members = self.groups_with_members[group]
                company = self.groups_with_companies[group]
                for member in members:
                    artists_with_company_info.append({
                        'artist': member,
                        'company': company,
                        'group': group,
                        'hops': 2,
                        'direct': False
                    })
        
        # Artists with direct company relationship (1-hop)
        for artist, company in self.artists_with_companies.items():
            artists_with_company_info.append({
                'artist': artist,
                'company': company,
                'group': None,
                'hops': 1,
                'direct': True
            })
        
        if not artists_with_company_info:
            return questions
        
        for _ in range(count):
            info = random.choice(artists_with_company_info)
            member = info['artist']
            company = info['company']
            group = info['group']
            is_direct = info['direct']
            
            # Templating cho câu hỏi
            if is_direct:
                true_templates = [
                    lambda: f"{member} thuộc công ty {company}.",
                    lambda: f"{member} có phải trực thuộc {company} không?",
                    lambda: f"{member} do {company} quản lý.",
                    lambda: f"{member} được quản lý bởi {company}.",
                ]
                false_templates = [
                    lambda wc: f"{member} thuộc công ty {wc}.",
                    lambda wc: f"{member} có phải trực thuộc {wc} không?",
                    lambda wc: f"{member} do {wc} quản lý.",
                    lambda wc: f"{member} được quản lý bởi {wc}.",
                ]
            else:
                true_templates = [
                    lambda: f"{member} thuộc công ty {company}.",
                    lambda: f"{member} có phải trực thuộc {company} qua nhóm {group} không?",
                    lambda: f"{member} (nhóm {group}) do {company} quản lý.",
                    lambda: f"{member} là thành viên {group} thuộc {company}.",
                ]
                false_templates = [
                    lambda wc: f"{member} thuộc công ty {wc}.",
                    lambda wc: f"{member} có phải trực thuộc {wc} qua nhóm {group} không?",
                    lambda wc: f"{member} (nhóm {group}) do {wc} quản lý.",
                    lambda wc: f"{member} là thành viên {group} thuộc {wc}.",
                ]
            
            if random.random() > 0.5:
                # True
                question = random.choice(true_templates)()
                answer = "Đúng"
                if is_direct:
                    explanation = f"{member} trực thuộc {company}."
                else:
                    explanation = f"{member} là thành viên của {group}, và {group} thuộc {company}."
            else:
                # False
                all_companies = list(set(list(self.companies_with_groups.keys()) + list(self.companies_with_artists.keys())))
                other_companies = [c for c in all_companies if c != company]
                if other_companies:
                    wrong_company = random.choice(other_companies)
                    question = random.choice(false_templates)(wrong_company)
                    answer = "Sai"
                    if is_direct:
                        explanation = f"{member} thuộc {company}, không phải {wrong_company}."
                    else:
                        explanation = f"{member} thuộc {group} → {company}, không phải {wrong_company}."
                else:
                    continue
                    
            entities = [member, company]
            relationships = ["MANAGED_BY"]
            if group:
                entities.insert(1, group)
                relationships = ["MEMBER_OF", "MANAGED_BY"]
                    
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="true_false",
                answer=answer,
                choices=[],
                hops=info['hops'],
                entities=entities,
                relationships=relationships,
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
                templates_yes = [
                    lambda: f"{group1} và {group2} có cùng công ty quản lý không?",
                    lambda: f"{group1} và {group2} đều trực thuộc {company} phải không?",
                    lambda: f"Cả {group1} và {group2} có chung công ty {company} chứ?",
                ]
                question = random.choice(templates_yes)()
                answer = "Có"
                explanation = f"Có, cả {group1} và {group2} đều thuộc {company}."
            else:
                # Different company - No
                if len(companies) < 2:
                    continue
                company1, company2 = random.sample(companies, 2)
                group1 = random.choice(self.companies_with_groups[company1])
                group2 = random.choice(self.companies_with_groups[company2])
                templates_no = [
                    lambda: f"{group1} và {group2} có cùng công ty quản lý không?",
                    lambda: f"{group1} có chung công ty với {group2} chứ?",
                    lambda: f"{group1} và {group2} cùng thuộc một công ty phải không?",
                ]
                question = random.choice(templates_no)()
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
            
            templates_mc = [
                lambda: f"Nhóm nào cùng công ty với {group1}?",
                lambda: f"Nhóm nào là đồng công ty với {group1} dưới {company}?",
                lambda: f"Nhóm nào khác cũng thuộc {company} giống {group1}?",
            ]
            question = random.choice(templates_mc)()
            
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
                templates_yes = [
                    lambda: f"{member1} và {member2} có cùng nhóm nhạc không?",
                    lambda: f"{member1} và {member2} đều thuộc nhóm {group} phải không?",
                    lambda: f"Cả {member1} và {member2} đều là thành viên của {group}, đúng không?",
                ]
                question = random.choice(templates_yes)()
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
                templates_no = [
                    lambda: f"{member1} và {member2} có cùng nhóm nhạc không?",
                    lambda: f"{member1} có chung nhóm với {member2} không?",
                    lambda: f"{member1} và {member2} thuộc cùng một nhóm chứ?",
                ]
                question = random.choice(templates_no)()
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
    
    def generate_3hop_song_company_tf(self, count: int = 100) -> List[EvaluationQuestion]:
        """
        Generate TF: Chuỗi Song→Artist→Group→Company (3 cạnh).
        """
        questions = []
        
        # Chuẩn bị candidates: song, artist, group, company
        candidates = []
        
        # Method 1: Artists with songs (direct or through groups)
        for artist, songs in self.entity_songs.items():
            # Check if this entity is actually an artist in a group
            for group, members in self.groups_with_members.items():
                if artist in members and group in self.groups_with_companies:
                    company = self.groups_with_companies[group]
                    for song in songs:
                        candidates.append((song, artist, group, company))
        
        # Method 2: Groups with songs directly (get songs from graph)
        for group in self.groups_with_companies.keys():
            if group in self.groups_with_members:
                company = self.groups_with_companies[group]
                members = self.groups_with_members[group]
                # Get songs for this group from graph
                group_songs = self.kg.get_group_songs(group)
                for member in members:
                    for song in group_songs:
                        candidates.append((song, member, group, company))
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        if not candidates:
            # Debug: Check why no candidates
            artists_in_groups = set()
            for group, members in self.groups_with_members.items():
                artists_in_groups.update(members)
            artists_with_songs = set(self.entity_songs.keys())
            intersection = artists_in_groups & artists_with_songs
            print(f"    ⚠️ No 3-hop candidates found")
            print(f"      - Entities with songs: {len(artists_with_songs)}")
            print(f"      - Artists in groups: {len(artists_in_groups)}")
            print(f"      - Intersection: {len(intersection)}")
            # Try alternative: groups with songs
            groups_with_songs = [e for e in self.entity_songs.keys() if e in self.groups_with_members]
            print(f"      - Groups with songs: {len(groups_with_songs)}")
            if len(groups_with_songs) > 0:
                sample_group = groups_with_songs[0]
                print(f"      - Sample group: {sample_group}, has company: {sample_group in self.groups_with_companies}")
            # Try getting songs directly from graph for groups
            groups_with_songs_from_graph = 0
            total_songs_from_graph = 0
            for group in list(self.groups_with_companies.keys())[:5]:  # Check first 5
                songs = self.kg.get_group_songs(group)
                if songs:
                    groups_with_songs_from_graph += 1
                    total_songs_from_graph += len(songs)
            print(f"      - Groups with songs (from graph, sample 5): {groups_with_songs_from_graph}, total songs: {total_songs_from_graph}")
            return questions
        
        print(f"    ✅ Found {len(candidates)} candidates for 3-hop TF questions")
        
        # Limit count to available candidates
        actual_count = min(count, len(candidates))
        if actual_count < count:
            print(f"    ⚠️ Only {actual_count} candidates available, generating {actual_count} questions instead of {count}")
        
        for _ in range(actual_count):
            song, artist, group, company = random.choice(candidates)
            true_templates = [
                lambda: f"{song} do {artist} (nhóm {group}) thực hiện, nhóm đó thuộc công ty {company}.",
                lambda: f"{song} là bài của {artist} (nhóm {group}); nhóm này trực thuộc {company}.",
                lambda: f"{song} do {artist} hát trong nhóm {group}; nhóm {group} thuộc {company}.",
                lambda: f"{artist} của nhóm {group} hát {song}; {group} được quản lý bởi {company}.",
            ]
            false_templates = [
                lambda wc: f"{song} do {artist} (nhóm {group}) thực hiện, nhóm đó thuộc công ty {wc}.",
                lambda wc: f"{song} là bài của {artist} (nhóm {group}); nhóm này trực thuộc {wc}.",
                lambda wc: f"{song} do {artist} hát trong nhóm {group}; nhóm {group} thuộc {wc}.",
                lambda wc: f"{artist} của nhóm {group} hát {song}; {group} được quản lý bởi {wc}.",
            ]
            
            if random.random() > 0.5:
                question = random.choice(true_templates)()
                answer = "Đúng"
                explanation = f"Bài hát {song} do {artist} (nhóm {group}) trình bày; nhóm {group} thuộc công ty {company}."
            else:
                other_companies = [c for c in self.companies_with_groups.keys() if c != company]
                if not other_companies:
                    continue
                wrong_company = random.choice(other_companies)
                question = random.choice(false_templates)(wrong_company)
                answer = "Sai"
                explanation = f"Bài hát {song} do {artist} (nhóm {group}) trình bày; nhóm {group} thuộc {company}, không phải {wrong_company}."
            
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="true_false",
                answer=answer,
                choices=[],
                hops=3,
                entities=[song, artist, group, company],
                relationships=["SINGS", "MEMBER_OF", "MANAGED_BY"],
                explanation=explanation,
                difficulty="medium",
                category="artist_company"
            ))
        
        return questions
        
    def generate_3hop_song_company_mc(self, count: int = 100) -> List[EvaluationQuestion]:
        """
        Generate MC: Công ty nào liên quan đến bài hát X qua Artist→Group→Company (3 cạnh).
        """
        questions = []
        
        candidates = []
        
        # Method 1: Artists with songs (direct or through groups)
        for artist, songs in self.entity_songs.items():
            # Check if this entity is actually an artist in a group
            for group, members in self.groups_with_members.items():
                if artist in members and group in self.groups_with_companies:
                    company = self.groups_with_companies[group]
                    for song in songs:
                        candidates.append((song, artist, group, company))
        
        # Method 2: Groups with songs directly (get songs from graph)
        for group in self.groups_with_companies.keys():
            if group in self.groups_with_members:
                company = self.groups_with_companies[group]
                members = self.groups_with_members[group]
                # Get songs for this group from graph
                group_songs = self.kg.get_group_songs(group)
                for member in members:
                    for song in group_songs:
                        candidates.append((song, member, group, company))
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        if not candidates:
            print(f"    ⚠️ No 3-hop candidates found for MC questions")
            return questions
        
        print(f"    ✅ Found {len(candidates)} candidates for 3-hop MC questions")
        
        companies = list(self.companies_with_groups.keys())
        
        # Limit count to available candidates
        actual_count = min(count, len(candidates))
        if actual_count < count:
            print(f"    ⚠️ Only {actual_count} candidates available, generating {actual_count} questions instead of {count}")
        
        for _ in range(actual_count):
            song, artist, group, company = random.choice(candidates)
            wrong_companies = [c for c in companies if c != company]
            if len(wrong_companies) < 3:
                continue
            wrong_choices = random.sample(wrong_companies, 3)
            all_choices = [company] + wrong_choices
            random.shuffle(all_choices)
            correct_index = all_choices.index(company)
            
            templates_mc = [
                lambda: f"{song} do {artist} (nhóm {group}) thực hiện, nhóm đó thuộc công ty nào?",
                lambda: f"{song} là bài của {artist} (nhóm {group}); nhóm này trực thuộc công ty nào?",
                lambda: f"{song} do {artist} hát trong nhóm {group}; nhóm {group} thuộc hãng nào?",
                lambda: f"{artist} của nhóm {group} hát {song}; {group} do công ty nào quản lý?",
            ]
            question = random.choice(templates_mc)()
            
            questions.append(EvaluationQuestion(
                id=self._next_id(),
                question=question,
                question_type="multiple_choice",
                answer=chr(65 + correct_index),
                choices=all_choices,
                hops=3,
                entities=[song, artist, group, company],
                relationships=["SINGS", "MEMBER_OF", "MANAGED_BY"],
                explanation=f"Bài hát {song} do {artist} (nhóm {group}) trình bày; nhóm {group} thuộc công ty {company}.",
                difficulty="hard",
                category="artist_company"
            ))
        
        return questions
        
    # =========== MAIN GENERATION ===========
    
    def generate_with_chatgpt(
        self,
        num_questions: int = 500,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ) -> List[EvaluationQuestion]:
        """
        Generate questions using ChatGPT (optional).
        
        Args:
            num_questions: Number of questions to generate
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use
            
        Returns:
            List of generated questions
        """
        if not OPENAI_AVAILABLE:
            print("⚠️ OpenAI library not installed. Install with: pip install openai")
            return []
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key parameter")
            return []
        
        openai.api_key = api_key
        
        # Prepare knowledge graph info
        stats = self.kg.get_statistics()
        sample_groups = list(self.kg.get_entities_by_type("Group"))[:10]
        sample_artists = list(self.kg.get_entities_by_type("Artist"))[:10]
        
        context = f"""
        Knowledge Graph về K-pop:
        - Tổng số entities: {stats['total_nodes']}
        - Entity types: {', '.join(list(stats['entity_types'].keys())[:5])}
        - Relationship types: {', '.join(list(stats['relationship_types'].keys())[:5])}
        - Sample groups: {', '.join(sample_groups)}
        - Sample artists: {', '.join(sample_artists)}
        """
        
        questions = []
        batch_size = 20  # Smaller batches for better quality
        
        print(f"  🤖 Generating {num_questions} questions with ChatGPT...")
        
        for i in range(0, num_questions, batch_size):
            current_batch = min(batch_size, num_questions - i)
            
            prompt = f"""
            {context}
            
            Tạo {current_batch} câu hỏi đánh giá về K-pop với yêu cầu:
            1. Câu hỏi phải dựa trên thông tin trong knowledge graph trên
            2. Câu hỏi phải yêu cầu multi-hop reasoning (1-hop, 2-hop, hoặc 3-hop)
            3. Các loại câu hỏi:
               - True/False: "Jungkook là thành viên của BTS." → Đúng/Sai
               - Yes/No: "Jungkook có phải thành viên của BTS không?" → Có/Không
               - Multiple Choice: "Jungkook thuộc công ty nào?" với 4 lựa chọn → A/B/C/D
            
            4. Phân bố hops: 40% 1-hop, 40% 2-hop, 20% 3-hop
            
            Trả về JSON format (chỉ JSON, không có markdown):
            {{
                "questions": [
                    {{
                        "question": "...",
                        "question_type": "true_false|yes_no|multiple_choice",
                        "answer": "Đúng|Sai|Có|Không|A|B|C|D",
                        "choices": ["option1", "option2", "option3", "option4"] (chỉ cho multiple_choice),
                        "hops": 1 hoặc 2 hoặc 3,
                        "explanation": "...",
                        "category": "membership|company|song|...",
                        "difficulty": "easy|medium|hard"
                    }}
                ]
            }}
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Bạn là chuyên gia tạo câu hỏi đánh giá về K-pop. Trả về JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                
                data = json.loads(content)
                batch_questions = data.get("questions", [])
                
                # Convert to EvaluationQuestion
                for q_data in batch_questions:
                    try:
                        q = EvaluationQuestion(
                            id=self._next_id(),
                            question=q_data.get("question", ""),
                            question_type=q_data.get("question_type", "true_false"),
                            answer=q_data.get("answer", ""),
                            choices=q_data.get("choices", []),
                            hops=q_data.get("hops", 1),
                            entities=[],  # Will be extracted later
                            relationships=[],  # Will be extracted later
                            explanation=q_data.get("explanation", ""),
                            difficulty=q_data.get("difficulty", "medium"),
                            category=q_data.get("category", "general")
                        )
                        questions.append(q)
                    except Exception as e:
                        print(f"    ⚠️ Skipping invalid question: {e}")
                        continue
                
                print(f"    Generated {len(questions)}/{num_questions} questions...")
                
                # Rate limiting
                import time
                time.sleep(1)
                
            except Exception as e:
                print(f"    ⚠️ Error generating batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"  ✅ Generated {len(questions)} questions with ChatGPT")
        return questions
    
    def generate_full_dataset(
        self,
        target_count: int = 2000,
        output_path: str = "data/evaluation_dataset.json",
        use_chatgpt: bool = False,
        chatgpt_ratio: float = 0.2  # 20% from ChatGPT, 80% from graph
    ) -> Dict:
        """
        Generate full evaluation dataset.
        
        Args:
            target_count: Target number of questions (minimum 2000)
            output_path: Path to save dataset
            use_chatgpt: Whether to use ChatGPT for some questions
            chatgpt_ratio: Ratio of questions from ChatGPT (0.0-1.0)
            
        Returns:
            Dataset statistics
        """
        print("🔄 Generating evaluation dataset...")
        
        all_questions = []
        
        # Calculate distribution
        if use_chatgpt and OPENAI_AVAILABLE:
            chatgpt_count = int(target_count * chatgpt_ratio)
            graph_count = target_count - chatgpt_count
            print(f"  📊 Distribution: {graph_count} from graph, {chatgpt_count} from ChatGPT")
        else:
            graph_count = target_count
            chatgpt_count = 0
            if use_chatgpt:
                print("  ⚠️ ChatGPT requested but not available. Using graph-only generation.")
        
        # Generate questions from graph
        # Adjust counts to reach graph_count
        if graph_count >= 2000:
            # Full generation: ưu tiên 2-hop nhiều hơn 3-hop
            print("  📝 Generating 2-hop questions...")
            all_questions.extend(self.generate_2hop_artist_company_tf(1200))
            all_questions.extend(self.generate_2hop_same_company_yn(900))
            all_questions.extend(self.generate_2hop_labelmates_mc(900))
            all_questions.extend(self.generate_2hop_same_group_yn(800))
            
            print("  📝 Generating 3-hop questions (chuỗi Song→Artist→Group→Company)...")
            all_questions.extend(self.generate_3hop_song_company_tf(500))
            all_questions.extend(self.generate_3hop_song_company_mc(500))
        else:
            # Proportional generation
            ratio_2hop = 0.75
            ratio_3hop = 0.25
            
            print("  📝 Generating questions from graph...")
            all_questions.extend(self.generate_2hop_artist_company_tf(int(graph_count * ratio_2hop * 0.35)))
            all_questions.extend(self.generate_2hop_same_company_yn(int(graph_count * ratio_2hop * 0.30)))
            all_questions.extend(self.generate_2hop_labelmates_mc(int(graph_count * ratio_2hop * 0.20)))
            all_questions.extend(self.generate_2hop_same_group_yn(int(graph_count * ratio_2hop * 0.15)))
            
            all_questions.extend(self.generate_3hop_song_company_tf(int(graph_count * ratio_3hop * 0.5)))
            all_questions.extend(self.generate_3hop_song_company_mc(int(graph_count * ratio_3hop * 0.5)))
        
        # Generate with ChatGPT if requested
        if chatgpt_count > 0:
            chatgpt_questions = self.generate_with_chatgpt(num_questions=chatgpt_count)
            all_questions.extend(chatgpt_questions)
            print(f"  ✅ Added {len(chatgpt_questions)} questions from ChatGPT")
        
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
    # Tăng target_count để tạo nhiều câu hỏi hơn
    stats = generator.generate_full_dataset(target_count=4800, output_path="data/kpop_eval_2000_multihop_max3hop.json")
    
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

