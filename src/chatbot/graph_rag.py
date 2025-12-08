"""
GraphRAG Module for K-pop Knowledge Graph

This module implements Graph-based Retrieval Augmented Generation (GraphRAG)
for the K-pop knowledge graph. It combines graph traversal with semantic
similarity search to retrieve relevant context for answering questions.

Key Features:
- Entity extraction from queries
- Graph-based context retrieval
- Semantic similarity matching
- Multi-hop relationship traversal
- Context ranking and filtering
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import os
import re

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Using keyword-based retrieval.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ faiss not installed. Using numpy-based similarity search.")

from .knowledge_graph import KpopKnowledgeGraph


class GraphRAG:
    """
    Graph-based Retrieval Augmented Generation for K-pop Knowledge Graph.
    
    ✅ GraphRAG = Retrieval layer trên đồ thị tri thức (Knowledge Graph)
    
    🎯 NHIỆM VỤ DUY NHẤT: TÌM từ đồ thị những thông tin liên quan nhất tới câu hỏi
    
    GraphRAG LÀM:
    ✅ 1. Tìm thực thể chính trong câu hỏi (Entity extraction)
    ✅ 2. Tìm neighbors / hàng xóm gần nhất (Graph traversal)
    ✅ 3. Tìm đường đi (paths) giữa các entity (Path finding)
    ✅ 4. Chuyển thành "context" cho LLM (Format triples/text)
    
    GraphRAG KHÔNG LÀM:
    ❌ Không diễn giải
    ❌ Không tóm tắt
    ❌ Không bịa thông tin
    ❌ Không suy luận multi-hop (do MultiHopReasoner làm)
    ❌ Không tạo câu trả lời (do LLM làm)
    
    📌 GraphRAG chỉ là "Retrieval layer" của chatbot.
    Reasoning và answer generation do các component khác thực hiện.
    
    Combines:
    1. Entity extraction from natural language queries
    2. Graph traversal for structured context
    3. Semantic embedding for similarity matching
    4. Multi-hop path finding
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KpopKnowledgeGraph] = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_cache: bool = True,
        llm_for_understanding: Optional[Any] = None
    ):
        """
        Initialize GraphRAG.
        
        Args:
            knowledge_graph: Pre-built knowledge graph (will create if None)
            embedding_model: Sentence transformer model for embeddings
            use_cache: Whether to cache embeddings
            llm_for_understanding: Optional LLM for understanding queries (entity extraction + intent detection)
        """
        self.kg = knowledge_graph or KpopKnowledgeGraph()
        self.embedding_model_name = embedding_model
        self.use_cache = use_cache
        self.llm_for_understanding = llm_for_understanding  # LLM để hiểu câu hỏi
        
        # Initialize embedding model
        self.embedder = None
        self.entity_embeddings = None
        self.entity_ids = []
        self.faiss_index = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_embeddings()
        else:
            print("⚠️ Running in keyword-only mode (no semantic embeddings)")
            
        # Entity patterns for extraction
        self._init_entity_patterns()
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """
        Normalize entity name bằng cách remove suffixes trong parentheses.
        
        Ví dụ:
        - "Lisa (ca sĩ)" → "Lisa"
        - "BLACKPINK (nhóm nhạc)" → "BLACKPINK"
        - "BTS (rapper)" → "BTS"
        
        Args:
            entity_name: Entity name có thể có đuôi
            
        Returns:
            Base name (không có đuôi)
        """
        # Remove suffixes trong parentheses: (ca sĩ), (nhóm nhạc), (rapper), etc.
        # Pattern: (.*) ở cuối string
        import re
        # Match pattern: space + (anything) ở cuối
        normalized = re.sub(r'\s*\([^)]+\)\s*$', '', entity_name)
        return normalized.strip()
        
    def _init_embeddings(self):
        """Initialize sentence transformer and build entity embeddings."""
        print(f"🔄 Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        # Check for cached embeddings
        cache_path = "data/entity_embeddings.npz"
        if self.use_cache and os.path.exists(cache_path):
            print("📂 Loading cached embeddings...")
            data = np.load(cache_path, allow_pickle=True)
            self.entity_embeddings = data['embeddings']
            self.entity_ids = data['entity_ids'].tolist()
        else:
            print("🔄 Building entity embeddings...")
            self._build_entity_embeddings()
            if self.use_cache:
                np.savez(
                    cache_path,
                    embeddings=self.entity_embeddings,
                    entity_ids=self.entity_ids
                )
                
        # Build FAISS index
        self._build_faiss_index()
        
    def _build_entity_embeddings(self):
        """Build embeddings for all entities."""
        texts = []
        self.entity_ids = []
        
        for node_id, data in self.kg.graph.nodes(data=True):
            # Create text representation of entity
            text = self._entity_to_text(node_id, data)
            texts.append(text)
            self.entity_ids.append(node_id)
            
        # Batch encode
        self.entity_embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=64
        )
        
        print(f"✅ Built embeddings for {len(self.entity_ids)} entities")
        
    def _entity_to_text(self, entity_id: str, data: Dict) -> str:
        """Convert entity to text representation for embedding."""
        parts = [entity_id]
        
        # Add type
        if 'label' in data:
            parts.append(f"loại: {data['label']}")
            
        # Add title if different
        title = data.get('title', '')
        if title and title != entity_id:
            parts.append(title)
            
        # Add infobox info
        infobox = data.get('infobox', {})
        if infobox:
            # Key fields
            for key in ['Thể loại', 'Năm hoạt động', 'Hãng đĩa', 'Thành viên']:
                if key in infobox and infobox[key]:
                    parts.append(f"{key}: {infobox[key]}")
                    
        return " | ".join(parts)
        
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE or self.entity_embeddings is None:
            return
            
        dim = self.entity_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        normalized = self.entity_embeddings / np.linalg.norm(
            self.entity_embeddings, axis=1, keepdims=True
        )
        self.faiss_index.add(normalized.astype('float32'))
        
        print(f"✅ Built FAISS index with {self.faiss_index.ntotal} vectors")
        
    def _init_entity_patterns(self):
        """Initialize regex patterns for entity extraction."""
        # Common K-pop group and artist name patterns
        self.entity_patterns = {
            'group': [
                r'\b(BTS|EXO|BLACKPINK|TWICE|NCT|SEVENTEEN|Stray Kids|ITZY|aespa|NewJeans)\b',
                r'\b(Red Velvet|Girls\' Generation|Super Junior|Big Bang|2NE1|f\(x\))\b',
                r'\b(ENHYPEN|TXT|LE SSERAFIM|IVE|NMIXX|(G)I-dle|Kep1er)\b',
                r'\b(GOT7|Monsta X|iKON|WINNER|MAMAMOO|GFRIEND|LOONA)\b',
                r'\b(SHINee|2PM|B.A.P|Block B|VIXX|BTOB|BEAST|Highlight)\b',
            ],
            'company': [
                r'\b(SM Entertainment|JYP Entertainment|YG Entertainment|HYBE|Big Hit)\b',
                r'\b(Cube Entertainment|Starship Entertainment|Pledis Entertainment)\b',
                r'\b(FNC Entertainment|Woollim Entertainment|RBW Entertainment)\b',
            ]
        }
        
    def extract_entities(self, query: str) -> List[Dict]:
        """
        Extract potential entities from a natural language query.
        
        Sử dụng 2 phương pháp:
        1. Pattern matching + Semantic search (nhanh, chính xác cho entity names)
        2. LLM understanding (nếu có) - hiểu ngữ cảnh tốt hơn, xử lý câu hỏi phức tạp
        
        Args:
            query: User's question
            
        Returns:
            List of extracted entities with types
        """
        entities = []
        query_lower = query.lower()
        
        # ============================================
        # PHƯƠNG PHÁP 1: Rule + KG + Semantic Search (ƯU TIÊN - Fast, An toàn)
        # ============================================
        # ✅ CHIẾN LƯỢC: Ưu tiên rule-based và KG lookup trước
        # - Pattern matching: Regex patterns cho groups, companies
        # - KG lookup: Tìm entities trong Knowledge Graph (quoted, capitalized, context patterns)
        # - Semantic search: FAISS + embeddings (nếu available)
        # Tất cả đều có threshold/validation để đảm bảo chất lượng
        
        # 1a. Pattern-based extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match,
                        'type': entity_type,
                        'method': 'pattern'
                    })
                    
        # 1b. Knowledge graph lookup - Tìm tất cả entities có thể có trong query
        # Extract potential entity names từ nhiều ngữ cảnh khác nhau:
        # - Quoted strings: "BTS", 'BLACKPINK'
        # - Capitalized words: BTS, BLACKPINK, Lisa, Jennie
        # - Words after keywords: "nhóm BTS", "ca sĩ Lisa", "công ty YG"
        # - Words before keywords: "BTS là nhóm", "Lisa thuộc nhóm"
        
        # Method 1: Quoted strings
        quoted_names = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        for match in quoted_names:
            name = match[0] or match[1]
            if name:
                results = self.kg.search_entities(name, limit=1)
                if results and results[0]['score'] > 0.7:
                    entities.append({
                        'text': results[0]['id'],
                        'type': results[0]['type'],
                        'method': 'kg_lookup_quoted',
                        'score': results[0]['score']
                    })
        
        # Method 2: Capitalized words (tên riêng)
        capitalized_words = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', query)
        for name in capitalized_words:
            # Skip common words
            if name.lower() not in ['có', 'không', 'và', 'với', 'của', 'là', 'thuộc', 'trong', 'từ']:
                results = self.kg.search_entities(name, limit=1)
                if results and results[0]['score'] > 0.7:
                    entities.append({
                        'text': results[0]['id'],
                        'type': results[0]['type'],
                        'method': 'kg_lookup_capitalized',
                        'score': results[0]['score']
                    })
        
        # Method 3: Tìm entities sau keywords (ngữ cảnh: "nhóm X", "ca sĩ Y", "công ty Z")
        context_patterns = [
            (r'(nhóm|group|ban nhạc)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', 'Group'),
            (r'(ca sĩ|nghệ sĩ|artist|singer|idol)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', 'Artist'),
            (r'(công ty|company|label|hãng đĩa)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', 'Company'),
            (r'(bài hát|song|ca khúc|track)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', 'Song'),
        ]
        for pattern, entity_type in context_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                name = match[1] if isinstance(match, tuple) else match
                if name:
                    results = self.kg.search_entities(name, limit=1)
                    if results and results[0]['score'] > 0.6:
                        entities.append({
                            'text': results[0]['id'],
                            'type': results[0]['type'],
                            'method': f'kg_lookup_context_{entity_type.lower()}',
                            'score': results[0]['score']
                        })
        
        # Method 4: Tìm entities trước keywords (ngữ cảnh: "X là nhóm", "Y thuộc công ty")
        before_keyword_patterns = [
            (r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(là|thuộc|belongs to|is)\s+(nhóm|group|ban nhạc)', 'Group'),
            (r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(là|thuộc|belongs to|is)\s+(ca sĩ|nghệ sĩ|artist)', 'Artist'),
            (r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(thuộc|belongs to|is)\s+(công ty|company)', 'Company'),
        ]
        for pattern, entity_type in before_keyword_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                if name:
                    results = self.kg.search_entities(name, limit=1)
                    if results and results[0]['score'] > 0.6:
                        entities.append({
                            'text': results[0]['id'],
                            'type': results[0]['type'],
                            'method': f'kg_lookup_before_keyword_{entity_type.lower()}',
                            'score': results[0]['score']
                        })
        
        # Method 5: Tìm tất cả nodes trong KG và check xem có trong query không (fuzzy match)
        # QUAN TRỌNG: Xử lý lowercase names như "jennie", "jisoo", "lisa"
        # Lấy tất cả entity names từ KG (cached để tránh chậm)
        if not hasattr(self, '_all_entity_names'):
            self._all_entity_names = list(self.kg.graph.nodes())
        
        # Cache lowercase mapping để tìm nhanh hơn
        # QUAN TRỌNG: Xử lý node có đuôi như "Lisa (ca sĩ)", "BLACKPINK (nhóm nhạc)"
        if not hasattr(self, '_entity_lowercase_map'):
            self._entity_lowercase_map = {}
            self._entity_base_name_map = {}  # Map base name (không có đuôi) → full name
            
            for name in self._all_entity_names:
                # Map full name lowercase
                self._entity_lowercase_map[name.lower()] = name
                
                # Extract base name (remove suffixes như "(ca sĩ)", "(nhóm nhạc)")
                base_name = self._normalize_entity_name(name)
                if base_name != name:
                    # Map base name → full name
                    if base_name.lower() not in self._entity_base_name_map:
                        self._entity_base_name_map[base_name.lower()] = []
                    self._entity_base_name_map[base_name.lower()].append(name)
        
        query_words = query_lower.split()
        # Tìm từng word trong query (case-insensitive)
        for word in query_words:
            if len(word) < 3:  # Skip short words
                continue
            
            # Method 5a: Exact match (case-insensitive) - với full name
            if word in self._entity_lowercase_map:
                entity_name = self._entity_lowercase_map[word]
                # Check xem đã có chưa
                if not any(e['text'].lower() == entity_name.lower() for e in entities):
                    entity_data = self.kg.get_entity(entity_name)
                    if entity_data:
                        entities.append({
                            'text': entity_name,
                            'type': entity_data.get('label', 'Unknown'),
                            'method': 'kg_lookup_fuzzy_exact',
                            'score': 0.9
                        })
                        if len(entities) >= 5:  # Đủ rồi
                            break
                    continue
            
            # Method 5a2: Match với base name (không có đuôi)
            # Ví dụ: query "lisa" → match với "Lisa (ca sĩ)"
            if word in self._entity_base_name_map:
                for entity_name in self._entity_base_name_map[word]:
                    if not any(e['text'].lower() == entity_name.lower() for e in entities):
                        entity_data = self.kg.get_entity(entity_name)
                        if entity_data:
                            entities.append({
                                'text': entity_name,
                                'type': entity_data.get('label', 'Unknown'),
                                'method': 'kg_lookup_base_name',
                                'score': 0.95  # High score vì match chính xác base name
                            })
                            if len(entities) >= 5:  # Đủ rồi
                                break
            
            # Method 5b: Partial match - word là substring của entity name (hoặc base name)
            for entity_name in self._all_entity_names[:1000]:  # Limit để tránh chậm
                entity_lower = entity_name.lower()
                base_name = self._normalize_entity_name(entity_name).lower()
                
                # Check nếu word match với full name hoặc base name
                if (word in entity_lower and len(word) >= 3) or (word in base_name and len(word) >= 3):
                    # Check xem đã có chưa
                    if not any(e['text'].lower() == entity_name.lower() for e in entities):
                        entity_data = self.kg.get_entity(entity_name)
                        if entity_data:
                            # Chỉ thêm nếu là Artist hoặc Group (tránh false positives)
                            entity_type = entity_data.get('label', '')
                            if entity_type in ['Artist', 'Group', 'Company']:
                                entities.append({
                                    'text': entity_name,
                                    'type': entity_type,
                                    'method': 'kg_lookup_fuzzy_partial',
                                    'score': 0.7
                                })
                                if len(entities) >= 5:  # Đủ rồi
                                    break
                    
        # 1c. Semantic similarity search (if available)
        if self.embedder:
            similar_entities = self.semantic_search(query, top_k=3)
            for entity, score in similar_entities:
                if score > 0.5:  # Threshold
                    entities.append({
                        'text': entity,
                        'type': self.kg.get_entity_type(entity),
                        'method': 'semantic',
                        'score': score
                    })
        
        # ============================================
        # PHƯƠNG PHÁP 2: LLM Understanding (FALLBACK/AUGMENTATION + INTENT DETECTION)
        # ============================================
        # ✅ CHIẾN LƯỢC: LLM dùng để:
        # 1. FALLBACK: Khi rule/semantic không tìm đủ entities (< 2)
        # 2. AUGMENTATION: Khi confidence thấp hoặc cần normalize (lowercase names)
        # 3. INTENT DETECTION: Detect intent chính xác hơn rule-based (xử lý biến thể ngôn ngữ)
        #    - "cùng một nhóm nhạc" → same_group (rule có thể miss từ "một")
        #    - "thuộc nhóm nhạc nào" → membership (rule có thể miss biến thể)
        # - Parse: Extract entities, detect intent, detect hop depth
        # 
        # ⚠️ QUAN TRỌNG: 
        # - LLM CHỈ parse câu hỏi → KHÔNG làm reasoning
        # - Tất cả kết quả từ LLM PHẢI được validate với KG + threshold
        llm_intent = None
        llm_metadata = {}
        if self.llm_for_understanding:
            # ✅ LUÔN gọi LLM để detect intent (quan trọng cho biến thể ngôn ngữ)
            # Gọi LLM trong các trường hợp:
            # 1. Không tìm đủ entities (< 2) - rule/semantic không đủ
            # 2. Query có lowercase names (jungkook, lisa) - pattern matching có thể miss
            # 3. Query có comparison keywords - cần detect intent chính xác
            # 4. Query có từ "một", "các", "nào" - biến thể ngôn ngữ tự nhiên
            should_use_llm = (
                not entities or 
                len(entities) < 2 or
                any(word.islower() and len(word) >= 4 for word in query_lower.split()) or  # Có lowercase words dài
                any(kw in query_lower for kw in ['và', 'and', 'cùng', 'same', 'có phải', 'phải', 'một', 'các', 'nào'])  # Câu hỏi so sánh hoặc biến thể
            )
            
            if should_use_llm:
                try:
                    llm_entities = self._extract_entities_with_llm(query)
                    # ✅ ALWAYS VALIDATE: Kết quả từ LLM phải được validate với KG + threshold
                    # Chiến lược an toàn: LLM làm fallback, nhưng phải validate trước khi dùng
                    for llm_entity in llm_entities:
                        # Chỉ thêm nếu chưa có và đã được validate với KG
                        if not any(e['text'].lower() == llm_entity['text'].lower() for e in entities):
                            entity_id = llm_entity.get('text', '')
                            if entity_id:
                                # Validate 1: Check entity tồn tại trong KG
                                entity_data = self.kg.get_entity(entity_id)
                                if entity_data:
                                    # Validate 2: Check confidence threshold (nếu có)
                                    llm_score = llm_entity.get('score', 0.5)
                                    # Nếu LLM trả về score thấp, verify thêm bằng KG search
                                    if llm_score < 0.6:
                                        kg_results = self.kg.search_entities(entity_id, limit=1)
                                        if kg_results and kg_results[0]['score'] > 0.6:
                                            # KG search confirm → dùng với score từ KG
                                            llm_entity['score'] = kg_results[0]['score']
                                            entities.append(llm_entity)
                                    else:
                                        # LLM score đủ cao → dùng luôn (đã validate với KG)
                                        entities.append(llm_entity)
                except Exception as e:
                    # Nếu LLM fail, fallback về pattern matching (an toàn)
                    pass
                    
        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity['text'] not in seen:
                seen.add(entity['text'])
                unique_entities.append(entity)
                
        return unique_entities
    
    def _extract_entities_with_llm(self, query: str) -> List[Dict]:
        """
        Sử dụng LLM nhỏ để hiểu đầu vào (intent + entity extraction).
        
        ✅ LLM nhỏ (≤1B params) phù hợp cho nhiệm vụ này vì:
        - Hiểu câu tiếng Việt tự nhiên tốt hơn rule
        - Normalize câu hỏi → mapping về template
        - Xử lý đa dạng ngôn ngữ tự nhiên
        
        LLM sẽ:
        - Detect intent (loại câu hỏi: membership, company, same group, comparison, etc.)
        - Extract entities (nghệ sĩ, nhóm, công ty) trong câu hỏi
        - Extract relations (MEMBER_OF, MANAGED_BY, FRIENDS_WITH, etc.)
        - Detect multi-hop depth (1-hop, 2-hop, 3-hop)
        - Hiểu ngữ cảnh phức tạp (lowercase names, nhiều entities, so sánh)
        
        ⚠️ QUAN TRỌNG: LLM CHỈ dùng để parse câu → KHÔNG làm reasoning
        Reasoning vẫn do đồ thị thực hiện (graph traversal, path search)
        
        Args:
            query: User's question
            
        Returns:
            List of extracted entities with types
        """
        if not self.llm_for_understanding:
            return []
        
        # Prompt cho LLM để hiểu đầu vào - CẢI THIỆN để detect intent, relations, multi-hop depth
        prompt = f"""Bạn là trợ lý AI chuyên về K-pop. Nhiệm vụ của bạn là HIỂU CÂU HỎI (parse input), không phải trả lời.

Câu hỏi: "{query}"

NHIỆM VỤ CỦA BẠN:
1. Detect Intent (loại câu hỏi):
   - membership: "X có phải thành viên Y không?", "X thuộc nhóm nào?"
   - same_group: "X và Y có cùng nhóm không?", "X và Y có cùng ban nhạc không?"
   - same_company: "X và Y có cùng công ty không?", "X và Y có cùng hãng đĩa không?"
   - company: "X thuộc công ty nào?", "Công ty nào quản lý X?"
   - song: "X hát bài nào?", "Bài hát nào của X?"
   - album: "X phát hành album nào?"
   - comparison: "X và Y có liên quan gì?", "So sánh X và Y"
   - multi_hop: "Bạn của X là ai?", "Những người cùng công ty với người hợp tác với X?"

2. Extract Entities (tìm TẤT CẢ entities):
   - Xử lý lowercase names: "jungkook" → "Jungkook", "lisa" → "Lisa"
   - Xử lý tên có đuôi: "Lisa (ca sĩ)" → "Lisa"
   - Hiểu ngữ cảnh: "jungkook và lisa" → cả 2 đều là entities
   - Tìm tất cả: nghệ sĩ, nhóm, công ty, bài hát, album

3. Extract Relations (loại quan hệ):
   - MEMBER_OF: "thành viên", "thuộc nhóm", "member"
   - MANAGED_BY: "công ty", "hãng đĩa", "quản lý", "company"
   - FRIENDS_WITH: "bạn", "quen", "chơi chung"
   - SINGS: "hát", "trình bày", "ca khúc"
   - RELEASED: "phát hành", "album"

4. Detect Multi-hop Depth:
   - 1-hop: "X thuộc nhóm nào?" (X → Group)
   - 2-hop: "X thuộc công ty nào?" (X → Group → Company)
   - 3-hop: "Bạn của X là ai?" (X → Friend → Friend's Friend)

Trả lời theo format JSON:
{{
    "intent": "membership|same_group|same_company|company|song|album|comparison|multi_hop",
    "entities": [
        {{"name": "tên thực thể", "type": "Artist|Group|Company|Song|Album"}}
    ],
    "relations": ["MEMBER_OF|MANAGED_BY|FRIENDS_WITH|SINGS|RELEASED"],
    "multi_hop_depth": 1 hoặc 2 hoặc 3,
    "question_type": "yes_no|true_false|fact|comparison"
}}

Chỉ trả về JSON, không thêm text khác."""

        try:
            response = self.llm_for_understanding.generate(
                prompt,
                context="",
                max_new_tokens=300,  # Tăng để đủ cho intent, relations, multi_hop_depth
                temperature=0.1  # Low temperature để output nhất quán
            )
            
            # Parse JSON response
            import json
            # Extract JSON from response (có thể có text thêm)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Lưu intent, relations, multi_hop_depth để dùng sau
                # (có thể lưu vào context hoặc return cùng với entities)
                intent = data.get('intent', '')
                relations = data.get('relations', [])
                multi_hop_depth = data.get('multi_hop_depth', 2)
                question_type = data.get('question_type', 'fact')
                
                # Extract entities
                entities = []
                for item in data.get('entities', []):
                    name = item.get('name', '').strip()
                    entity_type = item.get('type', '').strip()
                    if name:
                        # Tìm entity trong knowledge graph
                        results = self.kg.search_entities(name, limit=1)
                        if results and results[0]['score'] > 0.6:
                            entity_dict = {
                                'text': results[0]['id'],
                                'type': results[0]['type'],
                                'method': 'llm_understanding',
                                'score': results[0]['score']
                            }
                            # Thêm metadata từ LLM understanding
                            entity_dict['intent'] = intent
                            entity_dict['relations'] = relations
                            entity_dict['multi_hop_depth'] = multi_hop_depth
                            entity_dict['question_type'] = question_type
                            entities.append(entity_dict)
                        else:
                            # Nếu không tìm thấy trong KG, vẫn thêm với type từ LLM
                            entity_dict = {
                                'text': name,
                                'type': entity_type,
                                'method': 'llm_understanding',
                                'score': 0.5
                            }
                            # Thêm metadata
                            entity_dict['intent'] = intent
                            entity_dict['relations'] = relations
                            entity_dict['multi_hop_depth'] = multi_hop_depth
                            entity_dict['question_type'] = question_type
                            entities.append(entity_dict)
                
                return entities
        except Exception as e:
            # Nếu LLM fail, return empty list
            return []
        
        return []
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search entities by semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (entity_id, score) tuples
        """
        if not self.embedder:
            return []
            
        # Encode query
        query_embedding = self.embedder.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if FAISS_AVAILABLE and self.faiss_index:
            # Fast FAISS search
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                top_k
            )
            results = [
                (self.entity_ids[idx], float(dist))
                for idx, dist in zip(indices[0], distances[0])
            ]
        else:
            # Numpy fallback
            normalized = self.entity_embeddings / np.linalg.norm(
                self.entity_embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(normalized, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [
                (self.entity_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
        return results
        
    def retrieve_context(
        self,
        query: str,
        max_entities: int = 5,
        max_hops: int = 2,
        include_paths: bool = True
    ) -> Dict:
        """
        Retrieve relevant context for a query using GraphRAG.
        
        ✅ GraphRAG = 3 bước:
        1. Semantic Search: Tìm node gần nhất bằng vector search (FAISS + embeddings)
        2. Expand Subgraph: Từ node tìm được → mở rộng hàng xóm 1-2 hop → lấy subgraph
        3. Build Context: Chuyển subgraph → text/triples để feed vào LLM
        
        Args:
            query: User's question
            max_entities: Maximum number of entities to retrieve
            max_hops: Maximum hops for graph traversal (subgraph expansion)
            include_paths: Whether to include relationship paths
            
        Returns:
            Context dictionary with entities, relationships, and facts
        """
        context = {
            'query': query,
            'entities': [],
            'relationships': [],
            'facts': [],
            'paths': []
        }
        
        seen_entities = set()
        
        # ============================================
        # BƯỚC 1: SEMANTIC SEARCH
        # Tìm các node gần nhất với câu hỏi bằng vector search (FAISS + embeddings)
        # ============================================
        seed_entities = []
        
        # 1a. Pattern-based extraction (fallback nếu không có embeddings)
        extracted = self.extract_entities(query)
        for entity_info in extracted[:max_entities]:
            entity_id = entity_info['text']
            if entity_id not in seen_entities:
                seed_entities.append((entity_id, entity_info.get('score', 1.0), 'pattern'))
                seen_entities.add(entity_id)
        
        # 1b. Semantic Search (ưu tiên - tìm node gần nhất bằng FAISS)
        if self.embedder:
            similar_entities = self.semantic_search(query, top_k=max_entities)
            for entity_id, score in similar_entities:
                if entity_id not in seen_entities and score > 0.5:  # Threshold
                    seed_entities.append((entity_id, score, 'semantic'))
                    seen_entities.add(entity_id)
        
        # Sort by relevance (semantic search results first)
        seed_entities.sort(key=lambda x: (x[2] == 'semantic', x[1]), reverse=True)
        seed_entities = seed_entities[:max_entities]
        
        # ============================================
        # BƯỚC 2: EXPAND SUBGRAPH (multi-hop)
        # Từ node tìm được → mở rộng hàng xóm 1-2 hop → lấy subgraph liên quan
        # ============================================
        subgraph_entities = set()
        subgraph_relationships = []
        
        for entity_id, relevance, method in seed_entities:
            # Mở rộng subgraph từ entity này (1-2 hop)
            entity_context = self.kg.get_entity_context(entity_id, max_depth=max_hops)
            
            if entity_context:
                # Add main entity
                entity_data = entity_context.get('entity', {})
                context['entities'].append({
                    'id': entity_id,
                    'type': entity_data.get('label'),
                    'info': entity_data.get('infobox', {}),
                    'relevance': relevance,
                    'method': method
                })
                subgraph_entities.add(entity_id)
                
                # Add relationships (edges trong subgraph)
                # QUAN TRỌNG: Giới hạn số lượng relationships để tránh context quá lớn
                relationships = entity_context.get('relationships', [])
                # Chỉ lấy top 10 relationships quan trọng nhất cho mỗi entity
                # Ưu tiên relationships liên quan đến query
                query_lower = query.lower()
                scored_rels = []
                for rel in relationships:
                    score = 0.0
                    # Boost score nếu entity names trong relationship xuất hiện trong query
                    if rel.get('source', '').lower() in query_lower:
                        score += 1.0
                    if rel.get('target', '').lower() in query_lower:
                        score += 1.0
                    # Boost score cho các relationship types quan trọng
                    rel_type = rel.get('type', '')
                    if rel_type in ['MEMBER_OF', 'MANAGED_BY', 'SINGS', 'RELEASED']:
                        score += 0.5
                    scored_rels.append((rel, score))
                
                # Sort và lấy top 10
                scored_rels.sort(key=lambda x: x[1], reverse=True)
                for rel, _ in scored_rels[:10]:  # CHỈ LẤY TOP 10 RELATIONSHIPS
                    rel_key = (rel['source'], rel['type'], rel['target'])
                    if rel_key not in subgraph_relationships:
                        subgraph_relationships.append(rel_key)
                        context['relationships'].append(rel)
                        # Thêm các entities trong relationship vào subgraph
                        subgraph_entities.add(rel['source'])
                        subgraph_entities.add(rel['target'])
                        
                        # Giới hạn tổng số relationships
                        if len(context['relationships']) >= 30:  # Tối đa 30 relationships
                            break
                
                # Add connected entities (hàng xóm trong subgraph)
                # QUAN TRỌNG: Giới hạn số lượng để tránh context quá lớn
                connected = entity_context.get('connected_entities', {})
                # Chỉ lấy top 5 neighbors quan trọng nhất cho mỗi seed entity
                sorted_neighbors = sorted(
                    connected.items(),
                    key=lambda x: x[1].get('depth', 999),  # Ưu tiên 1-hop neighbors
                    reverse=False
                )[:5]  # CHỈ LẤY TOP 5 NEIGHBORS
                
                for neighbor_id, neighbor_info in sorted_neighbors:
                    if neighbor_id not in subgraph_entities:
                        neighbor_data = self.kg.get_entity(neighbor_id)
                        if neighbor_data:
                            context['entities'].append({
                                'id': neighbor_id,
                                'type': neighbor_info.get('type'),
                                'info': neighbor_data.get('infobox', {}),
                                'relevance': relevance * 0.8,  # Giảm relevance cho hàng xóm
                                'method': f'subgraph_expansion_{neighbor_info.get("depth", 1)}-hop'
                            })
                            subgraph_entities.add(neighbor_id)
                            
                            # Giới hạn tổng số entities trong context
                            if len(context['entities']) >= 30:  # Tối đa 30 entities
                                break
                    
                    if len(context['entities']) >= 30:  # Tối đa 30 entities
                        break
                
                # Generate facts from entity data
                facts = self._generate_facts(entity_id, entity_data)
                context['facts'].extend(facts)
        
        # Find paths between seed entities (multi-hop paths trong subgraph)
        if include_paths and len(seed_entities) >= 2:
            for i in range(len(seed_entities) - 1):
                for j in range(i + 1, min(i + 3, len(seed_entities))):
                    source = seed_entities[i][0]
                    target = seed_entities[j][0]
                    paths = self.kg.find_all_paths(source, target, max_hops=max_hops)
                    for path in paths[:3]:  # Limit paths
                        path_details = self.kg.get_path_details(path)
                        context['paths'].append({
                            'from': source,
                            'to': target,
                            'path': path,
                            'details': path_details
                        })
                        
        # ============================================
        # BƯỚC 2.5: GRAPH RANKING (Module B)
        # Xếp hạng độ liên quan của triples và lọc
        # ============================================
        context = self._rank_and_filter_context(context, query)
        
        return context
    
    def _rank_and_filter_context(self, context: Dict, query: str) -> Dict:
        """
        🔶 MODULE B - GRAPH RANKING
        Xếp hạng độ liên quan của triples và lọc.
        
        Lọc bằng:
        1. Similarity giữa node label với câu hỏi
        2. Độ quan trọng (degree / PageRank)
        3. Loại quan hệ phù hợp với câu hỏi
        
        Args:
            context: Context dictionary với entities, relationships, facts
            query: User's question
            
        Returns:
            Filtered và ranked context
        """
        query_lower = query.lower()
        
        # 1. Rank relationships (triples) by relevance
        ranked_relationships = []
        for rel in context['relationships']:
            score = 0.0
            
            # 1a. Similarity giữa node label với câu hỏi
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']
            
            # Check if entity names appear in query
            if source.lower() in query_lower:
                score += 0.3
            if target.lower() in query_lower:
                score += 0.3
            
            # 1b. Độ quan trọng (degree - số lượng connections)
            source_degree = len(list(self.kg.graph.neighbors(source))) if source in self.kg.graph else 0
            target_degree = len(list(self.kg.graph.neighbors(target))) if target in self.kg.graph else 0
            # Normalize degree score (0-0.2)
            degree_score = min((source_degree + target_degree) / 50.0, 0.2)
            score += degree_score
            
            # 1c. Loại quan hệ phù hợp với câu hỏi
            # Map query keywords to relevant relationship types
            rel_keywords = {
                'MEMBER_OF': ['thành viên', 'member', 'nhóm', 'group', 'thuộc', 'belongs'],
                'MANAGED_BY': ['công ty', 'company', 'hãng đĩa', 'label', 'quản lý', 'manage'],
                'SINGS': ['hát', 'sing', 'bài hát', 'song', 'ca khúc'],
                'RELEASED': ['phát hành', 'release', 'album', 'single'],
                'COLLAB_WITH': ['hợp tác', 'collab', 'collaborate', 'cùng'],
                'PRODUCED_BY': ['sản xuất', 'produce', 'producer']
            }
            
            for rel_type_key, keywords in rel_keywords.items():
                if rel_type == rel_type_key:
                    for keyword in keywords:
                        if keyword in query_lower:
                            score += 0.3
                            break
            
            ranked_relationships.append({
                'relationship': rel,
                'score': score
            })
        
        # Sort by score và lọc top relationships
        ranked_relationships.sort(key=lambda x: x['score'], reverse=True)
        # Giữ top 15 relationships có score > 0.1
        filtered_relationships = [
            item['relationship'] 
            for item in ranked_relationships 
            if item['score'] > 0.1
        ][:15]
        
        # 2. Rank entities by relevance
        ranked_entities = []
        for entity in context['entities']:
            score = entity.get('relevance', 0.0)
            entity_id = entity['id']
            
            # Boost score nếu entity name xuất hiện trong query
            if entity_id.lower() in query_lower:
                score += 0.5
            
            # Boost score nếu entity type phù hợp với query
            entity_type = entity.get('type', '')
            type_keywords = {
                'Group': ['nhóm', 'group', 'band'],
                'Artist': ['ca sĩ', 'artist', 'singer', 'idol'],
                'Song': ['bài hát', 'song', 'ca khúc'],
                'Company': ['công ty', 'company', 'label', 'hãng đĩa']
            }
            
            for type_key, keywords in type_keywords.items():
                if entity_type == type_key:
                    for keyword in keywords:
                        if keyword in query_lower:
                            score += 0.3
                            break
            
            ranked_entities.append({
                'entity': entity,
                'score': score
            })
        
        # Sort entities by score
        ranked_entities.sort(key=lambda x: x['score'], reverse=True)
        # QUAN TRỌNG: Giới hạn số lượng entities để tránh context quá lớn (1969 entities!)
        # CHỈ LẤY TOP 20 ENTITIES - đủ để trả lời nhưng không quá nhiều
        filtered_entities = [
            item['entity'] 
            for item in ranked_entities 
            if item['score'] > 0.1  # Chỉ lấy entities có score > 0.1
        ][:20]  # Tối đa 20 entities
        
        # 3. Filter facts (keep top 10 most relevant)
        facts = context['facts'][:10]
        
        # Update context với ranked và filtered data
        context['entities'] = filtered_entities
        context['relationships'] = filtered_relationships
        
        return context
        
    def _generate_facts(self, entity_id: str, entity_data: Dict) -> List[str]:
        """
        Generate natural language facts from entity data.
        
        ⚠️ LƯU Ý: Đây KHÔNG phải reasoning, chỉ là format dữ liệu từ đồ thị.
        Method này chỉ chuyển đổi thông tin từ entity data (infobox, relationships)
        thành câu văn tự nhiên để đưa vào context cho LLM.
        
        Tất cả facts đều lấy từ Knowledge Graph, không tự nghĩ ra.
        """
        facts = []
        entity_type = entity_data.get('label', 'Entity')
        infobox = entity_data.get('infobox', {})
        
        # Type-specific fact generation
        if entity_type == 'Group':
            if 'Thành viên' in infobox and infobox['Thành viên']:
                facts.append(f"{entity_id} có các thành viên: {infobox['Thành viên']}")
            if 'Năm hoạt động' in infobox:
                facts.append(f"{entity_id} hoạt động từ {infobox['Năm hoạt động']}")
            if 'Hãng đĩa' in infobox:
                facts.append(f"{entity_id} thuộc công ty {infobox['Hãng đĩa']}")
            if 'Thể loại' in infobox:
                facts.append(f"{entity_id} chơi nhạc {infobox['Thể loại']}")
                
            # Get members from relationships
            members = self.kg.get_group_members(entity_id)
            if members:
                facts.append(f"Thành viên của {entity_id}: {', '.join(members[:10])}")
                
        elif entity_type == 'Artist':
            groups = self.kg.get_artist_groups(entity_id)
            if groups:
                facts.append(f"{entity_id} là thành viên của: {', '.join(groups)}")
                
        elif entity_type == 'Company':
            groups = self.kg.get_company_groups(entity_id)
            if groups:
                facts.append(f"Các nhóm nhạc thuộc {entity_id}: {', '.join(groups[:10])}")
                
        return facts
        
    def format_context_for_llm(self, context: Dict, max_tokens: int = 20000) -> str:
        """
        BƯỚC 3: BUILD CONTEXT CHO LLM
        Chuyển subgraph → text/triples để feed vào mô hình 1B.
        
        Format retrieved context (subgraph) as a prompt for the LLM.
        Chuyển đổi subgraph (entities, relationships, paths) thành text format.
        
        Args:
            context: Retrieved context dictionary (từ subgraph expansion)
            max_tokens: Maximum tokens for context (default 20000, leaving room for query + response)
            
        Returns:
            Formatted context string (text/triples format cho LLM)
        """
        parts = []
        
        # ============================================
        # Format 1: Entities (Nodes trong subgraph)
        # ============================================
        if context['entities']:
            parts.append("=== THÔNG TIN THỰC THỂ (Từ Subgraph) ===")
            # Sort by relevance và giới hạn số lượng
            sorted_entities = sorted(context['entities'], key=lambda x: x.get('relevance', 0), reverse=True)
            # Giới hạn: chỉ lấy top 10 entities quan trọng nhất
            for entity in sorted_entities[:10]:
                entity_str = f"\n📍 {entity['id']} (Loại: {entity['type']})"
                if entity.get('method'):
                    entity_str += f" [Tìm bằng: {entity['method']}]"
                info = entity.get('info', {})
                if info:
                    # Giới hạn: chỉ lấy 3 fields quan trọng nhất
                    for key, value in list(info.items())[:3]:
                        if value:
                            entity_str += f"\n  • {key}: {value}"
                parts.append(entity_str)
                
        # ============================================
        # Format 2: Facts (Triples từ subgraph)
        # ============================================
        if context['facts']:
            parts.append("\n=== SỰ KIỆN (Triples từ Subgraph) ===")
            # Giới hạn: chỉ lấy top 5 facts quan trọng nhất
            for fact in context['facts'][:5]:
                parts.append(f"• {fact}")
                
        # ============================================
        # Format 3: Relationships (Edges trong subgraph - Triples format)
        # ============================================
        if context['relationships']:
            parts.append("\n=== MỐI QUAN HỆ (Edges trong Subgraph - Triples) ===")
            seen_rels = set()
            # Giới hạn: chỉ lấy top 10 relationships quan trọng nhất
            for rel in context['relationships'][:10]:
                rel_key = (rel['source'], rel['type'], rel['target'])
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    # Format as triple: (source, relationship, target)
                    parts.append(f"• ({rel['source']}, {rel['type']}, {rel['target']})")
                    
        # ============================================
        # Format 4: Paths (Multi-hop paths trong subgraph)
        # ============================================
        if context['paths']:
            parts.append("\n=== ĐƯỜNG DẪN QUAN HỆ (Multi-hop Paths trong Subgraph) ===")
            # Giới hạn: chỉ lấy top 3 paths quan trọng nhất
            for path_info in context['paths'][:3]:
                path = path_info['path']
                path_str = " → ".join(path)
                parts.append(f"• Path: {path_str}")
                # Không thêm path details để giảm độ dài
                
        context_text = "\n".join(parts)
        
        # ============================================
        # Truncate nếu quá dài (ước tính tokens)
        # ============================================
        # Ước tính: 1 token ≈ 4 characters (tiếng Việt)
        max_chars = max_tokens * 4
        if len(context_text) > max_chars:
            # Truncate và thêm thông báo
            context_text = context_text[:max_chars]
            # Cắt ở dòng cuối cùng hoàn chỉnh
            last_newline = context_text.rfind('\n')
            if last_newline > max_chars * 0.9:  # Nếu có newline gần cuối
                context_text = context_text[:last_newline]
            context_text += "\n\n[... Context đã được rút gọn để phù hợp với giới hạn model ...]"
        
        return context_text
        
    def get_multi_hop_context(
        self,
        query: str,
        hop_questions: List[str] = None,
        max_hops: int = 3
    ) -> Dict:
        """
        Get context for multi-hop reasoning questions.
        
        Args:
            query: Main query
            hop_questions: Intermediate questions for each hop
            max_hops: Maximum reasoning hops
            
        Returns:
            Multi-hop context with intermediate results
        """
        multi_hop_context = {
            'query': query,
            'hops': [],
            'final_context': None
        }
        
        # Extract entities from main query
        entities = self.extract_entities(query)
        current_entities = [e['text'] for e in entities]
        
        for hop in range(max_hops):
            hop_result = {
                'hop_number': hop + 1,
                'entities': current_entities,
                'context': {},
                'next_entities': []
            }
            
            # Get context for current entities
            for entity_id in current_entities[:3]:
                entity_context = self.kg.get_entity_context(entity_id, max_depth=1)
                if entity_context:
                    hop_result['context'][entity_id] = entity_context
                    
                    # Find next entities to explore
                    for rel in entity_context.get('relationships', []):
                        next_entity = rel['target'] if rel['source'] == entity_id else rel['source']
                        if next_entity not in current_entities:
                            hop_result['next_entities'].append(next_entity)
                            
            multi_hop_context['hops'].append(hop_result)
            
            # Move to next hop entities
            if not hop_result['next_entities']:
                break
            current_entities = list(set(hop_result['next_entities']))[:5]
            
        # Compile final context
        multi_hop_context['final_context'] = self.retrieve_context(query, max_hops=max_hops)
        
        return multi_hop_context


def main():
    """Test GraphRAG."""
    print("🔄 Initializing GraphRAG...")
    rag = GraphRAG()
    
    # Test queries
    test_queries = [
        "BTS có bao nhiêu thành viên?",
        "Công ty nào quản lý BLACKPINK?",
        "Ai là thành viên của (G)I-dle?",
        "BTS và SEVENTEEN có cùng công ty không?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"❓ Query: {query}")
        
        # Extract entities
        entities = rag.extract_entities(query)
        print(f"📍 Extracted entities: {[e['text'] for e in entities]}")
        
        # Retrieve context
        context = rag.retrieve_context(query)
        print(f"📚 Facts: {context['facts'][:3]}")
        
        # Format for LLM
        formatted = rag.format_context_for_llm(context)
        print(f"📝 Formatted context preview:\n{formatted[:500]}...")


if __name__ == "__main__":
    main()



