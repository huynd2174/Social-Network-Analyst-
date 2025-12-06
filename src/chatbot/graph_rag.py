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
    
    Combines:
    1. Entity extraction from natural language queries
    2. Graph traversal for structured context
    3. Semantic embedding for similarity matching
    4. Multi-hop reasoning support
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KpopKnowledgeGraph] = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_cache: bool = True
    ):
        """
        Initialize GraphRAG.
        
        Args:
            knowledge_graph: Pre-built knowledge graph (will create if None)
            embedding_model: Sentence transformer model for embeddings
            use_cache: Whether to cache embeddings
        """
        self.kg = knowledge_graph or KpopKnowledgeGraph()
        self.embedding_model_name = embedding_model
        self.use_cache = use_cache
        
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
        
        Args:
            query: User's question
            
        Returns:
            List of extracted entities with types
        """
        entities = []
        query_lower = query.lower()
        
        # 1. Pattern-based extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match,
                        'type': entity_type,
                        'method': 'pattern'
                    })
                    
        # 2. Knowledge graph lookup
        # Extract potential entity names (capitalized words, quoted strings)
        potential_names = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', query)
        for match in potential_names:
            name = match[0] or match[1] or match[2]
            if name:
                # Search in knowledge graph
                results = self.kg.search_entities(name, limit=1)
                if results and results[0]['score'] > 0.7:
                    entities.append({
                        'text': results[0]['id'],
                        'type': results[0]['type'],
                        'method': 'kg_lookup',
                        'score': results[0]['score']
                    })
                    
        # 3. Semantic similarity search (if available)
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
                    
        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity['text'] not in seen:
                seen.add(entity['text'])
                unique_entities.append(entity)
                
        return unique_entities
        
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
        
        Args:
            query: User's question
            max_entities: Maximum number of entities to retrieve
            max_hops: Maximum hops for graph traversal
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
        
        # 1. Extract entities from query
        extracted = self.extract_entities(query)
        
        # 2. Get context for each entity
        seen_entities = set()
        for entity_info in extracted[:max_entities]:
            entity_id = entity_info['text']
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            
            # Get entity context from knowledge graph
            entity_context = self.kg.get_entity_context(entity_id, max_depth=max_hops)
            
            if entity_context:
                # Add main entity
                entity_data = entity_context.get('entity', {})
                context['entities'].append({
                    'id': entity_id,
                    'type': entity_data.get('label'),
                    'info': entity_data.get('infobox', {}),
                    'relevance': entity_info.get('score', 1.0)
                })
                
                # Add relationships
                for rel in entity_context.get('relationships', []):
                    context['relationships'].append(rel)
                    
                # Generate facts from entity data
                facts = self._generate_facts(entity_id, entity_data)
                context['facts'].extend(facts)
                
        # 3. Find paths between entities (for multi-hop)
        if include_paths and len(extracted) >= 2:
            for i in range(len(extracted) - 1):
                for j in range(i + 1, min(i + 3, len(extracted))):
                    source = extracted[i]['text']
                    target = extracted[j]['text']
                    paths = self.kg.find_all_paths(source, target, max_hops=max_hops)
                    for path in paths[:3]:  # Limit paths
                        path_details = self.kg.get_path_details(path)
                        context['paths'].append({
                            'from': source,
                            'to': target,
                            'path': path,
                            'details': path_details
                        })
                        
        # 4. Semantic expansion (if available)
        if self.embedder:
            # Find additional relevant entities
            similar = self.semantic_search(query, top_k=3)
            for entity_id, score in similar:
                if entity_id not in seen_entities and score > 0.6:
                    entity_data = self.kg.get_entity(entity_id)
                    if entity_data:
                        context['entities'].append({
                            'id': entity_id,
                            'type': entity_data.get('label'),
                            'info': entity_data.get('infobox', {}),
                            'relevance': score,
                            'method': 'semantic_expansion'
                        })
                        
        return context
        
    def _generate_facts(self, entity_id: str, entity_data: Dict) -> List[str]:
        """Generate natural language facts from entity data."""
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
        
    def format_context_for_llm(self, context: Dict) -> str:
        """
        Format retrieved context as a prompt for the LLM.
        
        Args:
            context: Retrieved context dictionary
            
        Returns:
            Formatted context string
        """
        parts = []
        
        # Entities
        if context['entities']:
            parts.append("=== THÔNG TIN THỰC THỂ ===")
            for entity in context['entities']:
                entity_str = f"\n📍 {entity['id']} (Loại: {entity['type']})"
                info = entity.get('info', {})
                if info:
                    for key, value in list(info.items())[:5]:
                        if value:
                            entity_str += f"\n  • {key}: {value}"
                parts.append(entity_str)
                
        # Facts
        if context['facts']:
            parts.append("\n=== SỰ KIỆN ===")
            for fact in context['facts'][:10]:
                parts.append(f"• {fact}")
                
        # Relationships
        if context['relationships']:
            parts.append("\n=== MỐI QUAN HỆ ===")
            seen_rels = set()
            for rel in context['relationships'][:15]:
                rel_key = (rel['source'], rel['type'], rel['target'])
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    parts.append(f"• {rel['source']} --[{rel['type']}]--> {rel['target']}")
                    
        # Paths (for multi-hop)
        if context['paths']:
            parts.append("\n=== ĐƯỜNG DẪN QUAN HỆ ===")
            for path_info in context['paths'][:5]:
                path_str = " → ".join(path_info['path'])
                parts.append(f"• {path_str}")
                
        return "\n".join(parts)
        
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

