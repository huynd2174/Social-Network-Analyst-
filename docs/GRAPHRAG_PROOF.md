# 📋 Chứng minh: GraphRAG trên Đồ thị Tri thức

Tài liệu này chỉ ra các đoạn code thể hiện việc:
1. **Biểu diễn mạng xã hội dưới hình thức đồ thị tri thức**
2. **Áp dụng kỹ thuật GraphRAG** (RAG dùng biểu diễn trên đồ thị)

---

## 1. Biểu diễn Mạng Xã Hội dưới hình thức Đồ thị Tri thức

### 1.1. Xây dựng Knowledge Graph (File: `src/chatbot/knowledge_graph.py`)

#### 1.1.1. Class Definition

```15:23:src/chatbot/knowledge_graph.py
class KpopKnowledgeGraph:
    """
    Knowledge Graph for K-pop entities.
    
    Supports:
    - Entity types: Group, Artist, Song, Album, Company, Genre, Occupation, Instrument
    - Relationship types: MEMBER_OF, SINGS, RELEASED, MANAGED_BY, SUBUNIT_OF, etc.
    - Multi-hop traversal and reasoning
    """
```

**Giải thích:** Class `KpopKnowledgeGraph` xây dựng đồ thị tri thức từ dữ liệu K-pop (mạng xã hội).

#### 1.1.2. Sử dụng NetworkX để biểu diễn đồ thị

```8:9:src/chatbot/knowledge_graph.py
import networkx as nx
```

```28:28:src/chatbot/knowledge_graph.py
        self.graph = nx.DiGraph()
```

**Giải thích:** Sử dụng `NetworkX.DiGraph()` (Directed Graph) để biểu diễn đồ thị tri thức.

#### 1.1.3. Load dữ liệu từ mạng xã hội

```39:51:src/chatbot/knowledge_graph.py
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
```

**Giải thích:** Load dữ liệu từ `merged_kpop_data.json` (đã được xây dựng từ mạng xã hội) và chuyển thành nodes và edges.

#### 1.1.4. Build Graph từ Nodes và Edges

```53:81:src/chatbot/knowledge_graph.py
    def _build_graph(self):
        """Build NetworkX graph from nodes and edges."""
        # Add nodes
        for node_id, node_data in self.nodes.items():
            self.graph.add_node(
                node_id,
                label=node_data.get('label', 'Unknown'),
                title=node_data.get('title', node_id),
                infobox=node_data.get('infobox', {}),
                url=node_data.get('url', ''),
                depth=node_data.get('depth', 0)
            )
            
        # Add edges
        for edge in self.edges:
            source = edge.get('source')
            target = edge.get('target')
            rel_type = edge.get('type', 'RELATED')
            
            if source and target and source in self.graph and target in self.graph:
                self.graph.add_edge(
                    source, 
                    target,
                    type=rel_type,
                    confidence=edge.get('confidence', 1.0),
                    method=edge.get('method', 'unknown')
                )
                
        print(f"✅ Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
```

**Giải thích:**
- **Nodes**: Các entities (Group, Artist, Song, Album, Company, etc.)
- **Edges**: Các relationships (MEMBER_OF, SINGS, MANAGED_BY, etc.)
- Mỗi edge có `type`, `confidence`, `method` để thể hiện quan hệ trong mạng xã hội

---

## 2. Áp dụng GraphRAG (Graph-based RAG)

### 2.1. Class GraphRAG (File: `src/chatbot/graph_rag.py`)

#### 2.1.1. Class Definition và Documentation

```1:14:src/chatbot/graph_rag.py
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
```

**Giải thích:** Module này implement **GraphRAG** (không phải RAG thông thường), kết hợp graph traversal với semantic search.

#### 2.1.2. Class GraphRAG

```40:49:src/chatbot/graph_rag.py
class GraphRAG:
    """
    Graph-based Retrieval Augmented Generation for K-pop Knowledge Graph.
    
    Combines:
    1. Entity extraction from natural language queries
    2. Graph traversal for structured context
    3. Semantic embedding for similarity matching
    4. Multi-hop reasoning support
    """
```

**Giải thích:** Class `GraphRAG` kết hợp:
1. Entity extraction
2. **Graph traversal** (điểm khác biệt với RAG thông thường)
3. Semantic embedding
4. Multi-hop reasoning

#### 2.1.3. Khởi tạo với Knowledge Graph

```51:65:src/chatbot/graph_rag.py
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
```

**Giải thích:** GraphRAG được khởi tạo với `knowledge_graph` - đây là điểm khác biệt với RAG thông thường (không dùng graph).

---

### 2.2. Graph Traversal trong GraphRAG

#### 2.2.1. Method `retrieve_context` - Sử dụng Graph Traversal

```289:380:src/chatbot/graph_rag.py
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
```

**Giải thích:**
- **Dòng 328**: `self.kg.get_entity_context(entity_id, max_depth=max_hops)` - **Graph traversal** để lấy context từ đồ thị
- **Dòng 354**: `self.kg.find_all_paths(source, target, max_hops=max_hops)` - **Graph traversal** để tìm paths giữa entities
- **Dòng 356**: `self.kg.get_path_details(path)` - Lấy chi tiết về path trong đồ thị

**Điểm khác biệt với RAG thông thường:**
- RAG thông thường: Chỉ dùng vector similarity search
- **GraphRAG**: Dùng **graph traversal** để tìm relationships và paths trong đồ thị

---

### 2.3. Graph Traversal Methods trong Knowledge Graph

#### 2.3.1. `get_entity_context` - Traverse graph để lấy context

```334:366:src/chatbot/knowledge_graph.py
    def get_entity_context(self, entity_id: str, max_depth: int = 2) -> Dict:
        """
        Get context for an entity by traversing the graph.
        
        Args:
            entity_id: Entity to get context for
            max_depth: Maximum depth for traversal
            
        Returns:
            Dictionary with entity info, relationships, and neighbors
        """
        if entity_id not in self.graph:
            return None
            
        context = {
            'entity': self.get_entity(entity_id),
            'relationships': [],
            'neighbors': []
        }
        
        # Get relationships (edges)
        relationships = self.get_relationships(entity_id)
        context['relationships'] = relationships[:20]  # Limit
        
        # Get neighbors up to max_depth
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            
            if current != entity_id:
                neighbor_data = self.get_entity(current)
                if neighbor_data:
                    context['neighbors'].append({
                        'id': current,
                        'type': neighbor_data.get('label'),
                        'depth': depth
                    })
            
            if depth < max_depth:
                for neighbor, _, _ in self.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        return context
```

**Giải thích:**
- **BFS traversal**: Sử dụng queue để traverse đồ thị theo BFS
- **max_depth**: Giới hạn độ sâu traversal (multi-hop)
- **Neighbors**: Lấy các neighbors ở các độ sâu khác nhau

#### 2.3.2. `find_all_paths` - Tìm paths trong đồ thị

```170:176:src/chatbot/knowledge_graph.py
    def find_all_paths(self, source: str, target: str, max_hops: int = 3) -> List[List[str]]:
        """Find all simple paths between two entities (up to max_hops)."""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_hops))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
```

**Giải thích:**
- Sử dụng `nx.all_simple_paths()` của NetworkX để tìm **tất cả paths** giữa 2 entities
- Đây là **graph traversal** thuần túy, không phải vector search

#### 2.3.3. `find_path` - Tìm shortest path

```160:168:src/chatbot/knowledge_graph.py
    def find_path(self, source: str, target: str, max_hops: int = 5) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        try:
            path = nx.shortest_path(self.graph, source, target)
            if len(path) - 1 <= max_hops:
                return path
        except nx.NetworkXNoPath:
            pass
        return None
```

**Giải thích:**
- Sử dụng `nx.shortest_path()` để tìm **shortest path** trong đồ thị
- Đây là **graph algorithm** thuần túy

---

### 2.4. Sử dụng GraphRAG trong Chatbot

#### 2.4.1. Khởi tạo GraphRAG

```95:101:src/chatbot/chatbot.py
        # 2. GraphRAG
        if verbose:
            print("  🔍 Initializing GraphRAG...")
        self.rag = GraphRAG(
            knowledge_graph=self.kg,
            use_cache=True
        )
```

**Giải thích:** Chatbot khởi tạo `GraphRAG` với `knowledge_graph` (không phải vector store thông thường).

#### 2.4.2. Sử dụng GraphRAG để retrieve context

```171:172:src/chatbot/chatbot.py
        # 1. Retrieve context using GraphRAG
        context = self.rag.retrieve_context(query, max_entities=5, max_hops=3)
```

**Giải thích:** Chatbot sử dụng `GraphRAG.retrieve_context()` để lấy context, method này sử dụng **graph traversal** (không phải chỉ vector search).

---

## 3. So sánh GraphRAG vs RAG thông thường

| Tính năng | RAG thông thường | GraphRAG (Đã implement) |
|-----------|------------------|------------------------|
| **Retrieval** | Vector similarity search | ✅ Vector search + **Graph traversal** |
| **Context** | Chunks từ documents | ✅ Entities + Relationships + **Paths** |
| **Multi-hop** | Không hỗ trợ | ✅ Hỗ trợ qua graph traversal |
| **Relationships** | Không rõ ràng | ✅ Rõ ràng qua edges trong graph |
| **Paths** | Không có | ✅ Tìm paths giữa entities |

---

## 4. Tóm tắt

### ✅ 1. Biểu diễn mạng xã hội dưới hình thức đồ thị tri thức:

| Yếu tố | Vị trí trong code |
|--------|-------------------|
| **Load dữ liệu** | `knowledge_graph.py:39-51` |
| **Build graph** | `knowledge_graph.py:53-81` |
| **Sử dụng NetworkX** | `knowledge_graph.py:28` (nx.DiGraph) |
| **Nodes (Entities)** | Groups, Artists, Songs, Albums, Companies |
| **Edges (Relationships)** | MEMBER_OF, SINGS, MANAGED_BY, etc. |

### ✅ 2. Áp dụng GraphRAG (không phải RAG thông thường):

| Yếu tố | Vị trí trong code |
|--------|-------------------|
| **Class GraphRAG** | `graph_rag.py:40-49` |
| **Graph traversal** | `graph_rag.py:328` (get_entity_context) |
| **Find paths** | `graph_rag.py:354` (find_all_paths) |
| **BFS traversal** | `knowledge_graph.py:334-366` (get_entity_context) |
| **Shortest path** | `knowledge_graph.py:160-168` (find_path) |
| **Sử dụng trong chatbot** | `chatbot.py:98-101, 171-172` |

---

## 5. Cách Verify

### 5.1. Chạy Demo

```bash
python src/demo_chatbot.py
# Xem phần "2. DEMO: GraphRAG trên Đồ thị Tri thức"
```

### 5.2. Test GraphRAG

```python
from chatbot import GraphRAG

rag = GraphRAG()
context = rag.retrieve_context("BTS có bao nhiêu thành viên?")

print(f"Entities: {len(context['entities'])}")
print(f"Relationships: {len(context['relationships'])}")
print(f"Paths: {len(context['paths'])}")  # GraphRAG có paths!
```

### 5.3. Kiểm tra Graph Traversal

```python
from chatbot import KpopKnowledgeGraph

kg = KpopKnowledgeGraph()
# Graph traversal
context = kg.get_entity_context("BTS", max_depth=2)
print(f"Neighbors: {len(context['neighbors'])}")

# Find paths
paths = kg.find_all_paths("BTS", "BLACKPINK", max_hops=3)
print(f"Paths found: {len(paths)}")
```

---

## 6. Kết luận

✅ **ĐẠT YÊU CẦU:**

1. **Biểu diễn mạng xã hội dưới hình thức đồ thị tri thức:**
   - ✅ Sử dụng NetworkX.DiGraph
   - ✅ Nodes = Entities (Groups, Artists, etc.)
   - ✅ Edges = Relationships (MEMBER_OF, SINGS, etc.)
   - ✅ Load từ dữ liệu mạng xã hội

2. **Áp dụng GraphRAG (ưu tiên GraphRAG):**
   - ✅ Class `GraphRAG` (không phải RAG thông thường)
   - ✅ Sử dụng **graph traversal** (get_entity_context, find_all_paths)
   - ✅ Tìm paths giữa entities
   - ✅ Multi-hop reasoning qua graph traversal
   - ✅ Kết hợp graph traversal với semantic search

---

## 7. Các file liên quan

1. **`src/chatbot/knowledge_graph.py`** - Xây dựng knowledge graph
2. **`src/chatbot/graph_rag.py`** - Implement GraphRAG
3. **`src/chatbot/chatbot.py`** - Sử dụng GraphRAG trong chatbot
4. **`data/merged_kpop_data.json`** - Dữ liệu mạng xã hội đã được chuyển thành graph








