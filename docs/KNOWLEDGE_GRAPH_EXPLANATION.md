# 📚 Giải thích: Chatbot và Đồ thị Tri thức

Tài liệu này giải thích chi tiết:
1. **Làm sao biết chatbot lấy thông tin từ đồ thị tri thức?**
2. **Làm sao biết mạng xã hội đã được xây thành đồ thị tri thức?**

---

## Phần 1: Mạng Xã Hội → Đồ thị Tri thức

### 1.1. Dữ liệu Nguồn (Mạng Xã Hội)

Dữ liệu ban đầu được thu thập từ mạng xã hội (Wikipedia, v.v.) và lưu trong các file:

```
data/
├── korean_artists_graph_bfs.json    # Dữ liệu từ BFS crawl Wikipedia
├── kpop_ner_result.json            # Entities từ NER
└── merged_kpop_data.json            # Đồ thị tri thức đã merge
```

### 1.2. Quá trình Chuyển đổi

#### Bước 1: Load dữ liệu

```python
# File: src/chatbot/knowledge_graph.py

def _load_data(self):
    """Load merged K-pop data from JSON."""
    with open(self.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    self.nodes = data.get('nodes', {})  # Entities
    self.edges = data.get('edges', [])  # Relationships
```

**Giải thích:**
- `nodes`: Các entities (Groups, Artists, Songs, Albums, Companies)
- `edges`: Các relationships (MEMBER_OF, SINGS, MANAGED_BY, etc.)

#### Bước 2: Build Graph với NetworkX

```python
# File: src/chatbot/knowledge_graph.py

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
        
        if source and target:
            self.graph.add_edge(
                source, 
                target,
                type=rel_type,
                confidence=edge.get('confidence', 1.0),
                method=edge.get('method', 'unknown')
            )
```

**Giải thích:**
- Sử dụng `NetworkX.DiGraph()` để tạo đồ thị có hướng
- Mỗi node = một entity (BTS, Jungkook, etc.)
- Mỗi edge = một relationship (MEMBER_OF, SINGS, etc.)

### 1.3. Cấu trúc Đồ thị Tri thức

```
Nodes (Entities):
├── Group: BTS, BLACKPINK, EXO, ...
├── Artist: Jungkook, RM, J-Hope, ...
├── Song: "Dynamite", "Butter", ...
├── Album: "BE", "Love Yourself", ...
└── Company: HYBE, SM Entertainment, ...

Edges (Relationships):
├── MEMBER_OF: Jungkook → BTS
├── SINGS: "Dynamite" → BTS
├── MANAGED_BY: BTS → HYBE
└── RELEASED: BTS → "BE"
```

---

## Phần 2: Chatbot Lấy Thông tin từ Đồ thị Tri thức

### 2.1. Quy trình Chatbot Trả lời Câu hỏi

```
User Query
    ↓
1. GraphRAG.retrieve_context()
    ├── Extract entities từ query
    ├── get_entity_context() → Graph traversal
    ├── find_all_paths() → Graph traversal
    └── semantic_search() → Embedding (optional)
    ↓
2. MultiHopReasoner.reason()
    ├── Traverse graph theo hops
    ├── find_path() → Graph algorithm
    └── get_neighbors() → Graph traversal
    ↓
3. Format context
    ↓
4. Generate response (LLM hoặc reasoning)
```

### 2.2. Chứng minh: GraphRAG sử dụng Graph Traversal

#### Code trong `graph_rag.py`:

```python
def retrieve_context(self, query: str, ...):
    # 1. Extract entities
    extracted = self.extract_entities(query)
    
    # 2. Get context từ Knowledge Graph (GRAPH TRAVERSAL)
    for entity_info in extracted:
        entity_id = entity_info['text']
        
        # ⭐ ĐÂY LÀ ĐIỂM QUAN TRỌNG: get_entity_context() traverse graph
        entity_context = self.kg.get_entity_context(entity_id, max_depth=max_hops)
        
        # Add relationships từ graph
        for rel in entity_context.get('relationships', []):
            context['relationships'].append(rel)
    
    # 3. Find paths trong graph (GRAPH TRAVERSAL)
    paths = self.kg.find_all_paths(source, target, max_hops=max_hops)
```

**Chứng minh:**
- `self.kg.get_entity_context()` → **Graph traversal** (BFS)
- `self.kg.find_all_paths()` → **Graph algorithm** (NetworkX)
- Không có vector search thuần túy, có kết hợp graph traversal

### 2.3. Chứng minh: Multi-hop Reasoning sử dụng Graph

#### Code trong `multi_hop_reasoning.py`:

```python
def reason(self, query: str, start_entities: List[str], max_hops: int):
    # Traverse graph theo hops
    for hop in range(max_hops):
        # Get neighbors từ graph
        neighbors = self.kg.get_neighbors(current_entity)
        
        # Find paths trong graph
        path = self.kg.find_path(source, target, max_hops)
```

**Chứng minh:**
- `self.kg.get_neighbors()` → **Graph traversal**
- `self.kg.find_path()` → **Graph algorithm** (shortest path)
- Tất cả đều dựa trên đồ thị, không phải vector search

### 2.4. Chứng minh: Chatbot KHÔNG dùng nguồn khác

#### Kiểm tra trong `chatbot.py`:

```python
class KpopChatbot:
    def __init__(self, ...):
        # 1. Knowledge Graph (DUY NHẤT nguồn dữ liệu)
        self.kg = KpopKnowledgeGraph(data_path)
        
        # 2. GraphRAG (sử dụng knowledge_graph)
        self.rag = GraphRAG(knowledge_graph=self.kg)
        
        # 3. Multi-hop Reasoner (sử dụng knowledge_graph)
        self.reasoner = MultiHopReasoner(self.kg)
        
        # 4. LLM (chỉ để generate text, KHÔNG có knowledge)
        self.llm = get_llm(llm_model) if llm_model else None
```

**Chứng minh:**
- ✅ `self.kg` = Knowledge Graph (duy nhất nguồn dữ liệu)
- ✅ `self.rag.kg` = cùng knowledge graph
- ✅ `self.reasoner.kg` = cùng knowledge graph
- ❌ Không có external API
- ❌ Không có database khác
- ❌ Không có web scraping

---

## Phần 3: Cách Verify

### 3.1. Chạy Script Verify

```bash
python src/verify_knowledge_graph.py
```

Script này sẽ:
1. ✅ Kiểm tra dữ liệu nguồn → đồ thị tri thức
2. ✅ Trace quá trình chatbot trả lời
3. ✅ Chứng minh chatbot dùng graph traversal
4. ✅ Chứng minh không dùng nguồn khác

### 3.2. Verify Thủ công

#### Test 1: Kiểm tra Graph Structure

```python
from chatbot import KpopKnowledgeGraph

kg = KpopKnowledgeGraph()

# Check graph
print(f"Nodes: {kg.graph.number_of_nodes()}")
print(f"Edges: {kg.graph.number_of_edges()}")

# Check entity
bts = kg.get_entity('BTS')
print(f"BTS type: {bts['label']}")
print(f"BTS info: {bts['infobox']}")

# Check relationships
rels = kg.get_relationships('BTS')
print(f"BTS relationships: {len(rels)}")
```

#### Test 2: Trace GraphRAG

```python
from chatbot import GraphRAG

rag = GraphRAG()
context = rag.retrieve_context("BTS có bao nhiêu thành viên?")

# Check entities (từ graph nodes)
print(f"Entities: {context['entities']}")

# Check relationships (từ graph edges)
print(f"Relationships: {context['relationships']}")

# Check paths (từ graph traversal)
print(f"Paths: {context['paths']}")
```

#### Test 3: Trace Multi-hop Reasoning

```python
from chatbot import MultiHopReasoner

reasoner = MultiHopReasoner()
result = reasoner.reason("BTS có bao nhiêu thành viên?", ["BTS"], max_hops=2)

# Check steps (từ graph traversal)
for step in result.steps:
    print(f"Step: {step.operation}")
    print(f"  Source: {step.source_entities}")
    print(f"  Relationship: {step.relationship}")
    print(f"  Target: {step.target_entities}")
```

---

## Phần 4: So sánh với Chatbot Thông thường

| Tính năng | Chatbot Thông thường | Chatbot của bạn (Graph-based) |
|-----------|---------------------|-------------------------------|
| **Nguồn dữ liệu** | Vector database, Documents | ✅ Knowledge Graph (nodes + edges) |
| **Retrieval** | Vector similarity search | ✅ Graph traversal + Vector search |
| **Relationships** | Không rõ ràng | ✅ Rõ ràng qua edges |
| **Multi-hop** | Không hỗ trợ | ✅ Hỗ trợ qua graph traversal |
| **Paths** | Không có | ✅ Tìm paths giữa entities |
| **Structure** | Flat documents | ✅ Structured graph |

---

## Phần 5: Kết luận

### ✅ Chứng minh 1: Mạng xã hội → Đồ thị tri thức

1. **Dữ liệu nguồn:** `korean_artists_graph_bfs.json`, `kpop_ner_result.json`
2. **Quá trình:** Load → Build NetworkX graph → Nodes + Edges
3. **Kết quả:** `merged_kpop_data.json` với structure:
   ```json
   {
     "nodes": {
       "BTS": {
         "label": "Group",
         "infobox": {...}
       }
     },
     "edges": [
       {
         "source": "Jungkook",
         "target": "BTS",
         "type": "MEMBER_OF"
       }
     ]
   }
   ```

### ✅ Chứng minh 2: Chatbot lấy thông tin từ đồ thị tri thức

1. **GraphRAG:**
   - `get_entity_context()` → Graph traversal (BFS)
   - `find_all_paths()` → Graph algorithm
   - Không dùng external API

2. **Multi-hop Reasoning:**
   - `get_neighbors()` → Graph traversal
   - `find_path()` → Graph algorithm
   - Tất cả dựa trên graph

3. **Chatbot:**
   - `self.kg` = Knowledge Graph (duy nhất nguồn)
   - `self.rag.kg` = cùng graph
   - `self.reasoner.kg` = cùng graph
   - Không có nguồn khác

---

## Phần 6: Các File Quan trọng

1. **`src/chatbot/knowledge_graph.py`**
   - Xây dựng đồ thị tri thức từ dữ liệu
   - Graph traversal methods

2. **`src/chatbot/graph_rag.py`**
   - GraphRAG implementation
   - Sử dụng graph traversal

3. **`src/chatbot/chatbot.py`**
   - Chatbot sử dụng knowledge graph
   - Không có external sources

4. **`src/merge_and_import_neo4j.py`**
   - Merge dữ liệu thành đồ thị tri thức

5. **`src/verify_knowledge_graph.py`**
   - Script verify tất cả claims

---

## Phần 7: Cách Demo

### Demo 1: Show Graph Structure

```bash
python src/verify_knowledge_graph.py
# Xem phần "1. CHỨNG MINH: Mạng xã hội → Đồ thị tri thức"
```

### Demo 2: Trace Chatbot

```bash
python src/verify_knowledge_graph.py
# Xem phần "2. CHỨNG MINH: Chatbot lấy thông tin từ Đồ thị tri thức"
```

### Demo 3: Show No External Sources

```bash
python src/verify_knowledge_graph.py
# Xem phần "4. CHỨNG MINH: Chatbot KHÔNG dùng nguồn khác"
```

---

## Tóm tắt

✅ **Mạng xã hội → Đồ thị tri thức:**
- Dữ liệu từ Wikipedia, NER → `merged_kpop_data.json`
- Build NetworkX graph với nodes (entities) và edges (relationships)

✅ **Chatbot lấy thông tin từ đồ thị tri thức:**
- GraphRAG sử dụng `get_entity_context()` (graph traversal)
- Multi-hop reasoning sử dụng `get_neighbors()`, `find_path()` (graph algorithms)
- Không có external API, database, hoặc web scraping

✅ **Verify:**
- Chạy `python src/verify_knowledge_graph.py`
- Tất cả thông tin đều trace được về Knowledge Graph




