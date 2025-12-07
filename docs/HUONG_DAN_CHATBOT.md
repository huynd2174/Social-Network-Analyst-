# 🎤 Hướng dẫn K-pop Knowledge Graph Chatbot

## Mục lục
1. [Tổng quan](#tổng-quan)
2. [Cài đặt](#cài-đặt)
3. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
4. [Sử dụng](#sử-dụng)
5. [GraphRAG](#graphrag)
6. [Multi-hop Reasoning](#multi-hop-reasoning)
7. [Evaluation Dataset](#evaluation-dataset)
8. [So sánh Chatbot](#so-sánh-chatbot)

---

## Tổng quan

Hệ thống chatbot K-pop sử dụng đồ thị tri thức với các thành phần chính:

| Thành phần | Mô tả |
|------------|-------|
| **Knowledge Graph** | Đồ thị tri thức K-pop với 4596 nodes và 6107 edges |
| **GraphRAG** | Kỹ thuật RAG dựa trên đồ thị để truy xuất context |
| **Multi-hop Reasoning** | Suy luận đa bước (1-3 hop) trên đồ thị |
| **Small LLM** | Mô hình ngôn ngữ nhỏ (Qwen2-0.5B, ≤1B params) |
| **Evaluation** | Tập dữ liệu 2000+ câu hỏi đánh giá |

### Đặc điểm nổi bật

✅ **Small LLM (≤1B params)**: Sử dụng Qwen2-0.5B với 500M tham số
✅ **GraphRAG**: Kết hợp graph traversal với semantic search
✅ **Multi-hop Reasoning**: Hỗ trợ suy luận 1-3 hop
✅ **Evaluation Dataset**: 2000+ câu hỏi Đúng/Sai, Có/Không, Trắc nghiệm
✅ **Comparison Framework**: So sánh với ChatGPT và baseline

---

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements/requirements_chatbot.txt
```

### 2. Các thư viện chính

```
transformers>=4.36.0      # Hugging Face Transformers
torch>=2.0.0              # PyTorch
sentence-transformers     # Sentence embeddings
faiss-cpu                 # Vector search
networkx                  # Graph operations
gradio                    # Web UI
```

### 3. Tải mô hình (tự động)

Mô hình Qwen2-0.5B sẽ được tải tự động khi chạy lần đầu.

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       GraphRAG Module                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │Entity       │  │Semantic      │  │Graph               │  │
│  │Extraction   │→ │Search        │→ │Traversal           │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Multi-hop Reasoning Engine                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │Query        │  │Reasoning     │  │Answer              │  │
│  │Analysis     │→ │Steps         │→ │Generation          │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Small LLM (Qwen2-0.5B)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │Context      │  │Prompt        │  │Response            │  │
│  │Formatting   │→ │Engineering   │→ │Generation          │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Response                             │
└─────────────────────────────────────────────────────────────┘
```

### Cấu trúc thư mục

```
src/chatbot/
├── __init__.py           # Package exports
├── knowledge_graph.py    # Knowledge graph management
├── graph_rag.py          # GraphRAG implementation
├── multi_hop_reasoning.py # Multi-hop reasoning engine
├── small_llm.py          # Small LLM integration
├── chatbot.py            # Main chatbot interface
├── evaluation.py         # Evaluation dataset generator
├── comparison.py         # Chatbot comparison framework
└── app.py                # Gradio web UI
```

---

## Sử dụng

### 1. Chạy CLI Mode

```bash
python src/run_chatbot.py --mode cli
```

Các lệnh CLI:
- `members <group>`: Xem thành viên nhóm
- `company <group>`: Xem công ty quản lý
- `same <group1> <group2>`: Kiểm tra cùng công ty
- `path <entity1> <entity2>`: Tìm đường đi
- `stats`: Xem thống kê
- `quit`: Thoát

### 2. Chạy Web UI

```bash
python src/run_chatbot.py --mode ui
```

Truy cập: http://localhost:7860

### 3. Sử dụng trong code

```python
from chatbot import KpopChatbot

# Khởi tạo
chatbot = KpopChatbot()

# Chat thông thường
response = chatbot.chat("BTS có bao nhiêu thành viên?")
print(response['response'])

# Hỏi Có/Không
result = chatbot.answer_yes_no("BTS thuộc HYBE đúng không?")
print(result['answer'])  # "Có" hoặc "Không"

# Trắc nghiệm
result = chatbot.answer_multiple_choice(
    "Công ty nào quản lý BTS?",
    ["SM Entertainment", "HYBE", "YG Entertainment", "JYP Entertainment"]
)
print(result['selected_letter'])  # "B"

# Các method đặc biệt
chatbot.get_group_members("BTS")
chatbot.get_group_company("BTS")
chatbot.check_same_company("BTS", "SEVENTEEN")
chatbot.get_labelmates("BTS")
chatbot.find_path("Jungkook", "HYBE")
```

---

## GraphRAG

### Quy trình GraphRAG

1. **Entity Extraction**: Trích xuất thực thể từ câu hỏi
2. **Semantic Search**: Tìm entities tương tự bằng embeddings
3. **Graph Traversal**: Duyệt đồ thị để lấy context
4. **Context Ranking**: Xếp hạng và lọc context
5. **Prompt Generation**: Tạo prompt cho LLM

### Ví dụ

```python
from chatbot.graph_rag import GraphRAG

rag = GraphRAG()

# Truy xuất context
context = rag.retrieve_context(
    "BTS có cùng công ty với SEVENTEEN không?",
    max_entities=5,
    max_hops=2
)

# Format cho LLM
formatted = rag.format_context_for_llm(context)
print(formatted)
```

Output:
```
=== THÔNG TIN THỰC THỂ ===
📍 BTS (Loại: Group)
  • Thành viên: RM, Jin, Suga, J-Hope, Jimin, V, Jungkook
  • Hãng đĩa: HYBE

📍 SEVENTEEN (Loại: Group)
  • Thành viên: S.Coups, Jeonghan, Joshua, ...
  • Hãng đĩa: Pledis Entertainment

=== SỰ KIỆN ===
• BTS thuộc công ty HYBE
• SEVENTEEN thuộc công ty Pledis Entertainment

=== MỐI QUAN HỆ ===
• BTS --[MANAGED_BY]--> HYBE
• SEVENTEEN --[MANAGED_BY]--> Pledis Entertainment
```

---

## Multi-hop Reasoning

### Các loại suy luận

| Loại | Mô tả | Ví dụ |
|------|-------|-------|
| **1-hop** | Quan hệ trực tiếp | "BTS có bao nhiêu thành viên?" |
| **2-hop** | 1 thực thể trung gian | "Jungkook thuộc công ty nào?" |
| **3-hop** | 2 thực thể trung gian | "Jungkook và Jennie có cùng công ty không?" |

### Chiến lược suy luận

1. **Chain Reasoning**: A → B → C
2. **Aggregation**: A → {B1, B2, ...} → count/list
3. **Comparison**: So sánh A và B
4. **Intersection**: Tìm điểm chung

### Ví dụ

```python
from chatbot.multi_hop_reasoning import MultiHopReasoner

reasoner = MultiHopReasoner()

# 1-hop: Thành viên của BTS
result = reasoner.get_group_members("BTS")
print(result.answer_text)
# "BTS có 7 thành viên: RM, Jin, Suga, J-Hope, Jimin, V, Jungkook"

# 2-hop: Công ty của Jungkook (Artist → Group → Company)
result = reasoner.get_artist_company("Jungkook")
print(result.answer_text)
# "Jungkook thuộc công ty: HYBE"

# 3-hop: Kiểm tra cùng công ty
result = reasoner.check_same_company("BTS", "SEVENTEEN")
print(result.answer_text)
# "Không, BTS thuộc HYBE, còn SEVENTEEN thuộc Pledis Entertainment"
```

---

## Evaluation Dataset

### Tạo dataset

```bash
python src/run_chatbot.py --mode eval --num-questions 2000
```

Hoặc trong code:

```python
from chatbot.evaluation import EvaluationDatasetGenerator

generator = EvaluationDatasetGenerator()
stats = generator.generate_full_dataset(
    target_count=2000,
    output_path="data/evaluation_dataset.json"
)
```

### Cấu trúc dataset

```json
{
  "metadata": {
    "total_questions": 2000,
    "by_hops": {"1": 700, "2": 700, "3": 600},
    "by_type": {
      "true_false": 600,
      "yes_no": 600,
      "multiple_choice": 800
    }
  },
  "questions": [
    {
      "id": "Q00001",
      "question": "Jungkook là thành viên của BTS.",
      "question_type": "true_false",
      "answer": "Đúng",
      "choices": [],
      "hops": 1,
      "entities": ["Jungkook", "BTS"],
      "relationships": ["MEMBER_OF"],
      "explanation": "Jungkook thực sự là thành viên của BTS.",
      "difficulty": "easy",
      "category": "membership"
    }
  ]
}
```

### Phân bố câu hỏi

| Loại | 1-hop | 2-hop | 3-hop | Tổng |
|------|-------|-------|-------|------|
| True/False | 200 | 200 | 200 | 600 |
| Yes/No | 200 | 200 | 200 | 600 |
| Multiple Choice | 300 | 300 | 200 | 800 |
| **Tổng** | **700** | **700** | **600** | **2000** |

---

## So sánh Chatbot

### Chạy so sánh

```bash
python src/run_chatbot.py --mode compare --max-compare 500
```

### Các chatbot được so sánh

1. **K-pop Knowledge Graph Chatbot** (Của chúng ta)
2. **ChatGPT** (OpenAI API - cần API key)
3. **Random Baseline** (Đoán ngẫu nhiên)

### Metrics đánh giá

- **Accuracy**: Tỷ lệ câu trả lời đúng
- **Accuracy by Hops**: Độ chính xác theo số hop
- **Accuracy by Type**: Độ chính xác theo loại câu hỏi
- **Response Time**: Thời gian phản hồi

### Ví dụ kết quả

```
======================================================================
                     📊 COMPARISON RESULTS                     
======================================================================
Chatbot                        Accuracy     1-hop    2-hop    3-hop
----------------------------------------------------------------------
K-pop Knowledge Graph Chatbot     85.2%    92.1%    84.5%    78.3%
ChatGPT (gpt-3.5-turbo)          72.4%    78.2%    71.6%    67.8%
Random Baseline                   33.3%    32.8%    33.5%    33.6%
======================================================================

🏆 Best performer: K-pop Knowledge Graph Chatbot (85.2% accuracy)
```

### Phân tích

- **Ưu điểm của Knowledge Graph Chatbot**:
  - Độ chính xác cao hơn nhờ dữ liệu có cấu trúc
  - Suy luận multi-hop chính xác hơn
  - Thời gian phản hồi nhanh (không cần API)
  
- **Hạn chế**:
  - Chỉ trả lời được câu hỏi trong phạm vi knowledge graph
  - Không thể trả lời câu hỏi mở rộng

---

## Troubleshooting

### Lỗi thường gặp

1. **CUDA out of memory**
   ```python
   # Sử dụng 4-bit quantization
   llm = SmallLLM(model_key="qwen2-0.5b")
   ```

2. **Model không tải được**
   ```bash
   pip install accelerate bitsandbytes
   ```

3. **Gradio không chạy**
   ```bash
   pip install gradio>=4.0.0
   ```

### Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| RAM | 8GB | 16GB |
| GPU VRAM | 4GB | 8GB |
| Storage | 5GB | 10GB |

---

## API Reference

### KpopChatbot

```python
class KpopChatbot:
    def __init__(self, data_path, llm_model, use_embeddings, verbose)
    def chat(self, query, session_id, use_multi_hop, max_hops, return_details)
    def answer_yes_no(self, query, return_details)
    def answer_multiple_choice(self, query, choices, return_details)
    def get_group_members(self, group_name)
    def get_group_company(self, group_name)
    def check_same_company(self, entity1, entity2)
    def get_labelmates(self, entity)
    def find_path(self, source, target)
    def get_statistics(self)
```

### GraphRAG

```python
class GraphRAG:
    def __init__(self, knowledge_graph, embedding_model, use_cache)
    def extract_entities(self, query)
    def semantic_search(self, query, top_k)
    def retrieve_context(self, query, max_entities, max_hops, include_paths)
    def format_context_for_llm(self, context)
    def get_multi_hop_context(self, query, hop_questions, max_hops)
```

### MultiHopReasoner

```python
class MultiHopReasoner:
    def __init__(self, knowledge_graph)
    def reason(self, query, start_entities, max_hops)
    def get_group_members(self, group_name)
    def get_company_of_group(self, group_name)
    def get_artist_company(self, artist_name)
    def check_same_company(self, entity1, entity2)
    def get_labelmates(self, artist_or_group)
```

---

*Made with ❤️ for K-pop fans*




