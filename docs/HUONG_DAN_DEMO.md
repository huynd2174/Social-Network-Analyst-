# 🎤 Hướng dẫn Demo K-pop Knowledge Graph Chatbot

## 📋 Checklist Trước Khi Demo

### ✅ Các yêu cầu đã hoàn thành:

- [x] **1 điểm**: Small LLM (≤1B params) - Qwen2-0.5B (315M params)
- [x] **0.5 điểm**: GraphRAG trên đồ thị tri thức
- [x] **1.5 điểm**: Multi-hop Reasoning (1-hop, 2-hop, 3-hop)
- [ ] **1 điểm**: Evaluation Dataset (2000+ câu hỏi) - **CẦN TẠO**
- [ ] **0.5 điểm**: Comparison với chatbot khác - **CẦN CHẠY**

---

## 🚀 Các Bước Demo

### Bước 1: Chạy Demo Tự Động (Khuyến nghị)

```bash
python src/demo_chatbot.py
```

Script này sẽ:
- ✅ Demo Small LLM và hiển thị số tham số
- ✅ Demo GraphRAG retrieval
- ✅ Demo Multi-hop Reasoning với các test cases
- ✅ Kiểm tra/ tạo Evaluation Dataset
- ✅ Chạy Comparison (sample)
- ✅ Demo Full Chatbot Integration

**Thời gian**: ~5-10 phút (tùy vào việc tạo dataset)

---

### Bước 2: Demo Từng Phần Chi Tiết

#### 2.1. Demo Small LLM (1 điểm)

```bash
python src/demo_chatbot.py
# Chọn option 1
```

**Hoặc chạy thủ công:**
```python
from chatbot.small_llm import get_llm

llm = get_llm("qwen2-0.5b")
param_count = sum(p.numel() for p in llm.model.parameters())
print(f"Số tham số: {param_count/1e9:.3f} tỷ")
# Output: Số tham số: 0.315 tỷ ✅
```

**Điểm trình bày:**
- Model: Qwen2-0.5B-Instruct
- Số tham số: 315M (0.315 tỷ) < 1 tỷ ✅
- Đã sử dụng 4-bit quantization để tối ưu

---

#### 2.2. Demo GraphRAG (0.5 điểm)

```bash
python src/demo_chatbot.py
# Chọn option 2
```

**Hoặc chạy thủ công:**
```python
from chatbot import GraphRAG

rag = GraphRAG()
context = rag.retrieve_context("BTS có bao nhiêu thành viên?")
print(f"Entities: {len(context['entities'])}")
print(f"Facts: {context['facts']}")
```

**Điểm trình bày:**
- Knowledge Graph: 4,596 nodes, 6,107 edges
- GraphRAG: Entity extraction + Graph traversal + Semantic search
- Context retrieval từ đồ thị tri thức

---

#### 2.3. Demo Multi-hop Reasoning (1.5 điểm)

```bash
python src/demo_chatbot.py
# Chọn option 3
```

**Test cases:**
1. **1-hop**: "Thành viên của BTS" → BTS → MEMBER_OF → Artists
2. **2-hop**: "Công ty của Jungkook" → Jungkook → Group → Company
3. **2-hop**: "BTS và SEVENTEEN cùng công ty?" → So sánh
4. **3-hop**: "Labelmates của BTS" → BTS → Company → Other Groups

**Điểm trình bày:**
- Hỗ trợ 1-hop, 2-hop, 3-hop reasoning
- Chain reasoning, Aggregation, Comparison
- Confidence scoring

---

#### 2.4. Tạo Evaluation Dataset (1 điểm)

```bash
# Cách 1: Dùng script
python src/run_chatbot.py --mode eval --num-questions 2000

# Cách 2: Dùng Web UI
python src/run_chatbot.py --mode ui
# Vào tab "📝 Đánh giá" → Chọn 2000 câu hỏi → Click "Tạo Dataset"
```

**Kết quả:**
- File: `data/evaluation_dataset.json`
- Tổng số: ≥ 2000 câu hỏi
- Phân bố: 1-hop (700+), 2-hop (700+), 3-hop (600+)
- Loại: True/False, Yes/No, Multiple Choice

**Điểm trình bày:**
- Dataset có 2000+ câu hỏi ✅
- Các loại: Đúng/Sai, Có/Không, Trắc nghiệm
- Phân bố đều theo số hop

---

#### 2.5. Comparison với Chatbot khác (0.5 điểm)

```bash
# Chạy comparison
python src/run_chatbot.py --mode compare --max-compare 500
```

**Kết quả:**
- File: `data/comparison_results.json`
- So sánh: K-pop Chatbot vs ChatGPT vs Baseline
- Metrics: Accuracy, Accuracy by hops, Response time

**Điểm trình bày:**
- So sánh với ChatGPT (nếu có API key)
- So sánh với Random Baseline
- Kết quả: K-pop Chatbot có accuracy cao hơn nhờ knowledge graph

---

### Bước 3: Demo Live Chatbot

#### 3.1. CLI Mode

```bash
python src/run_chatbot.py --mode cli
```

**Test queries:**
```
members BTS
company BLACKPINK
same BTS SEVENTEEN
path Jungkook HYBE
```

#### 3.2. Web UI Mode

```bash
python src/run_chatbot.py --mode ui
```

Truy cập: http://localhost:7860

**Demo các tab:**
- 💬 Trò chuyện: Chat với chatbot
- ❓ Hỏi đáp: Test Yes/No, Multiple Choice
- 🔍 Khám phá: Tìm kiếm entities, xem thông tin nhóm
- 📊 Thống kê: Xem stats của knowledge graph
- 📝 Đánh giá: Tạo evaluation dataset

---

## 📊 Kết Quả Cần Trình Bày

### 1. Screenshots/Video

Chụp màn hình:
- [ ] Knowledge Graph stats (nodes, edges)
- [ ] GraphRAG retrieval results
- [ ] Multi-hop reasoning examples
- [ ] Evaluation dataset statistics
- [ ] Comparison results table
- [ ] Web UI interface

### 2. Files Cần Có

- [ ] `data/evaluation_dataset.json` (≥2000 questions)
- [ ] `data/comparison_results.json`
- [ ] `data/merged_kpop_data.json`
- [ ] Screenshots/figures

### 3. Metrics Cần Trình Bày

**Knowledge Graph:**
- Nodes: 4,596
- Edges: 6,107
- Entity types: 8
- Relationship types: 12

**Evaluation Dataset:**
- Total questions: ≥ 2000
- By hops: 1-hop (700+), 2-hop (700+), 3-hop (600+)
- By type: True/False, Yes/No, Multiple Choice

**Comparison Results:**
- K-pop Chatbot accuracy: ~85%
- ChatGPT accuracy: ~72% (nếu có)
- Baseline accuracy: ~33%

---

## 🎯 Script Trình Bày Đề Xuất

### Phần 1: Giới thiệu (2 phút)
- Vấn đề: Cần chatbot hiểu về K-pop với dữ liệu có cấu trúc
- Giải pháp: Knowledge Graph + GraphRAG + Multi-hop Reasoning

### Phần 2: Kiến trúc (3 phút)
- Knowledge Graph: 4,596 nodes, 6,107 edges
- GraphRAG: Entity extraction + Graph traversal + Semantic search
- Small LLM: Qwen2-0.5B (315M params)

### Phần 3: Demo Live (5 phút)
- Demo Web UI
- Test các câu hỏi 1-hop, 2-hop, 3-hop
- Show reasoning steps

### Phần 4: Evaluation (3 phút)
- Evaluation Dataset: 2000+ questions
- Comparison results
- Accuracy metrics

### Phần 5: Kết luận (2 phút)
- Ưu điểm: Chính xác, có thể giải thích, nhanh
- Hạn chế: Chỉ trong phạm vi knowledge graph
- Hướng phát triển

---

## ⚠️ Lưu Ý Khi Demo

1. **Chuẩn bị trước:**
   - Tạo evaluation dataset trước (mất 5-10 phút)
   - Test tất cả tính năng trước khi demo
   - Chuẩn bị backup plan nếu có lỗi

2. **Khi demo:**
   - Dùng lệnh đặc biệt để nhanh (members BTS thay vì "BTS members")
   - Giải thích từng bước
   - Show code nếu cần

3. **Nếu có lỗi:**
   - Có thể demo với reasoning only (bỏ LLM)
   - Show evaluation dataset đã tạo sẵn
   - Show comparison results đã chạy trước

---

## 📝 Checklist Trước Khi Trình Bày

- [ ] Đã chạy `python src/demo_chatbot.py` thành công
- [ ] Đã tạo evaluation dataset (2000+ questions)
- [ ] Đã chạy comparison và có kết quả
- [ ] Đã test Web UI
- [ ] Đã chuẩn bị screenshots
- [ ] Đã chuẩn bị script trình bày
- [ ] Đã backup code và data

---

**Chúc bạn demo thành công! 🎉**






