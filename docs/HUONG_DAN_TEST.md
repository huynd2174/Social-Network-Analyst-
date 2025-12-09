# 🧪 Hướng dẫn Test Chatbot

## Tổng quan

Chatbot có 3 chế độ test:

1. **Fast Mode** (Reasoning-only): Nhanh (1-5s), chính xác, không dùng LLM
2. **Slow Mode** (Với LLM): Chậm (10-30s), tự nhiên, dùng Small LLM
3. **Hybrid Mode**: Thử Fast Mode trước, nếu không đủ thì dùng Slow Mode

---

## Cách Test

### 1. Test Script Tự Động (Khuyến nghị)

```bash
# Test tất cả chế độ
python src/test_chatbot.py

# Chọn test cụ thể:
# - 1: Fast Mode only (nhanh)
# - 2: Slow Mode only (chậm)
# - 3: Hybrid Mode
# - 4: Kiểm tra yêu cầu bài tập
# - 5: Tất cả
```

### 2. CLI Interactive Mode

```bash
# Chạy CLI mode
python src/run_chatbot.py --mode cli

# Fast Mode (mặc định):
# - Câu hỏi đơn giản sẽ tự động dùng Fast Mode
# - VD: "BTS có bao nhiêu thành viên?"

# Slow Mode (tự động khi cần):
# - Câu hỏi phức tạp sẽ tự động dùng Slow Mode
# - VD: "Giới thiệu về BTS"
```

### 3. Web UI Mode

```bash
# Gradio UI
python src/run_chatbot.py --mode ui

# Streamlit UI (nhẹ hơn)
python src/run_chatbot.py --mode streamlit
```

### 4. Full Demo

```bash
# Demo tất cả tính năng
python src/demo_chatbot.py
```

---

## Test Cases theo Yêu Cầu Bài Tập

### ✅ 1. Small LLM (≤1B params) - 1 điểm

```python
from chatbot.small_llm import get_llm

llm = get_llm("qwen2-0.5b")
param_count = sum(p.numel() for p in llm.model.parameters())
print(f"Số tham số: {param_count/1e9:.3f} tỷ")
# Kết quả: ~0.5 tỷ (✅ ĐẠT)
```

**Test:**
```bash
python src/test_chatbot.py
# Chọn option 4 để kiểm tra
```

### ✅ 2. GraphRAG - 0.5 điểm

**Test:**
```python
from chatbot import GraphRAG

rag = GraphRAG()
context = rag.retrieve_context("BTS có bao nhiêu thành viên?")
print(f"Entities: {len(context['entities'])}")
print(f"Facts: {len(context['facts'])}")
```

**Hoặc:**
```bash
python src/demo_chatbot.py
# Xem phần "2. DEMO: GraphRAG"
```

### ✅ 3. Multi-hop Reasoning - 1.5 điểm

**Test Cases:**

1. **1-hop**: "BTS có bao nhiêu thành viên?"
2. **2-hop**: "Công ty nào quản lý Jungkook?" (Artist → Group → Company)
3. **2-hop**: "BTS và SEVENTEEN có cùng công ty không?"
4. **3-hop**: "Các nhóm cùng công ty với BTS"

**Test:**
```bash
python src/test_chatbot.py
# Chọn option 1 (Fast Mode) để test multi-hop
```

### ✅ 4. Evaluation Dataset (2000+ questions) - 1 điểm

**Tạo dataset:**
```bash
python src/run_chatbot.py --mode eval --num-questions 2000
```

**Kiểm tra:**
```bash
python -c "import json; data=json.load(open('data/evaluation_dataset.json')); print(f\"Total: {data['metadata']['total_questions']} questions\")"
```

**Kết quả:** 2415 câu hỏi (✅ ĐẠT)

### ✅ 5. Comparison - 0.5 điểm

**Chạy comparison:**
```bash
python src/run_chatbot.py --mode compare --max-compare 500
```

**Kết quả:** `data/comparison_results.json`

---

## Test Fast Mode vs Slow Mode

### Fast Mode (Reasoning-only)

**Ưu điểm:**
- ⚡ Nhanh: 1-5 giây
- ✅ Chính xác: Dựa trên knowledge graph
- 💾 Không cần LLM

**Phù hợp:**
- Câu hỏi về thành viên: "BTS có bao nhiêu thành viên?"
- Câu hỏi về công ty: "Công ty nào quản lý BLACKPINK?"
- Câu hỏi Yes/No: "Jungkook có phải thành viên BTS không?"
- Câu hỏi so sánh: "BTS và SEVENTEEN có cùng công ty không?"

**Test:**
```bash
python src/test_chatbot.py
# Chọn option 1
```

### Slow Mode (Với LLM)

**Ưu điểm:**
- 🗣️ Tự nhiên: LLM tạo câu trả lời tự nhiên
- 📝 Tổng hợp: Có thể tổng hợp nhiều thông tin

**Nhược điểm:**
- 🐌 Chậm: 10-30 giây
- ⚠️ Có thể hallucination

**Phù hợp:**
- Câu hỏi phức tạp: "Giới thiệu về BTS"
- Câu hỏi tổng hợp: "So sánh BTS và BLACKPINK"
- Câu hỏi cần context: "Kể về lịch sử K-pop"

**Test:**
```bash
python src/test_chatbot.py
# Chọn option 2
```

### Hybrid Mode (Khuyến nghị)

**Cách hoạt động:**
1. Thử Fast Mode trước (nhanh)
2. Nếu response không đủ tốt → dùng Slow Mode

**Test:**
```bash
python src/test_chatbot.py
# Chọn option 3
```

---

## Test Cases Mẫu

### Câu hỏi đơn giản (Fast Mode)

```python
queries = [
    "BTS có bao nhiêu thành viên?",
    "Công ty nào quản lý BLACKPINK?",
    "Jungkook có phải thành viên BTS không?",
    "BTS và SEVENTEEN có cùng công ty không?",
    "Các nhóm cùng công ty với BTS",
    "Nhóm nhạc đã hợp tác với BTS"
]
```

### Câu hỏi phức tạp (Slow Mode)

```python
queries = [
    "Giới thiệu về BTS",
    "So sánh BTS và BLACKPINK",
    "Kể về lịch sử phát triển của K-pop"
]
```

---

## Troubleshooting

### LLM không load được

**Giải pháp:**
- Chatbot vẫn hoạt động với Fast Mode (không cần LLM)
- Test với: `python src/test_chatbot.py` → chọn option 1

### Response quá chậm

**Giải pháp:**
- Dùng Fast Mode cho câu hỏi đơn giản
- Dùng lệnh nhanh: `members BTS`, `company BLACKPINK`

### Response không chính xác

**Giải pháp:**
- Kiểm tra knowledge graph: `data/merged_kpop_data.json`
- Kiểm tra reasoning steps trong response
- Dùng Fast Mode thay vì Slow Mode cho câu hỏi đơn giản

---

## Checklist Test

- [ ] Fast Mode hoạt động (1-5s)
- [ ] Slow Mode hoạt động (10-30s)
- [ ] Hybrid Mode tự động chuyển đổi
- [ ] Multi-hop reasoning (1-hop, 2-hop, 3-hop)
- [ ] GraphRAG retrieval
- [ ] Evaluation dataset (2000+ questions)
- [ ] Comparison framework
- [ ] Small LLM (≤1B params)

---

## Liên kết

- **Test Script**: `src/test_chatbot.py`
- **Demo Script**: `src/demo_chatbot.py`
- **CLI Runner**: `src/run_chatbot.py`
- **Documentation**: `docs/HUONG_DAN_CHATBOT.md`








