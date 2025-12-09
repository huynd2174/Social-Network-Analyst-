# 📝 Giải thích: Tập dữ liệu Đánh giá (Evaluation Dataset)

## Tóm tắt ngắn gọn

**Cách làm:** Tự động generate câu hỏi từ Knowledge Graph (KHÔNG dùng ChatGPT/NotebookLM)

**Quy trình:**
1. Load Knowledge Graph (nodes + edges)
2. Generate questions dựa trên relationships trong graph
3. Phân loại: True/False, Yes/No, Multiple Choice
4. Phân bố: 1-hop, 2-hop, 3-hop
5. Tổng số: **2415 câu hỏi** (≥ 2000)

---

## Chi tiết

### 1. Cách Generate Questions

#### Từ Knowledge Graph:

```python
# File: src/chatbot/evaluation.py

class EvaluationDatasetGenerator:
    def __init__(self):
        self.kg = KpopKnowledgeGraph()
        
        # Cache data từ graph
        self.groups_with_members = {}  # Group → Members
        self.groups_with_companies = {}  # Group → Company
        self.companies_with_groups = {}  # Company → Groups
```

**Giải thích:** Load Knowledge Graph và cache các relationships để generate questions.

#### Generate 1-hop Questions:

```python
def generate_1hop_membership_tf(self, count: int = 100):
    """Generate True/False: 'Jungkook là thành viên của BTS'"""
    group = random.choice(groups)
    members = self.groups_with_members[group]
    
    if random.random() > 0.5:
        # True: Chọn member thực sự
        member = random.choice(members)
        question = f"{member} là thành viên của {group}."
        answer = "Đúng"
    else:
        # False: Chọn member từ group khác
        other_member = random.choice(other_group_members)
        question = f"{other_member} là thành viên của {group}."
        answer = "Sai"
```

**Giải thích:** 
- Lấy relationships từ graph (Group → Members)
- Generate True/False questions
- True: Dùng relationship thực tế
- False: Dùng relationship sai

#### Generate 2-hop Questions:

```python
def generate_2hop_artist_company_tf(self, count: int = 100):
    """Generate: 'Jungkook thuộc công ty HYBE' (Artist → Group → Company)"""
    # 2-hop: Artist → Group → Company
    artist = random.choice(artists)
    group = self.kg.get_artist_groups(artist)[0]
    company = self.kg.get_group_company(group)
    
    question = f"{artist} thuộc công ty {company}."
    answer = "Đúng"
```

**Giải thích:**
- Traverse graph 2 hops: Artist → Group → Company
- Generate questions cần multi-hop reasoning

#### Generate 3-hop Questions:

```python
def generate_3hop_artist_labelmate_tf(self, count: int = 100):
    """Generate: 'Jungkook và Lisa cùng công ty' (Artist → Group → Company ← Group ← Artist)"""
    # 3-hop: Artist1 → Group1 → Company ← Group2 ← Artist2
    company = random.choice(companies)
    group1, group2 = random.sample(company_groups, 2)
    artist1 = random.choice(group1_members)
    artist2 = random.choice(group2_members)
    
    question = f"{artist1} và {artist2} thuộc cùng công ty."
    answer = "Đúng"
```

**Giải thích:**
- Traverse graph 3 hops
- Generate questions phức tạp hơn

---

### 2. Các Loại Questions

#### True/False:

```python
question = "Jungkook là thành viên của BTS."
answer = "Đúng"  # hoặc "Sai"
```

#### Yes/No:

```python
question = "Jungkook có phải thành viên của BTS không?"
answer = "Có"  # hoặc "Không"
```

#### Multiple Choice:

```python
question = "Jungkook thuộc công ty nào?"
choices = ["HYBE", "SM Entertainment", "JYP Entertainment", "YG Entertainment"]
answer = "A"  # HYBE
```

---

### 3. Phân bố Questions

```python
def generate_full_dataset(self, target_count: int = 2000):
    all_questions = []
    
    # 1-hop: 840 questions
    all_questions.extend(self.generate_1hop_membership_tf(120))
    all_questions.extend(self.generate_1hop_membership_yn(120))
    all_questions.extend(self.generate_1hop_membership_mc(120))
    all_questions.extend(self.generate_1hop_company_tf(120))
    all_questions.extend(self.generate_1hop_company_mc(120))
    all_questions.extend(self.generate_1hop_member_count(240))
    
    # 2-hop: 840 questions
    all_questions.extend(self.generate_2hop_artist_company_tf(140))
    all_questions.extend(self.generate_2hop_same_company_yn(140))
    all_questions.extend(self.generate_2hop_labelmates_mc(140))
    all_questions.extend(self.generate_2hop_same_group_yn(420))
    
    # 3-hop: 750 questions
    all_questions.extend(self.generate_3hop_artist_labelmate_tf(250))
    all_questions.extend(self.generate_3hop_company_of_artist_mc(500))
    
    # Total: 2415 questions
```

**Kết quả:**
- 1-hop: 840 questions
- 2-hop: 840 questions
- 3-hop: 750 questions
- **Tổng: 2415 questions** (≥ 2000) ✅

---

### 4. Cách Chạy

```bash
# Generate dataset
python src/run_chatbot.py --mode eval --num-questions 2000

# Hoặc trong code
from chatbot.evaluation import EvaluationDatasetGenerator

generator = EvaluationDatasetGenerator()
stats = generator.generate_full_dataset(
    target_count=2000,
    output_path="data/evaluation_dataset.json"
)
```

**Output:** `data/evaluation_dataset.json` với 2415 questions

---

## So sánh với ChatGPT/NotebookLM

| Cách | Ưu điểm | Nhược điểm |
|------|---------|------------|
| **ChatGPT/NotebookLM** | Câu hỏi tự nhiên | Cần API, tốn phí, không đảm bảo đúng |
| **Tự generate từ Graph** (Đã làm) | ✅ Miễn phí, đảm bảo đúng, có thể verify | Câu hỏi có thể đơn giản hơn |

**Lý do chọn tự generate:**
- ✅ Đảm bảo questions dựa trên Knowledge Graph thực tế
- ✅ Có thể verify answer từ graph
- ✅ Không cần API key
- ✅ Có thể generate số lượng lớn (2000+)

---

## Tóm tắt

✅ **Cách làm:** Tự động generate từ Knowledge Graph

✅ **Quy trình:**
1. Load Knowledge Graph
2. Cache relationships (Group-Members, Group-Company, etc.)
3. Generate questions theo patterns:
   - 1-hop: Direct relationships
   - 2-hop: 1 intermediate
   - 3-hop: 2 intermediates
4. Phân loại: True/False, Yes/No, Multiple Choice

✅ **Kết quả:** 2415 questions (≥ 2000) ✅

✅ **File:** `data/evaluation_dataset.json`








