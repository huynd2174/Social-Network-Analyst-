ừ# 🔑 Hướng dẫn lấy OpenAI API Key

## Cách 1: Lấy API Key từ OpenAI (Miễn phí có giới hạn)

### Bước 1: Đăng ký tài khoản OpenAI

1. Truy cập: https://platform.openai.com/
2. Click "Sign up" hoặc "Log in"
3. Đăng nhập bằng email hoặc Google/Microsoft account

### Bước 2: Tạo API Key

1. Sau khi đăng nhập, vào: https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Đặt tên cho key (ví dụ: "K-pop Chatbot Evaluation")
4. Copy API key ngay lập tức (chỉ hiện 1 lần!)

### Bước 3: Kiểm tra Credit

1. Vào: https://platform.openai.com/account/billing
2. Kiểm tra "Available credits"
3. **Lưu ý:** OpenAI có free tier với $5 credit (đủ để generate ~1000-2000 questions)

---

## Cách 2: Sử dụng API Key trong Code

### Option 1: Set Environment Variable (Khuyến nghị)

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-...
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Permanent (Windows):**
1. Mở "Environment Variables" trong System Properties
2. Thêm `OPENAI_API_KEY` với value là API key của bạn

### Option 2: Tạo file `.env`

Tạo file `.env` trong thư mục gốc:
```
OPENAI_API_KEY=sk-your-api-key-here
```

Sau đó install python-dotenv:
```bash
pip install python-dotenv
```

Và load trong code:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Option 3: Pass trực tiếp trong code

```python
from chatbot.evaluation import EvaluationDatasetGenerator

generator = EvaluationDatasetGenerator()
stats = generator.generate_full_dataset(
    target_count=2000,
    use_chatgpt=True,
    chatgpt_ratio=0.2  # 20% từ ChatGPT
)
```

---

## Cách 3: Chạy với ChatGPT

### Chạy với API key từ environment:

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Generate dataset với ChatGPT
python src/run_chatbot.py --mode eval --use-chatgpt
```

### Hoặc trong code:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

from chatbot.evaluation import EvaluationDatasetGenerator

generator = EvaluationDatasetGenerator()
stats = generator.generate_full_dataset(
    target_count=2000,
    use_chatgpt=True,
    chatgpt_ratio=0.2  # 20% từ ChatGPT, 80% từ graph
)
```

---

## Chi phí ước tính

### GPT-3.5-turbo:
- Input: ~$0.0015 per 1K tokens
- Output: ~$0.002 per 1K tokens
- **1 question ≈ 500 tokens**
- **2000 questions ≈ $2-3**

### GPT-4 (đắt hơn):
- Input: ~$0.03 per 1K tokens
- Output: ~$0.06 per 1K tokens
- **2000 questions ≈ $30-50**

**Khuyến nghị:** Dùng GPT-3.5-turbo (rẻ và đủ tốt)

---

## Lưu ý bảo mật

⚠️ **QUAN TRỌNG:**
- ❌ KHÔNG commit API key vào Git
- ❌ KHÔNG chia sẻ API key công khai
- ✅ Dùng environment variables
- ✅ Thêm `.env` vào `.gitignore`

---

## Troubleshooting

### Lỗi: "OpenAI API key not found"
- Kiểm tra: `echo $OPENAI_API_KEY` (Linux/Mac) hoặc `echo %OPENAI_API_KEY%` (Windows)
- Đảm bảo đã set đúng

### Lỗi: "Insufficient quota"
- Kiểm tra credit tại: https://platform.openai.com/account/billing
- Có thể cần add payment method

### Lỗi: "Rate limit exceeded"
- OpenAI có rate limit
- Code đã có `time.sleep(1)` để tránh rate limit
- Nếu vẫn lỗi, tăng sleep time

---

## Alternative: Dùng NotebookLM (Không cần API)

NotebookLM không có public API, nhưng có thể:
1. Upload knowledge graph info vào NotebookLM
2. Ask nó generate questions
3. Export kết quả
4. Parse và merge vào dataset

---

## Kết luận

✅ **Cách đơn giản nhất:**
1. Lấy API key từ https://platform.openai.com/api-keys
2. Set environment variable: `export OPENAI_API_KEY="sk-..."`
3. Chạy: `python src/run_chatbot.py --mode eval --use-chatgpt`

✅ **Hoặc không dùng ChatGPT:**
- Code hiện tại đã generate 2415 questions từ graph
- Đủ để đáp ứng yêu cầu ≥ 2000 questions
- Không cần API key






