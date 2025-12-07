# 📋 Chứng minh: Small LLM (≤1B Parameters)

Tài liệu này chỉ ra các đoạn code thể hiện việc sử dụng mô hình ngôn ngữ nhỏ với số lượng tham số ≤ 1 tỷ.

---

## 1. Định nghĩa Model (File: `src/chatbot/small_llm.py`)

### 1.1. Model được chọn: Qwen2-0.5B-Instruct

```49:49:src/chatbot/small_llm.py
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
```

**Giải thích:** Model mặc định là `Qwen2-0.5B-Instruct` với **0.5 tỷ tham số** (500M parameters).

### 1.2. Cấu hình Model

```63:67:src/chatbot/small_llm.py
    "qwen2-0.5b": LLMConfig(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        max_new_tokens=512,
        temperature=0.7
    ),
```

**Giải thích:** Model key `"qwen2-0.5b"` được định nghĩa với model `Qwen2-0.5B-Instruct` (0.5B = 500M parameters).

### 1.3. Class Documentation

```101:107:src/chatbot/small_llm.py
class SmallLLM:
    """
    Small Language Model wrapper for K-pop chatbot.
    
    Uses quantized models (≤1B parameters) for efficient inference
    while maintaining good response quality for Vietnamese K-pop Q&A.
    """
```

**Giải thích:** Class `SmallLLM` được document rõ ràng là sử dụng models **≤1B parameters**.

---

## 2. Tính toán và Hiển thị Số Tham Số (File: `src/chatbot/small_llm.py`)

### 2.1. Method tính số tham số

```230:241:src/chatbot/small_llm.py
    def _get_model_size(self) -> str:
        """Get model size in human-readable format."""
        if self.model is None:
            return "Unknown"
            
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count >= 1e9:
            return f"{param_count / 1e9:.2f}B parameters"
        elif param_count >= 1e6:
            return f"{param_count / 1e6:.2f}M parameters"
        else:
            return f"{param_count} parameters"
```

**Giải thích:** 
- `sum(p.numel() for p in self.model.parameters())` - Tính tổng số tham số
- Hiển thị dạng "B" (tỷ) nếu ≥ 1e9, "M" (triệu) nếu ≥ 1e6

### 2.2. Hiển thị khi load model

```223:224:src/chatbot/small_llm.py
            print(f"✅ Model loaded successfully!")
            print(f"   Model size: {self._get_model_size()}")
```

**Giải thích:** Khi load model, sẽ in ra số tham số của model.

---

## 3. Verification trong Demo (File: `src/demo_chatbot.py`)

### 3.1. Demo kiểm tra số tham số

```30:47:src/demo_chatbot.py
def demo_1_small_llm():
    """Demo 1: Small LLM (≤1B params) - 1 điểm"""
    print_section("1. DEMO: Small LLM (≤1B Parameters)")
    
    from chatbot.small_llm import SmallLLM, get_llm
    
    print("🔄 Đang khởi tạo Small LLM...")
    try:
        llm = get_llm("qwen2-0.5b")
        
        # Get model size
        param_count = sum(p.numel() for p in llm.model.parameters())
        param_count_b = param_count / 1e9
        
        print(f"\n✅ Model: Qwen2-0.5B-Instruct")
        print(f"✅ Số tham số: {param_count_b:.3f} tỷ ({param_count/1e6:.1f}M)")
        print(f"✅ Yêu cầu: ≤ 1 tỷ tham số")
        print(f"✅ Kết quả: {'✅ ĐẠT' if param_count_b <= 1.0 else '❌ KHÔNG ĐẠT'}")
```

**Giải thích:**
- Load model `qwen2-0.5b`
- Tính số tham số: `param_count = sum(p.numel() for p in llm.model.parameters())`
- Chuyển đổi sang tỷ: `param_count_b = param_count / 1e9`
- **Verify:** `param_count_b <= 1.0` → ĐẠT yêu cầu

---

## 4. Verification trong Test (File: `src/test_chatbot.py`)

### 4.1. Test kiểm tra số tham số

```248:255:src/test_chatbot.py
    # 1. Small LLM (≤1B params)
    print("1. ✅ Small LLM (≤1B params):")
    if chatbot.llm:
        param_count = sum(p.numel() for p in chatbot.llm.model.parameters())
        param_count_b = param_count / 1e9
        print(f"   - Model: Qwen2-0.5B-Instruct")
        print(f"   - Số tham số: {param_count_b:.3f} tỷ")
        print(f"   - Yêu cầu: ≤ 1 tỷ → {'✅ ĐẠT' if param_count_b <= 1.0 else '❌ KHÔNG ĐẠT'}")
```

**Giải thích:** Tương tự demo, test script cũng verify số tham số ≤ 1 tỷ.

---

## 5. Sử dụng trong Chatbot (File: `src/chatbot/chatbot.py`)

### 5.1. Khởi tạo với model mặc định

```67:73:src/chatbot/chatbot.py
    def __init__(
        self,
        data_path: str = "data/merged_kpop_data.json",
        llm_model: str = "qwen2-0.5b",
        use_embeddings: bool = True,
        verbose: bool = True
    ):
```

**Giải thích:** Chatbot mặc định sử dụng `llm_model="qwen2-0.5b"` (0.5B parameters).

### 5.2. Load LLM

```108:119:src/chatbot/chatbot.py
        # 4. Small LLM (optional)
        self.llm = None
        if llm_model:
            if verbose:
                print(f"  🤖 Loading LLM: {llm_model}...")
            try:
                self.llm = get_llm(llm_model)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ LLM loading failed: {e}")
                    print("  💡 Using fallback mode (context-based responses)")
                self.llm = None
```

**Giải thích:** Load LLM thông qua `get_llm(llm_model)` với model key `"qwen2-0.5b"`.

---

## 6. Cách Chạy và Verify

### 6.1. Chạy Demo

```bash
python src/demo_chatbot.py
```

**Output mẫu:**
```
✅ Model: Qwen2-0.5B-Instruct
✅ Số tham số: 0.500 tỷ (500.0M)
✅ Yêu cầu: ≤ 1 tỷ tham số
✅ Kết quả: ✅ ĐẠT
```

### 6.2. Chạy Test

```bash
python src/test_chatbot.py
# Chọn option 4: Kiểm tra yêu cầu bài tập
```

**Output mẫu:**
```
1. ✅ Small LLM (≤1B params):
   - Model: Qwen2-0.5B-Instruct
   - Số tham số: 0.500 tỷ
   - Yêu cầu: ≤ 1 tỷ → ✅ ĐẠT
```

### 6.3. Verify trực tiếp trong code

```python
from chatbot.small_llm import get_llm

llm = get_llm("qwen2-0.5b")
param_count = sum(p.numel() for p in llm.model.parameters())
param_count_b = param_count / 1e9

print(f"Số tham số: {param_count_b:.3f} tỷ")
print(f"Yêu cầu: ≤ 1 tỷ → {'✅ ĐẠT' if param_count_b <= 1.0 else '❌ KHÔNG ĐẠT'}")
```

---

## 7. Tóm tắt

| Yếu tố | Giá trị | Vị trí trong code |
|--------|---------|-------------------|
| **Model được chọn** | Qwen2-0.5B-Instruct | `small_llm.py:49, 64` |
| **Số tham số** | 0.5 tỷ (500M) | Model specification |
| **Tính toán số tham số** | `sum(p.numel() for p in model.parameters())` | `small_llm.py:235` |
| **Verification** | `param_count_b <= 1.0` | `demo_chatbot.py:47`, `test_chatbot.py:255` |
| **Sử dụng trong chatbot** | `llm_model="qwen2-0.5b"` | `chatbot.py:70` |

---

## 8. Kết luận

✅ **ĐẠT YÊU CẦU:** 
- Model: Qwen2-0.5B-Instruct
- Số tham số: **0.5 tỷ (500M)** < **1 tỷ**
- Code có verification rõ ràng: `param_count_b <= 1.0`
- Có demo và test để chứng minh

---

## 9. Các file liên quan

1. **`src/chatbot/small_llm.py`** - Định nghĩa và load model
2. **`src/demo_chatbot.py`** - Demo verification
3. **`src/test_chatbot.py`** - Test verification
4. **`src/chatbot/chatbot.py`** - Sử dụng model trong chatbot




