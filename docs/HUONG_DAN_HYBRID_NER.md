# HƯỚNG DẪN SỬ DỤNG HYBRID NER (RULE-BASED + ML)

## 📋 Tổng quan

Hệ thống NER hiện tại sử dụng **Hybrid Approach** kết hợp:
1. **Rule-based NER**: Sử dụng regex patterns và domain knowledge (đã có sẵn)
2. **ML-based NER**: Sử dụng pre-trained Vietnamese NER model từ HuggingFace

## 🚀 Cài đặt

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements_ml_ner.txt
```

**Lưu ý:**
- Nếu không có GPU, model vẫn chạy được trên CPU (chậm hơn)
- Model sẽ tự động tải về lần đầu tiên chạy (khoảng 500MB)

### Bước 2: Chạy NER

```bash
python run_ner.py
```

Hệ thống sẽ:
1. Tự động tải ML model (nếu chưa có)
2. Chạy cả rule-based và ML-based NER
3. Merge kết quả từ cả hai phương pháp
4. Lưu vào `kpop_ner_result.json`

## 📊 Kết quả

### Metadata trong output file:

```json
{
  "metadata": {
    "description": "Thực thể K-pop được nhận dạng và lọc (Hybrid: Rule-based + ML)",
    "ml_ner_enabled": true,
    "entities_by_method": {
      "rule-based": 2500,
      "ml-based": 278,
      "known_list": 50
    }
  }
}
```

### Entity format:

```json
{
  "text": "BTS",
  "type": "Group",
  "method": "ml-based",  // hoặc "rule-based"
  "confidence": 0.85,
  "source_node": "node_id",
  "sources": ["node_id1", "node_id2"]
}
```

## 🔧 Cấu hình

### Thay đổi model:

Trong `ml_ner.py`, bạn có thể thay đổi model:

```python
# Model mặc định
model = get_ner_model("NlpHUST/ner-vietnamese-electra-base")

# Hoặc dùng model khác
model = get_ner_model("vinai/phobert-base")
```

### Tắt ML NER:

Nếu không muốn dùng ML NER, chỉ cần không cài `transformers`:

```bash
# Không cài transformers
# Hệ thống sẽ tự động chỉ dùng rule-based
```

## 📈 So sánh kết quả

### Rule-based:
- ✅ Độ chính xác cao cho domain K-pop
- ✅ Không cần training data
- ✅ Dễ giải thích
- ⚠️ Có thể bỏ sót entities không khớp pattern

### ML-based:
- ✅ Tự động phát hiện entities
- ✅ Xử lý được các pattern phức tạp
- ✅ Bổ sung cho rule-based
- ⚠️ Có thể có false positives

### Hybrid:
- ✅ Kết hợp ưu điểm của cả hai
- ✅ Tăng recall (nhiều entities hơn)
- ✅ Vẫn giữ độ chính xác cao (nhờ rule-based)
- ✅ Dễ so sánh trong báo cáo

## 🐛 Xử lý lỗi

### Lỗi: "transformers không được cài đặt"
```bash
pip install transformers torch
```

### Lỗi: "Không thể tải model"
- Kiểm tra kết nối internet (model cần tải về)
- Thử lại sau vài phút
- Hệ thống sẽ tự động fallback về rule-based

### Model chạy chậm:
- Bình thường nếu chạy trên CPU
- Để tăng tốc, cài PyTorch với CUDA (nếu có GPU)

## 💡 Tips

1. **Lần đầu chạy**: Model sẽ tải về (~500MB), mất vài phút
2. **Kết quả tốt nhất**: Dùng cả rule-based và ML-based
3. **Báo cáo**: So sánh số lượng entities từ mỗi method trong metadata

## 📝 Ghi chú

- ML model sử dụng labels mặc định (PERSON, ORG, MISC...) và được map sang labels của chúng ta (Artist, Group, Company...)
- Entities từ ML model có confidence thấp hơn một chút so với rule-based (để ưu tiên rule-based)
- Nếu cùng một entity được tìm thấy bởi cả hai method, sẽ được merge và giữ confidence cao nhất


