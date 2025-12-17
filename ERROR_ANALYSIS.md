# Phân tích nguyên nhân lỗi các câu hỏi

## Các vấn đề chính đã xác định:

### 1. **Pronoun Resolution** (Q04265, Q04660, Q04541)
- **Vấn đề**: "nhóm đó", "nhóm này" không được resolve đúng
- **Nguyên nhân**: Entity extraction chưa tìm được group từ pattern "(nhóm X)"
- **Đã fix**: Thêm `_resolve_pronouns()` để resolve pronouns trước khi reasoning

### 2. **Pattern "đều trực thuộc Company_X"** (Q01861)
- **Vấn đề**: Pattern này chưa được xử lý
- **Nguyên nhân**: Chỉ có pattern "đều thuộc nhóm" chứ chưa có "đều trực thuộc Company"
- **Đã fix**: Thêm Pattern 3a để xử lý "đều trực thuộc Company_X"

### 3. **Pattern "đều thuộc nhóm Y"** (Q03182, Q03492, Q03564)
- **Vấn đề**: "X và Y đều thuộc nhóm Z" không match đúng
- **Nguyên nhân**: Logic kiểm tra membership chưa đầy đủ
- **Đã fix**: Cải thiện Pattern 3b để check tất cả artists trong context

### 4. **Company Name Matching** (Q00345, Q00509, Q00937, Q03953)
- **Vấn đề**: Company names không match đúng (ví dụ: "Company_Fnc" vs "Company_YG Entertainment")
- **Nguyên nhân**: 
  - Regex extraction không đủ flexible
  - Normalization không xử lý variations (Fnc vs FNC, JYP vs JYP Entertainment)
- **Đã fix**: 
  - Cải thiện regex để match "Company_X Entertainment"
  - Thêm alternative matching với title case
  - Word-based matching thay vì chỉ substring

### 5. **Same Company Logic** (Q01952, Q01585, Q01848, Q01309)
- **Vấn đề**: Trả "Có" khi nên trả "Không" (false positive)
- **Nguyên nhân**: Intersection logic có thể match nhầm hoặc có edge case
- **Cần kiểm tra**: Dữ liệu KG có đúng không, logic intersection có bug không

### 6. **Error Handling** (Q01720)
- **Vấn đề**: Trả về "Error" thay vì answer
- **Nguyên nhân**: Exception không được catch
- **Đã fix**: Thêm try-except trong `answer_yes_no()`

### 7. **Pattern "qua nhóm X"** (Q01051)
- **Vấn đề**: "HyunA có phải trực thuộc Company_JYP qua nhóm ITZY không"
- **Nguyên nhân**: 3-hop reasoning chưa xử lý pattern này
- **Cần fix**: Pattern matching cho "qua nhóm X"

### 8. **Multiple Choice với "nhóm này"** (Q04541, Q04660)
- **Vấn đề**: "nhóm này trực thuộc công ty nào?"
- **Nguyên nhân**: Pronoun resolution chưa hoạt động trong multiple choice
- **Cần fix**: Áp dụng pronoun resolution trong `answer_multiple_choice()`

## Các fix đã thực hiện:

1. ✅ Thêm `_resolve_pronouns()` method
2. ✅ Cải thiện company name extraction và matching
3. ✅ Thêm pattern "đều trực thuộc Company_X"
4. ✅ Thêm error handling
5. ✅ Cải thiện pattern matching cho "chung công ty với"

## Các fix còn cần làm:

1. ⚠️ Kiểm tra lại same_company intersection logic
2. ⚠️ Thêm pattern "qua nhóm X" 
3. ⚠️ Áp dụng pronoun resolution trong multiple choice
4. ⚠️ Test với dữ liệu thực để verify các fix








