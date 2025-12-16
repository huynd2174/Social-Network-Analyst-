# Phân tích Phát hiện Cộng đồng trong Mạng K-pop

## 1. Phương pháp

### 1.1. Thuật toán sử dụng

Hệ thống sử dụng **thuật toán Louvain** để phát hiện cộng đồng - một thuật toán heuristic tối ưu modularity theo hai giai đoạn:

1. **Tối ưu cục bộ**: Mỗi node được gán vào cộng đồng tăng modularity nhiều nhất
2. **Hợp nhất**: Các node cùng cộng đồng được hợp nhất thành siêu node

Công thức Modularity:
```
Q = (1/2m) × Σ[Aij - (ki×kj/2m)] × δ(ci, cj)
```

### 1.2. Phân tích ngữ nghĩa

Sau khi phát hiện cộng đồng, hệ thống thực hiện **phân tích ngữ nghĩa** để hiểu ý nghĩa thực tế:

- **Company Coherence**: Tỷ lệ nghệ sĩ cùng công ty nằm trong cùng cộng đồng
- **Group Coherence**: Tỷ lệ thành viên nhóm nằm cùng cộng đồng với nhóm
- **Genre Coherence**: Tỷ lệ nghệ sĩ cùng thể loại nằm trong cùng cộng đồng
- **Bridge Nodes**: Các node kết nối nhiều cộng đồng khác nhau

---

## 2. Kết quả

### 2.1. Thống kê cơ bản

| Metric | Giá trị |
|--------|---------|
| Số cộng đồng | 177 |
| Modularity | 0.5479 (> 0.5 = cấu trúc MẠNH) |
| Cộng đồng lớn nhất | 198 nodes (11.7%) |
| Kích thước trung bình | 9.6 nodes |

### 2.2. Kích thước và cấu trúc cộng đồng

Kết quả phân tích cho thấy đồ thị K-pop có cấu trúc cộng đồng rất rõ ràng, với 177 cộng đồng và giá trị modularity đạt 0.5479, vượt xa ngưỡng 0.5 thường gắn với các mạng xã hội có phân cụm mạnh. Điều này phản ánh tính phân tầng tự nhiên của ngành K-pop, nơi nghệ sĩ, nhóm nhạc, công ty và sản phẩm âm nhạc hình thành những cụm liên kết ổn định. Cộng đồng lớn nhất đạt 198 node (chiếm 11.7% toàn mạng), trong khi kích thước trung bình chỉ vào khoảng 9.6 node với trung vị 4 node.

Phân bố kích thước cộng đồng cho thấy sự bất đối xứng rõ rệt (right-skewed distribution): đa số cộng đồng có kích thước nhỏ chỉ vài node, nhưng vẫn tồn tại một số cộng đồng quy mô lớn đóng vai trò trung tâm. Cụ thể, khoảng 68% cộng đồng có kích thước nhỏ từ 1 đến 10 node, đại diện cho nghệ sĩ solo hoặc sản phẩm âm nhạc đơn lẻ. Khoảng 23% là cộng đồng vừa từ 11 đến 50 node, thường là một nhóm nhạc cùng với discography của họ. Chỉ có khoảng 6% là cộng đồng lớn từ 51 đến 100 node, đại diện cho các nhóm nhạc huyền thoại như BLACKPINK hay Big Bang. Và đặc biệt, chỉ có 4 mega-community với hơn 100 node, đại diện cho ecosystem của các công ty giải trí lớn như Pledis Entertainment (198 nodes), JYP Entertainment (188 nodes), Cube Entertainment (162 nodes) và Girls' Generation (119 nodes).

Phân tích cấu trúc nội bộ cho thấy mỗi cộng đồng có thành phần đa dạng với nhiều loại thực thể. Các company-based communities như Pledis, JYP, Cube và SM chủ yếu gồm Artist và Group (chiếm 50-70%), phản ánh mối quan hệ quản lý giữa công ty và nghệ sĩ. Trong khi đó, các group-centric communities như Girls' Generation chủ yếu gồm Album và Song (chiếm 70-80%), phản ánh discography phong phú của nhóm. Điều này cho thấy cấu trúc cộng đồng phản ánh sự đa dạng nhưng có tổ chức của hệ tri thức K-pop, nơi các cụm tri thức lớn được định hình bởi những thực thể có sức ảnh hưởng cao như các công ty giải trí lớn và các nhóm nhạc huyền thoại

---

### 2.3. Coherence Analysis

| Yếu tố | Coherence | Đánh giá |
|--------|-----------|----------|
| Công ty (Company) | 63.4% | ✓ Yếu tố MẠNH |
| Nhóm nhạc (Group) | ~70% | ✓ Yếu tố MẠNH |
| Thể loại (Genre) | 38.5% | ✗ Yếu tố YẾU |

### 2.3. Top 10 Cộng đồng lớn nhất

| # | Kích thước | Loại | Thực thể chính | Ý nghĩa |
|---|------------|------|----------------|---------|
| 1 | 198 nodes | Company-based | **Pledis Entertainment** | Cộng đồng nghệ sĩ Pledis (After School, SF9, NU'EST) |
| 2 | 188 nodes | Company-based | **JYP Entertainment** | Ecosystem JYP (TWICE, Stray Kids, ITZY, 2PM) |
| 3 | 162 nodes | Company-based | **Cube Entertainment** | Nghệ sĩ Cube (BTOB, (G)I-DLE, Pentagon) |
| 4 | 119 nodes | Group-centric | **Girls' Generation** | "Tiểu vũ trụ" SNSD + các album/bài hát liên quan |
| 5 | 107 nodes | Company-based | **SM Entertainment** | Ecosystem SM (EXO, NCT, Red Velvet, aespa) |
| 6 | 104 nodes | Company-based | **YG Entertainment** | Nghệ sĩ YG (BIGBANG, 2NE1, WINNER, iKON) |
| 7 | 85 nodes | Group-centric | **BLACKPINK** | "Tiểu vũ trụ" BP: 4 thành viên + 23 bài hát + 7 album |
| 8 | 80 nodes | Company-based | **HYBE** | Ecosystem HYBE (BTS, TXT, ENHYPEN) |
| 9 | 78 nodes | Group-centric | **Big Bang** | "Tiểu vũ trụ" BB: thành viên + 14 bài hát + 21 album |
| 10 | 70 nodes | Mixed | **T-ara, ITZY** | Cộng đồng hỗn hợp nhiều nhóm nhạc |

### 2.4. Giải thích chi tiết Top 9 cộng đồng

#### 🏢 Cộng đồng Company-based (6/9)

| # | Cộng đồng | Nodes | Giải thích |
|---|-----------|-------|------------|
| 1 | **Pledis Ent.** | 198 | Hệ sinh thái Pledis với After School, NU'EST, SEVENTEEN, SF9. Cộng đồng lớn nhất do Pledis có nhiều nhóm thế hệ khác nhau và nghệ sĩ solo. |
| 2 | **JYP Ent.** | 188 | "Big 3" company với TWICE, Stray Kids, ITZY, 2PM, GOT7. JYP nổi tiếng với chiến lược đào tạo nghệ sĩ toàn diện. |
| 3 | **Cube Ent.** | 162 | BTOB, (G)I-DLE, Pentagon, CLC. Cube tập trung vào nghệ sĩ tự sáng tác và biểu diễn. |
| 5 | **SM Ent.** | 107 | "Big 3" company với EXO, NCT, Red Velvet, aespa. SM nổi tiếng với concept độc đáo và visual. |
| 6 | **YG Ent.** | 104 | "Big 3" company với BIGBANG, 2NE1, WINNER, iKON. YG tập trung Hip-hop và style "swag". |
| 8 | **HYBE** | 80 | Công ty của BTS với TXT, ENHYPEN, LE SSERAFIM. HYBE (Big Hit) là công ty mới nổi thành "Big 4". |

#### 🎤 Cộng đồng Group-centric (3/9)

| # | Cộng đồng | Nodes | Giải thích |
|---|-----------|-------|------------|
| 4 | **Girls' Generation** | 119 | "Tiểu vũ trụ" SNSD - nhóm nhạc nữ huyền thoại thế hệ 2. Bao gồm 8 thành viên (Taeyeon, Tiffany, Seohyun...) + 19 bài hát + 17 album. SNSD có ảnh hưởng lớn đến toàn bộ industry. |
| 7 | **BLACKPINK** | 85 | "Tiểu vũ trụ" BP - nhóm nhạc nữ thành công nhất hiện tại. 4 thành viên (Jennie, Lisa, Rosé, Jisoo) + 23 bài hát + 7 album. BP có reach global lớn nhất K-pop. |
| 9 | **Big Bang** | 78 | "Tiểu vũ trụ" BB - nhóm nhạc nam huyền thoại "Kings of K-pop". 4 thành viên (G-Dragon, Taeyang, T.O.P, Daesung) + 14 bài hát + 21 album. BB định hình K-pop thế hệ 2. |

### 2.5. Tại sao các cộng đồng này lớn nhất?

1. **Company-based communities lớn** vì:
   - Công ty lớn có nhiều nhóm nhạc và nghệ sĩ solo
   - Nghệ sĩ cùng công ty chia sẻ producer, nhạc sĩ, công ty phân phối
   - Có collaboration nội bộ (SM Station, JYP collab stages...)

2. **Group-centric communities lớn** vì:
   - Nhóm nhạc huyền thoại có lịch sử hoạt động dài
   - Nhiều album, bài hát, concert, show truyền hình
   - Thành viên có hoạt động solo tạo thêm connections

3. **Điểm thú vị:**
   - SNSD, BLACKPINK, Big Bang là 3 nhóm duy nhất tạo thành cộng đồng riêng biệt
   - Điều này cho thấy **tầm ảnh hưởng đặc biệt** của 3 nhóm này trong industry
   - BTS không có cộng đồng riêng vì nằm trong cộng đồng HYBE (company-based)

---

## 3. Kết luận

### 3.1. Các phát hiện chính

1. **Cấu trúc cộng đồng RẤT MẠNH** (Modularity = 0.5479 > 0.5)

2. **Yếu tố hình thành cộng đồng theo thứ tự quan trọng:**
   - 🥇 **Công ty quản lý** (63.4% coherence) - Yếu tố quyết định nhất
   - 🥈 **Quan hệ nhóm nhạc** (~70% coherence) - Tạo "tiểu vũ trụ"
   - 🥉 **Thể loại âm nhạc** (38.5% coherence) - Ảnh hưởng yếu

3. **Bridge Nodes quan trọng:**
   - Genre (R&B, Dance-pop, Hip hop) - 50% bridge nodes
   - Occupation (Diễn viên, Nhạc sĩ) - 20% bridge nodes
   - Big Companies (SM, JYP) - 15% bridge nodes

### 3.2. Ý nghĩa thực tiễn

- **Nghệ sĩ cùng công ty** có xu hướng mạnh nằm trong cùng cộng đồng
- **Nhóm nhạc lớn** (BTS, BLACKPINK) tạo thành ecosystem riêng với thành viên, bài hát, album
- **Thể loại nhạc** KHÔNG phải yếu tố quyết định cấu trúc cộng đồng K-pop
- **Các công ty lớn** (SM, JYP, YG, HYBE) đóng vai trò cầu nối giữa các cộng đồng

---

## 4. Ứng dụng

1. **Gợi ý nghệ sĩ tương tự**: Dựa trên cộng đồng chung
2. **Phân tích xu hướng**: Đặc điểm chung trong cộng đồng
3. **Dự đoán hợp tác**: Nghệ sĩ cùng cộng đồng có tiềm năng collab cao

