# Script Thuyết Trình: Phân tích Phát hiện Cộng đồng trong Mạng K-pop

---

## 📌 SLIDE 1: Giới thiệu

**[Nói]:**
> "Phần tiếp theo của bài thuyết trình, em sẽ trình bày về kết quả phân tích phát hiện cộng đồng trong mạng tri thức K-pop. Đây là một trong những phân tích quan trọng nhất để hiểu cấu trúc và tổ chức của ngành công nghiệp K-pop."

---

## 📌 SLIDE 2: Phương pháp

**[Nói]:**
> "Để phát hiện cộng đồng, chúng em sử dụng thuật toán Louvain - một thuật toán tối ưu hóa modularity được đánh giá cao về tính hiệu quả và chất lượng kết quả. Thuật toán này hoạt động theo hai giai đoạn lặp: đầu tiên là tối ưu cục bộ, nơi mỗi node được gán vào cộng đồng làm tăng modularity nhiều nhất; sau đó là giai đoạn hợp nhất, nơi các node cùng cộng đồng được gộp thành siêu node. Quá trình này lặp lại cho đến khi modularity không còn tăng."

---

## 📌 SLIDE 3: Kết quả tổng quan

**[Nói]:**
> "Kết quả phân tích cho thấy đồ thị K-pop có cấu trúc cộng đồng RẤT RÕ RÀNG. Cụ thể, thuật toán đã phát hiện được 177 cộng đồng với giá trị modularity đạt 0.5479. Đây là một con số rất cao, vượt xa ngưỡng 0.5 thường được sử dụng để đánh giá các mạng xã hội có phân cụm mạnh.
>
> Điều này phản ánh tính phân tầng tự nhiên của ngành K-pop, nơi nghệ sĩ, nhóm nhạc, công ty và sản phẩm âm nhạc không phân bố ngẫu nhiên, mà hình thành những cụm liên kết ổn định và có ý nghĩa."

**[Chỉ vào số liệu]:**
- Số cộng đồng: 177
- Modularity: 0.5479 (> 0.5 = Cấu trúc MẠNH)
- Cộng đồng lớn nhất: 198 nodes (11.7% mạng)

---

## 📌 SLIDE 4: Phân bố kích thước cộng đồng

**[Nói]:**
> "Phân bố kích thước cộng đồng cho thấy một đặc điểm thú vị: sự bất đối xứng rõ rệt hay còn gọi là phân bố lệch phải. Cụ thể, đa số cộng đồng có kích thước nhỏ, chỉ vài node, trong khi cộng đồng lớn nhất đạt tới 198 node, chiếm khoảng 11.7% toàn mạng.
>
> Kích thước trung bình vào khoảng 9.6 node, nhưng trung vị chỉ 4 node - sự chênh lệch lớn giữa trung bình và trung vị này cho thấy phần lớn mạng phân mảnh thành nhiều cụm nhỏ. Tuy nhiên, vẫn tồn tại một số ít cộng đồng quy mô lớn đóng vai trò trung tâm, chi phối cấu trúc toàn mạng.
>
> Nếu nhìn vào phân bố chi tiết: khoảng 68% cộng đồng có kích thước nhỏ từ 1 đến 10 node, đại diện cho nghệ sĩ solo hoặc các sản phẩm âm nhạc đơn lẻ. Khoảng 23% là cộng đồng vừa từ 11 đến 50 node, thường là một nhóm nhạc cùng với discography của họ. Chỉ có khoảng 6% là cộng đồng lớn từ 51 đến 100 node, đại diện cho các nhóm nhạc huyền thoại như BLACKPINK hay Big Bang. Và đặc biệt, chỉ có 4 mega-community với hơn 100 node, đại diện cho ecosystem của các công ty giải trí lớn."

---

## 📌 SLIDE 5: Top 9 cộng đồng lớn nhất

**[Nói]:**
> "Bây giờ, chúng ta sẽ đi vào phân tích chi tiết 9 cộng đồng lớn nhất được phát hiện trong mạng. Đây là những cộng đồng có tầm ảnh hưởng lớn nhất đến cấu trúc của toàn bộ mạng K-pop."

**[Chỉ vào biểu đồ và giải thích từng cộng đồng]:**

> "Cộng đồng lớn nhất với 198 node là hệ sinh thái của Pledis Entertainment, bao gồm các nghệ sĩ như After School, NU'EST, SEVENTEEN và SF9. Đây là công ty có nhiều nhóm nhạc thuộc các thế hệ khác nhau, tạo nên mạng lưới kết nối phong phú.
>
> Cộng đồng thứ hai với 188 node thuộc về JYP Entertainment - một trong 'Big 3' của K-pop. Cộng đồng này bao gồm TWICE, Stray Kids, ITZY, 2PM, GOT7 và nhiều nghệ sĩ solo khác.
>
> Tiếp theo là Cube Entertainment với 162 node, nổi bật với các nhóm như BTOB, (G)I-DLE và Pentagon.
>
> Điều đáng chú ý là cộng đồng thứ 4 với 119 node lại không phải là một công ty mà là Girls' Generation - hay còn gọi là SNSD. Đây là nhóm nhạc nữ huyền thoại thế hệ 2, có tầm ảnh hưởng lớn đến mức tạo thành một 'tiểu vũ trụ' riêng biệt trong mạng, bao gồm 8 thành viên, 19 bài hát và 17 album.
>
> Tương tự, BLACKPINK với 85 node và Big Bang với 78 node cũng tạo thành các cộng đồng riêng biệt - không nằm trong cộng đồng công ty mẹ YG. Điều này cho thấy tầm ảnh hưởng đặc biệt của 3 nhóm nhạc này trong toàn bộ industry K-pop."

---

## 📌 SLIDE 6: Phân loại cộng đồng

**[Nói]:**
> "Phân tích sâu hơn cho thấy các cộng đồng có thể được phân thành hai loại chính:
>
> Thứ nhất là Company-based communities, chiếm khoảng 70% trong top 10. Đây là các cộng đồng được hình thành xung quanh một công ty giải trí như Pledis, JYP, SM, YG hay HYBE. Cấu trúc của các cộng đồng này chủ yếu gồm Artist và Group, chiếm 50-70%, phản ánh mối quan hệ quản lý giữa công ty và nghệ sĩ. Nghệ sĩ cùng công ty thường chia sẻ producer, nhạc sĩ, phong cách âm nhạc và chiến lược marketing, tạo nên sự liên kết chặt chẽ.
>
> Thứ hai là Group-centric communities, chiếm khoảng 30%. Đây là các 'tiểu vũ trụ' được hình thành xung quanh một nhóm nhạc cụ thể có tầm ảnh hưởng lớn. Điển hình là Girls' Generation, BLACKPINK và Big Bang. Cấu trúc của các cộng đồng này chủ yếu gồm Album và Song, chiếm 70-80%, phản ánh discography phong phú của nhóm. Sự tồn tại của các cộng đồng này cho thấy tầm quan trọng và ảnh hưởng đặc biệt của những nhóm nhạc huyền thoại này."

---

## 📌 SLIDE 7: Phân tích Coherence

**[Nói]:**
> "Để hiểu sâu hơn về ý nghĩa của các cộng đồng, chúng em thực hiện phân tích Coherence - đo lường mức độ các thực thể có cùng đặc điểm nằm trong cùng một cộng đồng.
>
> Kết quả cho thấy: Company Coherence đạt 63.4% - nghĩa là trung bình 63.4% nghệ sĩ cùng công ty sẽ nằm trong cùng một cộng đồng. Đây là con số khá cao, cho thấy quan hệ công ty là yếu tố MẠNH trong việc hình thành cộng đồng.
>
> Trong khi đó, Genre Coherence chỉ đạt 38.5% - nghĩa là việc nghệ sĩ cùng thể loại âm nhạc nằm trong cùng cộng đồng là khá ngẫu nhiên. Điều này cho thấy thể loại nhạc KHÔNG phải là yếu tố chính hình thành cộng đồng trong K-pop.
>
> Từ đây, chúng ta có thể kết luận: Cộng đồng K-pop được hình thành chủ yếu dựa trên quan hệ CÔNG TY và NHÓM NHẠC, chứ không phải dựa trên thể loại âm nhạc."

---

## 📌 SLIDE 8: Kết luận

**[Nói]:**
> "Tóm lại, qua phân tích phát hiện cộng đồng, chúng em rút ra được 3 kết luận chính:
>
> Thứ nhất, mạng K-pop có cấu trúc cộng đồng RẤT MẠNH với modularity 0.5479, phản ánh tính tổ chức cao của ngành công nghiệp này.
>
> Thứ hai, các yếu tố hình thành cộng đồng theo thứ tự quan trọng là: Quan hệ công ty đứng đầu với 63.4% coherence, tiếp theo là quan hệ nhóm nhạc với khoảng 70% coherence cho các nhóm lớn, và cuối cùng thể loại âm nhạc chỉ đạt 38.5% - ảnh hưởng yếu.
>
> Thứ ba, sự tồn tại của 3 group-centric communities độc lập - Girls' Generation, BLACKPINK và Big Bang - cho thấy tầm ảnh hưởng đặc biệt của những nhóm nhạc huyền thoại này trong toàn bộ industry K-pop.
>
> Kết quả phân tích này có thể ứng dụng vào việc gợi ý nghệ sĩ tương tự, phân tích xu hướng thị trường, và dự đoán khả năng hợp tác giữa các nghệ sĩ."

---

## 📌 SLIDE 9: Q&A

**[Nói]:**
> "Đó là toàn bộ phần trình bày của em về phân tích phát hiện cộng đồng. Cảm ơn thầy/cô và các bạn đã lắng nghe. Em xin sẵn sàng trả lời các câu hỏi."

---

## 📝 GHI CHÚ CHO NGƯỜI THUYẾT TRÌNH

### Thời gian dự kiến:
- Slide 1-2: 1 phút
- Slide 3-4: 2 phút
- Slide 5: 2 phút (phần quan trọng)
- Slide 6-7: 2 phút
- Slide 8-9: 1 phút
- **Tổng: 8-10 phút**

### Câu hỏi có thể gặp:
1. **Tại sao chọn Louvain thay vì thuật toán khác?**
   → Louvain có modularity cao nhất (0.5479) so với các thuật toán khác như Label Propagation (0.2655)

2. **Tại sao BTS không có cộng đồng riêng như BLACKPINK?**
   → BTS nằm trong cộng đồng HYBE (company-based), cho thấy mối liên kết chặt chẽ với công ty mẹ

3. **Ý nghĩa thực tiễn của phân tích này?**
   → Gợi ý nghệ sĩ, dự đoán collab, phân tích thị trường

