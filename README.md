## Social-network-analyst

Thu thập, chuẩn hóa và xây dựng mạng lưới nghệ sĩ/nhóm nhạc/album/bài hát Hàn Quốc từ Wikipedia tiếng Việt, kèm xuất dữ liệu sang Neo4j và notebook phân tích.

### 1) Yêu cầu môi trường
- Python 3.9+
- pip packages: requests, beautifulsoup4, pandas, matplotlib, networkx (tùy chọn), neo4j (tùy chọn)

```bash
pip install -r requirements.txt  # nếu có
# hoặc
pip install requests beautifulsoup4 pandas matplotlib networkx neo4j
```

### 2) Chạy crawler và lưu JSON
```bash
python korean_music_bfs.py \
  --max-nodes 3000 --top-k 40 --delay 0.2 \
  --output korean_artists_graph_bfs.json
```

Các tham số chính:
- `--seeds`: danh sách hạt giống (tiêu đề Wikipedia TV)
- `--max-nodes`: số node tối đa
- `--top-k`: số liên kết ưu tiên tối đa mỗi node
- `--delay`: độ trễ giữa các request (giây)

Crawler đã có nhiều bộ lọc để loại các thực thể ngoài lĩnh vực âm nhạc (công ty, phim, địa danh, esports, chính trị, lịch sử…), và logic siết chặt cho Artist (bắt buộc có trường nghề nghiệp thuộc âm nhạc trong infobox).

### 3) Xuất sang Neo4j (tùy chọn)
Khởi chạy Neo4j Desktop/Aura, sau đó:
```bash
python korean_music_bfs.py --output korean_artists_graph_bfs.json \
  --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass <password> \
  --neo4j-db neo4j --neo4j-batch 2000 --neo4j-create-constraints
```
Driver sẽ MERGE node theo `id`, ghi quan hệ theo `type`, và (nếu bật) tạo UNIQUE CONSTRAINT cho từng label chính.

### 4) Phân tích và trực quan hóa
Mở notebook `network_analysis.ipynb`:
- Đọc `korean_artists_graph_bfs.json`
- Thống kê phân bố label, loại quan hệ
- Top node theo độ kết nối (nếu đã cài networkx)

Chạy nhanh cell cài đặt nếu cần:
```python
!pip install pandas matplotlib networkx plotly
```

### 5) Cấu trúc dữ liệu JSON
```json
{
  "nodes": { "<title>": {"label": "Artist|Group|Album|Song|Genre|Instrument|Occupation", "infobox": {...}, ... }, ... },
  "edges": [ {"source": "<title>", "target": "<title>", "type": "IS_GENRE|PLAYS|MANAGED_BY|...", "text": "..."}, ... ],
  "statistics": { ... },
  "metadata": { "generation_date": "YYYY-MM-DD HH:MM:SS", ... }
}
```

### 6) Gợi ý xử lý sự cố
- Không thấy log: chạy Python ở chế độ unbuffered `python -u ...`
- Chậm: tăng `--delay` hoặc giảm `--top-k`; khi export Neo4j nhớ bật `--neo4j-create-constraints`
- Kết quả còn lọt thực thể ngoài âm nhạc: bổ sung từ khóa blacklist hoặc mở rộng bộ chỉ dấu trong `korean_music_bfs.py`

### 7) File chính
- `korean_music_bfs.py`: crawler + lọc + export Neo4j
- `korean_artists_graph_bfs.json`: kết quả mạng
- `network_analysis.ipynb`: notebook phân tích
- `BAO_CAO_MANG_LUOI_NGHE_SI_HAN_QUOC.md`: báo cáo tóm tắt


