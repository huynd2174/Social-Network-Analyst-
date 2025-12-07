# 📁 Cấu trúc thư mục dự án

## Tổng quan

Dự án được tổ chức theo cấu trúc modular, dễ quản lý và mở rộng:

```
Social-network-analyst/
├── data/                    # 📊 Dữ liệu JSON
├── src/                     # 💻 Source code Python
├── notebooks/               # 📓 Jupyter Notebooks
├── docs/                    # 📚 Tài liệu
├── outputs/                 # 🖼️ Kết quả phân tích (hình ảnh, JSON)
├── requirements/            # 📦 Dependencies
├── README.md                # Hướng dẫn chính
└── STRUCTURE.md             # File này
```

## Chi tiết từng thư mục

### 📊 `data/` - Dữ liệu

Chứa tất cả các file JSON dữ liệu:

- `korean_artists_graph_bfs.json` - Graph từ BFS crawl Wikipedia
- `kpop_ner_result.json` - Entities được nhận dạng (rule-based)
- `kpop_ner_ml_result.json` - Entities được nhận dạng (ML-based)
- `kpop_relationships_result.json` - Relationships được trích xuất
- `merged_kpop_data.json` - Dữ liệu đã merge từ 3 file trên
- `enrichment_text_data.json` - Dữ liệu text để làm giàu
- `infobox_members.json` - Thông tin members từ infobox
- `network_analysis_results.json` - Kết quả phân tích mạng

### 💻 `src/` - Source Code

Chứa tất cả các script Python:

**Crawler & Data Collection:**
- `korean_music_bfs.py` - Crawler Wikipedia BFS
- `data_collection.py` - Thu thập dữ liệu từ Neo4j
- `crawl_infobox_members.py` - Crawl infobox members

**NER & Relationship Extraction:**
- `run_ner.py` - NER chính (rule-based + ML-based)
- `ml_ner.py` - ML-based NER module
- `run_relationship_extraction.py` - Trích xuất quan hệ

**Analysis & Import:**
- `network_analysis_algorithms.py` - Phân tích mạng (Small World, PageRank, Community)
- `merge_and_import_neo4j.py` - Merge dữ liệu và import vào Neo4j

**Shortest Path:**
- `shortest_path_neo4j.py` - Tìm đường đi ngắn nhất (Neo4j native)
- `shortest_path_gds.py` - Tìm đường đi ngắn nhất (GDS)
- `batch_shortest_path_runner.py` - Chạy batch shortest path

### 📓 `notebooks/` - Jupyter Notebooks

- `network_analysis.ipynb` - Notebook phân tích mạng xã hội
  - Small World Analysis
  - PageRank Ranking
  - Community Detection

### 📚 `docs/` - Tài liệu

- `README.md` - Hướng dẫn chính (gốc)
- `BAO_CAO_MANG_LUOI_NGHE_SI_HAN_QUOC.md` - Báo cáo
- `HUONG_DAN_HYBRID_NER.md` - Hướng dẫn Hybrid NER
- `HUONG_DAN_MERGE_IMPORT.md` - Hướng dẫn Merge & Import
- `HUONG_DAN_SHORTEST_PATH.md` - Hướng dẫn Shortest Path
- `THU_VIEN_DO_THI.md` - Thư viện đồ thị
- `batch_shortest_paths_results.md` - Kết quả shortest paths

### 🖼️ `outputs/` - Kết quả phân tích

Chứa các hình ảnh và kết quả từ phân tích:

- `community_analysis.png` - Biểu đồ phân tích cộng đồng
- `pagerank_analysis.png` - Biểu đồ PageRank
- `small_world_analysis.png` - Biểu đồ Small World

### 📦 `requirements/` - Dependencies

- `requirements_graph_libs.txt` - Thư viện cho graph analysis
- `requirements_ml_ner.txt` - Thư viện cho ML-based NER

## 🔧 Lưu ý khi sử dụng

### Chạy script từ thư mục gốc

Tất cả các script được thiết kế để chạy từ thư mục gốc của project:

```bash
# ✅ Đúng
python src/run_ner.py

# ❌ Sai (nếu script tham chiếu file trong data/)
cd src
python run_ner.py
```

### Đường dẫn trong code

Các script sử dụng đường dẫn tương đối từ thư mục gốc:

```python
# Ví dụ trong run_ner.py
with open('data/kpop_ner_result.json', 'r') as f:
    # ...
```

### Cập nhật đường dẫn nếu cần

Nếu script nào đó không hoạt động, kiểm tra đường dẫn file trong code và cập nhật cho phù hợp với cấu trúc mới.

## 📝 Quy tắc đặt tên

- **Scripts**: `snake_case.py`
- **Data files**: `snake_case.json`
- **Documentation**: `UPPER_SNAKE_CASE.md` hoặc `Title Case.md`
- **Notebooks**: `snake_case.ipynb`

## 🚀 Workflow đề xuất

1. **Thu thập dữ liệu**: `src/korean_music_bfs.py` → `data/korean_artists_graph_bfs.json`
2. **NER**: `src/run_ner.py` → `data/kpop_ner_result.json`
3. **Relationships**: `src/run_relationship_extraction.py` → `data/kpop_relationships_result.json`
4. **Merge & Import**: `src/merge_and_import_neo4j.py` → `data/merged_kpop_data.json` + Neo4j
5. **Analysis**: `notebooks/network_analysis.ipynb` → `outputs/*.png`







