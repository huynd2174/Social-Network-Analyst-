# 🎵 Social Network Analyst - K-pop Network Analysis

Dự án phân tích mạng lưới nghệ sĩ/nhóm nhạc K-pop từ Wikipedia tiếng Việt, bao gồm:
- Thu thập dữ liệu từ Wikipedia
- Nhận dạng thực thể (NER) - Rule-based và ML-based
- Trích xuất quan hệ giữa các thực thể
- Phân tích mạng xã hội (Small World, PageRank, Community Detection)
- Lưu trữ vào Neo4j

## 📁 Cấu trúc thư mục

```
Social-network-analyst/
├── data/                    # Dữ liệu JSON
│   ├── korean_artists_graph_bfs.json      # Graph từ BFS crawl
│   ├── kpop_ner_result.json               # Entities (rule-based)
│   ├── kpop_ner_ml_result.json            # Entities (ML-based)
│   ├── kpop_relationships_result.json     # Relationships
│   ├── merged_kpop_data.json              # Dữ liệu đã merge
│   └── ...
│
├── src/                     # Source code Python
│   ├── korean_music_bfs.py               # Crawler Wikipedia BFS
│   ├── run_ner.py                         # NER chính
│   ├── run_relationship_extraction.py   # Trích xuất quan hệ
│   ├── ml_ner.py                          # ML-based NER
│   ├── merge_and_import_neo4j.py         # Merge & import Neo4j
│   ├── network_analysis_algorithms.py     # Phân tích mạng
│   └── ...
│
├── notebooks/               # Jupyter Notebooks
│   └── network_analysis.ipynb             # Phân tích mạng (Small World, PageRank, Community)
│
├── docs/                   # Tài liệu hướng dẫn
│   ├── README.md                          # Hướng dẫn chính (gốc)
│   ├── BAO_CAO_MANG_LUOI_NGHE_SI_HAN_QUOC.md
│   ├── HUONG_DAN_HYBRID_NER.md
│   ├── HUONG_DAN_MERGE_IMPORT.md
│   └── ...
│
├── outputs/                # Kết quả phân tích
│   ├── community_analysis.png
│   ├── pagerank_analysis.png
│   └── small_world_analysis.png
│
└── requirements/           # Dependencies
    ├── requirements_graph_libs.txt
    └── requirements_ml_ner.txt
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
# Cài đặt dependencies cơ bản
pip install requests beautifulsoup4 pandas matplotlib networkx neo4j

# Hoặc cài từ file requirements
pip install -r requirements/requirements_graph_libs.txt
pip install -r requirements/requirements_ml_ner.txt  # Nếu dùng ML-based NER
```

### 2. Thu thập dữ liệu

```bash
# Crawl Wikipedia và tạo graph
python src/korean_music_bfs.py \
  --max-nodes 3000 --top-k 40 --delay 0.2 \
  --output data/korean_artists_graph_bfs.json
```

### 3. Nhận dạng thực thể (NER)

```bash
# Rule-based NER
python src/run_ner.py

# Kết quả: data/kpop_ner_result.json (rule-based)
#          data/kpop_ner_ml_result.json (ML-based nếu có)
```

### 4. Trích xuất quan hệ

```bash
python src/run_relationship_extraction.py

# Kết quả: data/kpop_relationships_result.json
```

### 5. Merge và import vào Neo4j

```bash
python src/merge_and_import_neo4j.py \
  --neo4j-password YOUR_PASSWORD \
  --bfs-file data/korean_artists_graph_bfs.json \
  --ner-file data/kpop_ner_result.json \
  --relationships-file data/kpop_relationships_result.json \
  --output-file data/merged_kpop_data.json
```

### 6. Phân tích mạng xã hội

**Cách 1: Chạy script Python**
```bash
python src/network_analysis_algorithms.py
```

**Cách 2: Chạy Jupyter Notebook (Khuyến nghị)**
```bash
jupyter notebook notebooks/network_analysis.ipynb
```

Notebook bao gồm 3 phân tích chính:
- ✅ **Small World**: Chứng minh khái niệm thế giới nhỏ (APL, Clustering Coefficient)
- ✅ **PageRank**: Xếp hạng các node quan trọng nhất
- ✅ **Community Detection**: Phát hiện cộng đồng trong mạng

## 📊 Kết quả

- **Dữ liệu**: Lưu trong `data/`
- **Hình ảnh**: Lưu trong `outputs/`
- **Báo cáo**: Xem trong `docs/`

## 📚 Tài liệu chi tiết

- [Hướng dẫn Hybrid NER](docs/HUONG_DAN_HYBRID_NER.md)
- [Hướng dẫn Merge & Import Neo4j](docs/HUONG_DAN_MERGE_IMPORT.md)
- [Hướng dẫn Shortest Path](docs/HUONG_DAN_SHORTEST_PATH.md)
- [Thư viện đồ thị](docs/THU_VIEN_DO_THI.md)

## 🔧 Yêu cầu hệ thống

- Python 3.9+
- Neo4j (tùy chọn, để lưu trữ graph)
- Jupyter Notebook (để chạy notebook phân tích)

## 📝 Ghi chú

- Tất cả các script đã được cập nhật để sử dụng đường dẫn tương đối từ thư mục gốc
- File dữ liệu được lưu trong `data/`
- Kết quả phân tích được lưu trong `outputs/`

