# Hướng Dẫn Merge và Import Dữ Liệu vào Neo4j

Script này giúp bạn merge dữ liệu từ 3 file JSON và import vào Neo4j.

## Các File Đầu Vào

1. **korean_artists_graph_bfs.json**: Chứa nodes và edges từ BFS crawl Wikipedia
2. **kpop_ner_result.json**: Chứa entities được nhận dạng từ NER (rule-based)
3. **kpop_relationships_result.json**: Chứa relationships được trích xuất từ relationship extraction

## Cách Sử Dụng

### 1. Chỉ Merge (không import vào Neo4j)

```bash
python merge_and_import_neo4j.py --neo4j-password YOUR_PASSWORD --merge-only
```

Kết quả sẽ được lưu vào `merged_kpop_data.json`.

### 2. Merge và Import vào Neo4j

```bash
python merge_and_import_neo4j.py --neo4j-password YOUR_PASSWORD
```

### 3. Với các tùy chọn khác

```bash
python merge_and_import_neo4j.py \
  --bfs-file korean_artists_graph_bfs.json \
  --ner-file kpop_ner_result.json \
  --relationships-file kpop_relationships_result.json \
  --output-file merged_kpop_data.json \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password YOUR_PASSWORD \
  --neo4j-database neo4j \
  --batch-size 1000
```

## Các Tham Số

- `--bfs-file`: File BFS graph JSON (mặc định: `korean_artists_graph_bfs.json`)
- `--ner-file`: File NER result JSON (mặc định: `kpop_ner_result.json`)
- `--relationships-file`: File relationships result JSON (mặc định: `kpop_relationships_result.json`)
- `--output-file`: File output merged (mặc định: `merged_kpop_data.json`)
- `--neo4j-uri`: Neo4j URI (mặc định: `bolt://localhost:7687`)
- `--neo4j-user`: Neo4j username (mặc định: `neo4j`)
- `--neo4j-password`: Neo4j password (**bắt buộc**)
- `--neo4j-database`: Neo4j database name (mặc định: None = default database)
- `--batch-size`: Kích thước batch khi import (mặc định: 1000)
- `--no-constraints`: Không tạo constraints
- `--merge-only`: Chỉ merge, không import vào Neo4j

## Quy Trình Merge

1. **Thêm nodes từ BFS graph**: Tất cả nodes từ `korean_artists_graph_bfs.json`
2. **Thêm edges từ BFS graph**: Tất cả edges từ `korean_artists_graph_bfs.json`
3. **Thêm entities từ NER**: 
   - Tạo nodes mới nếu chưa tồn tại
   - Cập nhật properties nếu node đã tồn tại
4. **Thêm relationships**: 
   - Thêm relationships mới từ `kpop_relationships_result.json`
   - Bỏ qua relationships trùng lặp

## Cấu Trúc File Output

File `merged_kpop_data.json` có cấu trúc:

```json
{
  "metadata": {
    "merged_at": "2025-12-05T...",
    "source_files": [...],
    "total_nodes": 1234,
    "total_edges": 5678,
    "nodes_by_type": {...},
    "edges_by_type": {...}
  },
  "nodes": {
    "node_id": {
      "label": "Group",
      "title": "...",
      "infobox": {...},
      "properties": {...}
    }
  },
  "edges": [
    {
      "source": "...",
      "target": "...",
      "type": "MEMBER_OF",
      "text": "...",
      "properties": {...}
    }
  ]
}
```

## Import vào Neo4j

Script sẽ:
1. Tạo constraints cho các labels chính (Artist, Group, Album, Song, Company, Genre)
2. Import nodes theo từng label trong batch
3. Import relationships theo từng type trong batch

## Lưu Ý

- Script sẽ MERGE nodes dựa trên `id`, nên không lo trùng lặp
- Relationships cũng được MERGE, nên không lo trùng lặp
- Nếu node/relationship đã tồn tại, properties sẽ được cập nhật
- Batch size mặc định là 1000, có thể điều chỉnh tùy theo hiệu năng

