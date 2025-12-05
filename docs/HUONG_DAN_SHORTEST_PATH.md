# Hướng Dẫn Sử Dụng Thuật Toán Tìm Đường Đi Ngắn Nhất

## Tổng Quan

File `shortest_path_neo4j.py` cung cấp thuật toán tìm đường đi ngắn nhất giữa 2 node trong đồ thị lưu trữ trong Neo4j.

## Yêu Cầu

```bash
pip install neo4j networkx
```

## Cách Sử Dụng

### 1. Chạy từ Command Line

```bash
python shortest_path_neo4j.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password your_password \
  --source "BTS" \
  --target "Blackpink" \
  --method both
```

### 2. Các Tham Số

- `--uri`: Neo4j URI (mặc định: `bolt://localhost:7687`)
- `--user`: Username (mặc định: `neo4j`)
- `--password`: Password (bắt buộc)
- `--database`: Tên database (mặc định: `neo4j`)
- `--source`: Tên node nguồn (bắt buộc)
- `--target`: Tên node đích (bắt buộc)
- `--method`: Phương pháp (`cypher`, `networkx`, hoặc `both`) (mặc định: `both`)
- `--max-depth`: Độ sâu tối đa để tìm (mặc định: 10)

### 3. Ví Dụ

#### Tìm đường đi giữa 2 nghệ sĩ
```bash
python shortest_path_neo4j.py \
  --password your_password \
  --source "BTS" \
  --target "Blackpink"
```

#### Chỉ sử dụng Cypher
```bash
python shortest_path_neo4j.py \
  --password your_password \
  --source "BTS" \
  --target "Blackpink" \
  --method cypher
```

#### Chỉ sử dụng NetworkX
```bash
python shortest_path_neo4j.py \
  --password your_password \
  --source "BTS" \
  --target "Blackpink" \
  --method networkx
```

## Sử Dụng Trong Code Python

```python
from shortest_path_neo4j import ShortestPathFinder

# Khởi tạo
finder = ShortestPathFinder(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)

# Tìm đường đi bằng Cypher
path = finder.shortest_path_cypher("BTS", "Blackpink")
finder.print_path(path, "Cypher")

# Tìm đường đi bằng NetworkX
path = finder.shortest_path_networkx("BTS", "Blackpink")
finder.print_path(path, "NetworkX")

# Đóng kết nối
finder.close()
```

## Hai Phương Pháp

### 1. Neo4j Cypher Query
- Sử dụng query `shortestPath()` hoặc `allShortestPaths()` của Neo4j
- Nhanh, không cần load toàn bộ đồ thị vào memory
- Phù hợp với đồ thị lớn

### 2. NetworkX
- Load đồ thị từ Neo4j sang NetworkX
- Sử dụng `nx.shortest_path()` của NetworkX
- Có thể tái sử dụng graph object cho nhiều truy vấn
- Phù hợp khi cần nhiều thao tác trên cùng một graph

## Kết Quả

Kết quả bao gồm:
- Độ dài đường đi (số bước)
- Danh sách các node trong đường đi
- Danh sách các relationship giữa các node
- Thông tin chi tiết về từng node (name, labels)

## Lưu Ý

- Đảm bảo Neo4j đang chạy và có dữ liệu
- Node names phải khớp với property `name` hoặc `id` trong Neo4j
- Nếu không tìm thấy đường đi, sẽ trả về `None`

