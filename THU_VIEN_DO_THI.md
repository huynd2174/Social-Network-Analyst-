# Thư Viện Xử Lý Đồ Thị cho Dữ Liệu Mạng Nghệ Sĩ Hàn Quốc

## Tổng Quan

Dựa trên dữ liệu bài tập giữa kỳ (lưu trữ trong Neo4j), tài liệu này mô tả các thư viện Python có thể chạy các giải thuật đồ thị trên tập dữ liệu đã tạo ra.

## Cấu Trúc Dữ Liệu Hiện Tại

Dữ liệu được lưu trữ trong Neo4j với cấu trúc:
- **Nodes**: Các thực thể với labels như `Artist`, `Group`, `Album`, `Song`, `Genre`, `Instrument`, `Occupation`, `Company`
- **Relationships**: Các quan hệ có hướng với types như `MEMBER_OF`, `IS_GENRE`, `MANAGED_BY`, `RELEASED`, `SINGS`, `CONTAINS`, `PLAYS`, `HAS_OCCUPATION`
- **Properties**: Mỗi node có các thuộc tính như `id`, `name`, `url`, và các trường từ infobox

---

## 1. NetworkX - Thư Viện Phổ Biến Nhất

### Mô Tả
**NetworkX** là thư viện Python phổ biến nhất để tạo, thao tác và phân tích cấu trúc đồ thị. Được thiết kế cho các mạng lưới phức tạp với hàng nghìn đến hàng triệu node.

### Tính Năng Chính
- ✅ Hỗ trợ đồ thị có hướng và vô hướng
- ✅ Hơn 100 giải thuật đồ thị tích hợp sẵn
- ✅ Tích hợp tốt với NumPy, SciPy, Matplotlib
- ✅ Dễ sử dụng, tài liệu đầy đủ
- ✅ Hỗ trợ trực quan hóa
- ✅ Các giải thuật phổ biến: BFS, DFS, shortest path, centrality measures, community detection
- ⚠️ Chậm hơn với đồ thị rất lớn (>100K nodes)

### Các Giải Thuật Có Sẵn
- **Tìm đường đi**: `shortest_path()`, `all_shortest_paths()`, `shortest_path_length()`
- **Duyệt đồ thị**: `bfs_tree()`, `dfs_tree()`, `bfs_edges()`, `dfs_edges()`
- **Độ trung tâm**: `degree_centrality()`, `betweenness_centrality()`, `closeness_centrality()`, `pagerank()`
- **Thành phần liên thông**: `connected_components()`, `weakly_connected_components()`, `strongly_connected_components()`
- **Phát hiện cộng đồng**: `louvain_communities()`, `greedy_modularity_communities()`, `label_propagation_communities()`
- **Thống kê mạng**: `density()`, `average_clustering()`, `average_shortest_path_length()`

### Cài Đặt
```bash
pip install networkx matplotlib pandas numpy
```

### Ưu Điểm
- Dễ học và sử dụng
- Tài liệu phong phú, ví dụ nhiều
- Cộng đồng lớn, hỗ trợ tốt
- Tích hợp tốt với hệ sinh thái Python

### Nhược Điểm
- Chậm với đồ thị rất lớn (>100K nodes)
- Tốn bộ nhớ

---

## 2. igraph - Thư Viện Hiệu Năng Cao

### Mô Tả
**igraph** là thư viện đồ thị nhanh và hiệu quả, được viết bằng C với Python binding. Phù hợp cho đồ thị lớn và phân tích phức tạp.

### Tính Năng Chính
- ✅ Rất nhanh (C/C++ backend)
- ✅ Hỗ trợ đồ thị lớn (hàng triệu nodes)
- ✅ Nhiều giải thuật tối ưu
- ✅ Hỗ trợ trực quan hóa đẹp
- ✅ Có thể export sang nhiều định dạng
- ✅ Các giải thuật: shortest paths, centrality, community detection, layout algorithms
- ⚠️ API hơi khác NetworkX

### Các Giải Thuật Có Sẵn
- **Tìm đường đi**: `shortest_paths()`, `get_shortest_paths()`, `distances()`
- **Centrality**: `degree()`, `betweenness()`, `closeness()`, `pagerank()`, `eigenvector_centrality()`
- **Community**: `community_multilevel()` (Louvain), `community_infomap()`, `community_walktrap()`
- **Layout**: `layout_fruchterman_reingold()`, `layout_kamada_kawai()`, `layout_spring()`

### Cài Đặt
```bash
pip install python-igraph
```

### Ưu Điểm
- Rất nhanh
- Xử lý được đồ thị rất lớn
- Nhiều giải thuật tối ưu

### Nhược Điểm
- API khác NetworkX (cần học lại)
- Tài liệu ít hơn NetworkX

---

## 3. graph-tool - Thư Viện Cực Kỳ Nhanh

### Mô Tả
**graph-tool** là thư viện Python mạnh mẽ và nhanh nhất, sử dụng C++ và OpenMP để tăng tốc song song. Phù hợp cho đồ thị rất lớn.

### Tính Năng Chính
- ✅ Cực kỳ nhanh (C++ với OpenMP)
- ✅ Xử lý đồ thị hàng triệu nodes
- ✅ Hỗ trợ song song hóa
- ✅ Trực quan hóa tương tác
- ✅ Nhiều giải thuật tối ưu
- ⚠️ Cài đặt phức tạp hơn (cần C++ compiler)

### Các Giải Thuật Có Sẵn
- **Shortest paths**: `shortest_distance()`, `shortest_path()`
- **Centrality**: `pagerank()`, `betweenness()`, `closeness()`, `eigenvector()`
- **Community**: `minimize_blockmodel_dl()`, `minimize_nested_blockmodel_dl()`
- **Layout**: `sfdp_layout()`, `fruchterman_reingold_layout()`

### Cài Đặt
```bash
# Windows: Sử dụng conda (khuyến nghị)
conda install -c conda-forge graph-tool

# Hoặc build từ source (phức tạp)
```

### Ưu Điểm
- Nhanh nhất trong các thư viện Python
- Xử lý được đồ thị cực lớn

### Nhược Điểm
- Cài đặt phức tạp
- Tài liệu ít hơn

---

## 4. PyTorch Geometric (PyG) - Deep Learning trên Đồ Thị

### Mô Tả
**PyTorch Geometric** là thư viện cho deep learning trên đồ thị, phù hợp cho các bài toán như node classification, link prediction, graph classification.

### Tính Năng Chính
- ✅ Deep learning trên đồ thị
- ✅ Graph Neural Networks (GNN)
- ✅ Node/Link/Graph embedding
- ✅ Tích hợp PyTorch
- ✅ Nhiều kiến trúc GNN: GCN, GAT, GraphSAGE, Transformer
- ⚠️ Cần kiến thức về deep learning

### Các Kiến Trúc Hỗ Trợ
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GraphSAGE** (Graph Sample and Aggregate)
- **Transformer** cho đồ thị
- **Link Prediction** models

### Cài Đặt
```bash
pip install torch torch-geometric
```

### Ưu Điểm
- Mạnh mẽ cho deep learning
- Hỗ trợ nhiều kiến trúc GNN

### Nhược Điểm
- Cần kiến thức về deep learning
- Phức tạp hơn cho phân tích đồ thị cơ bản

---

## 5. Neo4j với Python Driver - Cơ Sở Dữ Liệu Đồ Thị

### Mô Tả
**Neo4j** là cơ sở dữ liệu đồ thị NoSQL, cho phép lưu trữ và truy vấn đồ thị lớn. Dữ liệu của bạn đang được lưu trữ trong Neo4j.

### Tính Năng Chính
- ✅ Lưu trữ đồ thị lớn
- ✅ Ngôn ngữ truy vấn Cypher mạnh mẽ
- ✅ Tích hợp với Python
- ✅ Trực quan hóa với Neo4j Browser/Bloom
- ✅ Hỗ trợ các giải thuật đồ thị (Graph Data Science Library)
- ✅ Shortest path queries với Cypher
- ✅ PageRank, Centrality, Community Detection tích hợp

### Các Giải Thuật Có Sẵn
- **Shortest Path**: Sử dụng Cypher query `shortestPath()` hoặc `allShortestPaths()`
- **Graph Data Science Library**: PageRank, Betweenness Centrality, Louvain Community Detection
- **Path Finding**: Dijkstra, A*, Yen's K-shortest paths

### Cài Đặt
```bash
pip install neo4j
# Cần cài Neo4j Desktop hoặc Neo4j Server riêng
```

### Ưu Điểm
- Lưu trữ và truy vấn đồ thị lớn
- Cypher dễ đọc và mạnh mẽ
- Trực quan hóa tốt với Neo4j Browser
- Hỗ trợ Graph Data Science Library cho các giải thuật phức tạp

### Nhược Điểm
- Cần cài đặt Neo4j server
- Tốn tài nguyên

---

## 6. Graph Data Science (GDS) Python Client - Thư Viện Chuyên Nghiệp của Neo4j

### Mô Tả
**Graph Data Science (GDS) Python Client** là thư viện chính thức của Neo4j, cung cấp API Python để sử dụng Graph Data Science Library. Đây là công cụ mạnh mẽ cho phép thực hiện các thuật toán đồ thị và phân tích dữ liệu trực tiếp trên Neo4j bằng Python mà không cần viết Cypher queries phức tạp.

### Tính Năng Chính
- ✅ **Hơn 65 thuật toán đồ thị** được tối ưu hóa
- ✅ **API thân thiện với Python** - không cần học Cypher
- ✅ **Dựng và quản lý đồ thị** - tạo graph projections từ dữ liệu Neo4j
- ✅ **Tích hợp học máy** - xây dựng và triển khai ML pipelines
- ✅ **Hiệu năng cao** - các thuật toán được tối ưu cho Neo4j
- ✅ **Tích hợp với Pandas** - kết quả trả về dạng DataFrame
- ✅ **Hỗ trợ đồ thị lớn** - xử lý hàng triệu nodes

### Các Giải Thuật Có Sẵn

#### Centrality Algorithms
- **PageRank**: `gds.pageRank.stream()`, `gds.pageRank.write()`
- **ArticleRank**: `gds.articleRank.stream()`
- **Betweenness Centrality**: `gds.betweenness.stream()`
- **Closeness Centrality**: `gds.closeness.stream()`
- **Harmonic Centrality**: `gds.harmonic.stream()`
- **Eigenvector Centrality**: `gds.eigenvector.stream()`

#### Community Detection
- **Louvain**: `gds.louvain.stream()`, `gds.louvain.write()`
- **Leiden**: `gds.leiden.stream()`
- **Label Propagation**: `gds.labelPropagation.stream()`
- **Triangle Count**: `gds.triangleCount.stream()`
- **Weakly Connected Components**: `gds.wcc.stream()`

#### Path Finding
- **Shortest Path (Dijkstra)**: `gds.shortestPath.dijkstra.stream()`
- **Shortest Path (Yen's K-shortest)**: `gds.shortestPath.yens.stream()`
- **All Pairs Shortest Path**: `gds.allShortestPaths.stream()`
- **Single Source Shortest Path**: `gds.shortestPath.dijkstra.stream()`

#### Similarity
- **Node Similarity**: `gds.nodeSimilarity.stream()`
- **K-Nearest Neighbors**: `gds.knn.stream()`

#### Embedding
- **FastRP**: `gds.fastRP.stream()`
- **GraphSAGE**: `gds.graphSage.train()`, `gds.graphSage.stream()`
- **Node2Vec**: `gds.node2vec.stream()`

#### Link Prediction
- **Link Prediction Pipeline**: `gds.beta.pipeline.linkPrediction.*`

### Cài Đặt
```bash
pip install graphdatascience
```

**Yêu cầu**: Neo4j Server với Graph Data Science Library plugin đã được cài đặt.

### Ví Dụ Sử Dụng Cơ Bản

```python
from graphdatascience import GraphDataScience

# Kết nối đến Neo4j
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# Tạo graph projection từ dữ liệu trong Neo4j
G, result = gds.graph.project(
    "myGraph",           # Tên graph projection
    "*",                 # Node labels (tất cả)
    "*"                  # Relationship types (tất cả)
)

print(f"Graph: {G.node_count()} nodes, {G.relationship_count()} edges")

# Chạy PageRank
pagerank_result = gds.pageRank.stream(G)
for row in pagerank_result:
    print(f"Node {row['nodeId']}: PageRank = {row['score']}")

# Chạy Louvain Community Detection
community_result = gds.louvain.stream(G)
for row in community_result:
    print(f"Node {row['nodeId']}: Community = {row['communityId']}")

# Shortest Path
path_result = gds.shortestPath.dijkstra.stream(
    G,
    sourceNode=source_node_id,
    targetNode=target_node_id
)
```

### Ưu Điểm
- **API Python thuần túy** - không cần viết Cypher
- **Hiệu năng cao** - các thuật toán được tối ưu cho Neo4j
- **Nhiều thuật toán** - hơn 65 thuật toán có sẵn
- **Tích hợp ML** - hỗ trợ xây dựng ML pipelines
- **Dễ sử dụng** - API mô phỏng các thủ tục Cypher của GDS
- **Kết quả dạng DataFrame** - dễ tích hợp với Pandas

### Nhược Điểm
- **Cần Neo4j với GDS plugin** - không thể dùng độc lập
- **Cần tạo graph projection** - tốn thời gian và bộ nhớ
- **Tài liệu** - ít hơn so với NetworkX
- **Phụ thuộc Neo4j** - không thể dùng với dữ liệu ngoài Neo4j

### So Sánh với Neo4j Driver Thông Thường

| Tính Năng | Neo4j Driver | GDS Python Client |
|-----------|--------------|-------------------|
| Viết queries | Cần viết Cypher | API Python thuần |
| Shortest Path | Cypher `shortestPath()` | `gds.shortestPath.dijkstra.stream()` |
| PageRank | Cypher `CALL gds.pageRank` | `gds.pageRank.stream()` |
| Dễ sử dụng | ⭐⭐ | ⭐⭐⭐⭐ |
| Hiệu năng | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Số thuật toán | Phụ thuộc GDS plugin | 65+ thuật toán |

### Khi Nào Sử Dụng GDS Python Client

✅ **Nên dùng khi**:
- Dữ liệu đã có trong Neo4j
- Cần nhiều thuật toán đồ thị phức tạp
- Muốn tránh viết Cypher queries
- Cần hiệu năng cao với đồ thị lớn
- Cần tích hợp với ML pipelines

❌ **Không nên dùng khi**:
- Dữ liệu không trong Neo4j
- Chỉ cần các thao tác đơn giản
- Không có Neo4j với GDS plugin
- Cần độc lập với Neo4j

---

## So Sánh và Khuyến Nghị

| Thư Viện | Tốc Độ | Độ Khó | Kích Thước Đồ Thị | Phù Hợp Cho |
|----------|--------|--------|-------------------|-------------|
| **NetworkX** | ⭐⭐⭐ | ⭐ | ~100K nodes | Phân tích cơ bản, học tập |
| **igraph** | ⭐⭐⭐⭐ | ⭐⭐ | ~1M nodes | Phân tích chuyên sâu, hiệu năng cao |
| **graph-tool** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~10M nodes | Đồ thị rất lớn, nghiên cứu |
| **PyG** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~1M nodes | Deep learning, embedding |
| **Neo4j Driver** | ⭐⭐⭐⭐ | ⭐⭐ | ~100M nodes | Lưu trữ, truy vấn, production |
| **GDS Python Client** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ~100M nodes | Phân tích chuyên nghiệp với Neo4j |

### Khuyến Nghị

1. **Bắt đầu với NetworkX**: Dễ học, đủ cho hầu hết bài toán, phù hợp với đồ thị ~2000 nodes
2. **Nếu cần tốc độ**: Chuyển sang igraph
3. **Nếu đồ thị rất lớn**: Dùng graph-tool hoặc Neo4j
4. **Nếu cần deep learning**: Dùng PyG
5. **Với dữ liệu trong Neo4j**: 
   - **GDS Python Client** (KHUYẾN NGHỊ): API Python thuần, nhiều thuật toán, hiệu năng cao
   - **Neo4j Driver + Cypher**: Linh hoạt, kiểm soát tốt
   - **Load sang NetworkX**: Khi cần các thuật toán đặc biệt của NetworkX

---

## Kết Luận

Với dữ liệu bài tập giữa kỳ của bạn (lưu trong Neo4j), có thể:

1. **GDS Python Client** (KHUYẾN NGHỊ cho dữ liệu trong Neo4j):
   - API Python thuần, dễ sử dụng
   - Hơn 65 thuật toán được tối ưu
   - Hiệu năng cao, phù hợp với đồ thị lớn
   - Tích hợp ML pipelines

2. **Neo4j Driver + Cypher**: 
   - Linh hoạt, kiểm soát tốt
   - Phù hợp cho các truy vấn đơn giản
   - Cần biết Cypher

3. **Load sang NetworkX**: 
   - Khi cần các thuật toán đặc biệt của NetworkX
   - Phù hợp cho phân tích tương tác
   - Dễ trực quan hóa

**Khuyến nghị cho dữ liệu trong Neo4j**:
- **GDS Python Client** là lựa chọn tốt nhất nếu có Neo4j với GDS plugin
- **NetworkX** là lựa chọn tốt nếu muốn độc lập hoặc cần các thuật toán đặc biệt
- **Kết hợp cả hai**: Dùng GDS cho các thuật toán phức tạp, NetworkX cho phân tích và trực quan hóa
