"""
Phân tích mạng xã hội K-pop với các thuật toán:
1. Chứng minh khái niệm thế giới nhỏ (Small World)
2. Xếp hạng node bằng PageRank
3. Phát hiện cộng đồng (Community Detection)

Dữ liệu từ file merged hoặc các file gốc
"""
import sys
import io
import json
import math
import random
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️  NetworkX chưa được cài đặt. Chạy: pip install networkx")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib chưa được cài đặt. Chạy: pip install matplotlib")


def load_graph_data(
    bfs_file: str = "korean_artists_graph_bfs.json",
    ner_file: str = "kpop_ner_result.json",
    relationships_file: str = "kpop_relationships_result.json"
) -> Tuple[Dict, List]:
    """Load và merge dữ liệu từ các file"""
    print("=" * 70)
    print("LOAD DỮ LIỆU GRAPH")
    print("=" * 70)
    
    nodes = {}
    edges = []
    
    # Load BFS data
    try:
        with open(bfs_file, 'r', encoding='utf-8') as f:
            bfs_data = json.load(f)
        bfs_nodes = bfs_data.get("nodes", {})
        bfs_edges = bfs_data.get("edges", [])
        nodes.update(bfs_nodes)
        edges.extend(bfs_edges)
        print(f"✓ Đã load {len(bfs_nodes)} nodes và {len(bfs_edges)} edges từ {bfs_file}")
    except FileNotFoundError:
        print(f"⚠️  Không tìm thấy {bfs_file}")
    
    # Load NER entities (tạo nodes mới nếu cần)
    try:
        with open(ner_file, 'r', encoding='utf-8') as f:
            ner_data = json.load(f)
        ner_entities = ner_data.get("entities", [])
        new_nodes = 0
        for entity in ner_entities:
            entity_id = entity.get("text", "")
            if entity_id and entity_id not in nodes:
                nodes[entity_id] = {
                    "label": entity.get("type", "Entity"),
                    "title": entity_id
                }
                new_nodes += 1
        print(f"✓ Đã load {len(ner_entities)} entities, thêm {new_nodes} nodes mới từ {ner_file}")
    except FileNotFoundError:
        print(f"⚠️  Không tìm thấy {ner_file}")
    
    # Load relationships
    try:
        with open(relationships_file, 'r', encoding='utf-8') as f:
            rel_data = json.load(f)
        relationships = rel_data.get("relationships", [])
        
        # Check duplicate
        existing_edges = set()
        for e in edges:
            existing_edges.add((e.get("source"), e.get("target"), e.get("type")))
        
        new_edges = 0
        for rel in relationships:
            key = (rel.get("source"), rel.get("target"), rel.get("type"))
            if key not in existing_edges:
                edges.append({
                    "source": rel.get("source"),
                    "target": rel.get("target"),
                    "type": rel.get("type", "RELATED_TO")
                })
                existing_edges.add(key)
                new_edges += 1
        print(f"✓ Đã load {len(relationships)} relationships, thêm {new_edges} edges mới từ {relationships_file}")
    except FileNotFoundError:
        print(f"⚠️  Không tìm thấy {relationships_file}")
    
    print(f"\n📊 Tổng cộng: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


def build_networkx_graph(nodes: Dict, edges: List, undirected: bool = True) -> 'nx.Graph':
    """Xây dựng NetworkX graph từ nodes và edges"""
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX không khả dụng")
    
    if undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    
    # Thêm nodes
    for node_id, node_data in nodes.items():
        G.add_node(node_id, **{
            'label': node_data.get('label', 'Entity'),
            'title': node_data.get('title', node_id)
        })
    
    # Thêm edges
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src and tgt and src in nodes and tgt in nodes:
            G.add_edge(src, tgt, type=edge.get('type', 'RELATED_TO'))
    
    return G


# =====================================================
# 1. SMALL WORLD - KHÁI NIỆM THẾ GIỚI NHỎ
# =====================================================
def analyze_small_world(G: 'nx.Graph') -> Dict[str, Any]:
    """
    Phân tích khái niệm Small World:
    - Tính Average Shortest Path Length (APL)
    - Tính Clustering Coefficient
    - So sánh với random graph cùng kích thước
    
    Small World có đặc điểm:
    - APL thấp (như random graph)
    - Clustering Coefficient cao (hơn random graph nhiều)
    """
    print("\n" + "=" * 70)
    print("1. PHÂN TÍCH KHÁI NIỆM THẾ GIỚI NHỎ (SMALL WORLD)")
    print("=" * 70)
    
    results = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
    }
    
    # Kiểm tra connected components
    if nx.is_connected(G):
        components = [G]
        largest_cc = G
        print(f"\n📊 Graph là connected với {G.number_of_nodes()} nodes")
    else:
        components = list(nx.connected_components(G))
        largest_cc = G.subgraph(max(components, key=len)).copy()
        print(f"\n📊 Graph có {len(components)} connected components")
        print(f"   Largest component: {largest_cc.number_of_nodes()} nodes ({100*largest_cc.number_of_nodes()/G.number_of_nodes():.1f}%)")
    
    results["num_components"] = len(components) if not nx.is_connected(G) else 1
    results["largest_component_size"] = largest_cc.number_of_nodes()
    results["largest_component_percentage"] = 100 * largest_cc.number_of_nodes() / G.number_of_nodes()
    
    # Tính Average Shortest Path Length trên largest component
    print("\n🔍 Tính Average Shortest Path Length (APL)...")
    
    n = largest_cc.number_of_nodes()
    m = largest_cc.number_of_edges()
    
    if n > 5000:
        # Sampling cho graph lớn
        print(f"   Graph lớn ({n} nodes), sử dụng sampling...")
        sample_size = min(1000, n)
        sample_nodes = random.sample(list(largest_cc.nodes()), sample_size)
        
        total_paths = 0
        path_count = 0
        
        for i, source in enumerate(sample_nodes):
            if i % 100 == 0:
                print(f"   Đang xử lý node {i}/{sample_size}...")
            lengths = nx.single_source_shortest_path_length(largest_cc, source)
            for target, length in lengths.items():
                if source != target:
                    total_paths += length
                    path_count += 1
        
        if path_count > 0:
            apl = total_paths / path_count
        else:
            apl = float('inf')
        print(f"   (Ước lượng từ {sample_size} nodes)")
    else:
        apl = nx.average_shortest_path_length(largest_cc)
    
    results["average_path_length"] = apl
    print(f"   ✓ Average Shortest Path Length: {apl:.4f}")
    
    # Tính Clustering Coefficient
    print("\n🔍 Tính Clustering Coefficient...")
    avg_clustering = nx.average_clustering(G)
    results["clustering_coefficient"] = avg_clustering
    print(f"   ✓ Average Clustering Coefficient: {avg_clustering:.4f}")
    
    # Tính diameter (đường kính)
    print("\n🔍 Tính Diameter...")
    if n <= 5000:
        diameter = nx.diameter(largest_cc)
    else:
        # Ước lượng diameter bằng sampling
        sample_nodes = random.sample(list(largest_cc.nodes()), min(100, n))
        max_dist = 0
        for node in sample_nodes:
            eccentricity = nx.eccentricity(largest_cc, v=node)
            max_dist = max(max_dist, eccentricity)
        diameter = max_dist
        print(f"   (Ước lượng từ sampling)")
    results["diameter"] = diameter
    print(f"   ✓ Diameter: {diameter}")
    
    # So sánh với Random Graph (Erdős–Rényi)
    print("\n🔍 So sánh với Random Graph (Erdős–Rényi)...")
    p = 2 * m / (n * (n - 1)) if n > 1 else 0  # Probability để có cùng số edges
    
    # Lý thuyết cho random graph:
    # APL_random ≈ ln(n) / ln(k) với k = average degree
    # Clustering_random ≈ p = k/n
    avg_degree = 2 * m / n if n > 0 else 0
    
    if avg_degree > 1:
        expected_apl_random = math.log(n) / math.log(avg_degree) if avg_degree > 1 else float('inf')
    else:
        expected_apl_random = float('inf')
    expected_clustering_random = avg_degree / n if n > 0 else 0
    
    results["random_graph_expected_apl"] = expected_apl_random
    results["random_graph_expected_clustering"] = expected_clustering_random
    results["average_degree"] = avg_degree
    
    print(f"   Average Degree: {avg_degree:.2f}")
    print(f"   Random Graph Expected APL: {expected_apl_random:.4f}")
    print(f"   Random Graph Expected Clustering: {expected_clustering_random:.6f}")
    
    # Small World Index
    # σ = (C/C_random) / (L/L_random)
    # σ > 1 indicates small world property
    if expected_clustering_random > 0 and expected_apl_random > 0 and expected_apl_random != float('inf'):
        sigma = (avg_clustering / expected_clustering_random) / (apl / expected_apl_random)
        results["small_world_sigma"] = sigma
        print(f"\n📊 Small World Sigma (σ): {sigma:.4f}")
        
        if sigma > 1:
            print("   ✓ σ > 1: Mạng có tính chất THẾ GIỚI NHỎ (Small World)")
        else:
            print("   ✗ σ ≤ 1: Mạng không thể hiện tính chất Small World rõ ràng")
    else:
        results["small_world_sigma"] = None
    
    # Kết luận
    print("\n" + "-" * 70)
    print("📋 KẾT LUẬN VỀ KHÁI NIỆM THẾ GIỚI NHỎ:")
    print("-" * 70)
    
    conclusions = []
    
    # APL thấp?
    if apl < math.log(n) * 2:
        conclusions.append(f"✓ APL = {apl:.2f} khá thấp (< 2*ln(n) = {math.log(n)*2:.2f})")
        results["low_apl"] = True
    else:
        conclusions.append(f"✗ APL = {apl:.2f} khá cao")
        results["low_apl"] = False
    
    # Clustering cao?
    if avg_clustering > expected_clustering_random * 10:
        conclusions.append(f"✓ Clustering = {avg_clustering:.4f} cao hơn random {avg_clustering/expected_clustering_random:.1f}x")
        results["high_clustering"] = True
    else:
        conclusions.append(f"✗ Clustering = {avg_clustering:.4f} không cao hơn random đáng kể")
        results["high_clustering"] = False
    
    # Six Degrees of Separation?
    if apl <= 6:
        conclusions.append(f"✓ APL ≤ 6: Tuân theo 'Six Degrees of Separation'")
        results["six_degrees"] = True
    else:
        conclusions.append(f"✗ APL > 6: Không tuân theo 'Six Degrees of Separation' nghiêm ngặt")
        results["six_degrees"] = False
    
    for c in conclusions:
        print(f"   {c}")
    
    if results.get("low_apl") and results.get("high_clustering"):
        print("\n🎯 KẾT LUẬN: Mạng K-pop THỎA MÃN tính chất THẾ GIỚI NHỎ (Small World)")
        print(f"   - Bất kỳ 2 node nào cũng có thể kết nối qua trung bình {apl:.1f} bước")
        print(f"   - Các node có xu hướng tạo thành các cụm (cluster) cục bộ")
        results["is_small_world"] = True
    else:
        print("\n🎯 KẾT LUẬN: Mạng có một số đặc điểm của Small World nhưng chưa hoàn toàn")
        results["is_small_world"] = False
    
    return results


# =====================================================
# 2. PAGERANK - XẾP HẠNG NODE
# =====================================================
def analyze_pagerank(G: 'nx.Graph', top_k: int = 50) -> Dict[str, Any]:
    """
    Xếp hạng nodes bằng thuật toán PageRank
    """
    print("\n" + "=" * 70)
    print("2. XẾP HẠNG NODE BẰNG PAGERANK")
    print("=" * 70)
    
    results = {}
    
    # Tính PageRank
    print("\n🔍 Đang tính PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    
    # Sắp xếp theo PageRank giảm dần
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    
    results["total_nodes"] = len(pagerank)
    results["top_nodes"] = []
    
    print(f"\n📊 TOP {top_k} NODES THEO PAGERANK:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Node':<40} {'PageRank':<12} {'Label'}")
    print("-" * 70)
    
    for i, (node, score) in enumerate(sorted_pagerank[:top_k], 1):
        label = G.nodes[node].get('label', 'Unknown')
        print(f"{i:<6} {node[:38]:<40} {score:.8f}   {label}")
        results["top_nodes"].append({
            "rank": i,
            "node": node,
            "pagerank": score,
            "label": label
        })
    
    # Thống kê theo label
    print("\n📊 PAGERANK TRUNG BÌNH THEO LABEL:")
    print("-" * 50)
    
    label_scores = defaultdict(list)
    for node, score in pagerank.items():
        label = G.nodes[node].get('label', 'Unknown')
        label_scores[label].append(score)
    
    label_avg = {}
    for label, scores in label_scores.items():
        avg = sum(scores) / len(scores)
        label_avg[label] = avg
    
    results["pagerank_by_label"] = {}
    for label, avg in sorted(label_avg.items(), key=lambda x: x[1], reverse=True):
        count = len(label_scores[label])
        print(f"  {label:<15}: {avg:.8f} (n={count})")
        results["pagerank_by_label"][label] = {
            "average": avg,
            "count": count
        }
    
    # Tính thêm các centrality khác để so sánh
    print("\n🔍 Đang tính các centrality khác...")
    
    # Degree Centrality
    degree_cent = nx.degree_centrality(G)
    sorted_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
    
    # Betweenness Centrality (sampling cho graph lớn)
    if G.number_of_nodes() > 1000:
        print("   Betweenness Centrality: sử dụng sampling...")
        betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
    else:
        betweenness = nx.betweenness_centrality(G)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 SO SÁNH TOP 10 THEO CÁC CENTRALITY:")
    print("-" * 90)
    print(f"{'Rank':<6} {'PageRank':<25} {'Degree':<25} {'Betweenness':<25}")
    print("-" * 90)
    
    for i in range(min(10, len(sorted_pagerank))):
        pr_node = sorted_pagerank[i][0][:23] if i < len(sorted_pagerank) else "-"
        deg_node = sorted_degree[i][0][:23] if i < len(sorted_degree) else "-"
        bet_node = sorted_betweenness[i][0][:23] if i < len(sorted_betweenness) else "-"
        print(f"{i+1:<6} {pr_node:<25} {deg_node:<25} {bet_node:<25}")
    
    results["degree_centrality_top10"] = [{"node": n, "score": s} for n, s in sorted_degree[:10]]
    results["betweenness_centrality_top10"] = [{"node": n, "score": s} for n, s in sorted_betweenness[:10]]
    
    # Kết luận
    print("\n" + "-" * 70)
    print("📋 KẾT LUẬN VỀ XẾP HẠNG:")
    print("-" * 70)
    
    top1 = sorted_pagerank[0] if sorted_pagerank else ("N/A", 0)
    print(f"   🏆 Node quan trọng nhất: {top1[0]} (PageRank: {top1[1]:.6f})")
    
    # Tìm top node theo từng label
    top_by_label = {}
    for node, score in sorted_pagerank:
        label = G.nodes[node].get('label', 'Unknown')
        if label not in top_by_label:
            top_by_label[label] = (node, score)
    
    print("\n   🏆 Top node theo từng loại:")
    for label, (node, score) in sorted(top_by_label.items(), key=lambda x: x[1][1], reverse=True):
        print(f"      - {label}: {node} ({score:.6f})")
    
    results["top_by_label"] = {label: {"node": node, "pagerank": score} for label, (node, score) in top_by_label.items()}
    
    return results


# =====================================================
# 3. COMMUNITY DETECTION - PHÁT HIỆN CỘNG ĐỒNG
# =====================================================
def analyze_communities(G: 'nx.Graph', top_k_communities: int = 10) -> Dict[str, Any]:
    """
    Phát hiện cộng đồng trong mạng sử dụng thuật toán Louvain
    """
    print("\n" + "=" * 70)
    print("3. PHÁT HIỆN CỘNG ĐỒNG (COMMUNITY DETECTION)")
    print("=" * 70)
    
    results = {}
    
    # Kiểm tra có thư viện community detection không
    try:
        from networkx.algorithms import community as nx_community
        HAS_LOUVAIN = hasattr(nx_community, 'louvain_communities')
    except ImportError:
        HAS_LOUVAIN = False
    
    if HAS_LOUVAIN:
        print("\n🔍 Sử dụng thuật toán Louvain...")
        communities = nx_community.louvain_communities(G, seed=42)
        method = "Louvain"
    else:
        print("\n🔍 Sử dụng thuật toán Greedy Modularity...")
        communities = list(nx_community.greedy_modularity_communities(G))
        method = "Greedy Modularity"
    
    # Chuyển thành list để sắp xếp
    communities = [set(c) for c in communities]
    communities.sort(key=len, reverse=True)
    
    results["method"] = method
    results["total_communities"] = len(communities)
    
    print(f"\n✓ Phát hiện được {len(communities)} cộng đồng")
    
    # Tính modularity
    try:
        modularity = nx_community.modularity(G, communities)
        results["modularity"] = modularity
        print(f"✓ Modularity: {modularity:.4f}")
        
        if modularity > 0.3:
            print("   → Modularity > 0.3: Cấu trúc cộng đồng RÕ RÀNG")
        elif modularity > 0.1:
            print("   → Modularity > 0.1: Cấu trúc cộng đồng TRUNG BÌNH")
        else:
            print("   → Modularity ≤ 0.1: Cấu trúc cộng đồng YẾU")
    except:
        results["modularity"] = None
    
    # Thống kê kích thước cộng đồng
    community_sizes = [len(c) for c in communities]
    results["community_sizes"] = {
        "min": min(community_sizes),
        "max": max(community_sizes),
        "mean": sum(community_sizes) / len(community_sizes),
        "median": sorted(community_sizes)[len(community_sizes)//2]
    }
    
    print(f"\n📊 THỐNG KÊ KÍCH THƯỚC CỘNG ĐỒNG:")
    print(f"   - Nhỏ nhất: {min(community_sizes)} nodes")
    print(f"   - Lớn nhất: {max(community_sizes)} nodes")
    print(f"   - Trung bình: {sum(community_sizes)/len(community_sizes):.1f} nodes")
    
    # Chi tiết top communities
    print(f"\n📊 TOP {top_k_communities} CỘNG ĐỒNG LỚN NHẤT:")
    print("-" * 70)
    
    results["top_communities"] = []
    
    for i, comm in enumerate(communities[:top_k_communities], 1):
        # Đếm labels trong community
        label_counts = defaultdict(int)
        for node in comm:
            label = G.nodes[node].get('label', 'Unknown')
            label_counts[label] += 1
        
        # Tìm label chủ đạo
        dominant_label = max(label_counts.items(), key=lambda x: x[1])
        
        # Lấy một số node mẫu
        sample_nodes = list(comm)[:5]
        
        print(f"\n🔹 Cộng đồng {i}: {len(comm)} nodes")
        print(f"   Label chủ đạo: {dominant_label[0]} ({dominant_label[1]} nodes, {100*dominant_label[1]/len(comm):.1f}%)")
        print(f"   Phân bố: {dict(label_counts)}")
        print(f"   Nodes mẫu: {', '.join(sample_nodes)}")
        
        results["top_communities"].append({
            "id": i,
            "size": len(comm),
            "dominant_label": dominant_label[0],
            "dominant_label_count": dominant_label[1],
            "dominant_label_percentage": 100 * dominant_label[1] / len(comm),
            "label_distribution": dict(label_counts),
            "sample_nodes": sample_nodes
        })
    
    # Phân tích các cộng đồng đặc biệt
    print("\n📊 PHÂN TÍCH CỘNG ĐỒNG:")
    print("-" * 70)
    
    # Tìm các cộng đồng có tính chất đặc biệt
    artist_communities = []
    group_communities = []
    mixed_communities = []
    
    for i, comm in enumerate(communities):
        label_counts = defaultdict(int)
        for node in comm:
            label = G.nodes[node].get('label', 'Unknown')
            label_counts[label] += 1
        
        total = len(comm)
        if label_counts.get('Artist', 0) / total > 0.7:
            artist_communities.append((i, len(comm)))
        elif label_counts.get('Group', 0) / total > 0.5:
            group_communities.append((i, len(comm)))
        else:
            mixed_communities.append((i, len(comm)))
    
    print(f"   - Cộng đồng chủ yếu Artist: {len(artist_communities)}")
    print(f"   - Cộng đồng chủ yếu Group: {len(group_communities)}")
    print(f"   - Cộng đồng hỗn hợp: {len(mixed_communities)}")
    
    results["community_types"] = {
        "artist_dominated": len(artist_communities),
        "group_dominated": len(group_communities),
        "mixed": len(mixed_communities)
    }
    
    # Kết luận
    print("\n" + "-" * 70)
    print("📋 KẾT LUẬN VỀ CẤU TRÚC CỘNG ĐỒNG:")
    print("-" * 70)
    
    print(f"   1. Mạng K-pop có {len(communities)} cộng đồng rõ ràng")
    
    if results.get("modularity", 0) > 0.3:
        print(f"   2. Modularity cao ({results.get('modularity', 0):.3f}) cho thấy cấu trúc cộng đồng mạnh")
    
    print(f"   3. Cộng đồng lớn nhất có {max(community_sizes)} nodes ({100*max(community_sizes)/G.number_of_nodes():.1f}% mạng)")
    
    # Diễn giải cộng đồng
    print("\n   💡 DIỄN GIẢI:")
    print("   - Các cộng đồng có thể đại diện cho:")
    print("     + Nghệ sĩ cùng công ty (SM, YG, JYP, HYBE...)")
    print("     + Thế hệ idol (1st, 2nd, 3rd, 4th generation)")
    print("     + Thể loại âm nhạc (Hip-hop, Ballad, Dance...)")
    print("     + Các mối quan hệ hợp tác, collab")
    
    return results


# =====================================================
# MAIN FUNCTION
# =====================================================
def main():
    """Hàm main chạy tất cả phân tích"""
    
    print("\n" + "=" * 70)
    print("🎵 PHÂN TÍCH MẠNG XÃ HỘI K-POP 🎵")
    print("=" * 70)
    print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not NETWORKX_AVAILABLE:
        print("\n❌ Không thể chạy phân tích vì NetworkX chưa được cài đặt")
        print("   Chạy: pip install networkx")
        return
    
    # Load dữ liệu
    nodes, edges = load_graph_data()
    
    if not nodes:
        print("\n❌ Không có dữ liệu để phân tích")
        return
    
    # Build NetworkX graph
    print("\n🔧 Đang xây dựng NetworkX graph...")
    G = build_networkx_graph(nodes, edges, undirected=True)
    print(f"✓ Graph có {G.number_of_nodes()} nodes và {G.number_of_edges()} edges")
    
    # Kết quả tổng hợp
    all_results = {
        "analysis_time": datetime.now().isoformat(),
        "graph_info": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges()
        }
    }
    
    # 1. Small World Analysis
    try:
        small_world_results = analyze_small_world(G)
        all_results["small_world"] = small_world_results
    except Exception as e:
        print(f"\n❌ Lỗi khi phân tích Small World: {e}")
        all_results["small_world"] = {"error": str(e)}
    
    # 2. PageRank Analysis
    try:
        pagerank_results = analyze_pagerank(G, top_k=50)
        all_results["pagerank"] = pagerank_results
    except Exception as e:
        print(f"\n❌ Lỗi khi phân tích PageRank: {e}")
        all_results["pagerank"] = {"error": str(e)}
    
    # 3. Community Detection
    try:
        community_results = analyze_communities(G, top_k_communities=10)
        all_results["communities"] = community_results
    except Exception as e:
        print(f"\n❌ Lỗi khi phát hiện cộng đồng: {e}")
        all_results["communities"] = {"error": str(e)}
    
    # Lưu kết quả
    output_file = "network_analysis_results.json"
    print(f"\n💾 Đang lưu kết quả vào {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"✓ Đã lưu kết quả vào {output_file}")
    
    # Tổng kết
    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT PHÂN TÍCH MẠNG XÃ HỘI K-POP")
    print("=" * 70)
    
    print("\n1️⃣  THẾ GIỚI NHỎ (SMALL WORLD):")
    if "small_world" in all_results and "error" not in all_results["small_world"]:
        sw = all_results["small_world"]
        print(f"    - Average Path Length: {sw.get('average_path_length', 'N/A'):.2f}")
        print(f"    - Clustering Coefficient: {sw.get('clustering_coefficient', 'N/A'):.4f}")
        print(f"    - Là Small World: {'✓ CÓ' if sw.get('is_small_world') else '✗ KHÔNG'}")
    
    print("\n2️⃣  PAGERANK (TOP 5):")
    if "pagerank" in all_results and "error" not in all_results["pagerank"]:
        pr = all_results["pagerank"]
        for node_info in pr.get("top_nodes", [])[:5]:
            print(f"    {node_info['rank']}. {node_info['node']} ({node_info['label']})")
    
    print("\n3️⃣  CỘNG ĐỒNG:")
    if "communities" in all_results and "error" not in all_results["communities"]:
        comm = all_results["communities"]
        print(f"    - Số cộng đồng: {comm.get('total_communities', 'N/A')}")
        print(f"    - Modularity: {comm.get('modularity', 'N/A'):.4f}" if comm.get('modularity') else "")
    
    print("\n" + "=" * 70)
    print("✓ HOÀN TẤT PHÂN TÍCH")
    print("=" * 70)


if __name__ == "__main__":
    main()

