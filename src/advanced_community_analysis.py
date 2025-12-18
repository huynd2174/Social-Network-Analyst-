"""
Phân tích cộng đồng nâng cao trong mạng xã hội K-pop

Các phân tích chuyên sâu:
1. So sánh nhiều thuật toán phát hiện cộng đồng
2. Đánh giá chất lượng cộng đồng (Internal Density, Conductance, NMI)
3. Phân tích cấu trúc cộng đồng (Hub nodes, Bridge nodes, Core-periphery)
4. Phân tích ngữ nghĩa (Company communities, Generation communities)
5. Hierarchical community structure

Author: K-pop Social Network Analysis Team
"""

import sys
import io
import json
import math
import random
from typing import Dict, List, Any, Tuple, Set, Optional
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
    from networkx.algorithms import community as nx_community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️  NetworkX chưa được cài đặt. Chạy: pip install networkx")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# =====================================================
# 1. MULTI-ALGORITHM COMMUNITY DETECTION
# =====================================================

def detect_communities_multi_algorithm(G: 'nx.Graph') -> Dict[str, List[Set]]:
    """
    Phát hiện cộng đồng bằng nhiều thuật toán khác nhau để so sánh.
    
    Returns:
        Dict với key là tên thuật toán, value là danh sách các communities
    """
    print("\n" + "=" * 70)
    print("🔬 PHÁT HIỆN CỘNG ĐỒNG - ĐA THUẬT TOÁN")
    print("=" * 70)
    
    results = {}
    
    # 1. Louvain Algorithm (tối ưu modularity)
    print("\n1️⃣  Thuật toán LOUVAIN (Modularity Optimization)...")
    try:
        communities_louvain = list(nx_community.louvain_communities(G, seed=42))
        results['louvain'] = communities_louvain
        print(f"   ✓ Phát hiện {len(communities_louvain)} cộng đồng")
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    # 2. Greedy Modularity Communities
    print("\n2️⃣  Thuật toán GREEDY MODULARITY (CNM Algorithm)...")
    try:
        communities_greedy = list(nx_community.greedy_modularity_communities(G))
        results['greedy_modularity'] = communities_greedy
        print(f"   ✓ Phát hiện {len(communities_greedy)} cộng đồng")
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    # 3. Label Propagation Algorithm (LPA)
    print("\n3️⃣  Thuật toán LABEL PROPAGATION (LPA)...")
    try:
        communities_lpa = list(nx_community.label_propagation_communities(G))
        results['label_propagation'] = communities_lpa
        print(f"   ✓ Phát hiện {len(communities_lpa)} cộng đồng")
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    # 4. Asynchronous Label Propagation
    print("\n4️⃣  Thuật toán ASYNC LABEL PROPAGATION...")
    try:
        communities_async_lpa = list(nx_community.asyn_lpa_communities(G, seed=42))
        results['async_lpa'] = communities_async_lpa
        print(f"   ✓ Phát hiện {len(communities_async_lpa)} cộng đồng")
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    # 5. Girvan-Newman (Edge Betweenness - chỉ cho graph nhỏ)
    if G.number_of_nodes() <= 500:
        print("\n5️⃣  Thuật toán GIRVAN-NEWMAN (Edge Betweenness)...")
        try:
            # Lấy k communities (dừng sau 10 iterations)
            gn_generator = nx_community.girvan_newman(G)
            communities_gn = None
            best_modularity = -1
            for i in range(min(10, G.number_of_nodes() // 10)):
                try:
                    communities_iter = next(gn_generator)
                    mod = nx_community.modularity(G, communities_iter)
                    if mod > best_modularity:
                        best_modularity = mod
                        communities_gn = list(communities_iter)
                except StopIteration:
                    break
            if communities_gn:
                results['girvan_newman'] = communities_gn
                print(f"   ✓ Phát hiện {len(communities_gn)} cộng đồng (best modularity)")
        except Exception as e:
            print(f"   ✗ Lỗi: {e}")
    else:
        print("\n5️⃣  Girvan-Newman: Bỏ qua (graph quá lớn, > 500 nodes)")
    
    # 6. K-Clique Communities (overlapping)
    print("\n6️⃣  Thuật toán K-CLIQUE PERCOLATION (Overlapping)...")
    try:
        # Thử k=3 (triangles)
        communities_kclique = list(nx_community.k_clique_communities(G, 3))
        if communities_kclique:
            results['k_clique_3'] = communities_kclique
            print(f"   ✓ Phát hiện {len(communities_kclique)} cộng đồng (k=3, có thể chồng lấp)")
        else:
            print(f"   ✗ Không tìm thấy k-clique communities (k=3)")
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    return results


# =====================================================
# 2. COMMUNITY QUALITY METRICS
# =====================================================

def calculate_internal_density(G: 'nx.Graph', community: Set) -> float:
    """
    Tính Internal Density của một cộng đồng.
    
    Internal Density = 2 * m_c / (n_c * (n_c - 1))
    
    Trong đó:
    - m_c: số edges bên trong cộng đồng
    - n_c: số nodes trong cộng đồng
    """
    subgraph = G.subgraph(community)
    n_c = subgraph.number_of_nodes()
    m_c = subgraph.number_of_edges()
    
    if n_c <= 1:
        return 0.0
    
    max_edges = n_c * (n_c - 1) / 2
    return m_c / max_edges if max_edges > 0 else 0.0


def calculate_external_density(G: 'nx.Graph', community: Set) -> float:
    """
    Tính External Density - tỷ lệ edges nối ra bên ngoài cộng đồng.
    
    External Density = edges_out / (n_c * (N - n_c))
    """
    n_c = len(community)
    N = G.number_of_nodes()
    
    if n_c == 0 or n_c == N:
        return 0.0
    
    edges_out = 0
    for node in community:
        for neighbor in G.neighbors(node):
            if neighbor not in community:
                edges_out += 1
    
    max_external_edges = n_c * (N - n_c)
    return edges_out / max_external_edges if max_external_edges > 0 else 0.0


def calculate_conductance(G: 'nx.Graph', community: Set) -> float:
    """
    Tính Conductance của một cộng đồng.
    
    Conductance = cut(S, S̄) / min(vol(S), vol(S̄))
    
    - cut(S, S̄): số edges cắt giữa community và phần còn lại
    - vol(S): tổng degree của nodes trong S
    
    Conductance thấp = cộng đồng tốt (ít liên kết ra ngoài)
    """
    if len(community) == 0 or len(community) == G.number_of_nodes():
        return 0.0
    
    cut = 0
    vol_s = 0
    vol_not_s = 0
    
    for node in G.nodes():
        degree = G.degree(node)
        if node in community:
            vol_s += degree
            for neighbor in G.neighbors(node):
                if neighbor not in community:
                    cut += 1
        else:
            vol_not_s += degree
    
    min_vol = min(vol_s, vol_not_s)
    return cut / min_vol if min_vol > 0 else 0.0


def calculate_cut_ratio(G: 'nx.Graph', community: Set) -> float:
    """
    Tính Cut Ratio của một cộng đồng.
    
    Cut Ratio = cut(S, S̄) / (|S| * |S̄|)
    """
    n_s = len(community)
    n_not_s = G.number_of_nodes() - n_s
    
    if n_s == 0 or n_not_s == 0:
        return 0.0
    
    cut = 0
    for node in community:
        for neighbor in G.neighbors(node):
            if neighbor not in community:
                cut += 1
    
    return cut / (n_s * n_not_s)


def evaluate_community_quality(G: 'nx.Graph', communities: List[Set]) -> Dict[str, Any]:
    """
    Đánh giá chất lượng toàn diện của các cộng đồng.
    """
    print("\n" + "=" * 70)
    print("📊 ĐÁNH GIÁ CHẤT LƯỢNG CỘNG ĐỒNG")
    print("=" * 70)
    
    results = {
        'num_communities': len(communities),
        'communities_detail': []
    }
    
    # Tính Modularity
    try:
        modularity = nx_community.modularity(G, communities)
        results['modularity'] = modularity
        print(f"\n✓ Modularity (Q): {modularity:.4f}")
        
        if modularity > 0.7:
            print("   → Modularity > 0.7: Cấu trúc cộng đồng RẤT MẠNH")
        elif modularity > 0.5:
            print("   → Modularity > 0.5: Cấu trúc cộng đồng MẠNH")
        elif modularity > 0.3:
            print("   → Modularity > 0.3: Cấu trúc cộng đồng RÕ RÀNG")
        elif modularity > 0.1:
            print("   → Modularity > 0.1: Cấu trúc cộng đồng TRUNG BÌNH")
        else:
            print("   → Modularity ≤ 0.1: Cấu trúc cộng đồng YẾU")
    except Exception as e:
        print(f"✗ Không thể tính Modularity: {e}")
    
    # Tính Coverage (tỷ lệ edges nội bộ)
    total_internal_edges = 0
    for comm in communities:
        subgraph = G.subgraph(comm)
        total_internal_edges += subgraph.number_of_edges()
    
    coverage = total_internal_edges / G.number_of_edges() if G.number_of_edges() > 0 else 0
    results['coverage'] = coverage
    print(f"✓ Coverage: {coverage:.4f} ({100*coverage:.1f}% edges nằm trong các cộng đồng)")
    
    # Tính metrics chi tiết cho từng cộng đồng lớn
    print(f"\n📊 METRICS CHI TIẾT CHO TOP 10 CỘNG ĐỒNG:")
    print("-" * 90)
    print(f"{'#':<4} {'Size':<8} {'Int.Dens':<12} {'Ext.Dens':<12} {'Conductance':<14} {'Cut Ratio':<12}")
    print("-" * 90)
    
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    all_internal_densities = []
    all_conductances = []
    
    for i, comm in enumerate(sorted_communities[:10], 1):
        int_dens = calculate_internal_density(G, comm)
        ext_dens = calculate_external_density(G, comm)
        conductance = calculate_conductance(G, comm)
        cut_ratio = calculate_cut_ratio(G, comm)
        
        all_internal_densities.append(int_dens)
        all_conductances.append(conductance)
        
        print(f"{i:<4} {len(comm):<8} {int_dens:<12.4f} {ext_dens:<12.4f} {conductance:<14.4f} {cut_ratio:<12.6f}")
        
        results['communities_detail'].append({
            'rank': i,
            'size': len(comm),
            'internal_density': int_dens,
            'external_density': ext_dens,
            'conductance': conductance,
            'cut_ratio': cut_ratio
        })
    
    # Thống kê tổng hợp
    print(f"\n📊 THỐNG KÊ TỔNG HỢP:")
    print("-" * 50)
    
    if all_internal_densities:
        avg_int_dens = sum(all_internal_densities) / len(all_internal_densities)
        results['avg_internal_density'] = avg_int_dens
        print(f"   - Internal Density trung bình (top 10): {avg_int_dens:.4f}")
    
    if all_conductances:
        avg_conductance = sum(all_conductances) / len(all_conductances)
        results['avg_conductance'] = avg_conductance
        print(f"   - Conductance trung bình (top 10): {avg_conductance:.4f}")
        
        if avg_conductance < 0.3:
            print("   → Conductance thấp: Các cộng đồng được phân tách tốt")
        elif avg_conductance < 0.5:
            print("   → Conductance trung bình: Các cộng đồng có một số liên kết ra ngoài")
        else:
            print("   → Conductance cao: Ranh giới cộng đồng không rõ ràng")
    
    return results


def compare_algorithms(G: 'nx.Graph', communities_dict: Dict[str, List[Set]]) -> Dict[str, Any]:
    """
    So sánh chất lượng của các thuật toán phát hiện cộng đồng.
    """
    print("\n" + "=" * 70)
    print("📊 SO SÁNH CÁC THUẬT TOÁN")
    print("=" * 70)
    
    comparison = {}
    
    print(f"\n{'Thuật toán':<25} {'Số CĐ':<10} {'Modularity':<12} {'Coverage':<12} {'Max Size':<10}")
    print("-" * 80)
    
    for algo_name, communities in communities_dict.items():
        try:
            modularity = nx_community.modularity(G, communities)
            
            # Coverage
            total_internal = sum(G.subgraph(c).number_of_edges() for c in communities)
            coverage = total_internal / G.number_of_edges() if G.number_of_edges() > 0 else 0
            
            # Max size
            max_size = max(len(c) for c in communities) if communities else 0
            
            comparison[algo_name] = {
                'num_communities': len(communities),
                'modularity': modularity,
                'coverage': coverage,
                'max_community_size': max_size
            }
            
            print(f"{algo_name:<25} {len(communities):<10} {modularity:<12.4f} {coverage:<12.4f} {max_size:<10}")
        except Exception as e:
            print(f"{algo_name:<25} Lỗi: {e}")
    
    # Tìm thuật toán tốt nhất theo Modularity
    if comparison:
        best_algo = max(comparison.items(), key=lambda x: x[1].get('modularity', 0))
        print(f"\n🏆 Thuật toán tốt nhất (theo Modularity): {best_algo[0]} (Q={best_algo[1]['modularity']:.4f})")
    
    return comparison


# =====================================================
# 3. COMMUNITY STRUCTURE ANALYSIS
# =====================================================

def analyze_community_structure(G: 'nx.Graph', communities: List[Set], top_k: int = 5) -> Dict[str, Any]:
    """
    Phân tích cấu trúc chi tiết của các cộng đồng:
    - Hub nodes (nodes quan trọng nhất trong mỗi community)
    - Bridge nodes (nodes kết nối giữa các communities)
    - Core-periphery structure
    """
    print("\n" + "=" * 70)
    print("🔬 PHÂN TÍCH CẤU TRÚC CỘNG ĐỒNG")
    print("=" * 70)
    
    results = {
        'hub_nodes': [],
        'bridge_nodes': [],
        'core_periphery': []
    }
    
    # Tạo mapping node -> community index
    node_to_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i
    
    # 1. TÌM HUB NODES TRONG MỖI COMMUNITY
    print(f"\n📊 HUB NODES (TOP {top_k} NODES QUAN TRỌNG NHẤT TRONG MỖI CỘNG ĐỒNG LỚN):")
    print("-" * 70)
    
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    for i, comm in enumerate(sorted_communities[:5], 1):
        subgraph = G.subgraph(comm)
        
        # Tính PageRank trong subgraph
        try:
            pr = nx.pagerank(subgraph)
            top_hubs = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n🔹 Cộng đồng {i} ({len(comm)} nodes):")
            for j, (node, score) in enumerate(top_hubs, 1):
                label = G.nodes[node].get('label', 'Unknown')
                print(f"   {j}. {node} ({label}) - Score: {score:.4f}")
            
            results['hub_nodes'].append({
                'community_id': i,
                'size': len(comm),
                'hubs': [{'node': n, 'score': s, 'label': G.nodes[n].get('label', 'Unknown')} for n, s in top_hubs]
            })
        except:
            pass
    
    # 2. TÌM BRIDGE NODES (nút cầu nối giữa các communities)
    print(f"\n📊 BRIDGE NODES (NÚT CẦU NỐI GIỮA CÁC CỘNG ĐỒNG):")
    print("-" * 70)
    
    bridge_scores = {}
    for node in G.nodes():
        if node not in node_to_community:
            continue
            
        my_comm = node_to_community[node]
        external_connections = defaultdict(int)
        
        for neighbor in G.neighbors(node):
            if neighbor in node_to_community:
                neighbor_comm = node_to_community[neighbor]
                if neighbor_comm != my_comm:
                    external_connections[neighbor_comm] += 1
        
        if external_connections:
            # Bridge score = số communities khác được kết nối * số connections
            bridge_scores[node] = {
                'communities_connected': len(external_connections),
                'total_external_edges': sum(external_connections.values()),
                'own_community': my_comm
            }
    
    # Sắp xếp theo số communities được kết nối
    top_bridges = sorted(
        bridge_scores.items(), 
        key=lambda x: (x[1]['communities_connected'], x[1]['total_external_edges']), 
        reverse=True
    )[:10]
    
    print(f"{'Node':<40} {'Label':<15} {'# CĐ kết nối':<15} {'# Edges ngoài':<15}")
    print("-" * 85)
    
    for node, info in top_bridges:
        label = G.nodes[node].get('label', 'Unknown')
        print(f"{node[:38]:<40} {label:<15} {info['communities_connected']:<15} {info['total_external_edges']:<15}")
    
    results['bridge_nodes'] = [
        {
            'node': node,
            'label': G.nodes[node].get('label', 'Unknown'),
            'communities_connected': info['communities_connected'],
            'total_external_edges': info['total_external_edges']
        }
        for node, info in top_bridges
    ]
    
    # 3. CORE-PERIPHERY ANALYSIS
    print(f"\n📊 CORE-PERIPHERY ANALYSIS (CẤU TRÚC LÕI-NGOẠI VI):")
    print("-" * 70)
    
    for i, comm in enumerate(sorted_communities[:3], 1):
        subgraph = G.subgraph(comm)
        
        # Phân loại nodes: Core (degree cao), Periphery (degree thấp)
        degrees = dict(subgraph.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        
        core_nodes = [n for n, d in degrees.items() if d >= avg_degree]
        periphery_nodes = [n for n, d in degrees.items() if d < avg_degree]
        
        print(f"\n🔹 Cộng đồng {i} ({len(comm)} nodes):")
        print(f"   - Core nodes (degree ≥ {avg_degree:.1f}): {len(core_nodes)} nodes ({100*len(core_nodes)/len(comm):.1f}%)")
        print(f"   - Periphery nodes: {len(periphery_nodes)} nodes ({100*len(periphery_nodes)/len(comm):.1f}%)")
        
        # Hiển thị một số core nodes
        top_core = sorted(core_nodes, key=lambda n: degrees[n], reverse=True)[:3]
        print(f"   - Top core nodes: {', '.join([f'{n} (deg={degrees[n]})' for n in top_core])}")
        
        results['core_periphery'].append({
            'community_id': i,
            'size': len(comm),
            'core_count': len(core_nodes),
            'periphery_count': len(periphery_nodes),
            'avg_degree': avg_degree,
            'top_core_nodes': [{'node': n, 'degree': degrees[n]} for n in top_core]
        })
    
    return results


# =====================================================
# 4. SEMANTIC COMMUNITY ANALYSIS
# =====================================================

def analyze_semantic_communities(G: 'nx.Graph', communities: List[Set]) -> Dict[str, Any]:
    """
    Phân tích ngữ nghĩa của các cộng đồng:
    - Xác định "chủ đề" của từng community
    - Tìm company-based communities
    - Tìm genre-based communities
    - Tìm generation-based communities
    """
    print("\n" + "=" * 70)
    print("🎯 PHÂN TÍCH NGỮ NGHĨA CỘNG ĐỒNG")
    print("=" * 70)
    
    results = {
        'company_communities': [],
        'genre_communities': [],
        'group_centric_communities': []
    }
    
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    # 1. PHÂN LOẠI CỘNG ĐỒNG THEO NỘI DUNG
    print(f"\n📊 PHÂN LOẠI CỘNG ĐỒNG THEO NỘI DUNG CHÍNH:")
    print("-" * 70)
    
    for i, comm in enumerate(sorted_communities[:15], 1):
        # Đếm theo label
        label_counts = defaultdict(int)
        companies = []
        groups = []
        genres = []
        
        for node in comm:
            label = G.nodes[node].get('label', 'Unknown')
            label_counts[label] += 1
            
            if label == 'Company':
                companies.append(node)
            elif label == 'Group':
                groups.append(node)
            elif label == 'Genre':
                genres.append(node)
        
        # Xác định loại cộng đồng
        total = len(comm)
        dominant_label, dominant_count = max(label_counts.items(), key=lambda x: x[1])
        dominant_pct = 100 * dominant_count / total
        
        # Phân loại
        comm_type = "Mixed"
        main_entity = None
        
        if companies:
            # Tìm company phổ biến nhất (dựa trên số connections)
            company_connections = {}
            for company in companies:
                connections = sum(1 for n in G.neighbors(company) if n in comm)
                company_connections[company] = connections
            if company_connections:
                main_company = max(company_connections.items(), key=lambda x: x[1])[0]
                comm_type = "Company-based"
                main_entity = main_company.replace("Company_", "")
                results['company_communities'].append({
                    'rank': i,
                    'size': len(comm),
                    'main_company': main_entity,
                    'groups': [g for g in groups[:5]],
                    'label_distribution': dict(label_counts)
                })
        
        if groups and comm_type == "Mixed":
            # Tìm group chính
            group_degrees = {g: G.degree(g) for g in groups}
            main_group = max(group_degrees.items(), key=lambda x: x[1])[0]
            comm_type = "Group-centric"
            main_entity = main_group
            results['group_centric_communities'].append({
                'rank': i,
                'size': len(comm),
                'main_group': main_entity,
                'label_distribution': dict(label_counts)
            })
        
        if genres and len(genres) >= 3:
            comm_type = "Genre-based"
            main_entity = ", ".join(g.replace("Genre_", "") for g in genres[:3])
            results['genre_communities'].append({
                'rank': i,
                'size': len(comm),
                'genres': [g.replace("Genre_", "") for g in genres],
                'label_distribution': dict(label_counts)
            })
        
        print(f"\n🔹 Cộng đồng {i} ({len(comm)} nodes) - {comm_type}")
        print(f"   - Label chủ đạo: {dominant_label} ({dominant_count} nodes, {dominant_pct:.1f}%)")
        if main_entity:
            print(f"   - Thực thể chính: {main_entity}")
        print(f"   - Phân bố: {dict(label_counts)}")
        
        # Sample nodes
        sample_by_label = {}
        for node in comm:
            label = G.nodes[node].get('label', 'Unknown')
            if label not in sample_by_label:
                sample_by_label[label] = node
            if len(sample_by_label) >= 4:
                break
        print(f"   - Mẫu: {list(sample_by_label.values())[:4]}")
    
    # 2. TÌM CÁC COMPANY CLUSTERS
    print(f"\n📊 COMPANY CLUSTERS (Nghệ sĩ theo công ty):")
    print("-" * 70)
    
    # Nhóm các artists theo công ty
    company_artists = defaultdict(list)
    for node in G.nodes():
        if G.nodes[node].get('label') == 'Artist':
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor].get('label') == 'Company':
                    company_artists[neighbor].append(node)
    
    # Top 5 công ty có nhiều nghệ sĩ nhất
    top_companies = sorted(company_artists.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    
    for company, artists in top_companies:
        company_name = company.replace("Company_", "")
        print(f"\n🏢 {company_name}: {len(artists)} nghệ sĩ")
        print(f"   Mẫu: {', '.join(artists[:5])}")
    
    results['top_companies_by_artists'] = [
        {'company': c.replace("Company_", ""), 'artist_count': len(a), 'sample_artists': a[:5]}
        for c, a in top_companies
    ]
    
    return results


# =====================================================
# 5. HIERARCHICAL COMMUNITY STRUCTURE
# =====================================================

def analyze_hierarchical_structure(G: 'nx.Graph', max_levels: int = 3) -> Dict[str, Any]:
    """
    Phân tích cấu trúc phân cấp của các cộng đồng.
    Sử dụng Girvan-Newman để tạo dendrogram.
    """
    print("\n" + "=" * 70)
    print("🌳 PHÂN TÍCH CẤU TRÚC PHÂN CẤP (HIERARCHICAL)")
    print("=" * 70)
    
    results = {'levels': []}
    
    if G.number_of_nodes() > 500:
        print("\n⚠️  Graph quá lớn cho phân tích hierarchical. Sử dụng subgraph của largest connected component (max 500 nodes)...")
        
        # Lấy largest connected component
        if nx.is_connected(G):
            subG = G
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc).copy()
        
        # Sample nếu vẫn quá lớn
        if subG.number_of_nodes() > 500:
            sample_nodes = random.sample(list(subG.nodes()), 500)
            subG = G.subgraph(sample_nodes).copy()
    else:
        subG = G
    
    print(f"\n🔍 Phân tích trên {subG.number_of_nodes()} nodes...")
    
    try:
        # Sử dụng Louvain với resolution khác nhau để mô phỏng hierarchical
        resolutions = [0.5, 1.0, 1.5, 2.0]
        
        for res in resolutions:
            try:
                communities = list(nx_community.louvain_communities(subG, resolution=res, seed=42))
                modularity = nx_community.modularity(subG, communities)
                
                print(f"\n📊 Resolution = {res}:")
                print(f"   - Số cộng đồng: {len(communities)}")
                print(f"   - Modularity: {modularity:.4f}")
                print(f"   - Kích thước: {[len(c) for c in sorted(communities, key=len, reverse=True)[:5]]}")
                
                results['levels'].append({
                    'resolution': res,
                    'num_communities': len(communities),
                    'modularity': modularity,
                    'sizes': [len(c) for c in sorted(communities, key=len, reverse=True)[:10]]
                })
            except Exception as e:
                print(f"   ✗ Lỗi tại resolution {res}: {e}")
        
        # Kết luận
        print(f"\n📋 PHÂN TÍCH HIERARCHICAL:")
        print("-" * 50)
        print("   - Resolution thấp → ít cộng đồng lớn (macro-level)")
        print("   - Resolution cao → nhiều cộng đồng nhỏ (micro-level)")
        
        if results['levels']:
            best_level = max(results['levels'], key=lambda x: x['modularity'])
            print(f"\n   🏆 Resolution tối ưu: {best_level['resolution']} (Modularity = {best_level['modularity']:.4f})")
    
    except Exception as e:
        print(f"✗ Lỗi khi phân tích hierarchical: {e}")
    
    return results


# =====================================================
# 6. VISUALIZATION
# =====================================================

def visualize_community_analysis(G: 'nx.Graph', communities: List[Set], 
                                  output_dir: str = "outputs") -> None:
    """
    Tạo các biểu đồ visualization cho phân tích cộng đồng.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib không khả dụng, bỏ qua visualization")
        return
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("📊 TẠO BIỂU ĐỒ VISUALIZATION")
    print("=" * 70)
    
    # 1. Community Size Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1.1 Histogram of community sizes
    sizes = [len(c) for c in communities]
    ax1 = axes[0, 0]
    ax1.hist(sizes, bins=min(50, len(sizes)), edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Community Size')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Phân bố Kích thước Cộng đồng')
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Top 20 communities bar chart
    ax2 = axes[0, 1]
    top_20_sizes = sorted(sizes, reverse=True)[:20]
    ax2.bar(range(1, len(top_20_sizes) + 1), top_20_sizes, color='coral')
    ax2.set_xlabel('Community Rank')
    ax2.set_ylabel('Size')
    ax2.set_title('Top 20 Cộng đồng Lớn nhất')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 1.3 Cumulative distribution
    ax3 = axes[1, 0]
    sorted_sizes = sorted(sizes, reverse=True)
    cumulative = [sum(sorted_sizes[:i+1]) / sum(sizes) * 100 for i in range(len(sorted_sizes))]
    ax3.plot(range(1, len(cumulative) + 1), cumulative, 'b-', linewidth=2)
    ax3.axhline(y=80, color='r', linestyle='--', label='80% coverage')
    ax3.set_xlabel('Number of Communities')
    ax3.set_ylabel('Cumulative % of Nodes')
    ax3.set_title('Phân bố Tích lũy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 1.4 Label distribution in top communities
    ax4 = axes[1, 1]
    sorted_communities = sorted(communities, key=len, reverse=True)[:5]
    
    labels_data = defaultdict(list)
    for i, comm in enumerate(sorted_communities):
        label_counts = defaultdict(int)
        for node in comm:
            label = G.nodes[node].get('label', 'Unknown')
            label_counts[label] += 1
        for label, count in label_counts.items():
            labels_data[label].append(count)
        for label in labels_data:
            if len(labels_data[label]) < i + 1:
                labels_data[label].append(0)
    
    x = range(1, 6)
    bottom = [0] * 5
    colors = plt.cm.tab10(range(len(labels_data)))
    
    for (label, counts), color in zip(labels_data.items(), colors):
        while len(counts) < 5:
            counts.append(0)
        ax4.bar(x, counts, bottom=bottom, label=label, color=color)
        bottom = [b + c for b, c in zip(bottom, counts)]
    
    ax4.set_xlabel('Community Rank')
    ax4.set_ylabel('Number of Nodes')
    ax4.set_title('Phân bố Label trong Top 5 Cộng đồng')
    ax4.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    output_path = f"{output_dir}/community_analysis_advanced.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ: {output_path}")
    plt.close()


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():
    """Hàm main chạy phân tích cộng đồng nâng cao"""
    
    print("\n" + "=" * 70)
    print("🎵 PHÂN TÍCH CỘNG ĐỒNG NÂNG CAO - MẠNG XÃ HỘI K-POP 🎵")
    print("=" * 70)
    print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not NETWORKX_AVAILABLE:
        print("\n❌ Không thể chạy phân tích vì NetworkX chưa được cài đặt")
        return
    
    # Load dữ liệu
    print("\n🔄 Đang load dữ liệu...")
    try:
        with open("data/korean_artists_graph_bfs.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        nodes = data.get('nodes', {})
        edges = data.get('edges', [])
        print(f"✓ Đã load {len(nodes)} nodes và {len(edges)} edges")
    except FileNotFoundError:
        print("❌ Không tìm thấy file dữ liệu")
        return
    
    # Build graph
    print("\n🔧 Đang xây dựng NetworkX graph...")
    G = nx.Graph()
    for node_id, node_data in nodes.items():
        G.add_node(node_id, **{
            'label': node_data.get('label', 'Entity'),
            'title': node_data.get('title', node_id)
        })
    for edge in edges:
        src, tgt = edge.get('source'), edge.get('target')
        if src and tgt and src in nodes and tgt in nodes:
            G.add_edge(src, tgt, type=edge.get('type', 'RELATED_TO'))
    
    print(f"✓ Graph có {G.number_of_nodes()} nodes và {G.number_of_edges()} edges")
    
    # Kết quả tổng hợp
    all_results = {
        'analysis_time': datetime.now().isoformat(),
        'graph_info': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        }
    }
    
    # 1. Phát hiện cộng đồng bằng nhiều thuật toán
    communities_dict = detect_communities_multi_algorithm(G)
    all_results['algorithms'] = {algo: len(comms) for algo, comms in communities_dict.items()}
    
    # 2. So sánh các thuật toán
    comparison = compare_algorithms(G, communities_dict)
    all_results['comparison'] = comparison
    
    # 3. Chọn thuật toán tốt nhất và phân tích chi tiết
    best_algo = max(comparison.items(), key=lambda x: x[1].get('modularity', 0))
    best_communities = communities_dict[best_algo[0]]
    
    print(f"\n🏆 Sử dụng kết quả từ {best_algo[0]} cho phân tích chi tiết...")
    
    # 4. Đánh giá chất lượng
    quality = evaluate_community_quality(G, best_communities)
    all_results['quality'] = quality
    
    # 5. Phân tích cấu trúc
    structure = analyze_community_structure(G, best_communities)
    all_results['structure'] = structure
    
    # 6. Phân tích ngữ nghĩa
    semantic = analyze_semantic_communities(G, best_communities)
    all_results['semantic'] = semantic
    
    # 7. Phân tích hierarchical
    hierarchical = analyze_hierarchical_structure(G)
    all_results['hierarchical'] = hierarchical
    
    # 8. Visualization
    visualize_community_analysis(G, best_communities)
    
    # Lưu kết quả
    output_path = "data/advanced_community_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert sets to lists for JSON serialization
        json_results = json.loads(json.dumps(all_results, default=str))
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu kết quả vào: {output_path}")
    
    # Tổng kết
    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT PHÂN TÍCH CỘNG ĐỒNG NÂNG CAO")
    print("=" * 70)
    
    print(f"\n1️⃣  ĐA THUẬT TOÁN:")
    for algo, count in all_results['algorithms'].items():
        mod = comparison.get(algo, {}).get('modularity', 0)
        print(f"    - {algo}: {count} cộng đồng (Q={mod:.4f})")
    
    print(f"\n2️⃣  CHẤT LƯỢNG (thuật toán {best_algo[0]}):")
    print(f"    - Modularity: {quality.get('modularity', 'N/A'):.4f}")
    print(f"    - Coverage: {quality.get('coverage', 'N/A'):.4f}")
    print(f"    - Avg Internal Density: {quality.get('avg_internal_density', 'N/A'):.4f}")
    
    print(f"\n3️⃣  CẤU TRÚC:")
    print(f"    - Bridge nodes: {len(structure.get('bridge_nodes', []))}")
    print(f"    - Hub nodes được phát hiện trong top communities")
    
    print(f"\n4️⃣  NGỮ NGHĨA:")
    print(f"    - Company communities: {len(semantic.get('company_communities', []))}")
    print(f"    - Group-centric communities: {len(semantic.get('group_centric_communities', []))}")
    
    print("\n" + "=" * 70)
    print("✓ HOÀN TẤT PHÂN TÍCH CỘNG ĐỒNG NÂNG CAO")
    print("=" * 70)


if __name__ == "__main__":
    main()



