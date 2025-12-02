#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để tạo 20 dữ liệu two-hop từ dataset korean_music_bfs
"""

import json
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import random

def load_graph(json_file: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load graph từ file JSON"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('nodes', {}), data.get('edges', [])

def build_graph(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Xây dựng graph từ danh sách edges"""
    graph = defaultdict(list)
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            graph[source].append(edge)
    return graph

def find_two_hop_paths(nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tìm tất cả các đường đi two-hop (2 bước, 3 nodes)"""
    graph = build_graph(edges)
    # Tạo graph ngược để tìm quan hệ nghịch đảo
    reverse_graph = defaultdict(list)
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            reverse_graph[target].append({
                **edge,
                'source': target,
                'target': source,
                'type': edge.get('type', 'RELATED_TO')
            })
    
    two_hop_paths = []
    
    # Tạo set các node để kiểm tra tồn tại
    node_set = set(nodes.keys())
    
    # Duyệt qua tất cả các node
    for start_node in node_set:
        # Tìm các node ở bước 1 (one-hop) - cả hướng thuận và ngược
        if start_node not in graph:
            continue
            
        # Hướng thuận
        for edge1 in graph[start_node]:
            middle_node = edge1['target']
            
            # Kiểm tra middle_node có tồn tại không
            if middle_node not in node_set:
                continue
            
            # Tìm các node ở bước 2 (two-hop) - cả hướng thuận và ngược
            for edge2 in graph.get(middle_node, []):
                end_node = edge2['target']
                
                # Kiểm tra end_node có tồn tại không
                if end_node not in node_set:
                    continue
                
                # Tránh vòng lặp (start != end)
                if start_node == end_node:
                    continue
                
                # Tạo path two-hop
                path = {
                    'start_node': start_node,
                    'middle_node': middle_node,
                    'end_node': end_node,
                    'edge1': edge1,
                    'edge2': edge2,
                    'start_label': nodes[start_node].get('label', 'Unknown'),
                    'middle_label': nodes[middle_node].get('label', 'Unknown'),
                    'end_label': nodes[end_node].get('label', 'Unknown')
                }
                two_hop_paths.append(path)
            
            # Hướng ngược từ middle_node
            for edge2 in reverse_graph.get(middle_node, []):
                end_node = edge2['target']
                
                # Kiểm tra end_node có tồn tại không
                if end_node not in node_set:
                    continue
                
                # Tránh vòng lặp (start != end)
                if start_node == end_node:
                    continue
                
                # Tạo path two-hop
                path = {
                    'start_node': start_node,
                    'middle_node': middle_node,
                    'end_node': end_node,
                    'edge1': edge1,
                    'edge2': edge2,
                    'start_label': nodes[start_node].get('label', 'Unknown'),
                    'middle_label': nodes[middle_node].get('label', 'Unknown'),
                    'end_label': nodes[end_node].get('label', 'Unknown')
                }
                two_hop_paths.append(path)
    
    return two_hop_paths

def generate_question(path: Dict[str, Any], nodes: Dict[str, Any]) -> str:
    """Tạo câu hỏi cho một two-hop path"""
    start = path['start_node']
    middle = path['middle_node']
    end = path['end_node']
    
    edge1_type = path['edge1']['type']
    edge2_type = path['edge2']['type']
    
    start_label = path['start_label']
    middle_label = path['middle_label']
    end_label = path['end_label']
    
    # Loại bỏ prefix "Genre_", "Company_", "Occupation_" để hiển thị đẹp hơn
    def clean_name(name):
        for prefix in ['Genre_', 'Company_', 'Occupation_']:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    
    clean_middle = clean_name(middle)
    clean_end = clean_name(end)
    
    # Tạo câu hỏi dựa trên loại quan hệ và labels - với nhiều biến thể để tránh trùng lặp
    import random
    
    # Tạo nhiều biến thể câu hỏi cho mỗi loại
    question_variants = {
        # Nghệ sĩ cùng nhóm
        ('MEMBER_OF', 'MEMBER_OF'): [
            lambda: f"Ai là thành viên khác của nhóm nhạc {clean_middle} cùng với {start}?",
            lambda: f"Nghệ sĩ nào khác cũng là thành viên của nhóm nhạc {clean_middle} giống như {start}?",
            lambda: f"Bên cạnh {start}, nghệ sĩ nào khác cũng thuộc nhóm nhạc {clean_middle}?",
        ],
        
        # Nhóm cùng công ty
        ('MANAGED_BY', 'MANAGED_BY'): [
            lambda: f"Nhóm nhạc nào khác cũng được quản lý bởi công ty {clean_middle} giống như {start}?",
            lambda: f"Cùng được quản lý bởi {clean_middle}, nhóm nhạc nào khác ngoài {start}?",
            lambda: f"Nhóm nhạc nào khác cũng thuộc công ty {clean_middle} như {start}?",
        ],
        
        # Nhóm cùng thể loại
        ('IS_GENRE', 'IS_GENRE'): [
            lambda: f"Nhóm nhạc nào khác cũng thuộc thể loại {clean_middle} giống như {start}?",
            lambda: f"Cùng thể loại {clean_middle}, nhóm nhạc nào khác ngoài {start}?",
            lambda: f"Nhóm nhạc nào khác cũng chơi nhạc {clean_middle} như {start}?",
        ],
        
        # Nghệ sĩ trình bày bài hát qua nhóm
        ('MEMBER_OF', 'SINGS'): [
            lambda: f"Bài hát nào mà nghệ sĩ {start} trình bày thông qua nhóm nhạc {clean_middle}?",
            lambda: f"Qua nhóm nhạc {clean_middle}, nghệ sĩ {start} đã trình bày bài hát nào?",
            lambda: f"Bài hát nào của nhóm nhạc {clean_middle} có sự tham gia của nghệ sĩ {start}?",
        ],
        
        # Nghệ sĩ phát hành album qua nhóm
        ('MEMBER_OF', 'RELEASED'): [
            lambda: f"Album nào mà nghệ sĩ {start} phát hành thông qua nhóm nhạc {clean_middle}?",
            lambda: f"Qua nhóm nhạc {clean_middle}, nghệ sĩ {start} đã phát hành album nào?",
            lambda: f"Album nào của nhóm nhạc {clean_middle} có sự tham gia của nghệ sĩ {start}?",
        ],
        
        # Nghệ sĩ liên quan đến thể loại qua nhóm
        ('MEMBER_OF', 'IS_GENRE'): [
            lambda: f"Thể loại nhạc nào mà nghệ sĩ {start} liên quan đến thông qua nhóm nhạc {clean_middle}?",
            lambda: f"Qua nhóm nhạc {clean_middle}, nghệ sĩ {start} liên quan đến thể loại nhạc nào?",
            lambda: f"Nhóm nhạc {clean_middle} mà {start} là thành viên thuộc thể loại nhạc nào?",
        ],
        
        # Nghệ sĩ liên quan đến công ty qua nhóm
        ('MEMBER_OF', 'MANAGED_BY'): [
            lambda: f"Công ty nào quản lý nghệ sĩ {start} thông qua nhóm nhạc {clean_middle}?",
            lambda: f"Qua nhóm nhạc {clean_middle}, nghệ sĩ {start} được quản lý bởi công ty nào?",
            lambda: f"Công ty nào quản lý nhóm nhạc {clean_middle} mà {start} là thành viên?",
        ],
        
        # Nghệ sĩ cùng nghề nghiệp
        ('HAS_OCCUPATION', 'HAS_OCCUPATION'): [
            lambda: f"Nghệ sĩ nào khác cũng có nghề nghiệp {clean_middle} giống như {start}?",
            lambda: f"Ai là nghệ sĩ khác cũng làm nghề {clean_middle} như {start}?",
            lambda: f"Cùng nghề nghiệp {clean_middle}, nghệ sĩ nào khác ngoài {start}?",
        ],
        
        # Bài hát trong album của nhóm
        ('RELEASED', 'CONTAINS'): [
            lambda: f"Bài hát nào được chứa trong album {clean_middle} mà nhóm nhạc {start} phát hành?",
            lambda: f"Album {clean_middle} của nhóm nhạc {start} chứa bài hát nào?",
            lambda: f"Bài hát nào nằm trong album {clean_middle} do nhóm nhạc {start} phát hành?",
        ],
        
        # Album chứa bài hát -> nhóm trình bày
        ('CONTAINS', 'SINGS'): [
            lambda: f"Nhóm nhạc nào trình bày bài hát {clean_middle} trong album {start}?",
            lambda: f"Bài hát {clean_middle} trong album {start} được trình bày bởi nhóm nhạc nào?",
            lambda: f"Ai là nhóm nhạc trình bày bài hát {clean_middle} từ album {start}?",
        ],
    }
    
    # Tạo câu hỏi dựa trên pattern
    key = (edge1_type, edge2_type)
    if key in question_variants:
        # Sử dụng hash của path để chọn biến thể câu hỏi nhất quán (không ngẫu nhiên)
        variants = question_variants[key]
        path_hash = hash((start, middle, end))
        variant_idx = abs(path_hash) % len(variants)
        question = variants[variant_idx]()
    else:
        # Tạo câu hỏi tổng quát dựa trên labels
        if start_label == 'Artist' and middle_label == 'Group' and end_label == 'Song':
            question = f"Bài hát nào mà nghệ sĩ {start} trình bày thông qua nhóm nhạc {clean_middle}?"
        elif start_label == 'Group' and middle_label == 'Company' and end_label == 'Group':
            question = f"Nhóm nhạc nào khác cũng được quản lý bởi {clean_middle} giống như {start}?"
        elif start_label == 'Group' and middle_label == 'Genre' and end_label == 'Group':
            question = f"Nhóm nhạc nào khác cũng thuộc thể loại {clean_middle} giống như {start}?"
        elif start_label == 'Artist' and middle_label == 'Group' and end_label == 'Artist':
            question = f"Nghệ sĩ nào khác cũng là thành viên của nhóm nhạc {clean_middle} giống như {start}?"
        elif start_label == 'Album' and middle_label == 'Song' and end_label == 'Group':
            question = f"Nhóm nhạc nào trình bày bài hát {clean_middle} trong album {start}?"
        elif start_label == 'Group' and middle_label == 'Album' and end_label == 'Song':
            question = f"Bài hát nào được chứa trong album {clean_middle} mà nhóm nhạc {start} phát hành?"
        else:
            # Tạo câu hỏi dựa trên quan hệ cụ thể
            if edge1_type == 'CONTAINS' and edge2_type == 'SINGS':
                question = f"Nhóm nhạc nào trình bày bài hát {clean_middle} trong album {start}?"
            elif edge1_type == 'RELEASED' and edge2_type == 'CONTAINS':
                question = f"Bài hát nào được chứa trong album {clean_middle} mà nhóm nhạc {start} phát hành?"
            else:
                question = f"Mối quan hệ giữa {start} và {clean_end} thông qua {clean_middle} là gì?"
    
    return question

def format_path_description(path: Dict[str, Any], nodes: Dict[str, Any]) -> str:
    """Định dạng mô tả đường đi"""
    start = path['start_node']
    middle = path['middle_node']
    end = path['end_node']
    
    edge1 = path['edge1']
    edge2 = path['edge2']
    
    start_label = path['start_label']
    middle_label = path['middle_label']
    end_label = path['end_label']
    
    description = f"""
Đường đi two-hop:
1. {start} [{start_label}]
   └─[{edge1['type']}]→ {edge1.get('text', '')}
2. {middle} [{middle_label}]
   └─[{edge2['type']}]→ {edge2.get('text', '')}
3. {end} [{end_label}]
"""
    return description.strip()

def main():
    import sys
    import io
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Load dữ liệu
    json_file = 'korean_artists_graph_bfs.json'
    print(f"Đang load dữ liệu từ {json_file}...")
    nodes, edges = load_graph(json_file)
    print(f"✓ Đã load {len(nodes)} nodes và {len(edges)} edges")
    
    # Tìm tất cả two-hop paths
    print("\nĐang tìm các đường đi two-hop...")
    all_paths = find_two_hop_paths(nodes, edges)
    print(f"✓ Tìm thấy {len(all_paths)} đường đi two-hop")
    
    # Lọc và chọn 20 paths đa dạng
    print("\nĐang chọn 20 paths đa dạng...")
    
    # Nhóm theo loại quan hệ để đảm bảo đa dạng
    paths_by_type = defaultdict(list)
    for path in all_paths:
        key = (path['edge1']['type'], path['edge2']['type'])
        paths_by_type[key].append(path)
    
    # Chọn paths từ mỗi nhóm
    selected_paths = []
    type_keys = list(paths_by_type.keys())
    random.shuffle(type_keys)
    
    # Đánh giá độ thú vị của từng path
    def score_path(path):
        """Đánh giá độ thú vị của path (cao hơn = thú vị hơn)"""
        edge1_type = path['edge1']['type']
        edge2_type = path['edge2']['type']
        start_label = path['start_label']
        middle_label = path['middle_label']
        end_label = path['end_label']
        
        score = 0
        
        # Các quan hệ rất thú vị (điểm cao)
        if (edge1_type, edge2_type) == ('MEMBER_OF', 'MEMBER_OF') and start_label == 'Artist' and end_label == 'Artist':
            score += 100  # Nghệ sĩ cùng nhóm
        elif (edge1_type, edge2_type) == ('MANAGED_BY', 'MANAGED_BY') and start_label == 'Group' and end_label == 'Group':
            score += 90  # Nhóm cùng công ty
        elif (edge1_type, edge2_type) == ('IS_GENRE', 'IS_GENRE') and start_label == 'Group' and end_label == 'Group':
            score += 85  # Nhóm cùng thể loại
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'SINGS') and start_label == 'Artist' and end_label == 'Song':
            score += 80  # Nghệ sĩ trình bày bài hát qua nhóm
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'RELEASED') and start_label == 'Artist' and end_label == 'Album':
            score += 75  # Nghệ sĩ phát hành album qua nhóm
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'IS_GENRE') and start_label == 'Artist' and end_label == 'Genre':
            score += 70  # Nghệ sĩ liên quan đến thể loại qua nhóm
        elif (edge1_type, edge2_type) == ('MEMBER_OF', 'MANAGED_BY') and start_label == 'Artist' and end_label == 'Company':
            score += 65  # Nghệ sĩ liên quan đến công ty qua nhóm
        elif (edge1_type, edge2_type) == ('HAS_OCCUPATION', 'HAS_OCCUPATION') and start_label == 'Artist' and end_label == 'Artist':
            score += 60  # Nghệ sĩ cùng nghề nghiệp
        elif (edge1_type, edge2_type) == ('RELEASED', 'CONTAINS') and start_label == 'Group' and end_label == 'Song':
            score += 55  # Bài hát trong album của nhóm
        elif (edge1_type, edge2_type) == ('SINGS', 'SINGS') and start_label == 'Song' and end_label == 'Song':
            score += 50  # Bài hát cùng nhóm trình bày
        
        # Trừ điểm cho các quan hệ không thú vị
        if edge1_type == 'PRODUCED_ALBUM' and edge2_type == 'CONTAINS':
            score -= 30  # Producer -> Album -> Song (quá nhiều, không thú vị)
        if start_label == end_label and start_label in ['Genre', 'Company', 'Occupation']:
            score -= 20  # Tránh Genre -> X -> Genre
        
        return score
    
    # Sắp xếp paths theo điểm số
    all_paths.sort(key=score_path, reverse=True)
    
    # Ưu tiên các loại quan hệ phổ biến và có ý nghĩa
    priority_types = [
        ('MEMBER_OF', 'MEMBER_OF'),  # Nghệ sĩ cùng nhóm
        ('MANAGED_BY', 'MANAGED_BY'),  # Nhóm cùng công ty
        ('IS_GENRE', 'IS_GENRE'),  # Nhóm cùng thể loại
        ('MEMBER_OF', 'SINGS'),  # Nghệ sĩ -> nhóm -> bài hát
        ('MEMBER_OF', 'RELEASED'),  # Nghệ sĩ -> nhóm -> album
        ('MEMBER_OF', 'IS_GENRE'),  # Nghệ sĩ -> nhóm -> thể loại
        ('MEMBER_OF', 'MANAGED_BY'),  # Nghệ sĩ -> nhóm -> công ty
        ('HAS_OCCUPATION', 'HAS_OCCUPATION'),  # Nghệ sĩ cùng nghề nghiệp
        ('RELEASED', 'CONTAINS'),  # Nhóm -> album -> bài hát
    ]
    
    # Lọc bỏ các paths không thú vị
    filtered_paths = []
    seen_combinations = set()
    
    for path in all_paths:
        edge1_type = path['edge1']['type']
        edge2_type = path['edge2']['type']
        
        # Bỏ qua các quan hệ không thú vị
        if edge1_type == 'PRODUCED_ALBUM':
            continue  # Loại bỏ hoàn toàn PRODUCED_ALBUM (không thú vị)
        
        # Tránh trùng lặp
        combo = (path['start_node'], path['end_node'], edge1_type, edge2_type)
        if combo in seen_combinations:
            continue
        seen_combinations.add(combo)
        
        filtered_paths.append(path)
    
    # Sắp xếp lại theo điểm số
    filtered_paths.sort(key=score_path, reverse=True)
    
    # Chọn 20 paths tốt nhất, đảm bảo đa dạng và không trùng lặp câu hỏi
    selected_paths = []
    type_counts = defaultdict(int)
    max_per_type = 2  # Tối đa 2 paths mỗi loại để đa dạng hơn
    seen_question_patterns = set()  # Lưu pattern câu hỏi để tránh trùng lặp
    seen_start_middle_pairs = set()  # Tránh trùng lặp cặp (start, middle)
    
    for path in filtered_paths:
        if len(selected_paths) >= 20:
            break
        
        key = (path['edge1']['type'], path['edge2']['type'])
        start_node = path['start_node']
        middle_node = path['middle_node']
        
        # Kiểm tra xem đã có đủ paths loại này chưa
        if type_counts[key] >= max_per_type:
            continue
        
        # Kiểm tra trùng lặp cặp (start, middle) - mỗi cặp chỉ xuất hiện 1 lần
        start_middle_pair = (start_node, middle_node)
        if start_middle_pair in seen_start_middle_pairs:
            continue
        
        # Kiểm tra không trùng lặp hoàn toàn
        is_duplicate = False
        for existing in selected_paths:
            if (existing['start_node'] == path['start_node'] and 
                existing['end_node'] == path['end_node'] and
                existing['middle_node'] == path['middle_node']):
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Tạo pattern câu hỏi để kiểm tra trùng lặp
            question_pattern = (key, start_node, middle_node)
            if question_pattern not in seen_question_patterns:
                selected_paths.append(path)
                type_counts[key] += 1
                seen_start_middle_pairs.add(start_middle_pair)
                seen_question_patterns.add(question_pattern)
    
    # Nếu chưa đủ 20, thêm từ các paths còn lại (không giới hạn loại)
    if len(selected_paths) < 20:
        for path in filtered_paths:
            if len(selected_paths) >= 20:
                break
            if path not in selected_paths:
                selected_paths.append(path)
    
    selected_paths = selected_paths[:20]
    print(f"✓ Đã chọn {len(selected_paths)} paths")
    
    # Tạo output
    output = []
    for i, path in enumerate(selected_paths, 1):
        question = generate_question(path, nodes)
        description = format_path_description(path, nodes)
        
        output.append({
            'id': i,
            'question': question,
            'path': {
                'start': {
                    'id': path['start_node'],
                    'label': path['start_label'],
                    'name': path['start_node']
                },
                'middle': {
                    'id': path['middle_node'],
                    'label': path['middle_label'],
                    'name': path['middle_node']
                },
                'end': {
                    'id': path['end_node'],
                    'label': path['end_label'],
                    'name': path['end_node']
                }
            },
            'edges': [
                {
                    'source': path['edge1']['source'],
                    'target': path['edge1']['target'],
                    'type': path['edge1']['type'],
                    'text': path['edge1'].get('text', '')
                },
                {
                    'source': path['edge2']['source'],
                    'target': path['edge2']['target'],
                    'type': path['edge2']['type'],
                    'text': path['edge2'].get('text', '')
                }
            ],
            'description': description
        })
    
    # Lưu vào file JSON
    output_file = 'two_hop_data_20.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu vào {output_file}")
    
    # In ra console
    print("\n" + "="*80)
    print("20 DỮ LIỆU TWO-HOP VỚI CÂU HỎI")
    print("="*80)
    
    for item in output:
        print(f"\n[{item['id']}] {item['question']}")
        print(item['description'])
        print("-" * 80)
    
    # Tạo file markdown
    md_file = 'two_hop_data_20.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 20 Dữ Liệu Two-Hop từ Dataset Korean Music\n\n")
        f.write("Tài liệu này chứa 20 dữ liệu two-hop (đường đi 2 bước) được tạo từ dataset korean_music_bfs.\n\n")
        f.write("---\n\n")
        
        for item in output:
            f.write(f"## {item['id']}. {item['question']}\n\n")
            f.write("### Đường đi:\n\n")
            f.write(f"**Bước 1:** {item['path']['start']['name']} [{item['path']['start']['label']}] ")
            f.write(f"─[{item['edges'][0]['type']}]→ {item['path']['middle']['name']} [{item['path']['middle']['label']}]\n\n")
            f.write(f"*{item['edges'][0]['text']}*\n\n")
            f.write(f"**Bước 2:** {item['path']['middle']['name']} [{item['path']['middle']['label']}] ")
            f.write(f"─[{item['edges'][1]['type']}]→ {item['path']['end']['name']} [{item['path']['end']['label']}]\n\n")
            f.write(f"*{item['edges'][1]['text']}*\n\n")
            f.write("### Dữ liệu JSON:\n\n")
            f.write("```json\n")
            f.write(json.dumps({
                'question': item['question'],
                'path': item['path'],
                'edges': item['edges']
            }, ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            f.write("---\n\n")
    
    print(f"\n✓ Đã tạo file markdown: {md_file}")

if __name__ == '__main__':
    main()

