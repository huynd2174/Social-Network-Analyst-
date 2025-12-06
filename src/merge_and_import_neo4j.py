"""
Script để merge dữ liệu từ 3 file JSON và đẩy vào Neo4j
- korean_artists_graph_bfs.json: nodes và edges từ BFS crawl
- kpop_ner_result.json: entities từ NER
- kpop_relationships_result.json: relationships từ relationship extraction
"""
import sys
import io
import json
import re
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict
from neo4j import GraphDatabase

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


def normalize_node_name(name: str) -> str:
    """
    Chuẩn hóa tên node để so sánh nhất quán.
    Giống với normalize_node_name trong run_relationship_extraction.py
    """
    if not name:
        return ""
    
    # Loại bỏ các pattern trong ngoặc đơn ở cuối
    name = re.sub(r'\s*\([^)]*(?:ca sĩ|nhóm nhạc|ban nhạc|nghệ sĩ|singer|group|band)[^)]*\)\s*$', '', name, flags=re.IGNORECASE)
    
    # Chuẩn hóa khoảng trắng
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    
    return name


def normalize_for_comparison(name: str) -> str:
    """
    Chuẩn hóa tên để so sánh (loại bỏ khoảng trắng, dấu gạch nối, lowercase)
    Dùng để match entities giữa NER và Relationships
    
    Xử lý các trường hợp:
    - "Ahn Ji-young" vs "Ahn Ji young" -> cùng một node
    - "Miyeon" vs "Miyeon (ca sĩ)" -> cùng một node
    """
    normalized = normalize_node_name(name)
    # Loại bỏ khoảng trắng, dấu gạch nối, và lowercase để so sánh
    # Điều này giúp match "Ahn Ji-young" với "Ahn Ji young"
    normalized = normalized.lower().replace(' ', '').replace('-', '').replace('_', '')
    return normalized


def load_json_file(filepath: str) -> Dict:
    """Đọc file JSON"""
    print(f"📖 Đang đọc {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✓ Đã đọc {filepath}")
    return data


def merge_data(
    bfs_data: Dict,
    ner_data: Dict,
    relationships_data: Dict,
    output_file: str = "merged_kpop_data.json"
) -> Dict:
    """
    Merge dữ liệu từ 3 file vào 1 file duy nhất
    
    Args:
        bfs_data: Dữ liệu từ korean_artists_graph_bfs.json
        ner_data: Dữ liệu từ kpop_ner_result.json
        relationships_data: Dữ liệu từ kpop_relationships_result.json
        output_file: File output
    
    Returns:
        Dict chứa merged data
    """
    print("\n" + "=" * 70)
    print("MERGE DỮ LIỆU TỪ 3 FILE JSON")
    print("=" * 70)
    
    # Khởi tạo merged data
    merged = {
        "metadata": {
            "merged_at": datetime.now().isoformat(),
            "source_files": [
                "korean_artists_graph_bfs.json",
                "kpop_ner_result.json",
                "kpop_relationships_result.json"
            ]
        },
        "nodes": {},
        "edges": []
    }
    
    # 1. Thêm nodes từ BFS graph
    print("\n📊 Bước 1: Thêm nodes từ BFS graph...")
    bfs_nodes = bfs_data.get("nodes", {})
    merged["nodes"].update(bfs_nodes)
    print(f"  ✓ Đã thêm {len(bfs_nodes)} nodes từ BFS graph")
    
    # 2. Thêm edges từ BFS graph
    print("\n📊 Bước 2: Thêm edges từ BFS graph...")
    bfs_edges = bfs_data.get("edges", [])
    merged["edges"].extend(bfs_edges)
    print(f"  ✓ Đã thêm {len(bfs_edges)} edges từ BFS graph")
    
    # 3. Thêm entities từ NER (tạo nodes mới nếu chưa có)
    print("\n📊 Bước 3: Thêm entities từ NER...")
    ner_entities = ner_data.get("entities", [])
    new_entities_count = 0
    existing_entities_count = 0
    
    # Tạo mapping từ (normalized name, type) -> original name để match chính xác
    # QUAN TRỌNG: Phải check cả type để tránh trùng lặp giữa các type khác nhau
    # QUAN TRỌNG: Lưu thông tin node nào từ BFS để ưu tiên giữ lại
    normalized_to_original: Dict[Tuple[str, str], str] = {}
    bfs_node_ids = set(bfs_nodes.keys())  # Lưu danh sách node IDs từ BFS
    for node_id in merged["nodes"].keys():
        normalized_key = normalize_for_comparison(node_id)
        node_type = merged["nodes"][node_id].get("label", "Entity")
        key = (normalized_key, node_type)
        if key not in normalized_to_original:
            normalized_to_original[key] = node_id
    
    for entity in ner_entities:
        entity_id = entity.get("text", "")
        entity_type = entity.get("type", "Entity")
        
        if not entity_id:
            continue
        
        # Chuẩn hóa để tìm node đã tồn tại
        normalized_key = normalize_for_comparison(entity_id)
        key = (normalized_key, entity_type)
        existing_node_id = normalized_to_original.get(key)
        
        if existing_node_id:
            # Node đã tồn tại VÀ CÙNG TYPE (có thể tên khác một chút do chuẩn hóa)
            existing_node = merged["nodes"][existing_node_id]
            existing_node_type = existing_node.get("label", "Entity")
            
            # Double check: phải cùng type mới cập nhật
            if existing_node_type == entity_type:
                # QUAN TRỌNG: Nếu node từ BFS graph (có infobox), chỉ cập nhật properties, KHÔNG ghi đè
                if existing_node_id in bfs_node_ids:
                    # Node từ BFS -> giữ nguyên, chỉ thêm NER properties
                    if "properties" not in existing_node:
                        existing_node["properties"] = {}
                    existing_node["properties"].update({
                        "ner_method": entity.get("method", "unknown"),
                        "ner_confidence": entity.get("confidence", 0.0),
                        "ner_source_node": entity.get("source_node", ""),
                        "ner_sources": entity.get("sources", [])
                    })
                else:
                    # Node không phải từ BFS -> cập nhật như bình thường
                    if "properties" not in existing_node:
                        existing_node["properties"] = {}
                    existing_node["properties"].update({
                        "ner_method": entity.get("method", "unknown"),
                        "ner_confidence": entity.get("confidence", 0.0),
                        "ner_source_node": entity.get("source_node", ""),
                        "ner_sources": entity.get("sources", [])
                    })
                existing_entities_count += 1
            else:
                # Tên giống nhưng type khác -> cho phép cả hai cùng tồn tại
                # Kiểm tra xem có node nào với cùng tên (entity_id) nhưng type khác không
                if entity_id in merged["nodes"]:
                    # Đã có node với tên này (có thể type khác) -> tạo node mới với tên khác
                    new_entity_id = f"{entity_id} ({entity_type})"
                    # Kiểm tra xem tên mới đã tồn tại chưa
                    if new_entity_id in merged["nodes"]:
                        # Tên đã tồn tại -> bỏ qua
                        continue
                    # Tạo node mới với tên khác
                    merged["nodes"][new_entity_id] = {
                        "label": entity_type,
                        "title": new_entity_id,
                        "properties": {
                            "method": entity.get("method", "unknown"),
                            "confidence": entity.get("confidence", 0.0),
                            "source_node": entity.get("source_node", ""),
                            "sources": entity.get("sources", []),
                            "original_name": entity_id  # Lưu tên gốc để reference
                        }
                    }
                    normalized_to_original[key] = new_entity_id
                    new_entities_count += 1
                else:
                    # Chưa có node với tên này -> tạo node mới với tên gốc
                    merged["nodes"][entity_id] = {
                        "label": entity_type,
                        "title": entity_id,
                        "properties": {
                            "method": entity.get("method", "unknown"),
                            "confidence": entity.get("confidence", 0.0),
                            "source_node": entity.get("source_node", ""),
                            "sources": entity.get("sources", [])
                        }
                    }
                    normalized_to_original[key] = entity_id
                    new_entities_count += 1
        else:
            # Node chưa tồn tại với key (normalized_name, type) này
            # Kiểm tra xem có node nào với cùng tên (entity_id) nhưng type khác không
            if entity_id in merged["nodes"]:
                # Đã có node với tên này (có thể type khác) -> tạo node mới với tên khác
                new_entity_id = f"{entity_id} ({entity_type})"
                # Kiểm tra xem tên mới đã tồn tại chưa
                if new_entity_id in merged["nodes"]:
                    # Tên đã tồn tại -> bỏ qua
                    continue
                # Tạo node mới với tên khác
                merged["nodes"][new_entity_id] = {
                    "label": entity_type,
                    "title": new_entity_id,
                    "properties": {
                        "method": entity.get("method", "unknown"),
                        "confidence": entity.get("confidence", 0.0),
                        "source_node": entity.get("source_node", ""),
                        "sources": entity.get("sources", []),
                        "original_name": entity_id  # Lưu tên gốc để reference
                    }
                }
                normalized_to_original[key] = new_entity_id
                new_entities_count += 1
            else:
                # Chưa có node với tên này -> tạo node mới với tên gốc
                merged["nodes"][entity_id] = {
                    "label": entity_type,
                    "title": entity_id,
                    "properties": {
                        "method": entity.get("method", "unknown"),
                        "confidence": entity.get("confidence", 0.0),
                        "source_node": entity.get("source_node", ""),
                        "sources": entity.get("sources", [])
                    }
                }
                normalized_to_original[key] = entity_id
                new_entities_count += 1
    
    print(f"  ✓ Đã thêm {new_entities_count} entities mới")
    print(f"  ✓ Đã cập nhật {existing_entities_count} entities đã tồn tại")
    
    # 4. Thêm relationships từ relationship extraction
    print("\n📊 Bước 4: Thêm relationships từ relationship extraction...")
    relationships = relationships_data.get("relationships", [])
    new_relationships_count = 0
    duplicate_relationships_count = 0
    skipped_missing_nodes = 0
    
    # Tạo set để check duplicate
    existing_edges_set: Set[tuple] = set()
    for edge in merged["edges"]:
        key = (
            edge.get("source", ""),
            edge.get("target", ""),
            edge.get("type", "")
        )
        existing_edges_set.add(key)
    
    # Tạo mapping normalized -> original cho tất cả nodes hiện có
    # Dùng để tìm node đã tồn tại dựa trên normalized name (check trùng lặp tốt hơn)
    # QUAN TRỌNG: Tạo mapping SAU KHI đã merge tất cả nodes (BFS + NER)
    normalized_to_original_rel: Dict[str, str] = {}
    # Tạo mapping với tất cả các biến thể tên có thể có
    # VÀ tạo mapping từ các phần tên (để match "Miyeon" với "Cho Mi-yeon")
    name_parts_to_nodes: Dict[str, List[str]] = defaultdict(list)
    
    for node_id in merged["nodes"].keys():
        normalized_key = normalize_for_comparison(node_id)
        # Nếu có nhiều node cùng normalized name, ưu tiên node từ BFS graph (tên gốc)
        if normalized_key not in normalized_to_original_rel:
            normalized_to_original_rel[normalized_key] = node_id
        else:
            # Ưu tiên node từ BFS graph nếu có
            existing_node_id = normalized_to_original_rel[normalized_key]
            if node_id in bfs_nodes and existing_node_id not in bfs_nodes:
                normalized_to_original_rel[normalized_key] = node_id
        
        # Tạo mapping từ các phần tên (tách theo khoảng trắng và dấu gạch nối)
        # Ví dụ: "Cho Mi-yeon" -> ["cho", "mi", "yeon"]
        # Để match "Miyeon" -> "miyeon" với "Cho Mi-yeon" -> "chomiyeon"
        name_parts = re.split(r'[\s\-_]+', normalized_key)
        for part in name_parts:
            if len(part) >= 3:  # Chỉ lưu các phần có độ dài >= 3
                name_parts_to_nodes[part].append(node_id)
        
        # THÊM: Lưu toàn bộ normalized name (nếu đủ dài) để tìm substring trực tiếp
        # Ví dụ: "chomiyeon" sẽ match với "miyeon" nếu "miyeon" được tìm trong name_parts_to_nodes
        if len(normalized_key) >= 3:
            name_parts_to_nodes[normalized_key].append(node_id)
    
    # Thống kê relationships bị bỏ qua để debug
    missing_source_stats = defaultdict(int)
    missing_target_stats = defaultdict(int)
    
    for rel in relationships:
        source_original = rel.get("source", "")
        target_original = rel.get("target", "")
        rel_type = rel.get("type", "")
        source_type = rel.get("source_type", "")
        target_type = rel.get("target_type", "")
        
        if not source_original or not target_original or not rel_type:
            continue
        
        # Chuẩn hóa để tìm node đã tồn tại (có thể tên khác một chút do chuẩn hóa)
        source_normalized = normalize_for_comparison(source_original)
        target_normalized = normalize_for_comparison(target_original)
        
        source_node_id = normalized_to_original_rel.get(source_normalized)
        target_node_id = normalized_to_original_rel.get(target_normalized)
        
        # Nếu không tìm thấy bằng normalized name, thử tìm bằng name parts
        # Ví dụ: "Miyeon" không khớp với "Cho Mi-yeon", nhưng "miyeon" có trong "chomiyeon"
        if not source_node_id:
            candidates = []
            
            # CÁCH 1: Tìm trực tiếp source_normalized trong name_parts_to_nodes
            # (nếu source_normalized là substring của một normalized name)
            if source_normalized in name_parts_to_nodes:
                candidates.extend(name_parts_to_nodes[source_normalized])
            
            # CÁCH 2: Tách source thành các phần và tìm
            source_parts = re.split(r'[\s\-_]+', source_normalized)
            for part in source_parts:
                if len(part) >= 3 and part in name_parts_to_nodes:
                    candidates.extend(name_parts_to_nodes[part])
            
            # CÁCH 3: Tìm trong tất cả nodes xem có node nào có normalized name chứa source_normalized
            if not candidates:
                for node_id in merged["nodes"].keys():
                    candidate_norm = normalize_for_comparison(node_id)
                    if source_normalized in candidate_norm:
                        candidate_type = merged["nodes"][node_id].get("label", "")
                        if not source_type or candidate_type == source_type:
                            candidates.append(node_id)
            
            # Loại bỏ duplicate candidates
            candidates = list(set(candidates))
            
            # Tìm node tốt nhất: node có normalized name chứa source hoặc ngược lại
            best_match = None
            best_score = 0
            
            for candidate in candidates:
                candidate_norm = normalize_for_comparison(candidate)
                candidate_type = merged["nodes"][candidate].get("label", "")
                
                # Kiểm tra type có khớp không
                if source_type and candidate_type != source_type:
                    continue
                
                # Tính điểm match:
                # - Nếu source là substring của candidate: điểm cao
                # - Nếu candidate là substring của source: điểm thấp hơn
                # - Nếu có nhiều phần khớp: điểm cao hơn
                score = 0
                if source_normalized in candidate_norm:
                    # Source là substring của candidate (ví dụ: "miyeon" trong "chomiyeon")
                    score = 100 + len(source_normalized)
                elif candidate_norm in source_normalized:
                    # Candidate là substring của source (ít phổ biến hơn)
                    score = 50 + len(candidate_norm)
                else:
                    # Đếm số phần khớp
                    matching_parts = sum(1 for part in source_parts if part in candidate_norm)
                    if matching_parts > 0:
                        score = matching_parts * 10
                
                if score > best_score:
                    best_score = score
                    best_match = candidate
            
            if best_match:
                source_node_id = best_match
        
        if not target_node_id:
            # Tương tự cho target
            candidates = []
            
            # CÁCH 1: Tìm trực tiếp target_normalized trong name_parts_to_nodes
            if target_normalized in name_parts_to_nodes:
                candidates.extend(name_parts_to_nodes[target_normalized])
            
            # CÁCH 2: Tách target thành các phần và tìm
            target_parts = re.split(r'[\s\-_]+', target_normalized)
            for part in target_parts:
                if len(part) >= 3 and part in name_parts_to_nodes:
                    candidates.extend(name_parts_to_nodes[part])
            
            # CÁCH 3: Tìm trong tất cả nodes
            if not candidates:
                for node_id in merged["nodes"].keys():
                    candidate_norm = normalize_for_comparison(node_id)
                    if target_normalized in candidate_norm:
                        candidate_type = merged["nodes"][node_id].get("label", "")
                        if not target_type or candidate_type == target_type:
                            candidates.append(node_id)
            
            # Loại bỏ duplicate candidates
            candidates = list(set(candidates))
            
            best_match = None
            best_score = 0
            
            for candidate in candidates:
                candidate_norm = normalize_for_comparison(candidate)
                candidate_type = merged["nodes"][candidate].get("label", "")
                
                # Kiểm tra type có khớp không
                if target_type and candidate_type != target_type:
                    continue
                
                # Tính điểm match tương tự như source
                score = 0
                if target_normalized in candidate_norm:
                    score = 100 + len(target_normalized)
                elif candidate_norm in target_normalized:
                    score = 50 + len(candidate_norm)
                else:
                    matching_parts = sum(1 for part in target_parts if part in candidate_norm)
                    if matching_parts > 0:
                        score = matching_parts * 10
                
                if score > best_score:
                    best_score = score
                    best_match = candidate
            
            if best_match:
                target_node_id = best_match
        
        # CHỈ thêm relationship nếu CẢ HAI nodes đều tồn tại
        # KHÔNG tạo node mới - chỉ dùng nodes đã có sẵn
        if not source_node_id:
            skipped_missing_nodes += 1
            missing_source_stats[source_type] += 1
            continue
        
        if not target_node_id:
            skipped_missing_nodes += 1
            missing_target_stats[target_type] += 1
            continue
        
        # Dùng tên node gốc đã tồn tại
        source = source_node_id
        target = target_node_id
        
        # Check duplicate (dùng tên gốc đã tồn tại)
        key = (source, target, rel_type)
        if key in existing_edges_set:
            duplicate_relationships_count += 1
            continue
        
        # Thêm relationship mới
        merged["edges"].append({
            "source": source,
            "target": target,
            "type": rel_type,
            "text": f"{source} {rel_type} {target}",
            "properties": {
                "confidence": rel.get("confidence", 0.0),
                "method": rel.get("method", "unknown"),
                "source_type": source_type,
                "target_type": target_type
            }
        })
        existing_edges_set.add(key)
        new_relationships_count += 1
    
    print(f"  ✓ Đã thêm {new_relationships_count} relationships mới")
    print(f"  ✓ Đã bỏ qua {duplicate_relationships_count} relationships trùng lặp")
    print(f"  ✓ Đã bỏ qua {skipped_missing_nodes} relationships (thiếu source/target node)")
    
    if skipped_missing_nodes > 0:
        print(f"\n  📊 Thống kê relationships bị bỏ qua:")
        if missing_source_stats:
            print(f"    - Thiếu source node:")
            for node_type, count in sorted(missing_source_stats.items(), key=lambda x: -x[1]):
                print(f"      • {node_type}: {count}")
        if missing_target_stats:
            print(f"    - Thiếu target node:")
            for node_type, count in sorted(missing_target_stats.items(), key=lambda x: -x[1]):
                print(f"      • {node_type}: {count}")
        print(f"\n  💡 Lưu ý: Các relationships này bị bỏ qua vì source/target node không tồn tại.")
        print(f"     Đảm bảo các Artist từ infobox đã được thêm vào NER result trước khi merge.")
    
    # 5. Cập nhật metadata
    merged["metadata"]["total_nodes"] = len(merged["nodes"])
    merged["metadata"]["total_edges"] = len(merged["edges"])
    merged["metadata"]["nodes_by_type"] = {}
    merged["metadata"]["edges_by_type"] = {}
    
    # Đếm nodes theo type
    for node_id, node_data in merged["nodes"].items():
        label = node_data.get("label", "Entity")
        merged["metadata"]["nodes_by_type"][label] = merged["metadata"]["nodes_by_type"].get(label, 0) + 1
    
    # Đếm edges theo type
    for edge in merged["edges"]:
        edge_type = edge.get("type", "RELATED_TO")
        merged["metadata"]["edges_by_type"][edge_type] = merged["metadata"]["edges_by_type"].get(edge_type, 0) + 1
    
    # 6. Lưu file
    print(f"\n💾 Đang lưu merged data vào {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Đã lưu merged data vào {output_file}")
    
    # In thống kê
    print("\n" + "=" * 70)
    print("THỐNG KÊ DỮ LIỆU SAU KHI MERGE")
    print("=" * 70)
    print(f"📊 Tổng số nodes: {merged['metadata']['total_nodes']}")
    print(f"📊 Tổng số edges: {merged['metadata']['total_edges']}")
    print("\n📊 Nodes theo type:")
    for label, count in sorted(merged["metadata"]["nodes_by_type"].items(), key=lambda x: -x[1]):
        print(f"  - {label}: {count}")
    print("\n📊 Edges theo type:")
    for edge_type, count in sorted(merged["metadata"]["edges_by_type"].items(), key=lambda x: -x[1]):
        print(f"  - {edge_type}: {count}")
    
    return merged


def import_to_neo4j(
    merged_data: Dict,
    uri: str,
    user: str,
    password: str,
    database: str = None,
    batch_size: int = 1000,
    create_constraints: bool = True
) -> None:
    """
    Đẩy dữ liệu merged vào Neo4j
    
    Args:
        merged_data: Dữ liệu đã merge
        uri: Neo4j URI (ví dụ: bolt://localhost:7687)
        user: Username
        password: Password
        database: Tên database (None = default)
        batch_size: Kích thước batch khi import
        create_constraints: Có tạo constraints không
    """
    print("\n" + "=" * 70)
    print("IMPORT DỮ LIỆU VÀO NEO4J")
    print("=" * 70)
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        def run_write(tx, query, parameters=None):
            return tx.run(query, parameters or {})
        
        # Chuẩn hóa key cho Neo4j property
        def _strip_diacritics_to_ascii(text: str) -> str:
            import unicodedata as _ud
            if not isinstance(text, str):
                text = str(text)
            norm = _ud.normalize('NFD', text)
            norm = norm.replace('đ', 'd').replace('Đ', 'D')
            return ''.join(ch for ch in norm if _ud.category(ch) != 'Mn')
        
        def _norm_key(s: str) -> str:
            import re as _re
            key = _strip_diacritics_to_ascii(str(s).strip().lower())
            key = _re.sub(r"[^a-z0-9]+", "_", key)
            key = _re.sub(r"_+", "_", key).strip('_')
            return key or "field"
        
        # Prepare nodes grouped by label
        print("\n📊 Bước 1: Chuẩn bị nodes...")
        label_to_nodes: Dict[str, List[Dict[str, Any]]] = {}
        nodes_data = merged_data.get("nodes", {})
        
        for node_id, node_data in nodes_data.items():
            label = node_data.get("label", "Entity")
            name = node_data.get("title") or node_id
            
            props = {
                "id": node_id,
                "name": name
            }
            
            # Thêm URL nếu có
            if "url" in node_data:
                props["url"] = node_data["url"]
            
            # Thêm properties nếu có
            if "properties" in node_data:
                for k, v in node_data["properties"].items():
                    if k not in props:
                        props[k] = v
            
            # Thêm infobox fields cho các labels chính
            if label in ("Artist", "Group", "Song", "Album", "Company", "Genre"):
                infobox = node_data.get("infobox") or {}
                if isinstance(infobox, dict) and infobox:
                    for raw_k, raw_v in infobox.items():
                        k_norm = _norm_key(raw_k)
                        if k_norm not in ("id", "name", "url"):
                            props[k_norm] = str(raw_v)
            
            label_to_nodes.setdefault(label, []).append({
                "id": node_id,
                "props": props
            })
        
        print(f"  ✓ Đã chuẩn bị {len(nodes_data)} nodes")
        for label, items in label_to_nodes.items():
            print(f"    - {label}: {len(items)} nodes")
        
        # Prepare relationships
        print("\n📊 Bước 2: Chuẩn bị relationships...")
        relationships: List[Dict[str, Any]] = []
        edges_data = merged_data.get("edges", [])
        
        for edge in edges_data:
            src = edge.get("source")
            tgt = edge.get("target")
            typ = edge.get("type") or "RELATED_TO"
            
            if not src or not tgt:
                continue
            
            rel_props = {"text": edge.get("text", "")}
            
            # Thêm properties nếu có
            if "properties" in edge:
                rel_props.update(edge["properties"])
            
            relationships.append({
                "sourceId": src,
                "targetId": tgt,
                "type": typ,
                "props": rel_props
            })
        
        print(f"  ✓ Đã chuẩn bị {len(relationships)} relationships")
        
        # Group relationships by type
        type_to_rels: Dict[str, List[Dict[str, Any]]] = {}
        for r in relationships:
            type_to_rels.setdefault(r["type"], []).append(r)
        
        for rel_type, rels in type_to_rels.items():
            print(f"    - {rel_type}: {len(rels)} relationships")
        
        # Cypher templates
        node_query_tpl = lambda label: f"""
        UNWIND $batch AS n
        MERGE (x:`{label}` {{id: n.id}})
        SET x += n.props
        """
        
        rel_query_tpl = lambda rel_type: f"""
        UNWIND $batch AS r
        MATCH (s {{id: r.sourceId}}), (t {{id: r.targetId}})
        MERGE (s)-[e:`{rel_type}`]->(t)
        SET e += r.props
        """
        
        # Open session
        print("\n📊 Bước 3: Kết nối Neo4j và import dữ liệu...")
        with driver.session(database=database) if database else driver.session() as session:
            # Tạo constraints nếu cần
            if create_constraints:
                print("  🔧 Đang tạo constraints...")
                constraints = [
                    "CREATE CONSTRAINT artist_id IF NOT EXISTS FOR (n:Artist) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT group_id IF NOT EXISTS FOR (n:Group) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT album_id IF NOT EXISTS FOR (n:Album) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT song_id IF NOT EXISTS FOR (n:Song) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT genre_id IF NOT EXISTS FOR (n:Genre) REQUIRE n.id IS UNIQUE",
                ]
                for q in constraints:
                    try:
                        session.execute_write(run_write, q)
                    except Exception as e:
                        # Constraint có thể đã tồn tại
                        pass
                print("  ✓ Đã tạo constraints")
            
            # Import nodes theo label
            print("\n📊 Bước 4: Đang import nodes...")
            total_nodes_imported = 0
            for label, items in label_to_nodes.items():
                print(f"  📥 Đang import {len(items)} {label} nodes...")
                for i in range(0, len(items), batch_size):
                    batch = items[i:i+batch_size]
                    session.execute_write(run_write, node_query_tpl(label), {"batch": batch})
                    total_nodes_imported += len(batch)
                    if (i // batch_size + 1) % 10 == 0:
                        print(f"    ✓ Đã import {total_nodes_imported} nodes...")
            print(f"  ✓ Đã import {total_nodes_imported} nodes")
            
            # Import relationships theo type
            print("\n📊 Bước 5: Đang import relationships...")
            total_rels_imported = 0
            for rel_type, rels in type_to_rels.items():
                print(f"  📥 Đang import {len(rels)} {rel_type} relationships...")
                for i in range(0, len(rels), batch_size):
                    batch = rels[i:i+batch_size]
                    session.execute_write(run_write, rel_query_tpl(rel_type), {"batch": batch})
                    total_rels_imported += len(batch)
                    if (i // batch_size + 1) % 10 == 0:
                        print(f"    ✓ Đã import {total_rels_imported} relationships...")
            print(f"  ✓ Đã import {total_rels_imported} relationships")
        
        print("\n" + "=" * 70)
        print("✓ HOÀN TẤT IMPORT VÀO NEO4J")
        print("=" * 70)
        
    finally:
        driver.close()


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge 3 file JSON và import vào Neo4j")
    parser.add_argument("--bfs-file", default="data/korean_artists_graph_bfs.json",
                        help="File BFS graph JSON")
    parser.add_argument("--ner-file", default="data/kpop_ner_result.json",
                        help="File NER result JSON")
    parser.add_argument("--relationships-file", default="data/kpop_relationships_result.json",
                        help="File relationships result JSON")
    parser.add_argument("--output-file", default="data/merged_kpop_data.json",
                        help="File output merged")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687",
                        help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j",
                        help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True,
                        help="Neo4j password")
    parser.add_argument("--neo4j-database", default=None,
                        help="Neo4j database name (None = default)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size cho import")
    parser.add_argument("--no-constraints", action="store_true",
                        help="Không tạo constraints")
    parser.add_argument("--merge-only", action="store_true",
                        help="Chỉ merge, không import vào Neo4j")
    
    args = parser.parse_args()
    
    # Load dữ liệu
    bfs_data = load_json_file(args.bfs_file)
    ner_data = load_json_file(args.ner_file)
    relationships_data = load_json_file(args.relationships_file)
    
    # Merge dữ liệu
    merged_data = merge_data(bfs_data, ner_data, relationships_data, args.output_file)
    
    # Import vào Neo4j nếu không phải merge-only
    if not args.merge_only:
        import_to_neo4j(
            merged_data,
            args.neo4j_uri,
            args.neo4j_user,
            args.neo4j_password,
            args.neo4j_database,
            args.batch_size,
            not args.no_constraints
        )


if __name__ == "__main__":
    main()

