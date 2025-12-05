"""
Script để merge dữ liệu từ 3 file JSON và đẩy vào Neo4j
- korean_artists_graph_bfs.json: nodes và edges từ BFS crawl
- kpop_ner_result.json: entities từ NER
- kpop_relationships_result.json: relationships từ relationship extraction
"""
import sys
import io
import json
from typing import Dict, List, Any, Set
from datetime import datetime
from neo4j import GraphDatabase

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


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
    
    for entity in ner_entities:
        entity_id = entity.get("text", "")
        entity_type = entity.get("type", "Entity")
        
        if not entity_id:
            continue
        
        # Nếu node chưa tồn tại, tạo mới
        if entity_id not in merged["nodes"]:
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
            new_entities_count += 1
        else:
            # Cập nhật properties nếu node đã tồn tại
            existing_node = merged["nodes"][entity_id]
            if "properties" not in existing_node:
                existing_node["properties"] = {}
            existing_node["properties"].update({
                "ner_method": entity.get("method", "unknown"),
                "ner_confidence": entity.get("confidence", 0.0),
                "ner_source_node": entity.get("source_node", ""),
                "ner_sources": entity.get("sources", [])
            })
            existing_entities_count += 1
    
    print(f"  ✓ Đã thêm {new_entities_count} entities mới")
    print(f"  ✓ Đã cập nhật {existing_entities_count} entities đã tồn tại")
    
    # 4. Thêm relationships từ relationship extraction
    print("\n📊 Bước 4: Thêm relationships từ relationship extraction...")
    relationships = relationships_data.get("relationships", [])
    new_relationships_count = 0
    duplicate_relationships_count = 0
    
    # Tạo set để check duplicate
    existing_edges_set: Set[tuple] = set()
    for edge in merged["edges"]:
        key = (
            edge.get("source", ""),
            edge.get("target", ""),
            edge.get("type", "")
        )
        existing_edges_set.add(key)
    
    for rel in relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "")
        
        if not source or not target or not rel_type:
            continue
        
        # Check duplicate
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
                "source_type": rel.get("source_type", ""),
                "target_type": rel.get("target_type", "")
            }
        })
        existing_edges_set.add(key)
        new_relationships_count += 1
    
    print(f"  ✓ Đã thêm {new_relationships_count} relationships mới")
    print(f"  ✓ Đã bỏ qua {duplicate_relationships_count} relationships trùng lặp")
    
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
    parser.add_argument("--bfs-file", default="korean_artists_graph_bfs.json",
                        help="File BFS graph JSON")
    parser.add_argument("--ner-file", default="kpop_ner_result.json",
                        help="File NER result JSON")
    parser.add_argument("--relationships-file", default="kpop_relationships_result.json",
                        help="File relationships result JSON")
    parser.add_argument("--output-file", default="merged_kpop_data.json",
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

