"""
Script kiểm tra relationships nào không được tạo trong Neo4j
và tại sao (source/target node không tồn tại)
"""
import sys
import io
import json
from collections import defaultdict
from neo4j import GraphDatabase

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


def check_missing_relationships(
    merged_file: str = "data/merged_kpop_data.json",
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = None,
    database: str = None
):
    """Kiểm tra relationships nào không tồn tại trong Neo4j"""
    
    print("=" * 70)
    print("KIỂM TRA RELATIONSHIPS THIẾU TRONG NEO4J")
    print("=" * 70)
    
    # Load merged data
    print(f"\n📖 Đang đọc {merged_file}...")
    with open(merged_file, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    edges = merged_data.get("edges", [])
    print(f"✓ Đã load {len(edges)} relationships từ file")
    
    # Kết nối Neo4j
    if not password:
        password = input("Nhập Neo4j password: ")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        def get_node_exists(tx, node_id):
            """Kiểm tra node có tồn tại không"""
            result = tx.run("MATCH (n {id: $node_id}) RETURN n LIMIT 1", node_id=node_id)
            return result.single() is not None
        
        def get_relationship_exists(tx, source_id, target_id, rel_type):
            """Kiểm tra relationship có tồn tại không"""
            result = tx.run(
                "MATCH (s {id: $source_id})-[r:`" + rel_type + "`]->(t {id: $target_id}) RETURN r LIMIT 1",
                source_id=source_id,
                target_id=target_id
            )
            return result.single() is not None
        
        def get_all_node_ids(tx):
            """Lấy tất cả node IDs trong Neo4j"""
            result = tx.run("MATCH (n) RETURN n.id as id")
            return {record["id"] for record in result if record["id"]}
        
        print("\n🔍 Đang kiểm tra nodes trong Neo4j...")
        with driver.session(database=database) if database else driver.session() as session:
            # Lấy tất cả node IDs
            all_node_ids = session.execute_read(get_all_node_ids)
            print(f"✓ Tìm thấy {len(all_node_ids)} nodes trong Neo4j")
            
            # Kiểm tra từng relationship
            print("\n🔍 Đang kiểm tra relationships...")
            missing_rels = []
            missing_source = []
            missing_target = []
            missing_both = []
            existing_rels = []
            
            for i, edge in enumerate(edges):
                if (i + 1) % 1000 == 0:
                    print(f"  Đang kiểm tra {i+1}/{len(edges)}...")
                
                source = edge.get("source", "")
                target = edge.get("target", "")
                rel_type = edge.get("type", "")
                
                if not source or not target:
                    continue
                
                source_exists = source in all_node_ids
                target_exists = target in all_node_ids
                
                if not source_exists and not target_exists:
                    missing_both.append({
                        "source": source,
                        "target": target,
                        "type": rel_type
                    })
                elif not source_exists:
                    missing_source.append({
                        "source": source,
                        "target": target,
                        "type": rel_type
                    })
                elif not target_exists:
                    missing_target.append({
                        "source": source,
                        "target": target,
                        "type": rel_type
                    })
                else:
                    # Cả hai đều tồn tại, kiểm tra relationship
                    rel_exists = session.execute_read(
                        get_relationship_exists, source, target, rel_type
                    )
                    if not rel_exists:
                        missing_rels.append({
                            "source": source,
                            "target": target,
                            "type": rel_type
                        })
                    else:
                        existing_rels.append(edge)
        
        # Thống kê
        print("\n" + "=" * 70)
        print("KẾT QUẢ KIỂM TRA")
        print("=" * 70)
        
        total_expected = len(edges)
        total_missing = len(missing_both) + len(missing_source) + len(missing_target) + len(missing_rels)
        total_existing = len(existing_rels)
        
        print(f"\n📊 Tổng quan:")
        print(f"   - Relationships trong file: {total_expected}")
        print(f"   - Relationships đã tồn tại: {total_existing}")
        print(f"   - Relationships thiếu: {total_missing}")
        print(f"   - Relationships trong Neo4j (theo query): {total_existing}")
        
        print(f"\n📊 Phân loại relationships thiếu:")
        print(f"   - Thiếu cả source và target: {len(missing_both)}")
        print(f"   - Thiếu source node: {len(missing_source)}")
        print(f"   - Thiếu target node: {len(missing_target)}")
        print(f"   - Nodes tồn tại nhưng relationship không có: {len(missing_rels)}")
        
        # Thống kê theo type
        print(f"\n📊 Relationships thiếu theo type:")
        missing_by_type = defaultdict(int)
        for rel in missing_both + missing_source + missing_target + missing_rels:
            missing_by_type[rel["type"]] += 1
        
        for rel_type, count in sorted(missing_by_type.items(), key=lambda x: -x[1]):
            print(f"   - {rel_type}: {count}")
        
        # Lưu kết quả
        result = {
            "total_expected": total_expected,
            "total_existing": total_existing,
            "total_missing": total_missing,
            "missing_both_nodes": len(missing_both),
            "missing_source": len(missing_source),
            "missing_target": len(missing_target),
            "missing_relationship_only": len(missing_rels),
            "missing_by_type": dict(missing_by_type),
            "missing_relationships": {
                "both_nodes_missing": missing_both[:100],  # Chỉ lưu 100 đầu tiên
                "source_missing": missing_source[:100],
                "target_missing": missing_target[:100],
                "relationship_missing": missing_rels[:100]
            }
        }
        
        output_file = "outputs/missing_relationships_report.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Đã lưu báo cáo vào {output_file}")
        
        # In một số ví dụ
        if missing_source:
            print(f"\n📋 Ví dụ relationships thiếu source node (5 đầu tiên):")
            for rel in missing_source[:5]:
                print(f"   - {rel['source']} -[{rel['type']}]-> {rel['target']}")
        
        if missing_target:
            print(f"\n📋 Ví dụ relationships thiếu target node (5 đầu tiên):")
            for rel in missing_target[:5]:
                print(f"   - {rel['source']} -[{rel['type']}]-> {rel['target']}")
        
        if missing_rels:
            print(f"\n📋 Ví dụ relationships thiếu (nodes tồn tại nhưng relationship không có) (5 đầu tiên):")
            for rel in missing_rels[:5]:
                print(f"   - {rel['source']} -[{rel['type']}]-> {rel['target']}")
        
    finally:
        driver.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kiểm tra relationships thiếu trong Neo4j")
    parser.add_argument("--merged-file", default="data/merged_kpop_data.json",
                        help="File merged data")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687",
                        help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j",
                        help="Neo4j username")
    parser.add_argument("--neo4j-password", default=None,
                        help="Neo4j password")
    parser.add_argument("--neo4j-database", default=None,
                        help="Neo4j database name")
    
    args = parser.parse_args()
    
    check_missing_relationships(
        args.merged_file,
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.neo4j_database
    )







