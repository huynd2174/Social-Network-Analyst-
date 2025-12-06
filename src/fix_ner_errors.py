"""
Script để sửa các lỗi NER trong merged_kpop_data.json

Các lỗi cần sửa:
1. "Kpop BTS" → "BTS" (loại bỏ tiền tố "Kpop")
2. "Lee Su ji" → Sửa type từ "Group" thành "Artist" (nếu đúng)
"""

import json
import os
from typing import Dict, List


def fix_entity_name(entity_name: str) -> str:
    """Fix entity name by removing common prefixes."""
    # Remove "Kpop" prefix
    if entity_name.startswith("Kpop "):
        entity_name = entity_name[5:]  # Remove "Kpop "
    
    # Remove other common prefixes
    prefixes = ["K-pop ", "K-Pop ", "KPOP "]
    for prefix in prefixes:
        if entity_name.startswith(prefix):
            entity_name = entity_name[len(prefix):]
    
    return entity_name.strip()


def fix_entity_type(entity_name: str, current_type: str, nodes: Dict) -> str:
    """Fix entity type if wrong."""
    # Known corrections
    corrections = {
        "Lee Su ji": "Artist",  # Should be Artist, not Group
        # Add more corrections here
    }
    
    if entity_name in corrections:
        return corrections[entity_name]
    
    return current_type


def fix_merged_data(input_path: str = "data/merged_kpop_data.json", 
                   output_path: str = "data/merged_kpop_data_fixed.json",
                   backup: bool = True):
    """Fix errors in merged_kpop_data.json."""
    
    print("🔄 Đang load merged_kpop_data.json...")
    
    # Backup original file
    if backup and os.path.exists(input_path):
        backup_path = input_path + ".backup"
        import shutil
        shutil.copy2(input_path, backup_path)
        print(f"✅ Đã backup file gốc: {backup_path}")
    
    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', {})
    edges = data.get('edges', [])
    
    print(f"📊 Loaded {len(nodes)} nodes and {len(edges)} edges")
    
    # Track fixes
    fixes = {
        'renamed_nodes': [],
        'type_corrections': [],
        'edge_updates': []
    }
    
    # Fix nodes
    print("\n🔧 Đang sửa nodes...")
    new_nodes = {}
    node_mapping = {}  # old_name -> new_name
    nodes_to_merge = {}  # Track nodes that will be merged
    
    for node_id, node_data in nodes.items():
        new_id = fix_entity_name(node_id)
        new_type = fix_entity_type(new_id, node_data.get('label', ''), nodes)
        
        # Track changes
        if new_id != node_id:
            fixes['renamed_nodes'].append({
                'old': node_id,
                'new': new_id,
                'reason': 'Removed prefix'
            })
            node_mapping[node_id] = new_id
        
        if new_type != node_data.get('label', ''):
            fixes['type_corrections'].append({
                'entity': new_id,
                'old_type': node_data.get('label', ''),
                'new_type': new_type
            })
        
        # Check if node already exists (merge case - e.g., "Kpop BTS" and "BTS")
        if new_id in new_nodes:
            # Node đã tồn tại - đánh dấu để xóa node cũ (giữ node gốc)
            nodes_to_merge[node_id] = new_id
            print(f"   ⚠️  Node trùng: '{node_id}' → '{new_id}' (sẽ merge edges)")
        else:
            # Update node
            node_data['label'] = new_type
            node_data['title'] = new_id if 'title' not in node_data else fix_entity_name(node_data.get('title', new_id))
            new_nodes[new_id] = node_data
    
    if nodes_to_merge:
        print(f"   ⚠️  Phát hiện {len(nodes_to_merge)} nodes trùng lặp sẽ được merge:")
        for old, new in list(nodes_to_merge.items())[:5]:
            print(f"      - '{old}' → '{new}' (giữ node '{new}')")
        if len(nodes_to_merge) > 5:
            print(f"      ... và {len(nodes_to_merge) - 5} nodes khác")
    
    # Fix edges - update source and target names
    print("🔧 Đang sửa edges...")
    new_edges = []
    seen_edges = set()  # Track duplicate edges
    
    for edge in edges:
        source = edge.get('source', '')
        target = edge.get('target', '')
        
        # Map to new names (check both mapping and merge list)
        if source in nodes_to_merge:
            new_source = nodes_to_merge[source]
        else:
            new_source = node_mapping.get(source, fix_entity_name(source))
        
        if target in nodes_to_merge:
            new_target = nodes_to_merge[target]
        else:
            new_target = node_mapping.get(target, fix_entity_name(target))
        
        # Skip if edge is duplicate
        edge_key = (new_source, edge.get('type', ''), new_target)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        
        if new_source != source or new_target != target:
            fixes['edge_updates'].append({
                'old': f"{source} -> {target}",
                'new': f"{new_source} -> {new_target}"
            })
        
        edge['source'] = new_source
        edge['target'] = new_target
        new_edges.append(edge)
    
    # Update data
    data['nodes'] = new_nodes
    data['edges'] = new_edges
    
    # Update metadata
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata']['fixed_at'] = __import__('datetime').datetime.now().isoformat()
    data['metadata']['fixes_applied'] = {
        'renamed_nodes': len(fixes['renamed_nodes']),
        'type_corrections': len(fixes['type_corrections']),
        'edge_updates': len(fixes['edge_updates'])
    }
    
    # Save fixed data
    print(f"\n💾 Đang lưu file đã sửa: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TÓM TẮT CÁC SỬA ĐỔI")
    print("="*70)
    print(f"\n✅ Đã đổi tên {len(fixes['renamed_nodes'])} nodes:")
    for fix in fixes['renamed_nodes'][:10]:  # Show first 10
        print(f"   - '{fix['old']}' → '{fix['new']}'")
    if len(fixes['renamed_nodes']) > 10:
        print(f"   ... và {len(fixes['renamed_nodes']) - 10} nodes khác")
    
    print(f"\n✅ Đã sửa type cho {len(fixes['type_corrections'])} entities:")
    for fix in fixes['type_corrections']:
        print(f"   - '{fix['entity']}': {fix['old_type']} → {fix['new_type']}")
    
    print(f"\n✅ Đã cập nhật {len(fixes['edge_updates'])} edges")
    
    print(f"\n📁 File đã sửa: {output_path}")
    print(f"📁 File backup: {input_path}.backup")
    print(f"\n💡 Để sử dụng file đã sửa, đổi tên:")
    print(f"   {output_path} → {input_path}")
    
    return fixes


def main():
    """Main function."""
    print("="*70)
    print("  🔧 FIX NER ERRORS IN MERGED DATA")
    print("="*70)
    print("\nScript này sẽ sửa các lỗi:")
    print("  1. Loại bỏ tiền tố 'Kpop' (ví dụ: 'Kpop BTS' → 'BTS')")
    print("  2. Sửa type sai (ví dụ: 'Lee Su ji' từ Group → Artist)")
    print("\n" + "="*70)
    
    # Check if file exists
    input_path = "data/merged_kpop_data.json"
    if not os.path.exists(input_path):
        print(f"❌ File không tồn tại: {input_path}")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  Sẽ tạo file mới: data/merged_kpop_data_fixed.json")
    print(f"   File gốc sẽ được backup: {input_path}.backup")
    response = input("\nTiếp tục? (y/n): ").strip().lower()
    
    if response != 'y':
        print("❌ Đã hủy")
        return
    
    # Run fix
    fixes = fix_merged_data()
    
    print("\n✅ Hoàn thành!")
    print("\n💡 Bước tiếp theo:")
    print("   1. Kiểm tra file merged_kpop_data_fixed.json")
    print("   2. Nếu OK, đổi tên: merged_kpop_data_fixed.json → merged_kpop_data.json")
    print("   3. Chạy lại chatbot để test")


if __name__ == "__main__":
    main()

