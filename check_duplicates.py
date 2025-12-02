import json

from collections import defaultdict

# Load NER result (đã merge & clean trong run_ner.py)
with open('kpop_ner_result.json', 'r', encoding='utf-8') as f:
    ner = json.load(f)

# Load graph gốc (cấu trúc mới: {"nodes": {...}} hoặc phẳng)
with open('korean_artists_graph_bfs.json', 'r', encoding='utf-8') as f:
    graph = json.load(f)

nodes = graph.get('nodes', graph)

# Lấy tất cả node names từ graph gốc (lowercase)
graph_nodes = set()
for node_id, node_data in nodes.items():
    graph_nodes.add(node_id.lower())
    if isinstance(node_data, dict):
        title = node_data.get('title')
        if title:
            graph_nodes.add(title.lower())

entities = ner.get('entities', [])

print(f"Số node trong graph gốc: {len(graph_nodes)}")
print(f"Số entity trong NER result: {len(entities)}")

# 1. Kiểm tra entity nào TRÙNG với graph gốc (theo text.lower())
duplicates_with_graph = []
for ent in entities:
    if ent.get('text', '').lower() in graph_nodes:
        duplicates_with_graph.append(ent)

print(f"\n=== ENTITY TRÙNG VỚI GRAPH GỐC: {len(duplicates_with_graph)} ===")
if duplicates_with_graph:
    for ent in duplicates_with_graph[:30]:
        print(f"  - {ent['text']} (Type: {ent['type']}, method: {ent.get('method')})")

# 2. Kiểm tra entity trùng nhau trong NER (cùng text, cùng type)
entity_map = defaultdict(list)
for ent in entities:
    key = (ent.get('text', '').lower(), ent.get('type'))
    entity_map[key].append(ent)

duplicates_same_type = {k: v for k, v in entity_map.items() if len(v) > 1}
print(f"\n=== ENTITY TRÙNG CÙNG TYPE (sau khi merge): {len(duplicates_same_type)} ===")
for (text, etype), ents in list(duplicates_same_type.items())[:20]:
    print(f"  - '{text}' ({etype}): {len(ents)} lần")
    for e in ents:
        print(f"      source: {e['source_node']}")

# 3. Kiểm tra entity trùng nhau KHÁC TYPE (hợp lệ - album/song cùng tên nhóm)
entity_text_map = defaultdict(list)
for ent in entities:
    entity_text_map[ent.get('text', '').lower()].append(ent)

duplicates_diff_type = {k: v for k, v in entity_text_map.items() if len(v) > 1}
print(f"\n=== ENTITY TRÙNG KHÁC TYPE (hợp lệ): {len(duplicates_diff_type)} ===")
for text, ents in list(duplicates_diff_type.items())[:20]:
    types = [e['type'] for e in ents]
    print(f"  - '{text}': {types}")

