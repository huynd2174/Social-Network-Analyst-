"""
Chạy shortest path cho nhiều cặp node và lưu kết quả ra file.
"""
from pathlib import Path

from shortest_path_neo4j import ShortestPathFinder

OUTPUT_FILE = Path("batch_shortest_paths_results.md")

NODE_PAIRS = [
    ("BTS", "BLACKPINK"),
    ("BLACKPINK", "AKMU"),
    ("Kill This Love (bài hát)", "Yet to Come (The Most Beautiful Moment)"),
    ("2 Cool 4 Skool", "Yet to Come (The Most Beautiful Moment)"),
    ("BTS", "Universal Music"),
    ("BLACKPINK", "Pop"),
    ("Pdogg", "Bekuh Boom"),
    ("RM (rapper)", "Bekuh Boom"),
    ("BTS", "Hip hop"),
    ("BLACKPINK", "YG Entertainment"),
]


def format_path(path_result) -> str:
    if not path_result:
        return "_Không tìm thấy đường đi._"

    lines = [
        f"- Độ dài: **{path_result['path_length']}** bước",
    ]
    lines.append("- Đường đi:")
    for idx, node in enumerate(path_result["nodes"]):
        label_str = ", ".join(node.get("labels", [])) or "Entity"
        line = f"  {idx+1}. {node.get('name') or node.get('id')} [{label_str}]"
        lines.append(line)
        if idx < len(path_result.get("relationships", [])):
            rel = path_result["relationships"][idx]
            lines.append(f"     └─[{rel.get('type', 'RELATED')}]→")
    return "\n".join(lines)


def main():
    finder = ShortestPathFinder(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="12345678",
        database="network",
        undirected=True,
    )

    lines = [
        "# Kết quả shortest path cho 10 cặp node",
        "",
        "_Chế độ: undirected, phương pháp: Cypher._",
        "",
    ]

    try:
        for idx, (src, tgt) in enumerate(NODE_PAIRS, start=1):
            lines.append(f"## {idx}. `{src}` → `{tgt}`")
            try:
                path = finder.shortest_path_cypher(src, tgt)
                lines.append(format_path(path))
            except Exception as exc:
                lines.append(f"_Lỗi: {exc}_")
            lines.append("")
    finally:
        finder.close()

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Đã ghi kết quả vào {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

