"""
Script crawl infobox từ Wikipedia cho các node Group/Artist.

Quy trình:
1. Load file korean_artists_graph_bfs.json
2. Tìm các node có label = "Group" hoặc "Artist"
3. Lấy URL Wikipedia của từng node
4. Truy cập Wikipedia API để lấy wikitext
5. Parse infobox, lấy các trường cần thiết:
   - Group: Thành viên, Cựu thành viên
   - Artist: Thành viên của, Cựu thành viên của
6. Lưu kết quả vào file mới: infobox_members.json
"""

import json
import re
import time
from urllib.parse import urlparse, unquote

import requests


# Wikipedia API endpoint
WIKI_API = "https://vi.wikipedia.org/w/api.php"

# User-Agent bắt buộc để Wikipedia không chặn 403
HEADERS = {
    "User-Agent": "KpopNetworkAnalyzer/1.0 (Educational project) Python/requests"
}

# Các trường cần lấy cho GROUP
GROUP_KEYS = [
    "Thành viên",
    "Cựu thành viên",
    "Thành viên hiện tại",
    "Thành viên cũ",
    "Thành viên ban đầu",
    "Members",
    "Former members",
    "Current members",
    "Past members",
]

# Các trường cần lấy cho ARTIST
ARTIST_KEYS = [
    "Thành viên của",
    "Cựu thành viên của",
    "Nhóm nhạc",
    "Group",
    "Groups",
    "Associated acts",
]


def get_title_from_url(url: str) -> str | None:
    """Lấy title Wikipedia từ URL."""
    if not url:
        return None
    try:
        path = urlparse(url).path  # /wiki/BTS
        if "/wiki/" not in path:
            return None
        title = path.split("/wiki/")[-1]
        return unquote(title) if title else None
    except Exception:
        return None


def fetch_wikitext(url: str) -> str | None:
    """
    Truy cập Wikipedia API để lấy wikitext của trang.
    """
    title = get_title_from_url(url)
    if not title:
        return None

    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2",
        "titles": title,
    }

    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None

        revs = pages[0].get("revisions", [])
        if not revs:
            return None

        slots = revs[0].get("slots", {})
        content = slots.get("main", {}).get("*") or slots.get("main", {}).get("content")
        return content

    except Exception as e:
        print(f"    [!] Lỗi: {e}")
        return None


def clean_value(value: str) -> str:
    """Làm sạch giá trị infobox."""
    text = value or ""

    # Bỏ <ref>...</ref>
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^/>]*/>", "", text, flags=re.IGNORECASE)

    # Thay <br> bằng dấu phẩy
    text = re.sub(r"<br\s*/?>", ", ", text, flags=re.IGNORECASE)

    # Bỏ HTML tags khác
    text = re.sub(r"</?[^>]+>", "", text)

    # Bỏ templates {{...}} (lặp nhiều lần để xử lý nested)
    for _ in range(5):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Xử lý wiki links [[...|display]] hoặc [[link]]
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)

    # Cắt bỏ phần dư sau '}}' hoặc '|module=...' / '|Past_members'... (template thừa)
    text = re.split(r"\}\}", text, 1)[0]
    text = re.split(r"\|\s*(module|Past_members|child|embed)\b", text, 1)[0]

    # Thay dấu * (bullet) thành dấu phẩy
    # Ví dụ: "* Jin * Suga * J-Hope" -> "Jin, Suga, J-Hope"
    text = re.sub(r"^\s*\*\s*", "", text)         # bỏ * đầu dòng
    text = re.sub(r"\s*\*\s*", ", ", text)        # các * còn lại -> dấu phẩy

    # Chuẩn hóa dấu phẩy và khoảng trắng
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip(" ,")

    # Sửa một số lỗi đặc biệt
    fixes = {
        "New , Jeans": "NewJeans",
        "New, Jeans": "NewJeans",
        "i , KON": "iKON",
        "i, KON": "iKON",
        "Gugudan Se , Mi , Na": "Gugudan SeMiNa",
        "Gugudan Se, Mi, Na": "Gugudan SeMiNa",
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)

    return text


def parse_infobox(wikitext: str, keys: list[str]) -> dict[str, str]:
    """
    Parse infobox từ wikitext, lấy các trường theo keys.
    """
    if not wikitext:
        return {}

    # Tìm vị trí bắt đầu của infobox
    match = re.search(r"\{\{[Ii]nfobox", wikitext)
    if not match:
        return {}

    # Đếm ngoặc để tìm điểm kết thúc
    start = match.start()
    depth = 0
    end = start
    i = start

    while i < len(wikitext):
        if wikitext[i:i+2] == "{{":
            depth += 1
            i += 2
        elif wikitext[i:i+2] == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                end = i
                break
        else:
            i += 1

    if end <= start:
        infobox_text = wikitext[start:start+3000]
    else:
        infobox_text = wikitext[start:end]

    # Parse các tham số | key = value
    params = {}
    pattern = r"^\|\s*([^=\n]+?)\s*=\s*"
    lines = infobox_text.split("\n")
    current_key = None
    current_value_lines = []

    for line in lines:
        m = re.match(pattern, line)
        if m:
            if current_key:
                params[current_key.strip()] = "\n".join(current_value_lines).strip()
            current_key = m.group(1)
            rest = line[m.end():]
            current_value_lines = [rest]
        elif current_key:
            current_value_lines.append(line)

    if current_key:
        params[current_key.strip()] = "\n".join(current_value_lines).strip()

    # Lấy các trường cần thiết
    result = {}
    for key in keys:
        variants = [key, key.lower(), key.replace(" ", "_"), key.replace(" ", "_").lower()]
        for var in variants:
            for param_key, param_val in params.items():
                if param_key.lower().strip() == var.lower().strip():
                    cleaned = clean_value(param_val)
                    if cleaned:
                        result[key] = cleaned
                    break
            if key in result:
                break

    return result


def main():
    print("=" * 60)
    print("CRAWL INFOBOX THÀNH VIÊN TỪ WIKIPEDIA")
    print("=" * 60)

    # 1. Load file gốc
    print("\n📂 Bước 1: Load file korean_artists_graph_bfs.json...")
    with open("korean_artists_graph_bfs.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", graph)
    print(f"   ✓ Tổng cộng {len(nodes)} nodes")

    # 2. Tìm các node Group và Artist có URL
    print("\n🔍 Bước 2: Tìm các node Group/Artist có URL...")
    groups_to_crawl = []
    artists_to_crawl = []

    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue

        label = node.get("label")
        url = node.get("url")

        if not url:
            continue

        if label == "Group":
            groups_to_crawl.append((node_id, url))
        elif label == "Artist":
            artists_to_crawl.append((node_id, url))

    print(f"   ✓ Tìm thấy {len(groups_to_crawl)} Groups có URL")
    print(f"   ✓ Tìm thấy {len(artists_to_crawl)} Artists có URL")

    # 3. Crawl infobox
    print("\n🌐 Bước 3: Crawl infobox từ Wikipedia...")
    results = {
        "groups": {},
        "artists": {},
    }

    # Crawl Groups
    print(f"\n   --- Crawling {len(groups_to_crawl)} Groups ---")
    for idx, (node_id, url) in enumerate(groups_to_crawl, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"   [{idx}/{len(groups_to_crawl)}] {node_id[:40]}...")

        wikitext = fetch_wikitext(url)
        if wikitext:
            info = parse_infobox(wikitext, GROUP_KEYS)
            if info:
                results["groups"][node_id] = {
                    "url": url,
                    "infobox": info,
                }

        time.sleep(0.5)  # Tránh spam API

    # Crawl Artists
    print(f"\n   --- Crawling {len(artists_to_crawl)} Artists ---")
    for idx, (node_id, url) in enumerate(artists_to_crawl, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"   [{idx}/{len(artists_to_crawl)}] {node_id[:40]}...")

        wikitext = fetch_wikitext(url)
        if wikitext:
            info = parse_infobox(wikitext, ARTIST_KEYS)
            if info:
                results["artists"][node_id] = {
                    "url": url,
                    "infobox": info,
                }

        time.sleep(0.5)

    # 4. Lưu kết quả
    print("\n💾 Bước 4: Lưu kết quả...")
    output_file = "infobox_members.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"   ✓ Đã lưu vào {output_file}")
    print(f"\n📊 Kết quả:")
    print(f"   - Groups có infobox thành viên: {len(results['groups'])}")
    print(f"   - Artists có infobox thành viên của: {len(results['artists'])}")
    print("\n🎉 Hoàn tất!")


if __name__ == "__main__":
    main()
