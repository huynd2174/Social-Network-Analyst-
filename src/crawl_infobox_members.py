# -*- coding: utf-8 -*-
"""
Script crawl infobox từ Wikipedia cho các node Group/Artist.
SỬ DỤNG BEAUTIFULSOUP ĐỂ PARSE HTML (giống korean_music_bfs.py)

Quy trình:
1. Load file korean_artists_graph_bfs.json
2. Tìm các node có label = "Group" hoặc "Artist" và có URL
3. Truy cập trang Wikipedia, parse HTML bằng BeautifulSoup
4. Lấy infobox, trích xuất các trường thành viên:
   - Group: Thành viên, Cựu thành viên, Current members, Past members...
   - Artist: Associated acts, Thành viên của...
5. Lưu kết quả vào file: infobox_members.json

ƯU ĐIỂM:
- Lấy được CẢ text có link và không có link
- Parse HTML rendered dễ hơn parse wikitext
- Chính xác hơn vì dùng cùng cách với korean_music_bfs.py
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote, quote

import requests
from bs4 import BeautifulSoup


# User-Agent để Wikipedia không chặn
HEADERS = {
    "User-Agent": "KpopNetworkAnalyzer/1.0 (Educational project) Python/requests"
}

# Các trường cần lấy cho GROUP
GROUP_KEYS = [
    "thành viên",
    "cựu thành viên", 
    "thành viên hiện tại",
    "thành viên cũ",
    "thành viên ban đầu",
    "members",
    "former members",
    "current members",
    "past members",
]

# Các trường cần lấy cho ARTIST
ARTIST_KEYS = [
    "thành viên của",
    "cựu thành viên của",
    "nhóm nhạc",
    "group",
    "groups",
    "associated acts",
]


def get_title_from_url(url: str) -> Optional[str]:
    """Lấy title Wikipedia từ URL."""
    if not url:
        return None
    try:
        path = urlparse(url).path
        if "/wiki/" not in path:
            return None
        title = path.split("/wiki/")[-1]
        return unquote(title) if title else None
    except Exception:
        return None


def fetch_page_soup(url: str) -> Optional[BeautifulSoup]:
    """
    Truy cập trang Wikipedia và trả về BeautifulSoup object.
    Giống cách làm của korean_music_bfs.py
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        print(f"    [!] Lỗi fetch: {e}")
        return None


def extract_infobox_from_soup(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Trích xuất infobox từ BeautifulSoup.
    Học hỏi từ korean_music_bfs.py nhưng đơn giản hóa.
    
    Lấy CẢ text có link và không có link.
    """
    if not soup:
        return {}
    
    # Tìm bảng infobox
    infobox = soup.find("table", class_="infobox")
    if not infobox:
        # Thử tìm các class khác
        infobox = soup.find("table", class_=re.compile(r"infobox", re.IGNORECASE))
    
    if not infobox:
        return {}
    
    result = {}
    
    # Duyệt qua tất cả các hàng trong infobox
    rows = infobox.find_all("tr")
    
    for row in rows:
        # Tìm header (th) và data (td)
        th = row.find("th")
        td = row.find("td")
        
        if not th or not td:
            continue
        
        # Lấy key từ header
        key = th.get_text(strip=True)
        if not key:
            continue
        
        # Lấy value từ td - LẤY CẢ TEXT CÓ LINK VÀ KHÔNG CÓ LINK
        value = extract_cell_value(td)
        
        if value:
            result[key] = value
    
    return result


def extract_cell_value(td) -> str:
    """
    Trích xuất giá trị từ một ô td trong infobox.
    Xử lý cả trường hợp có link và không có link.
    
    Ví dụ:
    - <a href="...">Xiumin</a> -> "Xiumin"
    - Suho (không có link) -> "Suho"
    - <a>Lay</a>, <a>Baekhyun</a> -> "Lay, Baekhyun"
    """
    if not td:
        return ""
    
    # Xóa các phần tử không cần thiết
    for elem in td.find_all(["sup", "style", "script"]):
        elem.decompose()
    
    # Thay thế <br> bằng dấu phân cách đặc biệt
    for br in td.find_all("br"):
        br.replace_with(" |SEPARATOR| ")
    
    # Xử lý <li> - mỗi li là một item
    for li in td.find_all("li"):
        li.insert_before(" |SEPARATOR| ")
    
    # Lấy text từ tất cả các phần tử
    # Ưu tiên lấy text từ <a> tags trước
    members = []
    
    # Tìm tất cả các link <a> - đây thường là tên thành viên
    links = td.find_all("a")
    if links:
        for link in links:
            text = link.get_text(strip=True)
            if text and len(text) >= 2:
                # Bỏ qua các link không phải tên người
                if text.lower() not in ['edit', 'sửa', 'xem', 'view', 'more']:
                    members.append(text)
    
    # Nếu không có link, lấy toàn bộ text và tách theo separator
    if not members:
        full_text = td.get_text(separator=" ")
        # Tách theo các dấu phân cách
        full_text = full_text.replace("|SEPARATOR|", ",")
        parts = re.split(r'[,•·*\n]+', full_text)
        for part in parts:
            part = part.strip()
            if part and len(part) >= 2:
                members.append(part)
    
    # Nếu vẫn rỗng, lấy raw text
    if not members:
        raw_text = td.get_text(strip=True)
        if raw_text:
            members = [raw_text]
    
    # Gộp thành chuỗi với dấu phẩy
    raw_text = ", ".join(members)
    
    # Làm sạch
    text = clean_member_text(raw_text)
    
    return text


def clean_member_text(text: str) -> str:
    """
    Làm sạch text chứa danh sách thành viên.
    """
    if not text:
        return ""
    
    # Bỏ các tham chiếu [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Bỏ separator marker
    text = text.replace("|SEPARATOR|", ",")
    
    # Bỏ các ghi chú trong ngoặc đơn có chứa năm hoặc thông tin phụ
    # Nhưng giữ lại tên nghệ sĩ trong ngoặc
    # Ví dụ: "Hana (Zinger)" -> giữ nguyên, "(2006-2011)" -> bỏ
    text = re.sub(r'\(\d{4}[–-]\d{0,4}\)', '', text)
    text = re.sub(r'\s*\(†\)\s*', ' (†)', text)  # Giữ lại dấu † cho người đã mất
    
    # Thay thế các dấu phân cách thành dấu phẩy
    text = re.sub(r'\s*[•·]\s*', ', ', text)
    text = re.sub(r'\s*\*\s*', ', ', text)
    
    # Chuẩn hóa dấu phẩy - đảm bảo có khoảng trắng sau dấu phẩy
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r',\s*,+', ', ', text)
    text = re.sub(r'^[,\s]+', '', text)
    text = re.sub(r'[,\s]+$', '', text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def filter_member_keys(infobox: Dict[str, str], target_keys: List[str]) -> Dict[str, str]:
    """
    Lọc chỉ lấy các trường liên quan đến thành viên từ infobox.
    
    Args:
        infobox: Dictionary chứa tất cả các trường từ infobox
        target_keys: Danh sách các key cần lấy (lowercase)
    
    Returns:
        Dictionary chỉ chứa các trường thành viên
    """
    result = {}
    
    for key, value in infobox.items():
        key_lower = key.lower().strip()
        
        for target in target_keys:
            if target in key_lower or key_lower in target:
                # Đã tìm thấy key phù hợp
                # Chuẩn hóa tên key
                if 'former' in key_lower or 'past' in key_lower or 'cựu' in key_lower or 'cũ' in key_lower:
                    normalized_key = "Past members" if 'member' in key_lower or 'thành viên' in key_lower else "Former members"
                elif 'current' in key_lower or 'hiện tại' in key_lower:
                    normalized_key = "Current members"
                elif 'thành viên' in key_lower or 'member' in key_lower:
                    normalized_key = "Current members"
                elif 'associated' in key_lower:
                    normalized_key = "Associated acts"
                else:
                    normalized_key = key  # Giữ nguyên
                
                # Chỉ thêm nếu value không rỗng
                if value and len(value) > 1:
                    result[normalized_key] = value
                break
    
    return result


def main():
    print("=" * 60)
    print("CRAWL INFOBOX THÀNH VIÊN TỪ WIKIPEDIA (HTML VERSION)")
    print("=" * 60)
    print("Sử dụng BeautifulSoup để parse HTML - lấy cả text có link và không link")

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

        if not url or "wikipedia.org" not in url:
            continue

        if label == "Group":
            groups_to_crawl.append((node_id, url))
        elif label == "Artist":
            artists_to_crawl.append((node_id, url))

    print(f"   ✓ Tìm thấy {len(groups_to_crawl)} Groups có URL")
    print(f"   ✓ Tìm thấy {len(artists_to_crawl)} Artists có URL")

    # 3. Crawl infobox
    print("\n🌐 Bước 3: Crawl infobox từ Wikipedia (HTML parsing)...")
    results = {
        "groups": {},
        "artists": {},
    }

    # Crawl Groups
    print(f"\n   --- Crawling {len(groups_to_crawl)} Groups ---")
    success_groups = 0
    
    for idx, (node_id, url) in enumerate(groups_to_crawl, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"   [{idx}/{len(groups_to_crawl)}] {node_id[:50]}...")

        soup = fetch_page_soup(url)
        if soup:
            full_infobox = extract_infobox_from_soup(soup)
            member_info = filter_member_keys(full_infobox, GROUP_KEYS)
            
            if member_info:
                results["groups"][node_id] = {
                    "url": url,
                    "infobox": member_info,
                }
                success_groups += 1

        time.sleep(0.3)  # Tránh spam

    print(f"   ✓ Crawl thành công {success_groups}/{len(groups_to_crawl)} Groups")

    # Crawl Artists
    print(f"\n   --- Crawling {len(artists_to_crawl)} Artists ---")
    success_artists = 0
    
    for idx, (node_id, url) in enumerate(artists_to_crawl, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"   [{idx}/{len(artists_to_crawl)}] {node_id[:50]}...")

        soup = fetch_page_soup(url)
        if soup:
            full_infobox = extract_infobox_from_soup(soup)
            member_info = filter_member_keys(full_infobox, ARTIST_KEYS)
            
            if member_info:
                results["artists"][node_id] = {
                    "url": url,
                    "infobox": member_info,
                }
                success_artists += 1

        time.sleep(0.3)

    print(f"   ✓ Crawl thành công {success_artists}/{len(artists_to_crawl)} Artists")

    # 4. Lưu kết quả
    print("\n💾 Bước 4: Lưu kết quả...")
    output_file = "infobox_members.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"   ✓ Đã lưu vào {output_file}")
    
    # 5. Hiển thị một số ví dụ
    print(f"\n📊 Kết quả:")
    print(f"   - Groups có infobox thành viên: {len(results['groups'])}")
    print(f"   - Artists có infobox liên quan: {len(results['artists'])}")
    
    # Hiển thị ví dụ
    print(f"\n📋 Một số ví dụ Groups:")
    count = 0
    for group_name, data in results["groups"].items():
        if count >= 5:
            break
        infobox = data.get("infobox", {})
        members = infobox.get("Current members", infobox.get("Past members", "N/A"))
        if len(members) > 80:
            members = members[:80] + "..."
        print(f"   • {group_name}: {members}")
        count += 1
    
    print("\n🎉 Hoàn tất!")


if __name__ == "__main__":
    main()
