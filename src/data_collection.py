# -*- coding: utf-8 -*-
"""
THU THẬP VÀ LỰA CHỌN TẬP DỮ LIỆU LÀM GIÀU
=============================================
Nguồn dữ liệu: Wikipedia tiếng Việt
Mục tiêu: Thu thập văn bản để làm đầu vào cho NER và Relation Extraction

Cấu trúc dữ liệu đầu ra (JSON):
{
    "node_id": "...",           # ID của node trong Neo4j
    "node_name": "...",         # Tên node
    "node_label": "...",        # Label (Artist, Group, Album, Song, Genre, Company)
    "wikipedia_url": "...",     # URL Wikipedia
    "text": "...",              # Văn bản đã làm sạch (dùng cho NER và RE)
    "sections": {               # Các phần chi tiết
        "intro": "...",         # Đoạn giới thiệu
        "career": "...",        # Sự nghiệp
        "discography": "...",   # Danh sách album/đĩa nhạc
        "awards": "..."         # Giải thưởng
    }
}
"""
import sys
import io
import json
import time
import re
from typing import Dict, List, Optional
from urllib.parse import unquote, quote
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from neo4j import GraphDatabase

# UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class WikipediaCollector:
    """Thu thập văn bản từ Wikipedia"""
    
    def __init__(self, delay: float = 0.3):
        self.base_url = "https://vi.wikipedia.org/wiki/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-requests'
        })
        self.delay = delay
    
    def collect(self, title: str) -> Dict:
        """Thu thập dữ liệu từ một trang Wikipedia"""
        try:
            url = self.base_url + quote(title.replace(' ', '_'))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            result = {
                'url': url,
                'title': title,
                'text': '',
                'sections': {
                    'intro': '',
                    'career': '',
                    'discography': '',
                    'awards': ''
                }
            }
            
            # 1. Lấy intro (đoạn đầu tiên)
            content = soup.find('div', class_='mw-parser-output')
            if content:
                # Lấy tất cả đoạn p trước heading đầu tiên
                intro_parts = []
                for elem in content.children:
                    if elem.name in ['h2', 'h3']:
                        break
                    if elem.name == 'p':
                        text = elem.get_text(strip=True)
                        if text:
                            intro_parts.append(text)
                result['sections']['intro'] = ' '.join(intro_parts)
            
            # 2. Lấy các section quan trọng
            headings = soup.find_all(['h2', 'h3'])
            for heading in headings:
                heading_text = heading.get_text(strip=True).lower()
                section_content = []
                
                # Lấy nội dung sau heading
                elem = heading.find_next_sibling()
                while elem and elem.name not in ['h2', 'h3']:
                    if elem.name == 'p':
                        text = elem.get_text(strip=True)
                        if text and len(text) > 20:
                            section_content.append(text)
                    elif elem.name == 'ul':
                        for li in elem.find_all('li'):
                            text = li.get_text(strip=True)
                            if text:
                                section_content.append(text)
                    elem = elem.find_next_sibling()
                
                section_text = ' '.join(section_content)
                
                # Phân loại vào các section
                if any(kw in heading_text for kw in ['sự nghiệp', 'career', 'hoạt động']):
                    result['sections']['career'] = section_text
                elif any(kw in heading_text for kw in ['album', 'discography', 'đĩa nhạc', 'tác phẩm']):
                    result['sections']['discography'] = section_text
                elif any(kw in heading_text for kw in ['giải thưởng', 'award', 'thành tích']):
                    result['sections']['awards'] = section_text
            
            # 3. Lấy full text (đã làm sạch)
            content_div = soup.find('div', id='mw-content-text')
            if content_div:
                # Loại bỏ các phần không cần
                for tag in content_div.find_all(['table', 'div', 'span'], 
                                                class_=['navbox', 'reference', 'mw-references-wrap', 
                                                       'mw-editsection', 'toc', 'infobox']):
                    tag.decompose()
                
                full_text = content_div.get_text(separator=' ', strip=True)
                full_text = re.sub(r'\[\d+\]', '', full_text)  # Loại bỏ [1], [2]...
                full_text = re.sub(r'\s+', ' ', full_text)
                result['text'] = full_text
            
            time.sleep(self.delay)
            return result
            
        except Exception as e:
            return {
                'url': '',
                'title': title,
                'text': '',
                'sections': {'intro': '', 'career': '', 'discography': '', 'awards': ''},
                'error': str(e)
            }


class DataCollector:
    """Thu thập dữ liệu từ các nodes trong Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.wiki = WikipediaCollector()
        print(f"✓ Đã kết nối Neo4j: {uri}")
    
    def close(self):
        self.driver.close()
    
    def collect_and_save(self, output_file: str = 'enrichment_text_data.json', limit: int = None, 
                         labels: List[str] = None):
        """Thu thập dữ liệu và lưu vào file JSON"""
        print("=" * 70)
        print("THU THẬP TẬP DỮ LIỆU LÀM GIÀU TỪ WIKIPEDIA")
        print("=" * 70)
        
        # Labels mặc định nếu không chỉ định
        if labels is None:
            labels = ['Artist', 'Group', 'Song', 'Album', 'Company']
        
        print(f"📌 Labels: {', '.join(labels)}")
        
        # Lấy nodes có URL Wikipedia và thuộc các labels chỉ định
        def get_nodes(tx):
            # Tạo điều kiện cho labels
            label_conditions = ' OR '.join([f'n:{label}' for label in labels])
            query = f"""
            MATCH (n)
            WHERE ({label_conditions})
            AND n.url IS NOT NULL AND n.url CONTAINS 'wikipedia.org'
            RETURN n.id as id, n.name as name, n.url as url, labels(n) as labels
            ORDER BY n.name
            """
            if limit:
                query += f" LIMIT {limit}"
            return [dict(r) for r in tx.run(query)]
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            nodes = session.execute_read(get_nodes)
        
        print(f"✓ Tìm thấy {len(nodes)} nodes có URL Wikipedia\n")
        
        # Thu thập dữ liệu
        collected = []
        
        for i, node in enumerate(nodes, 1):
            name = node['name']
            url = node['url']
            label = node['labels'][0] if node['labels'] else 'Entity'
            
            print(f"[{i}/{len(nodes)}] {name} ({label})")
            
            # Lấy title từ URL
            title = url.split('/wiki/')[-1] if '/wiki/' in url else name
            title = unquote(title).replace('_', ' ')
            
            # Thu thập từ Wikipedia
            data = self.wiki.collect(title)
            
            # Tạo record
            record = {
                'node_id': node['id'],
                'node_name': name,
                'node_label': label,
                'wikipedia_url': data.get('url', url),
                'text': data.get('text', ''),
                'sections': data.get('sections', {}),
                'text_length': len(data.get('text', ''))
            }
            
            if 'error' in data:
                record['error'] = data['error']
                print(f"  ⚠ Lỗi: {data['error']}")
            else:
                print(f"  ✓ {record['text_length']:,} ký tự")
            
            collected.append(record)
        
        # Lọc bỏ các record lỗi hoặc quá ngắn
        valid_data = [r for r in collected if r['text_length'] >= 500 and 'error' not in r]
        
        # Lưu file
        output = {
            'metadata': {
                'description': 'Tập dữ liệu văn bản từ Wikipedia để làm giàu đồ thị tri thức',
                'source': 'Wikipedia tiếng Việt',
                'collected_at': datetime.now().isoformat(),
                'total_nodes': len(nodes),
                'valid_records': len(valid_data),
                'total_characters': sum(r['text_length'] for r in valid_data)
            },
            'data': valid_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'=' * 70}")
        print(f"KẾT QUẢ THU THẬP")
        print(f"{'=' * 70}")
        print(f"📊 Tổng nodes xử lý: {len(nodes)}")
        print(f"📊 Records hợp lệ: {len(valid_data)}")
        print(f"📊 Tổng ký tự: {output['metadata']['total_characters']:,}")
        print(f"💾 Đã lưu: {output_file}")
        
        return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Thu thập dữ liệu làm giàu từ Wikipedia')
    parser.add_argument('--neo4j-uri', default='bolt://127.0.0.1:7687')
    parser.add_argument('--neo4j-user', default='neo4j')
    parser.add_argument('--neo4j-pass', default='12345678')
    parser.add_argument('--neo4j-db', default='network')
    parser.add_argument('--output', default='enrichment_text_data.json')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--labels', nargs='+', default=['Artist', 'Group', 'Song', 'Album', 'Company'],
                        help='Các labels cần thu thập (mặc định: Artist, Group, Song, Album, Company)')
    args = parser.parse_args()
    
    collector = DataCollector(args.neo4j_uri, args.neo4j_user, args.neo4j_pass, args.neo4j_db)
    try:
        collector.collect_and_save(output_file=args.output, limit=args.limit, labels=args.labels)
    finally:
        collector.close()


if __name__ == '__main__':
    main()

