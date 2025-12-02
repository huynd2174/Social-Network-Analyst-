# -*- coding: utf-8 -*-
"""
Hệ thống làm giàu dữ liệu đồ thị - Phiên bản đầy đủ
- Thu thập dữ liệu: Wikipedia (intro, lịch sử, sự nghiệp) + các nguồn khác
- NER: Phát hiện thực thể MỚI (nghệ sĩ, nhóm nhạc, album, bài hát, công ty...)
- Relation Extraction: Phát hiện quan hệ MỚI giữa các thực thể
- Mục tiêu: Tạo cả NODES MỚI và RELATIONSHIPS MỚI để làm giàu đồ thị
"""
import sys
import io
import time
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
from urllib.parse import unquote, quote

import requests
from bs4 import BeautifulSoup
from neo4j import GraphDatabase

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class TextDataCollector:
    """Thu thập dữ liệu văn bản từ Wikipedia và các nguồn khác"""
    
    def __init__(self, request_timeout: int = 10, request_delay: float = 0.3):
        self.base_url = "https://vi.wikipedia.org/wiki/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-requests'
        })
        self.request_timeout = request_timeout
        self.request_delay = request_delay
        
    def fetch_wikipedia_full_text(self, title: str) -> Dict[str, str]:
        """
        Lấy toàn bộ văn bản từ Wikipedia bao gồm:
        - Intro/Description
        - Lịch sử hoạt động
        - Sự nghiệp/Album/Âm nhạc
        - Infobox
        """
        try:
            url = self.base_url + quote(title.replace(' ', '_'))
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            result = {
                'intro': '',
                'career': '',
                'albums': '',
                'full_text': ''
            }
            
            # Lấy intro (đoạn đầu tiên)
            intro_para = soup.find('div', class_='mw-parser-output')
            if intro_para:
                first_p = intro_para.find('p')
                if first_p:
                    result['intro'] = first_p.get_text(strip=True)
            
            # Lấy các section về sự nghiệp, album, âm nhạc
            sections = soup.find_all(['h2', 'h3'])
            for section in sections:
                section_text = section.get_text(strip=True).lower()
                if any(keyword in section_text for keyword in ['sự nghiệp', 'career', 'album', 'âm nhạc', 'music', 'discography']):
                    content = []
                    next_elem = section.find_next_sibling()
                    while next_elem and next_elem.name not in ['h2', 'h3']:
                        if next_elem.name == 'p':
                            content.append(next_elem.get_text(strip=True))
                        next_elem = next_elem.find_next_sibling()
                    if content:
                        result['career'] = ' '.join(content)
            
            # Lấy full text (loại bỏ references, navbox)
            content_div = soup.find('div', id='mw-content-text')
            if content_div:
                # Loại bỏ các phần không cần thiết
                for element in content_div.find_all(['table', 'div', 'span'], 
                                                   class_=['navbox', 'reference', 'mw-references-wrap', 'mw-editsection']):
                    element.decompose()
                
                full_text = content_div.get_text(separator=' ', strip=True)
                full_text = re.sub(r'\s+', ' ', full_text)
                full_text = re.sub(r'\[\d+\]', '', full_text)
                result['full_text'] = full_text
            
            # Lấy infobox
            infobox = soup.find('table', class_='infobox')
            if infobox:
                result['infobox'] = infobox.get_text(separator=' ', strip=True)
            
            time.sleep(self.request_delay)
            return result
            
        except Exception as e:
            print(f"  ⚠ Lỗi khi lấy Wikipedia cho '{title}': {e}")
            return {'intro': '', 'career': '', 'albums': '', 'full_text': ''}


class KoreanMusicNER:
    """
    Mô hình NER để phát hiện thực thể MỚI trong văn bản
    Tập trung vào các thực thể liên quan đến âm nhạc Hàn Quốc
    """
    
    def __init__(self):
        # Từ khóa để nhận diện các loại thực thể
        self.artist_keywords = ['ca sĩ', 'singer', 'nghệ sĩ', 'artist', 'soloist', 'rapper', 'idol']
        self.group_keywords = ['nhóm nhạc', 'group', 'band', 'ban nhạc', 'boy group', 'girl group']
        self.album_keywords = ['album', 'mini album', 'ep', 'single album', 'full album', 'studio album']
        self.song_keywords = ['bài hát', 'song', 'ca khúc', 'đĩa đơn', 'single', 'track', 'ost']
        self.company_keywords = ['entertainment', 'công ty', 'company', 'label', 'agency', 'smtown', 'yg', 'jyp', 'hybe']
        self.genre_keywords = ['thể loại', 'genre', 'dòng nhạc', 'k-pop', 'ballad', 'hip hop', 'r&b', 'edm']
        
        # Pattern cho tên tiếng Hàn (Hangul)
        self.hangul_pattern = re.compile(r'[\uac00-\ud7af]+')
        
        # Pattern cho tên nghệ sĩ/nhóm nhạc (thường có chữ cái và số)
        self.name_pattern = re.compile(r'\b[A-Z][A-Z0-9\s&\.\']+\b')
        
    def extract_entities(self, text: str, existing_nodes: Dict[str, Dict] = None) -> List[Dict[str, Any]]:
        """
        Trích xuất các thực thể MỚI từ văn bản
        Trả về danh sách các entities với label và confidence
        """
        entities = []
        text_lower = text.lower()
        
        # 1. Tìm các nghệ sĩ (Artist)
        artists = self._extract_artists(text, text_lower)
        entities.extend(artists)
        
        # 2. Tìm các nhóm nhạc (Group)
        groups = self._extract_groups(text, text_lower)
        entities.extend(groups)
        
        # 3. Tìm các album
        albums = self._extract_albums(text, text_lower)
        entities.extend(albums)
        
        # 4. Tìm các bài hát
        songs = self._extract_songs(text, text_lower)
        entities.extend(songs)
        
        # 5. Tìm các công ty
        companies = self._extract_companies(text, text_lower)
        entities.extend(companies)
        
        # 6. Tìm các thể loại
        genres = self._extract_genres(text, text_lower)
        entities.extend(genres)
        
        # Loại bỏ trùng lặp và entities đã tồn tại
        if existing_nodes:
            existing_names = {node['name'].lower() for node in existing_nodes.values() if node.get('name')}
            entities = [e for e in entities if e['text'].lower() not in existing_names]
        
        # Merge trùng lặp
        return self._merge_duplicates(entities)
    
    def _extract_artists(self, text: str, text_lower: str) -> List[Dict]:
        """Trích xuất nghệ sĩ"""
        entities = []
        
        # Tìm các câu có từ khóa nghệ sĩ
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in self.artist_keywords):
                # Tìm tên có chữ Hàn hoặc tên tiếng Anh
                # Pattern: "tên" + keyword hoặc keyword + "tên"
                for kw in self.artist_keywords:
                    if kw in sentence_lower:
                        # Tìm tên trước hoặc sau keyword
                        parts = sentence_lower.split(kw)
                        for part in parts:
                            # Tìm tên trong phần này
                            name_match = self._extract_name_near_keyword(sentence, part, kw)
                            if name_match:
                                entities.append({
                                    'text': name_match,
                                    'label': 'Artist',
                                    'confidence': 0.75,
                                    'context': sentence[:200]
                                })
        
        # Tìm các tên có chữ Hàn (thường là nghệ sĩ Hàn Quốc)
        hangul_names = self.hangul_pattern.findall(text)
        for name in hangul_names:
            if len(name) >= 2 and len(name) <= 20:
                # Kiểm tra xem có phải nghệ sĩ không
                context = self._get_context(text, name, 100)
                if any(kw in context.lower() for kw in self.artist_keywords):
                    entities.append({
                        'text': name,
                        'label': 'Artist',
                        'confidence': 0.7,
                        'context': context
                    })
        
        return entities
    
    def _extract_groups(self, text: str, text_lower: str) -> List[Dict]:
        """Trích xuất nhóm nhạc"""
        entities = []
        
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in self.group_keywords):
                # Tìm tên nhóm (thường là chữ in hoa hoặc có chữ Hàn)
                name_match = self._extract_name_near_keyword(sentence, sentence_lower, 'nhóm nhạc')
                if name_match:
                    entities.append({
                        'text': name_match,
                        'label': 'Group',
                        'confidence': 0.75,
                        'context': sentence[:200]
                    })
        
        # Tìm các tên nhóm phổ biến (BTS, BLACKPINK, TWICE...)
        group_name_pattern = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z]+)?)\b')
        matches = group_name_pattern.findall(text)
        for match in matches:
            if len(match) >= 2 and len(match) <= 30:
                context = self._get_context(text, match, 100)
                if any(kw in context.lower() for kw in self.group_keywords):
                    entities.append({
                        'text': match,
                        'label': 'Group',
                        'confidence': 0.7,
                        'context': context
                    })
        
        return entities
    
    def _extract_albums(self, text: str, text_lower: str) -> List[Dict]:
        """Trích xuất album"""
        entities = []
        
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in self.album_keywords):
                # Tìm tên album (thường trong dấu ngoặc kép hoặc sau keyword)
                # Pattern: "Tên Album" hoặc album "Tên"
                quoted = re.findall(r'["\']([^"\']+)["\']', sentence)
                for name in quoted:
                    if len(name) >= 3:
                        entities.append({
                            'text': name,
                            'label': 'Album',
                            'confidence': 0.7,
                            'context': sentence[:200]
                        })
        
        return entities
    
    def _extract_songs(self, text: str, text_lower: str) -> List[Dict]:
        """Trích xuất bài hát"""
        entities = []
        
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in self.song_keywords):
                # Tìm tên bài hát trong dấu ngoặc kép
                quoted = re.findall(r'["\']([^"\']+)["\']', sentence)
                for name in quoted:
                    if len(name) >= 2:
                        entities.append({
                            'text': name,
                            'label': 'Song',
                            'confidence': 0.7,
                            'context': sentence[:200]
                        })
        
        return entities
    
    def _extract_companies(self, text: str, text_lower: str) -> List[Dict]:
        """Trích xuất công ty"""
        entities = []
        
        # Tìm các công ty phổ biến
        company_names = ['SM Entertainment', 'YG Entertainment', 'JYP Entertainment', 
                        'HYBE', 'Big Hit', 'CUBE', 'Starship', 'Pledis', 'FNC']
        
        for company in company_names:
            if company.lower() in text_lower:
                context = self._get_context(text, company, 100)
                entities.append({
                    'text': company,
                    'label': 'Company',
                    'confidence': 0.8,
                    'context': context
                })
        
        # Tìm pattern: "tên" + Entertainment/Company
        pattern = re.compile(r'([A-Z][A-Za-z\s]+)\s+(?:Entertainment|Company|Agency)')
        matches = pattern.findall(text)
        for match in matches:
            if len(match.strip()) >= 2:
                entities.append({
                    'text': match.strip(),
                    'label': 'Company',
                    'confidence': 0.7,
                    'context': self._get_context(text, match, 100)
                })
        
        return entities
    
    def _extract_genres(self, text: str, text_lower: str) -> List[Dict]:
        """Trích xuất thể loại"""
        entities = []
        
        # Thể loại phổ biến
        common_genres = ['K-pop', 'Ballad', 'Hip hop', 'R&B', 'EDM', 'Rock', 'Jazz', 
                        'Trot', 'Indie', 'Rap', 'Dance', 'Electronic']
        
        for genre in common_genres:
            if genre.lower() in text_lower:
                context = self._get_context(text, genre, 100)
                entities.append({
                    'text': genre,
                    'label': 'Genre',
                    'confidence': 0.7,
                    'context': context
                })
        
        return entities
    
    def _extract_name_near_keyword(self, sentence: str, part: str, keyword: str) -> Optional[str]:
        """Trích xuất tên gần keyword"""
        # Tìm từ/cụm từ có chữ Hàn hoặc chữ in hoa
        words = sentence.split()
        keyword_idx = -1
        for i, word in enumerate(words):
            if keyword.lower() in word.lower():
                keyword_idx = i
                break
        
        if keyword_idx >= 0:
            # Lấy 2-3 từ trước và sau keyword
            start = max(0, keyword_idx - 3)
            end = min(len(words), keyword_idx + 4)
            candidate = ' '.join(words[start:end])
            
            # Tìm tên trong candidate
            if self.hangul_pattern.search(candidate):
                return self.hangul_pattern.findall(candidate)[0]
            elif re.search(r'\b[A-Z][A-Z0-9\s&\.\']+\b', candidate):
                match = re.search(r'\b([A-Z][A-Z0-9\s&\.\']+)\b', candidate)
                if match:
                    return match.group(1).strip()
        
        return None
    
    def _get_context(self, text: str, entity: str, window: int = 100) -> str:
        """Lấy context xung quanh entity"""
        idx = text.lower().find(entity.lower())
        if idx == -1:
            return ''
        start = max(0, idx - window)
        end = min(len(text), idx + len(entity) + window)
        return text[start:end]
    
    def _merge_duplicates(self, entities: List[Dict]) -> List[Dict]:
        """Merge các entities trùng lặp"""
        seen = {}
        merged = []
        
        for entity in entities:
            key = entity['text'].lower().strip()
            if key not in seen:
                seen[key] = entity
                merged.append(entity)
            else:
                # Nếu confidence cao hơn, thay thế
                if entity['confidence'] > seen[key]['confidence']:
                    seen[key] = entity
        
        return list(seen.values())


class RelationExtractor:
    """Mô hình nhận dạng quan hệ giữa các thực thể"""
    
    def __init__(self):
        # Pattern cho các quan hệ
        self.patterns = {
            'MEMBER_OF': [
                r'(.+?)\s+(?:là|thành viên|member|của|of)\s+(.+?)(?:\.|,|$|và)',
                r'(.+?)\s+(?:gia nhập|joined)\s+(.+?)(?:\.|,|$)',
            ],
            'SINGS': [
                r'(.+?)\s+(?:hát|sings|performs|trình bày|ca khúc)\s+(.+?)(?:\.|,|$)',
                r'(.+?)\s+(?:bài hát|song|single)\s+(.+?)(?:\.|,|$)',
            ],
            'RELEASED': [
                r'(.+?)\s+(?:phát hành|released|ra mắt|tung ra)\s+(.+?)(?:\.|,|$)',
            ],
            'CONTAINS': [
                r'(.+?)\s+(?:chứa|contains|bao gồm|includes|có bài)\s+(.+?)(?:\.|,|$)',
            ],
            'IS_GENRE': [
                r'(.+?)\s+(?:thuộc thể loại|genre|thể loại|dòng nhạc)\s+(.+?)(?:\.|,|$)',
            ],
            'MANAGED_BY': [
                r'(.+?)\s+(?:được quản lý|managed by|signed to|thuộc|ký hợp đồng)\s+(.+?)(?:\.|,|$)',
            ],
            'COLLABORATED_WITH': [
                r'(.+?)\s+(?:hợp tác|collaborates|collaborated|ft\.|feat\.|với|và)\s+(.+?)(?:\.|,|$)',
            ],
            'PRODUCED_SONG': [
                r'(.+?)\s+(?:sản xuất|produced|producer)\s+(?:bài hát|song|ca khúc)\s+(.+?)(?:\.|,|$)',
            ],
            'WROTE': [
                r'(.+?)\s+(?:sáng tác|wrote|composed|tác giả)\s+(.+?)(?:\.|,|$)',
            ],
        }
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Trích xuất quan hệ giữa các entities"""
        relationships = []
        text_lower = text.lower()
        
        # Tìm các cặp entity gần nhau
        entity_pairs = self._find_entity_pairs(text, entities)
        
        for entity1, entity2, context in entity_pairs:
            rel_type = self._classify_relationship(entity1, entity2, context)
            if rel_type:
                relationships.append({
                    'source': entity1['text'],
                    'target': entity2['text'],
                    'source_label': entity1['label'],
                    'target_label': entity2['label'],
                    'type': rel_type,
                    'confidence': 0.75,
                    'context': context[:200]
                })
        
        return relationships
    
    def _find_entity_pairs(self, text: str, entities: List[Dict]) -> List[Tuple]:
        """Tìm các cặp entity xuất hiện gần nhau"""
        pairs = []
        
        # Tìm vị trí của mỗi entity trong text
        entity_positions = []
        for entity in entities:
            text_lower = text.lower()
            entity_lower = entity['text'].lower()
            idx = text_lower.find(entity_lower)
            if idx != -1:
                entity_positions.append({
                    'entity': entity,
                    'start': idx,
                    'end': idx + len(entity_lower)
                })
        
        # Sắp xếp theo vị trí
        entity_positions.sort(key=lambda x: x['start'])
        
        # Tìm các cặp gần nhau
        window_size = 300
        seen_pairs = set()
        
        for i in range(len(entity_positions)):
            for j in range(i + 1, len(entity_positions)):
                e1 = entity_positions[i]
                e2 = entity_positions[j]
                
                if e2['start'] - e1['end'] > window_size:
                    break
                
                pair_key = (e1['entity']['text'].lower(), e2['entity']['text'].lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                # Lấy context
                context_start = max(0, e1['start'] - 100)
                context_end = min(len(text), e2['end'] + 100)
                context = text[context_start:context_end]
                
                pairs.append((e1['entity'], e2['entity'], context))
        
        return pairs
    
    def _classify_relationship(self, entity1: Dict, entity2: Dict, context: str) -> Optional[str]:
        """Phân loại relationship"""
        context_lower = context.lower()
        label1 = entity1['label']
        label2 = entity2['label']
        
        # Kiểm tra patterns
        for rel_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, context_lower, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        e1_text = entity1['text'].lower()
                        e2_text = entity2['text'].lower()
                        
                        if (e1_text in groups[0].lower() and e2_text in groups[1].lower()) or \
                           (e1_text in groups[1].lower() and e2_text in groups[0].lower()):
                            if self._is_valid_relationship(label1, label2, rel_type):
                                return rel_type
        
        # Heuristic
        return self._heuristic_relationship(label1, label2, context_lower)
    
    def _is_valid_relationship(self, label1: str, label2: str, rel_type: str) -> bool:
        """Kiểm tra tính hợp lệ"""
        valid = {
            'MEMBER_OF': [('Artist', 'Group')],
            'SINGS': [('Artist', 'Song'), ('Group', 'Song')],
            'RELEASED': [('Artist', 'Album'), ('Group', 'Album')],
            'CONTAINS': [('Album', 'Song')],
            'IS_GENRE': [('Artist', 'Genre'), ('Group', 'Genre'), ('Song', 'Genre'), ('Album', 'Genre')],
            'MANAGED_BY': [('Artist', 'Company'), ('Group', 'Company')],
            'COLLABORATED_WITH': [('Artist', 'Artist'), ('Group', 'Group'), ('Artist', 'Group')],
            'PRODUCED_SONG': [('Artist', 'Song'), ('Group', 'Song')],
            'WROTE': [('Artist', 'Song'), ('Group', 'Song')],
        }
        
        valid_pairs = valid.get(rel_type, [])
        return (label1, label2) in valid_pairs
    
    def _heuristic_relationship(self, label1: str, label2: str, context: str) -> Optional[str]:
        """Heuristic dựa trên label"""
        if label1 == 'Artist' and label2 == 'Group':
            if any(kw in context for kw in ['thành viên', 'member']):
                return 'MEMBER_OF'
        elif label1 == 'Group' and label2 == 'Album':
            if any(kw in context for kw in ['phát hành', 'released']):
                return 'RELEASED'
        elif label1 == 'Album' and label2 == 'Song':
            if any(kw in context for kw in ['chứa', 'contains']):
                return 'CONTAINS'
        return None


class CompleteGraphEnricher:
    """Hệ thống làm giàu dữ liệu đầy đủ - tạo cả nodes mới và relationships mới"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.text_collector = TextDataCollector()
        self.ner = KoreanMusicNER()
        self.relation_extractor = RelationExtractor()
        
        print(f"✓ Đã kết nối với Neo4j: {uri}")
    
    def close(self):
        self.driver.close()
    
    def get_existing_nodes(self) -> Dict[str, Dict]:
        """Lấy nodes hiện có để tránh trùng lặp"""
        def get_nodes(tx):
            query = """
            MATCH (n)
            WHERE n.name IS NOT NULL AND n.id IS NOT NULL
            RETURN n.id as id, labels(n) as labels, n.name as name
            """
            result = tx.run(query)
            nodes = {}
            for record in result:
                nodes[record['id']] = {
                    'id': record['id'],
                    'label': record['labels'][0] if record['labels'] else 'Entity',
                    'name': record['name']
                }
            return nodes
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            return session.execute_read(get_nodes)
    
    def enrich_node(self, node_id: str, node_data: Dict, existing_nodes: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Làm giàu một node"""
        print(f"\n📝 Đang làm giàu node: {node_data.get('name', node_id)}")
        
        # Lấy text từ Wikipedia
        url = node_data.get('url')
        if not url or 'wikipedia.org' not in url:
            return [], []
        
        title = url.split('/wiki/')[-1] if '/wiki/' in url else None
        if not title:
            return [], []
        
        title = unquote(title).replace('_', ' ')
        text_data = self.text_collector.fetch_wikipedia_full_text(title)
        
        # Kết hợp tất cả text
        full_text = ' '.join([text_data.get('intro', ''), 
                             text_data.get('career', ''),
                             text_data.get('full_text', '')])
        
        if len(full_text) < 100:
            return [], []
        
        print(f"  ✓ Đã thu thập {len(full_text)} ký tự text")
        
        # 1. NER - Tìm entities MỚI
        new_entities = self.ner.extract_entities(full_text, existing_nodes)
        print(f"  ✓ Tìm thấy {len(new_entities)} entities MỚI")
        
        # 2. Relation Extraction - Tìm relationships
        # Kết hợp entities mới và entities cũ để tìm relationships
        all_entities = list(new_entities)
        for node in existing_nodes.values():
            all_entities.append({
                'text': node['name'],
                'label': node['label'],
                'confidence': 1.0
            })
        
        relationships = self.relation_extractor.extract_relationships(full_text, all_entities)
        print(f"  ✓ Tìm thấy {len(relationships)} relationships")
        
        return new_entities, relationships
    
    def update_neo4j(self, new_entities: List[Dict], relationships: List[Dict], batch_size: int = 50):
        """Cập nhật Neo4j với nodes mới và relationships mới"""
        def run_write(tx, query, parameters=None):
            return tx.run(query, parameters or {})
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            # 1. Tạo nodes mới
            if new_entities:
                node_query = """
                UNWIND $batch AS e
                MERGE (n:Entity {id: e.id})
                SET n.name = e.name,
                    n.label = e.label,
                    n.enriched = true,
                    n.enrichment_confidence = e.confidence,
                    n.enrichment_source = 'ner_wikipedia'
                """
                
                batch = []
                for entity in new_entities:
                    entity_id = f"ENRICHED_{abs(hash(entity['text']))}"
                    batch.append({
                        'id': entity_id,
                        'name': entity['text'],
                        'label': entity['label'],
                        'confidence': entity['confidence']
                    })
                
                if batch:
                    for i in range(0, len(batch), batch_size):
                        session.execute_write(run_write, node_query, {'batch': batch[i:i+batch_size]})
                    print(f"  ✓ Đã tạo {len(batch)} nodes MỚI")
            
            # 2. Tạo relationships mới
            if relationships:
                # Tạo mapping từ tên sang ID
                name_to_id = {}
                for entity in new_entities:
                    entity_id = f"ENRICHED_{abs(hash(entity['text']))}"
                    name_to_id[entity['text'].lower()] = entity_id
                
                # Lấy IDs từ existing nodes
                existing_result = session.run("MATCH (n) RETURN n.id as id, n.name as name")
                for record in existing_result:
                    name_to_id[record['name'].lower() if record['name'] else ''] = record['id']
                
                # Tạo relationships
                rel_batch = []
                for rel in relationships:
                    source_id = name_to_id.get(rel['source'].lower())
                    target_id = name_to_id.get(rel['target'].lower())
                    
                    if source_id and target_id:
                        rel_batch.append({
                            'source': source_id,
                            'target': target_id,
                            'type': rel['type'],
                            'context': rel.get('context', '')[:300],
                            'confidence': rel.get('confidence', 0.75)
                        })
                
                if rel_batch:
                    # Group by relationship type
                    for rel_type in set(r['type'] for r in rel_batch):
                        type_batch = [r for r in rel_batch if r['type'] == rel_type]
                        
                        query = f"""
                        UNWIND $batch AS r
                        MATCH (s {{id: r.source}}), (t {{id: r.target}})
                        WHERE s IS NOT NULL AND t IS NOT NULL
                        MERGE (s)-[rel:`{rel_type}` {{enriched: true}}]->(t)
                        SET rel.context = r.context,
                            rel.enrichment_confidence = r.confidence,
                            rel.enrichment_source = 'relation_extraction'
                        """
                        
                        for i in range(0, len(type_batch), batch_size):
                            session.execute_write(run_write, query, {'batch': type_batch[i:i+batch_size]})
                    
                    print(f"  ✓ Đã tạo {len(rel_batch)} relationships MỚI")
    
    def enrich_all(self, limit: int = None, batch_size: int = 10):
        """Làm giàu tất cả nodes"""
        print("=" * 80)
        print("BẮT ĐẦU LÀM GIÀU DỮ LIỆU ĐỒ THỊ (PHIÊN BẢN ĐẦY ĐỦ)")
        print("=" * 80)
        print("📌 Nguồn: Wikipedia (intro, career, full text)")
        print("📌 NER: Phát hiện entities MỚI")
        print("📌 Relation Extraction: Phát hiện relationships MỚI")
        print("=" * 80)
        
        # Lấy nodes hiện có
        print("\n📊 Đang lấy nodes hiện có...")
        existing_nodes = self.get_existing_nodes()
        print(f"✓ Tìm thấy {len(existing_nodes)} nodes hiện có")
        
        # Lấy nodes có URL Wikipedia
        def get_nodes_with_url(tx):
            query = """
            MATCH (n)
            WHERE n.url IS NOT NULL AND n.url CONTAINS 'wikipedia.org'
            RETURN n.id as id, n.name as name, n.url as url, labels(n) as labels
            """
            if limit:
                query += f" LIMIT {limit}"
            result = tx.run(query)
            return [dict(record) for record in result]
        
        with self.driver.session(database=self.database) if self.database else self.driver.session() as session:
            nodes_to_process = session.execute_read(get_nodes_with_url)
        
        print(f"✓ Có {len(nodes_to_process)} nodes có URL Wikipedia để xử lý")
        
        all_new_entities = []
        all_relationships = []
        processed = 0
        
        for node_data in nodes_to_process:
            try:
                node_id = node_data['id']
                entities, relationships = self.enrich_node(node_id, node_data, existing_nodes)
                all_new_entities.extend(entities)
                all_relationships.extend(relationships)
                
                # Cập nhật existing_nodes với entities mới
                for entity in entities:
                    entity_id = f"ENRICHED_{abs(hash(entity['text']))}"
                    existing_nodes[entity_id] = {
                        'id': entity_id,
                        'label': entity['label'],
                        'name': entity['text']
                    }
                
                processed += 1
                if processed % batch_size == 0:
                    print(f"\n💾 Đang cập nhật Neo4j (đã xử lý {processed}/{len(nodes_to_process)})...")
                    self.update_neo4j(all_new_entities, all_relationships)
                    all_new_entities = []
                    all_relationships = []
                    
            except Exception as e:
                print(f"  ❌ Lỗi: {e}")
                continue
        
        # Cập nhật lần cuối
        if all_new_entities or all_relationships:
            print(f"\n💾 Đang cập nhật Neo4j lần cuối...")
            self.update_neo4j(all_new_entities, all_relationships)
        
        print("\n" + "=" * 80)
        print("HOÀN TẤT LÀM GIÀU DỮ LIỆU")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description='Làm giàu dữ liệu đồ thị - Tạo nodes mới và relationships mới')
    parser.add_argument('--neo4j-uri', type=str, required=True)
    parser.add_argument('--neo4j-user', type=str, required=True)
    parser.add_argument('--neo4j-pass', type=str, required=True)
    parser.add_argument('--neo4j-db', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    enricher = CompleteGraphEnricher(args.neo4j_uri, args.neo4j_user, args.neo4j_pass, args.neo4j_db)
    try:
        enricher.enrich_all(limit=args.limit, batch_size=args.batch_size)
    finally:
        enricher.close()


if __name__ == '__main__':
    main()








