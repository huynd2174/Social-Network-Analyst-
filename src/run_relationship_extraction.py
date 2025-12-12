# -*- coding: utf-8 -*-
"""
MÔ HÌNH NHẬN DẠNG MỐI QUAN HỆ (RELATIONSHIP EXTRACTION)
=========================================================
Mô hình Rule-based để trích xuất các mối quan hệ giữa các thực thể K-pop

Các loại quan hệ được hỗ trợ (TẤT CẢ ĐỀU ĐÚNG CHIỀU):
- MEMBER_OF: Artist → Group (Artist là thành viên hoặc cựu thành viên của Group)
- RELEASED: Artist/Group → Album (Artist/Group phát hành Album)
- SINGS: Artist/Group → Song (Artist/Group hát Song)
- CONTAINS: Album → Song (Album chứa Song)
- MANAGED_BY: Artist/Group → Company (Artist/Group được quản lý bởi Company)
- PRODUCE_SONG: Artist → Song (Artist sản xuất Song)
- PRODUCE_ALBUM: Artist → Album (Artist sản xuất Album)
- WROTE: Artist → Song (Artist sáng tác Song)
- SUBUNIT_OF: Group → Group (Group là nhóm nhỏ của Group khác)
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict

# =====================================================
# HÀM CHUẨN HÓA TÊN NODE
# =====================================================
def normalize_node_name(name: str) -> str:
    """
    Chuẩn hóa tên node để loại bỏ suffix như "(ca sĩ)", "(nhóm nhạc)", etc.
    Và chuẩn hóa khoảng trắng để đảm bảo so sánh chính xác.
    
    Ví dụ:
    - "IU (ca sĩ)" -> "IU"
    - "BTS (nhóm nhạc)" -> "BTS"
    - "EXO (nhóm nhạc Hàn Quốc)" -> "EXO"
    - "Big  Bang" -> "Big Bang" (loại bỏ khoảng trắng thừa)
    
    Args:
        name: Tên node gốc
        
    Returns:
        Tên node đã được chuẩn hóa
    """
    if not name:
        return ""
    
    # Loại bỏ các pattern trong ngoặc đơn ở cuối
    # Pattern: (ca sĩ), (nhóm nhạc), (ca sĩ Hàn Quốc), etc.
    # NHƯNG giữ lại nếu là (album), (bài hát), (EP) - vì đó là thông tin quan trọng
    name = re.sub(r'\s*\([^)]*(?:ca sĩ|nhóm nhạc|ban nhạc|nghệ sĩ|singer|group|band)[^)]*\)\s*$', '', name, flags=re.IGNORECASE)
    
    # Chuẩn hóa khoảng trắng: loại bỏ khoảng trắng thừa (nhiều khoảng trắng liên tiếp -> 1 khoảng)
    name = re.sub(r'\s+', ' ', name)
    
    # Loại bỏ khoảng trắng ở đầu và cuối
    name = name.strip()
    
    return name

# =====================================================
# DANH SÁCH PRODUCERS VÀ SONGWRITERS (TỰ ĐỘNG TỪ NGHỀ NGHIỆP)
# =====================================================
# Danh sách cơ bản của các producer nổi tiếng (không phải idol)
# Sẽ được BỔ SUNG tự động từ field "Nghề nghiệp" trong graph nodes

BASE_KNOWN_PRODUCERS = {
    # SM Entertainment producers
    'yoo young-jin', 'kenzie', 'lee soo-man', 'hitchhiker',
    # JYP Entertainment producers
    'park jin-young', 'j.y. park', 'jyp',
    # YG Entertainment producers
    'teddy', 'teddy park', 'choice37', 'future bounce',
    # Big Hit / HYBE producers
    'bang si-hyuk', 'pdogg', 'slow rabbit', 'supreme boi', 'el capitxn',
    # Starship producers
    'black eyed pilseung', 'rado',
    # Cube producers
    'shinsadong tiger',
    # Các producer độc lập nổi tiếng
    'brave brothers', 'duble sidekick', 'e-tribe', 'sweetune', 'ryan jhun',
    'ldn noise', 'dem jointz', 'moonshine', 'full8loom',
}

BASE_KNOWN_SONGWRITERS = BASE_KNOWN_PRODUCERS | {
    # Các nhạc sĩ/producer độc lập
    'nell', 'crush', 'dean', 'heize', 'zion.t', 'tablo',
}

# Từ khóa để nhận dạng PRODUCER từ field nghề nghiệp
PRODUCER_OCCUPATION_KEYWORDS = [
    'nhà sản xuất', 'sản xuất âm nhạc', 'producer', 'music producer',
    'nhà sản xuất âm nhạc', 'producer âm nhạc',
]

# Từ khóa để nhận dạng SONGWRITER từ field nghề nghiệp  
SONGWRITER_OCCUPATION_KEYWORDS = [
    'nhạc sĩ', 'sáng tác nhạc', 'sáng tác', 'songwriter', 'composer',
    'lyricist', 'viết nhạc', 'tác giả nhạc',
]

def build_producers_songwriters_from_graph(graph_nodes: Dict) -> Tuple[Set[str], Set[str]]:
    """
    Xây dựng danh sách producer và songwriter từ field nghề nghiệp trong graph
    
    Returns:
        Tuple[Set[str], Set[str]]: (producers, songwriters)
    """
    producers = set(BASE_KNOWN_PRODUCERS)
    songwriters = set(BASE_KNOWN_SONGWRITERS)
    
    for node_id, node_data in graph_nodes.items():
        if node_data.get('label') != 'Artist':
            continue
            
        infobox = node_data.get('infobox', {})
        occupation = infobox.get('Nghề nghiệp', '').lower()
        
        if not occupation:
            continue
        
        node_name = node_data.get('name', node_data.get('title', node_id))
        node_name_lower = node_name.lower()
        
        # Kiểm tra có phải producer không
        for kw in PRODUCER_OCCUPATION_KEYWORDS:
            if kw in occupation:
                producers.add(node_name_lower)
                break
        
        # Kiểm tra có phải songwriter không
        for kw in SONGWRITER_OCCUPATION_KEYWORDS:
            if kw in occupation:
                songwriters.add(node_name_lower)
                break
    
    return producers, songwriters

# Placeholder - sẽ được cập nhật trong main()
KNOWN_PRODUCERS = set(BASE_KNOWN_PRODUCERS)
KNOWN_SONGWRITERS = set(BASE_KNOWN_SONGWRITERS)

# Map Album (normalized) -> set of allowed source names (normalized) từ graph gốc cho quan hệ RELEASED
RELEASED_ALLOWED_SOURCES_BY_ALBUM: Dict[str, Set[str]] = {}

# Các context mà entity 'solo' KHÔNG nên được dùng làm node (chỉ là tính từ/trạng thái)
SOLO_INVALID_CONTEXT_PHRASES = [
    'bài hát solo', 'ca khúc solo', 'song solo',
    'single solo', 'album solo', 'đĩa đơn solo',
    'sân khấu solo', 'stage solo', 'biểu diễn solo', 'performance solo',
    'nghệ sĩ solo', 'artist solo', 'ca sĩ solo', 'singer solo',
    'thành viên solo', 'member solo', 'hoạt động solo', 'activity solo',
    'solo track', 'solo title track',
    'buổi hòa nhạc solo', 'concert solo', 'concert tại mỹ solo',
    'solo artist', 'solo debut', 'go solo', 'solo day', 'nhảy solo', 'dance solo',
]

# Các context mà 'champion' không phải Song (ví dụ: Show Champion)
CHAMPION_INVALID_CONTEXT_PHRASES = [
    'show champion',
]

# =====================================================
# PATTERNS CHO RELATIONSHIP EXTRACTION
# =====================================================

# Patterns để nhận dạng quan hệ MEMBER_OF
MEMBER_OF_PATTERNS = [
    # Tiếng Việt
    r'(.+?)\s+(?:là\s+)?(?:thành viên|member)\s+(?:của|of|trong)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:thuộc|nằm trong)\s+(?:nhóm nhạc|nhóm|group)\s+(.+?)(?:\.|,|$)',
    r'(?:nhóm nhạc|nhóm|group)\s+(.+?)\s+(?:gồm|bao gồm|có|gồm có)\s+(?:các thành viên|thành viên)\s*:?\s*(.+?)(?:\.|$)',
    r'(.+?)\s+(?:gia nhập|tham gia|joined)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:là\s+)?(?:một trong|one of)\s+(?:các\s+)?(?:thành viên|members)\s+(?:của|of)\s+(.+?)(?:\.|,|$)',
    r'(?:các\s+thành viên|members)\s+(?:của|of)\s+(.+?)\s+(?:bao gồm|include|includes)\s+(.+?)(?:\.|,|$)',
    # Tiếng Anh
    r'(.+?)\s+(?:is|are|was|were)\s+(?:a\s+)?member(?:s)?\s+of\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+joined\s+(.+?)(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ FORMER_MEMBER_OF
FORMER_MEMBER_PATTERNS = [
    r'(.+?)\s+(?:là\s+)?(?:cựu thành viên|former member)\s+(?:của|of)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:đã rời|rời khỏi|left|departed from)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:was|were)\s+(?:a\s+)?former\s+member(?:s)?\s+of\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:từng là|used to be)\s+(?:thành viên|member)\s+(?:của|of)\s+(.+?)(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ RELEASED
RELEASED_PATTERNS = [
    r'(.+?)\s+(?:phát hành|ra mắt|released|dropped)\s+(?:album|EP|mini-album|single)\s+(.+?)(?:\.|,|$)',
    r'(?:album|EP|mini-album)\s+(.+?)\s+(?:của|by|from)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:tung|tung ra|comeback với)\s+(?:album|EP)\s+(.+?)(?:\.|,|$)',
    r'(?:album|EP|mini-album)\s+["\']?(.+?)["\']?\s+(?:được phát hành bởi|was released by|by)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:trở lại|returned)\s+(?:với|with)\s+(?:album|EP|mini-album)\s+(.+?)(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ SINGS
SINGS_PATTERNS = [
    # Ca sĩ / nhóm hát bài hát
    r'(.+?)\s+(?:hát|trình bày|performs?|sings?)\s+(?:ca khúc|bài hát|song)\s+["\']?(.+?)["\']?(?:\.|,|$)',
    r'(?:ca khúc|bài hát|song)\s+["\']?(.+?)["\']?\s+(?:của|by|from)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:với|with)\s+(?:ca khúc chủ đề|title track)\s+["\']?(.+?)["\']?(?:\.|,|$)',
    # Ca sĩ / nhóm phát hành single / bài hát -> SINGS (không dùng RELEASED cho Song)
    r'(.+?)\s+(?:phát hành|ra mắt|released|dropped)\s+(?:single|bài hát|ca khúc|song)\s+["\']?(.+?)["\']?(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ CONTAINS
CONTAINS_PATTERNS = [
    r'(?:album|EP)\s+(.+?)\s+(?:chứa|bao gồm|contains?|includes?)\s+(?:các bài hát|bài hát|ca khúc)\s+(.+?)(?:\.|,|$)',
    r'(?:album|EP)\s+(.+?)\s+(?:gồm|gồm có)\s+(\d+)\s+(?:bài hát|track)',
    r'(?:bài hát|ca khúc|track)\s+["\']?(.+?)["\']?\s+(?:nằm trong|thuộc|from|in)\s+(?:album|EP)\s+(.+?)(?:\.|,|$)',
    r'(?:album|EP)\s+(.+?)\s+(?:contains?|includes?)\s+["\']?(.+?)["\']?(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ MANAGED_BY
MANAGED_BY_PATTERNS = [
    r'(.+?)\s+(?:được quản lý bởi|managed by|under|thuộc|trực thuộc)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:ký hợp đồng với|signed with|signed to)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:là nghệ sĩ của|is an artist of|belongs to)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:ra mắt dưới|debuted under)\s+(?:hãng|label|company)\s+(.+?)(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ PRODUCE_SONG
PRODUCE_SONG_PATTERNS = [
    r'(.+?)\s+(?:sản xuất|produced?)\s+(?:bài hát|ca khúc|song)\s+["\']?(.+?)["\']?(?:\.|,|$)',
    r'(?:bài hát|ca khúc|song)\s+["\']?(.+?)["\']?\s+(?:được sản xuất bởi|produced by)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:là producer của|producer of)\s+(?:bài hát|ca khúc|song)\s+["\']?(.+?)["\']?(?:\.|,|$)',
    r'(.+?)\s+(?:đảm nhận|phụ trách)\s+(?:sản xuất|production)\s+(?:bài hát|ca khúc|song)\s+["\']?(.+?)["\']?(?:\.|,|$)',
    r'(.+?)\s+(?:produced?|co-produced?)\s+(?:the\s+)?(?:song|track)\s+["\']?(.+?)["\']?(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ PRODUCE_ALBUM
PRODUCE_ALBUM_PATTERNS = [
    r'(.+?)\s+(?:sản xuất|produced?)\s+(?:album|EP|mini-album)\s+(.+?)(?:\.|,|$)',
    r'(?:album|EP|mini-album)\s+(.+?)\s+(?:được sản xuất bởi|produced by)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:là producer của|producer of)\s+(?:album|EP|mini-album)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:đảm nhận|phụ trách)\s+(?:sản xuất|production)\s+(?:album|EP|mini-album)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:produced?|co-produced?)\s+(?:the\s+)?(?:album|EP|mini-album)\s+["\']?(.+?)["\']?(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ WROTE
WROTE_PATTERNS = [
    r'(.+?)\s+(?:sáng tác|viết|wrote|composed?)\s+(?:bài hát|ca khúc|lời|song|tác phẩm)?\s*["\']?(.+?)["\']?(?:\.|,|$)',
    r'(?:bài hát|ca khúc|tác phẩm)\s+(?:["\'])?(.+?)(?:["\'])?\s+(?:do|được viết bởi|written by|composed by)\s+(.+?)(?:\.|,|$)',
    r'do chính\s+(.+?)\s+(?:sáng tác|viết)\s+(?:bài hát|ca khúc|tác phẩm)?\s*["\']?(.+?)["\']?(?:\.|,|$)',
    r'(.+?)\s+(?:là tác giả|tác giả|songwriter|composer)\s+(?:của|of)?\s*(?:bài hát|ca khúc|song)?\s*["\']?(.+?)["\']?(?:\.|,|$)',
    r'(?:bài hát|ca khúc|song)\s+["\']?(.+?)["\']?\s+(?:có lời|lyrics)\s+(?:do|bởi|by)\s+(.+?)(?:\.|,|$)',
]

# Patterns để nhận dạng quan hệ SUBUNIT_OF
SUBUNIT_PATTERNS = [
    r'(.+?)\s+(?:là nhóm nhỏ|is a subunit|subunit)\s+(?:của|of)\s+(.+?)(?:\.|,|$)',
    r'(?:nhóm nhỏ|subunit)\s+(.+?)\s+(?:của|of|from)\s+(.+?)(?:\.|,|$)',
    r'(.+?)\s+(?:là\s+)?(?:subgroup|sub-group)\s+(?:của|of)\s+(.+?)(?:\.|,|$)',
]

# =====================================================
# KEYWORD-BASED RELATIONSHIP DETECTION
# =====================================================

# Từ khóa cho từng loại quan hệ
RELATIONSHIP_KEYWORDS = {
    'MEMBER_OF': {
        'vi': ['thành viên', 'member', 'gồm', 'bao gồm', 'có các thành viên', 
               'cựu thành viên', 'đã rời', 'rời khỏi', 'rời nhóm'],  # Bao gồm cả cựu thành viên
        'en': ['member', 'members', 'consists of', 'comprising',
               'former member', 'ex-member', 'left', 'departed', 'former'],
    },
    'RELEASED': {
        'vi': ['phát hành', 'ra mắt', 'tung ra', 'comeback'],
        'en': ['released', 'dropped', 'debut', 'comeback'],
    },
    'SINGS': {
        'vi': ['hát', 'trình bày', 'thể hiện', 'ca khúc chủ đề'],
        'en': ['sings', 'performs', 'title track'],
    },
    'CONTAINS': {
        'vi': ['chứa', 'bao gồm', 'gồm có', 'nằm trong album'],
        'en': ['contains', 'includes', 'track', 'from album'],
    },
    'MANAGED_BY': {
        'vi': ['quản lý', 'thuộc', 'trực thuộc', 'ký hợp đồng'],
        'en': ['managed by', 'under', 'signed with', 'belongs to'],
    },
    'PRODUCE_SONG': {
        'vi': ['sản xuất bài hát', 'sản xuất ca khúc', 'producer bài hát'],
        'en': ['produced song', 'song producer', 'producer of song'],
    },
    'PRODUCE_ALBUM': {
        'vi': ['sản xuất album', 'producer album'],
        'en': ['produced album', 'album producer', 'producer of album'],
    },
    'WROTE': {
        'vi': ['sáng tác', 'viết lời', 'tác giả'],
        'en': ['wrote', 'composed', 'written by', 'songwriter'],
    },
    'SUBUNIT_OF': {
        'vi': ['nhóm nhỏ', 'nhóm con'],
        'en': ['subunit', 'sub-unit', 'unit'],
    },
}

# Ma trận quan hệ hợp lệ: (source_type, target_type) -> [valid_relationships]
VALID_RELATIONSHIPS = {
    ('Artist', 'Group'): ['MEMBER_OF'],  # MEMBER_OF bao gồm cả cựu thành viên
    ('Group', 'Artist'): [],  # Không có quan hệ ngược
    ('Artist', 'Album'): ['RELEASED', 'PRODUCE_ALBUM'],
    ('Group', 'Album'): ['RELEASED'],
    ('Artist', 'Song'): ['SINGS', 'PRODUCE_SONG', 'WROTE'],
    ('Group', 'Song'): ['SINGS'],  # RELEASED chỉ cho Album, không cho Song
    ('Album', 'Song'): ['CONTAINS'],
    ('Song', 'Album'): [],  # Reverse - handled differently
    ('Artist', 'Company'): ['MANAGED_BY', 'SIGNED_WITH'],
    ('Group', 'Company'): ['MANAGED_BY', 'SIGNED_WITH'],
    ('Company', 'Artist'): [],  # Company không quản lý trực tiếp, dùng MANAGED_BY ngược lại
    ('Company', 'Group'): [],  # Company không quản lý trực tiếp, dùng MANAGED_BY ngược lại
    ('Group', 'Group'): ['SUBUNIT_OF'],
    ('Artist', 'Artist'): [],  # Không có quan hệ Artist-Artist
}

# =====================================================
# CLASS RELATIONSHIP EXTRACTOR
# =====================================================

class RelationshipExtractor:
    """Mô hình nhận dạng mối quan hệ giữa các thực thể K-pop"""
    
    def __init__(self):
        """Khởi tạo Relationship Extractor"""
        self.patterns = {
            'MEMBER_OF': MEMBER_OF_PATTERNS + FORMER_MEMBER_PATTERNS,  # Gộp patterns
            'RELEASED': RELEASED_PATTERNS,
            'SINGS': SINGS_PATTERNS,
            'CONTAINS': CONTAINS_PATTERNS,
            'MANAGED_BY': MANAGED_BY_PATTERNS,
            'PRODUCE_SONG': PRODUCE_SONG_PATTERNS,
            'PRODUCE_ALBUM': PRODUCE_ALBUM_PATTERNS,
            'WROTE': WROTE_PATTERNS,
            'SUBUNIT_OF': SUBUNIT_PATTERNS,
        }
        
        self.keywords = RELATIONSHIP_KEYWORDS
        self.valid_relationships = VALID_RELATIONSHIPS
        
        # Statistics
        self.stats = defaultdict(int)
        # Mapping chuẩn hóa -> tên gốc ưu tiên (sẽ được set từ main)
        # Key: (normalized_lower, type) -> original_name_from_graph_or_ner
        self.normalized_to_original: Dict[Tuple[str, str], str] = {}
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Trích xuất tất cả các mối quan hệ từ văn bản
        
        Args:
            text: Văn bản nguồn
            entities: Danh sách các entities đã được nhận dạng
            
        Returns:
            Danh sách các relationships được trích xuất
        """
        relationships = []
        
        if not text or not entities:
            return relationships
        
        # Tìm vị trí của các entities trong text
        entity_positions = self._find_entity_positions(text, entities)
        
        # Tìm các cặp entity gần nhau
        entity_pairs = self._find_entity_pairs(text, entity_positions)
        
        # Phân loại quan hệ cho từng cặp
        for entity1, entity2, context in entity_pairs:
            result = self._classify_relationship(entity1, entity2, context)
            if result:
                rel_type, source_entity, target_entity = result
                confidence = self._calculate_confidence(source_entity, target_entity, context, rel_type)
                # Lưu TÊN GỐC (original_text) thay vì normalized text
                source_original = source_entity.get('original_text', source_entity.get('text', ''))
                target_original = target_entity.get('original_text', target_entity.get('text', ''))
                
                # Chuẩn hóa về tên node gốc trong graph nếu có mapping
                source_type = source_entity.get('type', source_entity.get('label', 'Entity'))
                target_type = target_entity.get('type', target_entity.get('label', 'Entity'))
                try:
                    if self.normalized_to_original:
                        src_norm = normalize_node_name(source_original).lower()
                        tgt_norm = normalize_node_name(target_original).lower()
                        src_key = (src_norm, source_type)
                        tgt_key = (tgt_norm, target_type)
                        if src_key in self.normalized_to_original:
                            source_original = self.normalized_to_original[src_key]
                        if tgt_key in self.normalized_to_original:
                            target_original = self.normalized_to_original[tgt_key]
                except Exception:
                    # Nếu có lỗi ngoài ý muốn, fallback dùng original hiện tại
                    pass
                relationships.append({
                    'source': source_original,  # Tên gốc để lưu
                    'source_type': source_type,
                    'target': target_original,  # Tên gốc để lưu
                    'target_type': target_type,
                    'type': rel_type,
                    'confidence': confidence,
                    'context': context[:300],
                    'method': 'rule-based',
                })
                self.stats[rel_type] += 1
        
        # Loại bỏ duplicates
        relationships = self._remove_duplicate_relationships(relationships)
        
        # Lọc các quan hệ không hợp lệ
        relationships = self._filter_invalid_relationships(relationships)
        
        return relationships
    
    def _filter_invalid_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """
        Lọc các quan hệ không hợp lệ
        
        Args:
            relationships: Danh sách quan hệ
            
        Returns:
            Danh sách quan hệ đã được lọc
        """
        valid = []
        
        for rel in relationships:
            source_type = rel.get('source_type', '')
            target_type = rel.get('target_type', '')
            rel_type = rel.get('type', '')
            source = rel.get('source', '').strip()
            target = rel.get('target', '').strip()
            
            # Loại bỏ nếu source hoặc target rỗng
            if not source or not target:
                continue
            
            # Loại bỏ nếu source == target
            if source.lower() == target.lower():
                continue
            
            # Kiểm tra entity types hợp lệ
            valid_types = ['Artist', 'Group', 'Album', 'Song', 'Company']
            if source_type not in valid_types or target_type not in valid_types:
                continue
            
            # Kiểm tra quan hệ có hợp lệ với entity types không
            valid_pairs = self.valid_relationships.get((source_type, target_type), [])
            if rel_type not in valid_pairs:
                continue
            
            # Loại bỏ các quan hệ có confidence quá thấp (TĂNG LÊN 0.7)
            if rel.get('confidence', 0) < 0.7:
                continue
            
            # Loại bỏ các quan hệ có tên quá ngắn hoặc quá dài
            if len(source) < 2 or len(source) > 100:
                continue
            if len(target) < 2 or len(target) > 100:
                continue
            
            # Loại bỏ các từ chung chung không phải tên thực thể
            GENERIC_TERMS = {
                'thành viên', 'members', 'member', 'cựu thành viên', 'former members',
                'past members', 'current members', 'thành viên hiện tại', 'thành viên cũ',
                'danh sách', 'danh sách thành viên', 'danh sách cựu thành viên',
                'list', 'list of members', 'list of former members',
                'current', 'former', 'past', 'cựu'
            }
            source_lower = source.lower().strip()
            target_lower = target.lower().strip()
            if source_lower in GENERIC_TERMS or target_lower in GENERIC_TERMS:
                continue
            # Loại bỏ nếu chứa cụm từ chung chung (chỉ check các từ dài hơn 3 ký tự)
            if any(term in source_lower for term in GENERIC_TERMS if len(term) > 3):
                continue
            if any(term in target_lower for term in GENERIC_TERMS if len(term) > 3):
                continue
            
            # Loại bỏ các quan hệ có context quá dài (entities quá xa nhau)
            context = rel.get('context', '')
            
            # Cải thiện: Kiểm tra context với logic tốt hơn
            if context:
                context_lower = context.lower()
                source_lower = source.lower()
                target_lower = target.lower()
                
                # Kiểm tra xem source và target có thực sự xuất hiện trong context không
                if source_lower not in context_lower or target_lower not in context_lower:
                    continue
                
                # Đối với MEMBER_OF từ rule-based, yêu cầu chặt chẽ hơn
                if rel_type == 'MEMBER_OF' and rel.get('method') == 'rule-based':
                    if len(context) > 200:  # Context quá dài = entities quá xa nhau
                        continue
                    
                    # Kiểm tra khoảng cách trong context - CHẶT HƠN cho MEMBER_OF
                    source_pos = context_lower.find(source_lower)
                    target_pos = context_lower.find(target_lower)
                    if source_pos != -1 and target_pos != -1:
                        distance = abs(target_pos - source_pos)
                        if distance > 100:  # MEMBER_OF phải gần nhau hơn (< 100 ký tự)
                            continue
                    
                    # Cải thiện: Kiểm tra có từ khóa quan hệ trong context không
                    member_keywords = ['thành viên', 'member', 'cựu thành viên', 'former member', 
                                     'current member', 'past member', 'trưởng nhóm', 'leader']
                    has_member_keyword = any(kw in context_lower for kw in member_keywords)
                    if not has_member_keyword:
                        # Không có từ khóa quan hệ -> giảm confidence
                        rel['confidence'] = max(0.6, rel.get('confidence', 0.7) - 0.1)
                else:
                    # Các quan hệ khác: giữ logic cũ nhưng cải thiện
                    if len(context) > 300:
                        continue
                    
                    # Cải thiện: Kiểm tra có từ khóa quan hệ trong context không (tùy loại)
                    rel_keywords_map = {
                        'RELEASED': ['phát hành', 'release', 'ra mắt', 'debut'],
                        'SINGS': ['hát', 'sing', 'ca khúc', 'bài hát', 'song'],
                        'MANAGED_BY': ['quản lý', 'manage', 'công ty', 'company', 'entertainment'],
                        'CONTAINS': ['chứa', 'contain', 'bao gồm', 'include'],
                    }
                    rel_keywords = rel_keywords_map.get(rel_type, [])
                    if rel_keywords:
                        has_rel_keyword = any(kw in context_lower for kw in rel_keywords)
                        if not has_rel_keyword:
                            # Không có từ khóa quan hệ -> giảm confidence
                            rel['confidence'] = max(0.65, rel.get('confidence', 0.7) - 0.05)
                    
                    # Kiểm tra khoảng cách trong context
                    source_pos = context_lower.find(source_lower)
                    target_pos = context_lower.find(target_lower)
                    if source_pos != -1 and target_pos != -1:
                        distance = abs(target_pos - source_pos)
                        if distance > 150:  # Quá xa nhau
                            continue
            
            # Loại bỏ các quan hệ có confidence thấp (đã check ở trên nhưng double check)
            if rel.get('confidence', 0) < 0.7:
                continue
            
            valid.append(rel)
        
        return valid
    
    def _find_entity_positions(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Tìm vị trí của các entities trong text.
        
        Lưu ý:
        - Nếu có nhiều entity trùng/lồng nhau (ví dụ: "I Made" và "Made"),
          ưu tiên GIỮ entity dài hơn để tránh target sai (chọn "I Made" thay vì "Made").
        """
        positions = []
        text_lower = text.lower()
        
        # Bước 1: tìm tất cả vị trí xuất hiện (có thể trùng/lồng nhau)
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_lower = entity_text.lower()
            
            if not entity_lower:
                continue
            
            start = 0
            while True:
                idx = text_lower.find(entity_lower, start)
                if idx == -1:
                    break

                # CHỈ chấp nhận nếu match theo "từ đầy đủ" (word boundary),
                # tránh trường hợp entity là substring của từ khác (ví dụ: "Lay" trong "play").
                char_before = text_lower[idx - 1] if idx > 0 else ' '
                end_idx = idx + len(entity_lower)
                char_after = text_lower[end_idx] if end_idx < len(text_lower) else ' '
                is_word_boundary_before = not char_before.isalnum()
                is_word_boundary_after = not char_after.isalnum()

                if not (is_word_boundary_before and is_word_boundary_after):
                    start = idx + 1
                    continue

                # Loại bỏ một số context mà entity thực ra không phải node riêng biệt
                window_start = max(0, idx - 40)
                window_end = min(len(text_lower), end_idx + 40)
                window = text_lower[window_start:window_end]

                # 1) 'solo' chỉ là tính từ trong các cụm như "bài hát solo", "sân khấu solo", ...
                if entity_lower == 'solo':
                    if any(phrase in window for phrase in SOLO_INVALID_CONTEXT_PHRASES):
                        start = idx + 1
                        continue

                # 2) 'champion' nằm trong "show champion" -> không phải Song node
                if entity_lower == 'champion':
                    if any(phrase in window for phrase in CHAMPION_INVALID_CONTEXT_PHRASES):
                        start = idx + 1
                        continue

                positions.append({
                    'entity': entity,
                    'start': idx,
                    'end': end_idx,
                    'text': entity_text,
                })
                start = idx + 1
        
        if not positions:
            return []
        
        # Bước 2: ưu tiên entity dài hơn khi các entity chồng lấn nhau
        # Sort theo: start tăng dần, length giảm dần
        positions.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        filtered = []
        for pos in positions:
            overlap = False
            for kept in filtered:
                # Nếu khoảng [start, end) của pos chồng với kept -> bỏ pos (vì pos luôn ngắn hơn hoặc bằng)
                if not (pos['end'] <= kept['start'] or pos['start'] >= kept['end']):
                    overlap = True
                    break
            if not overlap:
                filtered.append(pos)
        
        # Cuối cùng sort lại theo start cho ổn định
        filtered.sort(key=lambda x: x['start'])
        return filtered
    
    def _find_entity_pairs(self, text: str, entity_positions: List[Dict], 
                          window_size: int = 200) -> List[Tuple]:
        """
        Tìm các cặp entity xuất hiện gần nhau trong văn bản
        CHỈ lấy các cặp thực sự gần nhau (trong cùng câu/đoạn ngắn)
        
        Args:
            text: Văn bản nguồn
            entity_positions: Danh sách vị trí entities
            window_size: Khoảng cách tối đa giữa 2 entities (giảm xuống 200 để chính xác hơn)
            
        Returns:
            Danh sách các cặp (entity1, entity2, context)
        """
        pairs = []
        seen_pairs = set()
        
        for i in range(len(entity_positions)):
            for j in range(i + 1, len(entity_positions)):
                e1 = entity_positions[i]
                e2 = entity_positions[j]
                
                # Kiểm tra khoảng cách - GIẢM XUỐNG 200 để chỉ lấy entities thực sự gần nhau
                distance = e2['start'] - e1['end']
                if distance > window_size:
                    break
                
                # BỎ QUA nếu entities quá xa nhau (hơn 150 ký tự)
                if distance > 150:
                    continue
                
                # Tránh cặp trùng lặp
                pair_key = (e1['text'].lower(), e2['text'].lower())
                reverse_key = (e2['text'].lower(), e1['text'].lower())
                
                if pair_key in seen_pairs or reverse_key in seen_pairs:
                    continue
                
                # Không ghép cặp entity với chính nó
                if e1['text'].lower() == e2['text'].lower():
                    continue
                
                # Kiểm tra xem có trong cùng một câu không (tùy chọn - nếu có dấu chấm giữa 2 entities thì bỏ qua)
                text_between = text[e1['end']:e2['start']]
                # Nếu có quá nhiều dấu chấm (có thể là nhiều câu) thì bỏ qua
                if text_between.count('.') > 1 or text_between.count('!') > 1 or text_between.count('?') > 1:
                    continue
                
                seen_pairs.add(pair_key)
                
                # Lấy context nhỏ hơn (chỉ 50 ký tự mỗi bên)
                context_start = max(0, e1['start'] - 50)
                context_end = min(len(text), e2['end'] + 50)
                context = text[context_start:context_end]
                
                pairs.append((e1['entity'], e2['entity'], context))
        
        return pairs
    
    def _classify_relationship(self, entity1: Dict, entity2: Dict, 
                               context: str) -> Optional[Tuple[str, Dict, Dict]]:
        """
        Phân loại loại quan hệ giữa 2 entities và đảm bảo chiều đúng
        
        Args:
            entity1: Entity đầu tiên
            entity2: Entity thứ hai
            context: Văn bản context chứa cả 2 entities
            
        Returns:
            Tuple (rel_type, source_entity, target_entity) hoặc None
            Đảm bảo source và target đúng chiều quan hệ
        """
        context_lower = context.lower()
        
        type1 = entity1.get('type', entity1.get('label', 'Entity'))
        type2 = entity2.get('type', entity2.get('label', 'Entity'))
        e1_text = entity1.get('text', '').lower()
        e2_text = entity2.get('text', '').lower()
        
        # Thử cả 2 chiều: (type1, type2) và (type2, type1)
        for source_entity, target_entity, source_type, target_type in [
            (entity1, entity2, type1, type2),
            (entity2, entity1, type2, type1)
        ]:
            valid_rels = self.valid_relationships.get((source_type, target_type), [])
            
            if not valid_rels:
                continue
            
            # Bước 1: Kiểm tra patterns với chiều đúng
            for rel_type in valid_rels:
                if rel_type in self.patterns:
                    for pattern in self.patterns[rel_type]:
                        match = re.search(pattern, context, re.IGNORECASE)
                        if match:
                            groups = match.groups()
                            if len(groups) >= 2:
                                source_text = source_entity.get('text', '').lower()
                                target_text = target_entity.get('text', '').lower()
                                
                                # Kiểm tra xem entities có trong match và đúng vị trí không
                                # CHẶT CHẼ HƠN: Phải là từ đầy đủ, không phải substring
                                source_in_first = self._is_full_word_match(source_text, groups[0].lower())
                                target_in_second = self._is_full_word_match(target_text, groups[1].lower())
                                
                                if source_in_first and target_in_second:
                                    # Kiểm tra khoảng cách giữa source và target trong match
                                    # Không được quá xa nhau (tối đa 100 ký tự)
                                    match_text = match.group(0).lower()
                                    source_pos_in_match = match_text.find(source_text)
                                    target_pos_in_match = match_text.find(target_text)
                                    
                                    if source_pos_in_match != -1 and target_pos_in_match != -1:
                                        distance_in_match = abs(target_pos_in_match - source_pos_in_match)
                                        if distance_in_match <= 100:
                                            # Validate chiều quan hệ
                                            if self._validate_relationship_direction(
                                                source_entity, target_entity, rel_type, context
                                            ):
                                                return (rel_type, source_entity, target_entity)
            
            # Bước 2: Kiểm tra keywords với chiều đúng (CHẶT CHẼ HƠN)
            source_text = source_entity.get('text', '').lower()
            target_text = target_entity.get('text', '').lower()
            
            for rel_type in valid_rels:
                if rel_type in self.keywords:
                    keywords = self.keywords[rel_type]['vi'] + self.keywords[rel_type]['en']
                    for keyword in keywords:
                        keyword_lower = keyword.lower()
                        if keyword_lower in context_lower:
                            # KIỂM TRA: Keyword phải nằm GIỮA 2 entities
                            source_pos = context_lower.find(source_text)
                            target_pos = context_lower.find(target_text)
                            keyword_pos = context_lower.find(keyword_lower)
                            
                            # Keyword phải nằm giữa source và target
                            if source_pos != -1 and target_pos != -1 and keyword_pos != -1:
                                if source_pos < keyword_pos < target_pos or target_pos < keyword_pos < source_pos:
                                    # Validate chiều quan hệ
                                    if self._validate_relationship_direction(
                                        source_entity, target_entity, rel_type, context
                                    ):
                                        return (rel_type, source_entity, target_entity)
            
            # Bước 3: Heuristic với chiều đúng (CHỈ dùng khi không có pattern/keyword match)
            # Chỉ dùng heuristic nếu không tìm thấy pattern hoặc keyword
            # (Đã được xử lý ở trên, nên bỏ qua heuristic để tránh false positive)
        
        return None
    
    def _is_full_word_match(self, word: str, text: str) -> bool:
        """
        Kiểm tra xem word có phải là từ đầy đủ trong text không (không phải substring)
        
        Args:
            word: Từ cần tìm
            text: Text để tìm
            
        Returns:
            True nếu là từ đầy đủ
        """
        if not word or not text:
            return False
        
        word_lower = word.lower()
        text_lower = text.lower()
        
        # Tìm vị trí của word trong text
        idx = text_lower.find(word_lower)
        if idx == -1:
            return False
        
        # Kiểm tra ký tự trước và sau
        char_before = text_lower[idx - 1] if idx > 0 else ' '
        char_after = text_lower[idx + len(word_lower)] if idx + len(word_lower) < len(text_lower) else ' '
        
        # Phải là ký tự không phải chữ/số (word boundary)
        is_word_boundary_before = not char_before.isalnum()
        is_word_boundary_after = not char_after.isalnum()
        
        return is_word_boundary_before and is_word_boundary_after
    
    def _validate_relationship_direction(self, source_entity: Dict, target_entity: Dict,
                                       rel_type: str, context: str) -> bool:
        """
        Validate chiều quan hệ có đúng không dựa trên context
        
        Args:
            source_entity: Entity nguồn
            target_entity: Entity đích
            rel_type: Loại quan hệ
            context: Context text
            
        Returns:
            True nếu chiều đúng, False nếu sai
        """
        source_text = source_entity.get('text', '').lower()
        target_text = target_entity.get('text', '').lower()
        source_type = source_entity.get('type', source_entity.get('label', 'Entity'))
        target_type = target_entity.get('type', target_entity.get('label', 'Entity'))
        context_lower = context.lower()
        
        # Validation rules cho từng loại quan hệ - YÊU CẦU BẰNG CHỨNG RÕ RÀNG
        if rel_type == 'MEMBER_OF':
            # Artist → Group: bao gồm cả thành viên hiện tại và cựu thành viên
            if source_type == 'Artist' and target_type == 'Group':
                source_pos = context_lower.find(source_text)
                target_pos = context_lower.find(target_text)
                
                if source_pos == -1 or target_pos == -1:
                    return False
                
                # Khoảng cách phải hợp lý (< 180 ký tự) - nới lỏng để bắt thêm câu dài
                if abs(target_pos - source_pos) > 180:
                    return False
                
                # Patterns RÕ RÀNG cho member
                strong_member_patterns = [
                    re.escape(source_text) + r'\s+(?:là\s+)?thành viên\s+(?:của\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:là\s+)?cựu thành viên\s+(?:của\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:is\s+)?(?:a\s+)?member\s+of\s+' + re.escape(target_text),
                    re.escape(target_text) + r'\s+(?:gồm|bao gồm|có|gồm có)\s+.*?' + re.escape(source_text),
                    re.escape(source_text) + r'\s+(?:đã\s+)?rời\s+(?:khỏi\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+left\s+' + re.escape(target_text),
                    r'thành viên\s+(?:của\s+)?' + re.escape(target_text) + r'.*?' + re.escape(source_text),
                    # === THÊM PATTERNS MỚI ĐỂ TĂNG RECALL ===
                    # "Nhóm hiện có X thành viên gồm A, B, C..."
                    r'nhóm\s+(?:hiện\s+)?(?:có|gồm)\s+\d+\s+thành viên\s+(?:gồm|bao gồm|là)\s*:?\s*.*?' + re.escape(source_text),
                    # "Bộ đôi này gồm A và B"
                    r'bộ đôi\s+(?:này\s+)?gồm\s+' + re.escape(source_text),
                    # "gồm hai thành viên là A và B"
                    r'gồm\s+\d+\s+thành viên\s+(?:là\s+)?' + re.escape(source_text),
                    # "thành viên bao gồm: A, B, C"
                    r'thành viên\s+bao gồm\s*:?\s*.*?' + re.escape(source_text),
                    # "Group có các thành viên A, B, C"
                    re.escape(target_text) + r'\s+có\s+(?:các\s+)?thành viên\s*:?\s*.*?' + re.escape(source_text),
                    # "gồm X thành viên: A, B, C"
                    r'gồm\s+(?:các\s+)?thành viên\s*:?\s*' + re.escape(source_text),
                    # "bao gồm A, B, C và D"
                    r'bao gồm\s+' + re.escape(source_text),
                    # "Group consists of A, B, C"
                    re.escape(target_text) + r'\s+(?:consists?\s+of|comprising)\s+.*?' + re.escape(source_text),
                    # "các thành viên của Group: A, B, C"
                    r'(?:các\s+)?thành viên\s+của\s+' + re.escape(target_text) + r'\s*:?\s*.*?' + re.escape(source_text),
                ]
                
                for pattern in strong_member_patterns:
                    if re.search(pattern, context_lower, re.IGNORECASE):
                        return True
                
                # Kiểm tra từ khóa mạnh NẰM GIỮA 2 entities
                min_pos = min(source_pos, target_pos)
                max_pos = max(source_pos, target_pos)
                between_text = context_lower[min_pos:max_pos]
                
                strong_member_keywords = ['thành viên', 'member', 'cựu thành viên', 'former member', 
                                          'gồm', 'bao gồm', 'gồm có', 'comprising', 'consists of']
                if any(kw in between_text for kw in strong_member_keywords):
                    return True
                
                return False
        
        elif rel_type == 'RELEASED':
            # RELEASED chỉ dành cho Album (KHÔNG dùng cho Song)
            if target_type == 'Song':
                return False
                
            if source_type in ['Group', 'Artist'] and target_type == 'Album':
                source_pos = context_lower.find(source_text)
                target_pos = context_lower.find(target_text)
                
                if source_pos == -1 or target_pos == -1:
                    return False
                
                # Khoảng cách phải hợp lý (< 150 ký tự) – nới nhẹ để bắt thêm câu dài
                if abs(target_pos - source_pos) > 150:
                    return False
                
                # Patterns RÕ RÀNG cho RELEASED (CHÍNH là nguồn high-confidence)
                strong_release_patterns = [
                    re.escape(source_text) + r'\s+(?:đã\s+)?phát hành\s+(?:album\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:đã\s+)?ra mắt\s+(?:album\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+released?\s+(?:the\s+)?(?:album\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+tung ra\s+(?:album\s+)?' + re.escape(target_text),
                    r'album\s+' + re.escape(target_text) + r'\s+(?:của|by|of)\s+' + re.escape(source_text),
                    # Dạng: album/EP/mini-album "X" được phát hành bởi/was released by/by A
                    r'(?:album|ep|mini-album)\s+' + re.escape(target_text) + r'\s+(?:được\s+phát hành\s+bởi|was released by|by)\s+' + re.escape(source_text),
                    # === THÊM PATTERNS MỚI ĐỂ TĂNG RECALL ===
                    # "là album tiếng Hàn thứ X của Group/Artist"
                    re.escape(target_text) + r'\s+là\s+(?:album|mini-album|ep).*?(?:của|by|of)\s+.*?' + re.escape(source_text),
                    # "mini-album đầu tay của nhóm nhạc X"
                    r'(?:album|mini-album|ep)\s+(?:đầu tay|thứ\s+\d+|đầu tiên)\s+(?:của\s+)?(?:nhóm\s+nhạc\s+|nhóm\s+|)?' + re.escape(source_text) + r'.*?' + re.escape(target_text),
                    # "Album được Group/Artist phát hành"
                    r'album\s+(?:được\s+)?' + re.escape(source_text) + r'\s+phát hành',
                    # "Album được phát hành bởi SM Entertainment" (Company phát hành) - chỉ cho Company
                    r'album\s+' + re.escape(target_text) + r'\s+được\s+.*?phát hành\s+(?:bởi\s+)?' + re.escape(source_text),
                    # "với mini-album đầu tay X"
                    re.escape(source_text) + r'\s+(?:ra mắt\s+)?(?:với\s+)?(?:mini-album|album|ep)\s+(?:đầu tay\s+)?' + re.escape(target_text),
                    # "album đầu tay của nhóm, X"
                    r'(?:album|mini-album|ep)\s+(?:đầu tay|đầu tiên|thứ\s+\w+)\s+(?:của\s+)?(?:nhóm\s+)?' + re.escape(source_text) + r'[\s,]+' + re.escape(target_text),
                    # "mini-album thứ hai của nhóm nhạc Hàn Quốc X"
                    r'(?:album|mini-album|ep)\s+(?:thứ\s+\w+|đầu tiên|đầu tay)\s+(?:của\s+)?(?:nhóm\s+nhạc\s+(?:nam\s+|nữ\s+)?(?:hàn\s+quốc\s+)?)?' + re.escape(source_text),
                    # "là mini-album thứ X của nhóm Y"
                    r'là\s+(?:album|mini-album|ep)\s+(?:thứ\s+\w+|đầu tiên|đầu tay).*?(?:của|by)\s+' + re.escape(source_text),
                    # "Album X của Group"
                    r'album\s+' + re.escape(target_text) + r'\s+của\s+' + re.escape(source_text),
                    # "Group ra mắt/trở lại với album X"
                    re.escape(source_text) + r'\s+(?:ra mắt|trở lại|debut|comeback)\s+(?:với\s+)?(?:album|mini-album|ep)\s+' + re.escape(target_text),
                ]
                
                for pattern in strong_release_patterns:
                    if re.search(pattern, context_lower, re.IGNORECASE):
                        src_norm = normalize_node_name(source_text).lower()
                        tgt_norm = normalize_node_name(target_text).lower()
                        
                        # 1) Nếu album đã có RELEASED trong graph gốc, CHỈ cho phép source
                        # trùng với một trong các source gốc đó
                        allowed_sources_graph = RELEASED_ALLOWED_SOURCES_BY_ALBUM.get(tgt_norm)
                        if allowed_sources_graph and src_norm not in allowed_sources_graph:
                            return False
                        
                        # 2) Nếu target là Album đến từ NER rule-based,
                        # ưu tiên CHỈ chấp nhận khi source trùng với ÍT NHẤT MỘT trong:
                        # - source_node
                        # - hoặc một trong các sources của entity đó.
                        ner_method = target_entity.get('ner_method', '')
                        ner_source_node = target_entity.get('ner_source_node', '')
                        ner_sources = target_entity.get('ner_sources', [])
                        if ner_method == 'rule-based' and (ner_source_node or ner_sources):
                            allowed = set()
                            if ner_source_node:
                                allowed.add(normalize_node_name(ner_source_node).lower())
                            for s in ner_sources:
                                allowed.add(normalize_node_name(s).lower())
                            if allowed and src_norm not in allowed:
                                return False
                        
                        return True
                
                # Nếu không có pattern rõ ràng, cho phép FALLBACK bằng keyword,
                # nhưng vẫn phải:
                # - target là Album
                # - khoảng cách và vị trí keyword hợp lý
                # - (nếu target từ NER rule-based) source phải thuộc allowed sources
                strong_release_keywords = ['phát hành', 'ra mắt', 'released', 'tung ra']
                for kw in strong_release_keywords:
                    kw_lower = kw.lower()
                    kw_pos = context_lower.find(kw_lower)
                    if kw_pos == -1:
                        continue
                    
                    # Keyword phải nằm giữa source và target
                    if not (min(source_pos, target_pos) < kw_pos < max(source_pos, target_pos)):
                        continue
                    
                    # Khoảng cách từ keyword tới từng entity không được quá xa (nới nhẹ lên 100 ký tự)
                    if abs(kw_pos - source_pos) > 100 or abs(kw_pos - target_pos) > 100:
                        continue
                    
                    # Kiểm tra nguồn NER nếu có + nguồn gốc từ graph
                    ner_method = target_entity.get('ner_method', '')
                    ner_source_node = target_entity.get('ner_source_node', '')
                    ner_sources = target_entity.get('ner_sources', [])
                    src_norm = normalize_node_name(source_text).lower()
                    tgt_norm = normalize_node_name(target_text).lower()
                    
                    # 1) Nếu album đã có RELEASED trong graph gốc, chỉ cho phép source trùng nguồn đó
                    allowed_sources_graph = RELEASED_ALLOWED_SOURCES_BY_ALBUM.get(tgt_norm)
                    if allowed_sources_graph and src_norm not in allowed_sources_graph:
                        continue
                    
                    # 2) Nếu target từ NER rule-based, kiểm tra thêm nguồn NER
                    if ner_method == 'rule-based' and (ner_source_node or ner_sources):
                        allowed = set()
                        if ner_source_node:
                            allowed.add(normalize_node_name(ner_source_node).lower())
                        for s in ner_sources:
                            allowed.add(normalize_node_name(s).lower())
                        if allowed and src_norm not in allowed:
                            continue
                    
                    return True
                
                return False
        
        elif rel_type == 'SINGS':
            # Group/Artist → Song: phải có bằng chứng rõ ràng
            if source_type in ['Group', 'Artist'] and target_type == 'Song':
                source_pos = context_lower.find(source_text)
                target_pos = context_lower.find(target_text)
                
                if source_pos == -1 or target_pos == -1:
                    return False
                
                # Khoảng cách phải hợp lý (< 120 ký tự) – nới nhẹ để bắt thêm câu dài
                if abs(target_pos - source_pos) > 120:
                    return False
                
                # Patterns RÕ RÀNG cho SINGS
                strong_sing_patterns = [
                    re.escape(source_text) + r'\s+(?:đã\s+)?(?:hát|trình bày|thể hiện)\s+(?:bài\s+|ca khúc\s+)?' + re.escape(target_text),
                    r'(?:bài hát|ca khúc|single|title track)\s+' + re.escape(target_text) + r'\s+(?:của|by|of)\s+' + re.escape(source_text),
                    re.escape(source_text) + r"'s\s+(?:song|single|track)\s+" + re.escape(target_text),
                    re.escape(target_text) + r'\s+(?:là\s+)?(?:bài hát|ca khúc|single)\s+(?:của\s+)?' + re.escape(source_text),
                    # Ca sĩ / nhóm phát hành single / bài hát "X"
                    re.escape(source_text) + r'\s+(?:đã\s+)?(?:phát hành|ra mắt|released|dropped)\s+(?:single|bài hát|ca khúc|song)\s+' + re.escape(target_text),
                    # === THÊM PATTERNS MỚI ĐỂ TĂNG RECALL ===
                    # "với ca khúc chủ đề X"
                    re.escape(source_text) + r'\s+(?:với\s+)?(?:ca khúc|bài hát)\s+chủ đề\s+["\']?' + re.escape(target_text),
                    # "ca khúc chủ đề X của Group/Artist"
                    r'(?:ca khúc|bài hát)\s+chủ đề\s+["\']?' + re.escape(target_text) + r'["\']?\s*(?:của\s+)?' + re.escape(source_text),
                    # "biểu diễn bài hát X"
                    re.escape(source_text) + r'\s+(?:biểu diễn|perform)\s+(?:bài hát|ca khúc)\s+["\']?' + re.escape(target_text),
                    # "ra mắt ngày X với single/ca khúc Y"
                    re.escape(source_text) + r'\s+(?:ra mắt|debut)\s+.*?(?:với\s+)?(?:single|ca khúc|bài hát)\s+["\']?' + re.escape(target_text),
                    # "đĩa đơn đầu tay X"
                    re.escape(source_text) + r'\s+.*?(?:đĩa đơn|single)\s+(?:đầu tay|đầu tiên)\s+["\']?' + re.escape(target_text),
                    # "title track X của album Y" - bắt Source hát Target
                    r'(?:title track|ca khúc chủ đề)\s+["\']?' + re.escape(target_text) + r'["\']?.*?(?:của\s+)?' + re.escape(source_text),
                    # "nhóm với bài hát X"
                    re.escape(source_text) + r'\s+với\s+(?:bài\s+(?:hát\s+)?|ca khúc\s+)?["\']?' + re.escape(target_text),
                    # "bài hát X được trình bày bởi Y"
                    r'(?:bài hát|ca khúc|single)\s+["\']?' + re.escape(target_text) + r'["\']?\s+(?:được\s+)?(?:trình bày|hát|thể hiện)\s+(?:bởi\s+)?' + re.escape(source_text),
                ]
                
                for pattern in strong_sing_patterns:
                    if re.search(pattern, context_lower, re.IGNORECASE):
                        # Nếu target là Song đến từ NER rule-based,
                        # ưu tiên CHỈ chấp nhận khi source trùng với ÍT NHẤT MỘT trong:
                        # - source_node
                        # - hoặc một trong các sources của entity đó.
                        ner_method = target_entity.get('ner_method', '')
                        ner_source_node = target_entity.get('ner_source_node', '')
                        ner_sources = target_entity.get('ner_sources', [])
                        if ner_method == 'rule-based' and (ner_source_node or ner_sources):
                            src_norm = normalize_node_name(source_text).lower()
                            allowed = set()
                            if ner_source_node:
                                allowed.add(normalize_node_name(ner_source_node).lower())
                            for s in ner_sources:
                                allowed.add(normalize_node_name(s).lower())
                            if allowed and src_norm not in allowed:
                                return False
                        return True
                
                # Kiểm tra từ khóa mạnh giữa 2 entities
                min_pos = min(source_pos, target_pos)
                max_pos = max(source_pos, target_pos)
                between_text = context_lower[min_pos:max_pos]
                
                strong_sing_keywords = [
                    'hát', 'trình bày', 'thể hiện',
                    'ca khúc của', 'bài hát của', 'title track',
                    # Trong ngữ cảnh single / bài hát, phát hành cũng ngầm hiểu là hát
                    'phát hành single', 'ra mắt single', 'released the single', 'released the song',
                ]
                if any(kw in between_text for kw in strong_sing_keywords):
                    return True
                
                return False
        
        elif rel_type == 'CONTAINS':
            # Album → Song: phải có bằng chứng rõ ràng
            if source_type == 'Album' and target_type == 'Song':
                source_pos = context_lower.find(source_text)
                target_pos = context_lower.find(target_text)
                
                if source_pos == -1 or target_pos == -1:
                    return False
                
                # Khoảng cách phải hợp lý (< 150 ký tự) - nới lỏng để bắt thêm
                if abs(target_pos - source_pos) > 150:
                    return False
                
                # Patterns RÕ RÀNG cho CONTAINS
                strong_contains_patterns = [
                    re.escape(source_text) + r'\s+(?:có\s+)?(?:chứa|bao gồm|gồm)\s+(?:bài\s+|ca khúc\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:includes?|contains?)\s+(?:the\s+)?(?:song\s+|track\s+)?' + re.escape(target_text),
                    r'(?:bài|track|ca khúc)\s+' + re.escape(target_text) + r'\s+(?:trong|in|from)\s+(?:album\s+)?' + re.escape(source_text),
                    re.escape(target_text) + r'\s+(?:nằm trong|trong album|from the album)\s+' + re.escape(source_text),
                    # === THÊM PATTERNS MỚI ĐỂ TĂNG RECALL ===
                    # "album bao gồm X bài hát" - skip (số lượng, không có tên)
                    # "album gồm các bài hát A, B, C"
                    re.escape(source_text) + r'\s+(?:gồm|bao gồm)\s+(?:các\s+)?(?:bài hát|ca khúc|track)\s*:?\s*.*?' + re.escape(target_text),
                    # "X là bài hát trong album Y"
                    re.escape(target_text) + r'\s+là\s+(?:bài hát|ca khúc|track)\s+(?:trong|thuộc|nằm trong)\s+(?:album\s+)?' + re.escape(source_text),
                    # "album X có bài hát Y"
                    r'album\s+' + re.escape(source_text) + r'\s+có\s+(?:bài hát|ca khúc)\s+' + re.escape(target_text),
                    # "được trích từ album X"
                    r'(?:trích từ|from|in)\s+(?:album\s+)?' + re.escape(source_text) + r'.*?' + re.escape(target_text),
                    # "title track/ca khúc chủ đề X thuộc album Y"
                    r'(?:title track|ca khúc chủ đề|bài hát chủ đề)\s+["\']?' + re.escape(target_text) + r'["\']?\s+(?:thuộc|trong|nằm trong)\s+(?:album\s+)?' + re.escape(source_text),
                    # "album Y với ca khúc chủ đề X"
                    r'(?:album|mini-album|ep)\s+' + re.escape(source_text) + r'\s+(?:với\s+)?(?:ca khúc|bài hát)\s+(?:chủ đề\s+)?["\']?' + re.escape(target_text),
                ]
                
                for pattern in strong_contains_patterns:
                    if re.search(pattern, context_lower, re.IGNORECASE):
                        return True
                
                # Kiểm tra từ khóa mạnh giữa 2 entities
                min_pos = min(source_pos, target_pos)
                max_pos = max(source_pos, target_pos)
                between_text = context_lower[min_pos:max_pos]
                
                strong_contains_keywords = ['chứa', 'bao gồm', 'gồm có', 'includes', 'contains', 'trong album', 'nằm trong',
                                            'ca khúc chủ đề', 'title track', 'trích từ', 'thuộc album', 'từ album']
                if any(kw in between_text for kw in strong_contains_keywords):
                    return True
                
                return False
        
        elif rel_type == 'MANAGED_BY':
            # Artist/Group → Company: phải có bằng chứng rõ ràng
            if source_type in ['Artist', 'Group'] and target_type == 'Company':
                source_pos = context_lower.find(source_text)
                target_pos = context_lower.find(target_text)
                
                if source_pos == -1 or target_pos == -1:
                    return False
                
                # Khoảng cách phải hợp lý (< 130 ký tự) - nới lỏng để bắt thêm
                if abs(target_pos - source_pos) > 130:
                    return False
                
                # Patterns RÕ RÀNG cho MANAGED_BY
                strong_manage_patterns = [
                    re.escape(source_text) + r'\s+(?:thuộc|trực thuộc)\s+(?:công ty\s+|hãng\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:được\s+)?quản lý\s+bởi\s+' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:is\s+)?(?:under|signed with|signed to)\s+' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:đã\s+)?ký hợp đồng\s+(?:với\s+)?' + re.escape(target_text),
                    r'(?:nghệ sĩ|nhóm|group|artist)\s+(?:của|of)\s+' + re.escape(target_text) + r'.*?' + re.escape(source_text),
                    re.escape(target_text) + r"'s\s+(?:artist|group)\s+" + re.escape(source_text),
                    # === THÊM PATTERNS MỚI ĐỂ TĂNG RECALL ===
                    # "được thành lập và quản lý bởi công ty X"
                    re.escape(source_text) + r'\s+(?:được\s+)?(?:thành lập\s+và\s+)?(?:quản lý|quản lý\s+bởi|managed\s+by)\s+(?:bởi\s+)?(?:công ty\s+)?' + re.escape(target_text),
                    # "được thành lập bởi X Entertainment"
                    re.escape(source_text) + r'\s+(?:được\s+)?(?:thành lập|thành lập\s+bởi|formed\s+by)\s+(?:bởi\s+)?(?:hãng\s+giải trí\s+|công ty\s+giải trí\s+|)?' + re.escape(target_text),
                    # "ký hợp đồng với X Entertainment"
                    re.escape(source_text) + r'\s+(?:đã\s+)?ký\s+(?:hợp đồng\s+)?(?:với\s+)?' + re.escape(target_text),
                    # "trực thuộc công ty giải trí X"
                    re.escape(source_text) + r'\s+trực thuộc\s+(?:công ty\s+(?:giải trí\s+)?|hãng\s+(?:giải trí\s+)?|label\s+)?' + re.escape(target_text),
                    # "X ra mắt dưới trướng Y"
                    re.escape(source_text) + r'\s+(?:ra mắt|debut)\s+dưới\s+(?:trướng|sự\s+quản lý\s+của)\s+' + re.escape(target_text),
                    # "là nghệ sĩ của X Entertainment"
                    re.escape(source_text) + r'\s+là\s+(?:nghệ sĩ|thực tập sinh|trainee)\s+(?:của|thuộc)\s+' + re.escape(target_text),
                    # "nhóm nhạc X thuộc/của công ty Y"
                    r'nhóm\s+(?:nhạc\s+)?' + re.escape(source_text) + r'\s+(?:thuộc|của)\s+(?:công ty\s+(?:giải trí\s+)?|hãng\s+)?' + re.escape(target_text),
                    # "được phát hành bởi X Entertainment" (label phát hành)
                    r'(?:được\s+)?(?:phát hành|released)\s+(?:bởi|by)\s+' + re.escape(target_text) + r'.*?' + re.escape(source_text),
                    # "dưới trướng X Entertainment"
                    r'dưới\s+(?:trướng|sự\s+quản lý)\s+(?:của\s+)?' + re.escape(target_text) + r'.*?' + re.escape(source_text),
                    # "Y (công ty) quản lý nhóm/nghệ sĩ X"
                    re.escape(target_text) + r'\s+(?:quản lý|manages?)\s+(?:nhóm\s+|nghệ sĩ\s+)?' + re.escape(source_text),
                ]
                
                for pattern in strong_manage_patterns:
                    if re.search(pattern, context_lower, re.IGNORECASE):
                        return True
                
                # Kiểm tra từ khóa mạnh giữa 2 entities
                min_pos = min(source_pos, target_pos)
                max_pos = max(source_pos, target_pos)
                between_text = context_lower[min_pos:max_pos]
                
                strong_manage_keywords = ['thuộc', 'trực thuộc', 'quản lý bởi', 'under', 'signed with', 'ký hợp đồng',
                                          'được thành lập', 'thành lập bởi', 'formed by', 'managed by', 
                                          'dưới trướng', 'công ty', 'hãng', 'entertainment']
                if any(kw in between_text for kw in strong_manage_keywords):
                    return True
                
                return False
        
        elif rel_type == 'PRODUCE_SONG':
            # Artist → Song: CHỈ PRODUCER CHUYÊN NGHIỆP mới có quan hệ này
            if source_type == 'Artist' and target_type == 'Song':
                # PHẢI là producer trong danh sách đã biết
                is_known_producer = source_text in KNOWN_PRODUCERS
                
                # Nếu KHÔNG phải producer đã biết -> TỪ CHỐI HOÀN TOÀN
                if not is_known_producer:
                    return False
                
                # Nếu là producer đã biết, cần có từ khóa produce trong context
                produce_keywords = ['sản xuất', 'produced', 'producer', 'produce']
                has_produce_keyword = any(kw in context_lower for kw in produce_keywords)
                if not has_produce_keyword:
                    return False

                # Nếu target là Song đến từ NER rule-based, ưu tiên CHỈ chấp nhận
                # khi source trùng với ÍT NHẤT MỘT trong source_node/sources của entity đó
                ner_method = target_entity.get('ner_method', '')
                ner_source_node = target_entity.get('ner_source_node', '')
                ner_sources = target_entity.get('ner_sources', [])
                if ner_method == 'rule-based' and (ner_source_node or ner_sources):
                    src_norm = normalize_node_name(source_text).lower()
                    allowed = set()
                    if ner_source_node:
                        allowed.add(normalize_node_name(ner_source_node).lower())
                    for s in ner_sources:
                        allowed.add(normalize_node_name(s).lower())
                    if allowed and src_norm not in allowed:
                        return False

                return True
            return False
        
        elif rel_type == 'PRODUCE_ALBUM':
            # Artist → Album: CHỈ PRODUCER CHUYÊN NGHIỆP mới có quan hệ này
            if source_type == 'Artist' and target_type == 'Album':
                # PHẢI là producer trong danh sách đã biết
                is_known_producer = source_text in KNOWN_PRODUCERS
                
                # Nếu KHÔNG phải producer đã biết -> TỪ CHỐI HOÀN TOÀN
                if not is_known_producer:
                    return False
                
                # Nếu là producer đã biết, cần có từ khóa produce trong context
                produce_keywords = ['sản xuất', 'produced', 'producer', 'produce', 'executive producer']
                has_produce_keyword = any(kw in context_lower for kw in produce_keywords)
                if not has_produce_keyword:
                    return False

                # Nếu target là Album đến từ NER rule-based, ưu tiên CHỈ chấp nhận
                # khi source trùng với ÍT NHẤT MỘT trong source_node/sources của entity đó
                ner_method = target_entity.get('ner_method', '')
                ner_source_node = target_entity.get('ner_source_node', '')
                ner_sources = target_entity.get('ner_sources', [])
                if ner_method == 'rule-based' and (ner_source_node or ner_sources):
                    src_norm = normalize_node_name(source_text).lower()
                    allowed = set()
                    if ner_source_node:
                        allowed.add(normalize_node_name(ner_source_node).lower())
                    for s in ner_sources:
                        allowed.add(normalize_node_name(s).lower())
                    if allowed and src_norm not in allowed:
                        return False

                return True
            return False
        
        elif rel_type == 'WROTE':
            # Artist → Song: CHỈ SONGWRITER CHUYÊN NGHIỆP mới có quan hệ này
            if source_type == 'Artist' and target_type == 'Song':
                # PHẢI là songwriter trong danh sách đã biết
                is_known_songwriter = source_text in KNOWN_SONGWRITERS
                
                # Nếu KHÔNG phải songwriter đã biết -> TỪ CHỐI HOÀN TOÀN
                if not is_known_songwriter:
                    return False
                
                # Tìm vị trí của source và target trong context
                source_pos = context_lower.find(source_text)
                target_pos = context_lower.find(target_text)
                
                if source_pos == -1 or target_pos == -1:
                    return False
                
                # Từ khóa sáng tác phải xuất hiện trong context
                write_keywords = ['sáng tác', 'viết lời', 'wrote', 'composed', 'tác giả', 
                                  'songwriter', 'composer', 'lyricist', 'written by', 'do chính', 'được viết bởi']
                
                # Tìm từ khóa gần nhất với source entity
                keyword_found = False
                keyword_pos = -1
                matched_keyword = None
                
                for kw in write_keywords:
                    pos = context_lower.find(kw)
                    if pos != -1:
                        # Từ khóa phải nằm gần source entity (trong vòng 50 ký tự trước hoặc sau)
                        # Và source phải đứng TRƯỚC từ khóa (là subject của động từ)
                        if source_pos < pos and (pos - source_pos) <= 50:
                            keyword_found = True
                            keyword_pos = pos
                            matched_keyword = kw
                            break
                
                if not keyword_found:
                    return False
                
                # Kiểm tra pattern rõ ràng: "Entity sáng tác Song" hoặc "do chính Entity sáng tác"
                # Pattern 1: "Entity sáng tác Song"
                pattern1 = re.escape(source_text) + r'\s+(?:sáng tác|viết|wrote|composed)\s+.*?' + re.escape(target_text)
                # Pattern 2: "do chính Entity sáng tác Song"
                pattern2 = r'do chính\s+' + re.escape(source_text) + r'\s+(?:sáng tác|viết).*?' + re.escape(target_text)
                # Pattern 3: "Song do Entity sáng tác"
                pattern3 = re.escape(target_text) + r'\s+do\s+' + re.escape(source_text) + r'\s+(?:sáng tác|viết)'
                # Pattern 4: "Song được viết bởi Entity"
                pattern4 = re.escape(target_text) + r'\s+được\s+(?:viết|sáng tác)\s+bởi\s+' + re.escape(source_text)
                
                strong_patterns = [pattern1, pattern2, pattern3, pattern4]
                for pattern in strong_patterns:
                    if re.search(pattern, context_lower, re.IGNORECASE):
                        return True
                
                # Nếu không match pattern rõ ràng, kiểm tra xem source có gần từ khóa hơn target không
                # Source phải đứng TRƯỚC từ khóa, target có thể đứng sau
                if source_pos < keyword_pos and keyword_pos < target_pos:
                    # Khoảng cách giữa source và keyword phải hợp lý (< 30 ký tự)
                    if (keyword_pos - source_pos) <= 30:
                        # Kiểm tra nguồn NER nếu có
                        ner_method = target_entity.get('ner_method', '')
                        ner_source_node = target_entity.get('ner_source_node', '')
                        ner_sources = target_entity.get('ner_sources', [])
                        if ner_method == 'rule-based' and (ner_source_node or ner_sources):
                            src_norm = normalize_node_name(source_text).lower()
                            allowed = set()
                            if ner_source_node:
                                allowed.add(normalize_node_name(ner_source_node).lower())
                            for s in ner_sources:
                                allowed.add(normalize_node_name(s).lower())
                            if allowed and src_norm not in allowed:
                                return False
                        return True

                return False
            return False
        
        # FORMER_MEMBER_OF đã được gộp vào MEMBER_OF
        # Validation cho cựu thành viên sẽ dùng chung với MEMBER_OF
        
        elif rel_type == 'SUBUNIT_OF':
            # SUBUNIT_OF: Subunit → Parent Group (EXO-CBX → EXO, không phải ngược lại)
            # Source phải là nhóm nhỏ, Target phải là nhóm lớn
            if source_type == 'Group' and target_type == 'Group':
                source_pos = context_lower.find(source_text)
                target_pos = context_lower.find(target_text)
                
                if source_pos == -1 or target_pos == -1:
                    return False
                
                # Khoảng cách phải hợp lý (< 130 ký tự) - nới lỏng để bắt thêm
                if abs(target_pos - source_pos) > 130:
                    return False
                
                # Patterns RÕ RÀNG cho SUBUNIT_OF
                strong_subunit_patterns = [
                    re.escape(source_text) + r'\s+(?:là\s+)?(?:nhóm nhỏ|subunit|sub-unit)\s+(?:của\s+)?' + re.escape(target_text),
                    re.escape(source_text) + r'\s+(?:is\s+)?(?:a\s+)?(?:subunit|sub-unit)\s+of\s+' + re.escape(target_text),
                    r'(?:nhóm nhỏ|subunit)\s+' + re.escape(source_text) + r'\s+(?:của|of)\s+' + re.escape(target_text),
                    # === THÊM PATTERNS MỚI ĐỂ TĂNG RECALL ===
                    # "X, nhóm nhỏ (chính thức) (thứ N) của nhóm nhạc nam/nữ Y"
                    re.escape(source_text) + r',?\s+(?:là\s+)?nhóm nhỏ\s+(?:chính thức\s+)?(?:thứ\s+\w+\s+)?(?:của\s+)?(?:nhóm nhạc\s+(?:nam|nữ)?\s+)?(?:hàn[\s\-–]?(?:trung\s+)?quốc\s+)?' + re.escape(target_text),
                    # "X là một trong hai/các nhóm nhỏ được tách ra từ nhóm nhạc Y"
                    re.escape(source_text) + r'\s+là\s+một\s+trong\s+(?:hai|các|những)\s+nhóm\s+(?:nhỏ\s+)?(?:được\s+)?(?:tách|tách ra|chia)\s+(?:ra\s+)?(?:từ\s+)?(?:nhóm\s+(?:nhạc\s+)?)?(?:nam\s+|nữ\s+)?(?:\d+\s+thành viên\s+)?' + re.escape(target_text),
                    # "X là một nhóm nhỏ của ban nhạc/nhóm nhạc Y"
                    re.escape(source_text) + r'\s+là\s+(?:một\s+)?(?:nhóm\s+nhỏ|subunit)\s+(?:của\s+)?(?:ban nhạc|nhóm nhạc|nhóm|group)\s+' + re.escape(target_text),
                    # "nhóm nhạc X được tách từ nhóm Y"
                    re.escape(source_text) + r'\s+(?:được\s+)?(?:tách|tách ra|chia)\s+(?:ra\s+)?(?:từ\s+)?(?:nhóm\s+(?:nhạc\s+)?)?(?:nam\s+|nữ\s+)?' + re.escape(target_text),
                    # "X, subunit của Y"
                    re.escape(source_text) + r',?\s+(?:là\s+)?(?:subunit|sub-unit|unit)\s+(?:của|of)\s+' + re.escape(target_text),
                    # "các thành viên nữ tách thành nhóm X" + context có target_text
                    r'các\s+thành viên\s+(?:nam|nữ)\s+.*?(?:tách|thành lập)\s+(?:thành\s+)?(?:nhóm\s+)?' + re.escape(source_text),
                ]
                
                for pattern in strong_subunit_patterns:
                    if re.search(pattern, context_lower, re.IGNORECASE):
                        return True
                
                # Heuristic MẠNH: Nếu source chứa tên target (là subunit) VÀ có keyword
                subunit_keywords = ['nhóm nhỏ', 'subunit', 'sub-unit', 'unit', 'tách ra từ', 'được tách', 
                                    'chia tách', 'spin-off', 'spinoff']
                if target_text in source_text and source_text != target_text:
                    if any(kw in context_lower for kw in subunit_keywords):
                        return True
                
                # Kiểm tra từ khóa mạnh nằm giữa 2 entities
                min_pos = min(source_pos, target_pos)
                max_pos = max(source_pos, target_pos)
                between_text = context_lower[min_pos:max_pos]
                
                strong_subunit_keywords = ['nhóm nhỏ', 'subunit', 'sub-unit', 'tách ra từ', 'được tách từ']
                if any(kw in between_text for kw in strong_subunit_keywords):
                    return True
                
                return False
        
        # MẶC ĐỊNH: KHÔNG CHẤP NHẬN nếu không có validation cụ thể
        # Phải có bằng chứng rõ ràng - không chấp nhận chung chung
        return False
    
    def _heuristic_relationship(self, type1: str, type2: str, 
                                context: str) -> Optional[str]:
        """
        Sử dụng heuristic để xác định quan hệ dựa trên entity types
        CHỈ trả về quan hệ khi có đủ bằng chứng trong context
        
        Args:
            type1: Loại entity nguồn
            type2: Loại entity đích
            context: Context đã lowercase
            
        Returns:
            Loại quan hệ hoặc None
        """
        # Artist + Group -> MEMBER_OF (chỉ khi có từ khóa rõ ràng)
        if type1 == 'Artist' and type2 == 'Group':
            member_keywords = ['thành viên', 'member', 'gồm', 'của nhóm', 'trong nhóm']
            if any(kw in context for kw in member_keywords):
                return 'MEMBER_OF'
        
        # Group + Album -> RELEASED (chỉ khi có từ khóa phát hành)
        if type1 == 'Group' and type2 == 'Album':
            release_keywords = ['phát hành', 'ra mắt', 'released', 'tung ra', 'comeback']
            if any(kw in context for kw in release_keywords):
                return 'RELEASED'
        
        # Artist + Album -> RELEASED (chỉ khi có từ khóa phát hành)
        if type1 == 'Artist' and type2 == 'Album':
            release_keywords = ['phát hành', 'ra mắt', 'released', 'tung ra']
            if any(kw in context for kw in release_keywords):
                return 'RELEASED'
        
        # Album + Song -> CONTAINS (chỉ khi có từ khóa chứa/gồm)
        if type1 == 'Album' and type2 == 'Song':
            contains_keywords = ['chứa', 'bao gồm', 'gồm', 'includes', 'contains', 'track']
            if any(kw in context for kw in contains_keywords):
                return 'CONTAINS'
        
        # Group/Artist + Company -> MANAGED_BY (chỉ khi có từ khóa quản lý)
        if type1 in ['Artist', 'Group'] and type2 == 'Company':
            manage_keywords = ['thuộc', 'quản lý', 'trực thuộc', 'under', 'signed', 'ký hợp đồng']
            if any(kw in context for kw in manage_keywords):
                return 'MANAGED_BY'
        
        # Group + Song -> SINGS (chỉ khi có từ khóa hát/ca khúc)
        if type1 == 'Group' and type2 == 'Song':
            sing_keywords = ['hát', 'ca khúc', 'bài hát', 'title track', 'song', 'trình bày']
            if any(kw in context for kw in sing_keywords):
                return 'SINGS'
        
        # Artist + Song -> SINGS (chỉ khi có từ khóa hát/ca khúc)
        if type1 == 'Artist' and type2 == 'Song':
            sing_keywords = ['hát', 'ca khúc', 'bài hát', 'song', 'trình bày']
            if any(kw in context for kw in sing_keywords):
                return 'SINGS'
        
        # Artist + Song -> PRODUCE_SONG (CHỈ producer mới có quan hệ này)
        # Heuristic không tự động trả về PRODUCE_SONG - phải qua validation
        
        # Artist + Song -> WROTE (CHỈ songwriter mới có quan hệ này)
        # Heuristic không tự động trả về WROTE - phải qua validation
        
        # Artist + Album -> PRODUCE_ALBUM (CHỈ producer mới có quan hệ này)
        # Heuristic không tự động trả về PRODUCE_ALBUM - phải qua validation
        
        # Group + Group -> SUBUNIT_OF (chỉ khi có từ khóa nhóm nhỏ)
        if type1 == 'Group' and type2 == 'Group':
            subunit_keywords = ['nhóm nhỏ', 'subunit', 'sub-unit']
            if any(kw in context for kw in subunit_keywords):
                return 'SUBUNIT_OF'
        
        return None
    
    def _calculate_confidence(self, entity1: Dict, entity2: Dict, 
                             context: str, rel_type: str) -> float:
        """
        Tính độ tin cậy của quan hệ
        
        Args:
            entity1: Entity nguồn
            entity2: Entity đích
            context: Context
            rel_type: Loại quan hệ
            
        Returns:
            Độ tin cậy (0.0 - 1.0)
        """
        base_confidence = 0.7
        
        # Tăng confidence nếu match pattern
        if rel_type in self.patterns:
            for pattern in self.patterns[rel_type]:
                if re.search(pattern, context, re.IGNORECASE):
                    base_confidence += 0.15
                    break
        
        # Tăng confidence nếu có nhiều keywords
        if rel_type in self.keywords:
            keywords = self.keywords[rel_type]['vi'] + self.keywords[rel_type]['en']
            keyword_count = sum(1 for kw in keywords if kw.lower() in context.lower())
            base_confidence += min(keyword_count * 0.05, 0.1)
        
        # Giảm confidence nếu context quá dài (entities xa nhau)
        if len(context) > 200:
            base_confidence -= 0.15
        if len(context) > 300:
            base_confidence -= 0.2
        
        # Giảm confidence nếu không có pattern match (chỉ dựa vào keyword)
        if rel_type in self.patterns:
            pattern_matched = any(re.search(p, context, re.IGNORECASE) for p in self.patterns[rel_type])
            if not pattern_matched:
                base_confidence -= 0.1
        
        return min(max(base_confidence, 0.0), 0.95)
    
    def _remove_duplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Loại bỏ các quan hệ trùng lặp"""
        seen = set()
        unique = []
        
        for rel in relationships:
            key = (
                rel['source'].lower(),
                rel['target'].lower(),
                rel['type']
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def extract_from_infobox(self, infobox: Dict, entity_name: str, 
                             entity_type: str, normalized_to_original: Dict[Tuple[str, str], str] = None) -> List[Dict[str, Any]]:
        """
        Trích xuất quan hệ từ infobox Wikipedia
        
        Args:
            infobox: Dữ liệu infobox
            entity_name: Tên entity chính (ĐÃ LÀ ORIGINAL NAME)
            entity_type: Loại entity chính
            normalized_to_original: Mapping từ normalized name → original_text để tìm tên gốc
            
        Returns:
            Danh sách relationships (với original names)
        """
        relationships = []
        
        if not infobox or not isinstance(infobox, dict):
            return relationships
        
        if normalized_to_original is None:
            normalized_to_original = {}
        
        # Trích xuất MEMBER_OF từ "Current members", "Thành viên"
        member_keys = ['Current members', 'Thành viên', 'Thành viên hiện tại', 'Members']
        for key in member_keys:
            if key in infobox:
                members = self._parse_member_list(infobox[key])
                for member in members:
                    # Normalize member name để tìm original_text
                    member_normalized = normalize_node_name(member).lower()
                    member_key = (member_normalized, 'Artist')
                    member_original = normalized_to_original.get(member_key, member)
                    
                    relationships.append({
                        'source': member_original,  # Dùng original name
                        'source_type': 'Artist',
                        'target': entity_name,  # Đã là original name
                        'target_type': 'Group',
                        'type': 'MEMBER_OF',
                        'confidence': 0.95,
                        'method': 'infobox',
                    })
        
        # Trích xuất MEMBER_OF từ "Past members", "Cựu thành viên" (gộp chung với MEMBER_OF)
        former_keys = ['Past members', 'Cựu thành viên', 'Former members']
        for key in former_keys:
            if key in infobox:
                members = self._parse_member_list(infobox[key])
                for member in members:
                    # Normalize member name để tìm original_text
                    member_normalized = normalize_node_name(member).lower()
                    member_key = (member_normalized, 'Artist')
                    member_original = normalized_to_original.get(member_key, member)
                    
                    relationships.append({
                        'source': member_original,  # Dùng original name
                        'source_type': 'Artist',
                        'target': entity_name,  # Đã là original name
                        'target_type': 'Group',
                        'type': 'MEMBER_OF',  # Gộp chung với MEMBER_OF
                        'confidence': 0.95,
                        'method': 'infobox',
                    })
        
        # Trích xuất MANAGED_BY từ "Label", "Công ty", "Agency"
        label_keys = ['Label', 'Hãng đĩa', 'Công ty', 'Agency', 'Associated acts']
        for key in label_keys:
            if key in infobox:
                companies = self._parse_company_list(infobox[key])
                for company in companies:
                    # Normalize company name để tìm original_text
                    company_normalized = normalize_node_name(company).lower()
                    company_key = (company_normalized, 'Company')
                    company_original = normalized_to_original.get(company_key, company)
                    
                    relationships.append({
                        'source': entity_name,  # Đã là original name
                        'source_type': entity_type,
                        'target': company_original,  # Dùng original name
                        'target_type': 'Company',
                        'type': 'MANAGED_BY',
                        'confidence': 0.9,
                        'method': 'infobox',
                    })
        
        return relationships
    
    def _parse_member_list(self, value: str) -> List[str]:
        """Parse danh sách thành viên từ infobox"""
        if not value:
            return []
        
        # Từ chung chung cần loại bỏ (không phải tên thành viên)
        GENERIC_TERMS = {
            'thành viên', 'members', 'member', 'cựu thành viên', 'former members',
            'past members', 'current members', 'thành viên hiện tại', 'thành viên cũ',
            'danh sách', 'danh sách thành viên', 'danh sách cựu thành viên',
            'list', 'list of members', 'list of former members',
            'current', 'former', 'past', 'cựu'
        }
        
        # Tách theo dấu phẩy hoặc dấu *
        members = []
        parts = re.split(r'[,*•]', value)
        
        for part in parts:
            part = part.strip()
            # Loại bỏ các ký tự không mong muốn
            part = re.sub(r'\[.*?\]', '', part)  # Loại bỏ [1], [2], etc.
            part = re.sub(r'\(.*?\)', '', part)  # Loại bỏ (notes)
            part = part.strip()
            
            if not part:
                continue
            
            # Loại bỏ các từ chung chung (không phải tên thành viên)
            part_lower = part.lower()
            if part_lower in GENERIC_TERMS:
                continue
            # Loại bỏ nếu chứa cụm từ chung chung
            if any(term in part_lower for term in GENERIC_TERMS if len(term) > 3):
                continue
            
            if len(part) >= 2 and len(part) <= 40:
                # Kiểm tra không phải số hoặc ký tự đặc biệt
                if re.match(r'^[A-Za-z\u3131-\u318E\u4E00-\u9FFF]', part):
                    members.append(part)
        
        return members
    
    def _parse_company_list(self, value: str) -> List[str]:
        """Parse danh sách công ty từ infobox"""
        if not value:
            return []
        
        companies = []
        parts = re.split(r'[,;]', value)
        
        for part in parts:
            part = part.strip()
            part = re.sub(r'\[.*?\]', '', part)
            part = re.sub(r'\(.*?\)', '', part)
            part = part.strip()
            
            if part and len(part) >= 2:
                # Kiểm tra có phải công ty không
                if 'entertainment' in part.lower() or 'music' in part.lower() or \
                   'records' in part.lower() or 'agency' in part.lower():
                    companies.append(part)
        
        return companies
    
    def get_statistics(self) -> Dict[str, int]:
        """Trả về thống kê các loại quan hệ đã trích xuất"""
        return dict(self.stats)


# =====================================================
# HỖ TRỢ TĂNG CONFIDENCE TỪ NER SOURCES
# =====================================================

def build_source_cooccurrence_map(ner_entities: List[Dict]) -> Dict[Tuple[str, str], int]:
    """
    Xây dựng bản đồ đồng xuất hiện (co-occurrence) giữa các entities dựa trên sources.
    
    Nếu 2 entities cùng xuất hiện trong nhiều sources, khả năng có quan hệ cao hơn.
    Đây là YẾU TỐ HỖ TRỢ, không phải quyết định.
    
    Args:
        ner_entities: Danh sách entities từ kpop_ner_result.json
        
    Returns:
        Dictionary {(entity1_normalized, entity2_normalized): count} - số lần đồng xuất hiện
        (dùng normalized để match chính xác)
    """
    # Tạo map: source -> list of entities (normalized)
    source_to_entities = defaultdict(set)
    
    for entity in ner_entities:
        entity_text = entity.get('text', '')
        # Normalize để match chính xác
        entity_normalized = normalize_node_name(entity_text).lower()
        sources = entity.get('sources', [])
        source_node = entity.get('source_node', '')
        
        if not entity_normalized:
            continue
        
        # Thêm entity vào tất cả các sources của nó
        for source in sources:
            source_lower = source.lower()
            source_to_entities[source_lower].add(entity_normalized)
        
        # Thêm source_node
        if source_node:
            source_to_entities[source_node.lower()].add(entity_normalized)
    
    # Đếm co-occurrence
    cooccurrence = defaultdict(int)
    
    for source, entities in source_to_entities.items():
        entity_list = list(entities)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                e1, e2 = entity_list[i], entity_list[j]
                # Sắp xếp để key nhất quán
                key = tuple(sorted([e1, e2]))
                cooccurrence[key] += 1
    
    return dict(cooccurrence)


def calculate_source_boost(entity1: str, entity2: str, 
                          cooccurrence_map: Dict[Tuple[str, str], int]) -> float:
    """
    Tính điểm boost confidence dựa trên số lần đồng xuất hiện trong sources.
    
    Args:
        entity1: Tên entity 1 (có thể là original hoặc normalized)
        entity2: Tên entity 2 (có thể là original hoặc normalized)
        cooccurrence_map: Bản đồ đồng xuất hiện (dùng normalized keys)
        
    Returns:
        Điểm boost (0.0 - 0.15)
    """
    # Normalize để match với cooccurrence_map
    e1_normalized = normalize_node_name(entity1).lower()
    e2_normalized = normalize_node_name(entity2).lower()
    key = tuple(sorted([e1_normalized, e2_normalized]))
    count = cooccurrence_map.get(key, 0)
    
    # Tính boost: mỗi lần đồng xuất hiện thêm 0.03, tối đa 0.15
    boost = min(count * 0.03, 0.15)
    
    return boost


def detect_subunit_relationships(graph_nodes: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    Phát hiện quan hệ SUBUNIT_OF dựa trên tên nhóm.
    
    Subunit thường có tên chứa tên parent group:
    - EXO-CBX, EXO-SC → EXO
    - Girls' Generation-TTS, Girls' Generation-Oh!GG → Girls' Generation
    - Super Junior-H, Super Junior-M, Super Junior-K.R.Y. → Super Junior
    
    Returns:
        Danh sách relationships SUBUNIT_OF với chiều đúng (Subunit → Parent)
    """
    relationships = []
    
    # Lấy tất cả groups với tên gốc và normalized
    groups = {}
    for node_id, node_data in graph_nodes.items():
        if node_data.get('label') == 'Group':
            original_name = node_data.get('name', node_id)
            normalized_name = normalize_node_name(original_name)
            groups[normalized_name] = {
                'original': original_name,
                'normalized': normalized_name
            }
    
    group_names = list(groups.keys())
    
    # Tìm các cặp parent-subunit (dùng normalized để so sánh)
    for potential_subunit_norm in group_names:
        for potential_parent_norm in group_names:
            if potential_subunit_norm == potential_parent_norm:
                continue
            
            # Kiểm tra xem potential_subunit có chứa tên potential_parent không (normalized)
            parent_lower = potential_parent_norm.lower()
            subunit_lower = potential_subunit_norm.lower()
            
            # Subunit phải chứa parent name và dài hơn
            if parent_lower in subunit_lower and len(subunit_lower) > len(parent_lower):
                # Kiểm tra thêm: subunit thường có dạng "Parent-Suffix" hoặc "Parent Suffix"
                # Loại bỏ false positive
                
                # Tìm vị trí của parent trong subunit
                idx = subunit_lower.find(parent_lower)
                
                # Parent phải ở đầu tên
                if idx == 0:
                    # Ký tự ngay sau parent phải là separator (-,  , _)
                    after_parent_idx = len(parent_lower)
                    if after_parent_idx < len(subunit_lower):
                        subunit_original = groups[potential_subunit_norm]['original']
                        char_after = subunit_original[after_parent_idx] if after_parent_idx < len(subunit_original) else ''
                        if char_after in ['-', ' ', '_', '(']:
                            relationships.append({
                                'source': groups[potential_subunit_norm]['original'],  # Tên gốc Subunit
                                'source_type': 'Group',
                                'target': groups[potential_parent_norm]['original'],  # Tên gốc Parent group
                                'target_type': 'Group',
                                'type': 'SUBUNIT_OF',
                                'confidence': 0.95,
                                'method': 'name_pattern',
                            })
    
    return relationships


# =====================================================
# MAIN PROCESSING
# =====================================================

def main():
    """Hàm chính để chạy Relationship Extraction"""
    
    print("=" * 70)
    print("MÔ HÌNH NHẬN DẠNG MỐI QUAN HỆ (RELATIONSHIP EXTRACTION)")
    print("=" * 70)
    
    # Load dữ liệu
    print("\n📂 Đang load dữ liệu...")
    
    # Load entities đã nhận dạng
    try:
        with open('data/kpop_ner_result.json', 'r', encoding='utf-8') as f:
            ner_data = json.load(f)
        entities = ner_data.get('entities', [])
        print(f"  ✓ Đã load {len(entities)} entities từ kpop_ner_result.json")
    except Exception as e:
        print(f"  ✗ Lỗi load kpop_ner_result.json: {e}")
        entities = []
    
    # Load text data
    try:
        with open('data/enrichment_text_data.json', 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        records = text_data.get('data', [])
        print(f"  ✓ Đã load {len(records)} records từ enrichment_text_data.json")
    except Exception as e:
        print(f"  ✗ Lỗi load enrichment_text_data.json: {e}")
        records = []
    
    # Load graph nodes và edges để làm entities reference và kiểm tra trùng lặp
    try:
        with open('data/korean_artists_graph_bfs.json', 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        graph_nodes = graph_data.get('nodes', {})
        graph_edges = graph_data.get('edges', [])
        print(f"  ✓ Đã load {len(graph_nodes)} nodes từ graph")
        print(f"  ✓ Đã load {len(graph_edges)} edges từ graph")
        
        # Xây dựng map Album -> allowed sources cho RELEASED từ graph gốc
        global RELEASED_ALLOWED_SOURCES_BY_ALBUM
        RELEASED_ALLOWED_SOURCES_BY_ALBUM = {}
        for edge in graph_edges:
            if not isinstance(edge, dict):
                continue
            edge_type = edge.get('type', '')
            target_type = edge.get('target_type', '')
            if edge_type != 'RELEASED' or target_type != 'Album':
                continue
            src_raw = edge.get('source', '')
            tgt_raw = edge.get('target', '')
            if not src_raw or not tgt_raw:
                continue
            album_norm = normalize_node_name(tgt_raw).lower()
            src_norm = normalize_node_name(src_raw).lower()
            if not album_norm or not src_norm:
                continue
            if album_norm not in RELEASED_ALLOWED_SOURCES_BY_ALBUM:
                RELEASED_ALLOWED_SOURCES_BY_ALBUM[album_norm] = set()
            RELEASED_ALLOWED_SOURCES_BY_ALBUM[album_norm].add(src_norm)
        print(f"  ✓ RELEASED_ALLOWED_SOURCES_BY_ALBUM: {len(RELEASED_ALLOWED_SOURCES_BY_ALBUM)} albums có source gốc")
    except Exception as e:
        print(f"  ✗ Lỗi load korean_artists_graph_bfs.json: {e}")
        graph_nodes = {}
        graph_edges = []
    
    # Load infobox members
    try:
        with open('data/infobox_members.json', 'r', encoding='utf-8') as f:
            infobox_data = json.load(f)
        print(f"  ✓ Đã load infobox_members.json")
    except Exception as e:
        print(f"  ✗ Lỗi load infobox_members.json: {e}")
        infobox_data = {}
    
    # Tạo danh sách entities từ cả NER result và graph nodes
    # QUAN TRỌNG:
    # - Normalize tên để match với text
    # - Giữ original_text để lưu
    # - GIỮ THÊM thông tin source_node / sources / method từ NER để ưu tiên quan hệ đúng
    all_entities = []
    
    # Từ NER result
    for ent in entities:
        original_text = ent.get('text', '')
        if not original_text:
            continue
        normalized_text = normalize_node_name(original_text)
        all_entities.append({
            'text': normalized_text,                  # Dùng normalized để match với text
            'original_text': original_text,           # Giữ tên gốc để lưu
            'type': ent.get('type', 'Entity'),
            'source': 'ner',
            # Thông tin bổ sung để ưu tiên quan hệ đúng
            'ner_method': ent.get('method', ''),
            'ner_source_node': ent.get('source_node', ''),  # ví dụ: (G)I-dle cho album/single rule-based
            'ner_sources': ent.get('sources', []),
        })
    
    # Từ graph nodes
    for node_id, node_data in graph_nodes.items():
        original_text = node_data.get('name', node_id)
        normalized_text = normalize_node_name(original_text)
        all_entities.append({
            'text': normalized_text,  # Dùng normalized để match với text
            'original_text': original_text,  # Giữ tên gốc để lưu
            'type': node_data.get('label', 'Entity'),
            'source': 'graph'
        })
    
    print(f"  ✓ Tổng cộng {len(all_entities)} entities để phân tích (đã normalize để match)")
    
    # Xây dựng danh sách producer/songwriter từ field nghề nghiệp trong graph
    global KNOWN_PRODUCERS, KNOWN_SONGWRITERS
    KNOWN_PRODUCERS, KNOWN_SONGWRITERS = build_producers_songwriters_from_graph(graph_nodes)
    print(f"  ✓ Đã xác định {len(KNOWN_PRODUCERS)} producers và {len(KNOWN_SONGWRITERS)} songwriters từ nghề nghiệp")
    
    # Khởi tạo extractor
    extractor = RelationshipExtractor()
    
    # Trích xuất quan hệ
    print("\n📊 Đang trích xuất quan hệ...")
    all_relationships = []
    
    # Tạo mapping từ (normalized name, type) → original_text để tìm tên gốc khi lưu relationships
    # ƯU TIÊN tên từ graph nodes nếu trùng normalized + type
    normalized_to_original: Dict[Tuple[str, str], str] = {}
    for entity in all_entities:
        norm_lower = entity['text'].lower()
        ent_type = entity.get('type', 'Entity')
        key = (norm_lower, ent_type)
        original = entity.get('original_text', entity['text'])
        src = entity.get('source', '')
        # Nếu chưa có, hoặc entry mới đến từ graph thì override
        if key not in normalized_to_original or src == 'graph':
            normalized_to_original[key] = original
    
    # Bước 1: Trích xuất từ infobox
    print("  📌 Bước 1: Trích xuất từ infobox...")
    infobox_groups = infobox_data.get('groups', {})
    for group_name, group_data in infobox_groups.items():
        infobox = group_data.get('infobox', {})
        if infobox:
            # Normalize group_name để tìm original_text (ưu tiên node gốc trong graph)
            group_normalized = normalize_node_name(group_name).lower()
            group_key = (group_normalized, 'Group')
            group_original = normalized_to_original.get(group_key, group_name)
            
            rels = extractor.extract_from_infobox(infobox, group_original, 'Group', normalized_to_original)
            all_relationships.extend(rels)
    
    print(f"    ✓ Trích xuất {len(all_relationships)} quan hệ từ infobox")
    
    # Bước 2: Xây dựng bản đồ co-occurrence từ NER sources (HỖ TRỢ TĂNG CONFIDENCE)
    print("  📌 Bước 2: Xây dựng bản đồ co-occurrence từ NER sources...")
    cooccurrence_map = build_source_cooccurrence_map(entities)
    print(f"    ✓ Đã xây dựng bản đồ với {len(cooccurrence_map)} cặp entity đồng xuất hiện")
    
    # Bước 3: Phát hiện SUBUNIT_OF từ tên nhóm
    print("  📌 Bước 3: Phát hiện SUBUNIT_OF từ tên nhóm...")
    subunit_relationships = detect_subunit_relationships(graph_nodes)
    all_relationships.extend(subunit_relationships)
    print(f"    ✓ Phát hiện {len(subunit_relationships)} quan hệ SUBUNIT_OF")
    
    # Bước 4: Trích xuất từ text (TỐI ƯU KHÔNG BỎ SÓT)
    print("  📌 Bước 4: Trích xuất từ văn bản...")
    text_relationships = []
    
    # Tạo index cho entities để tìm nhanh hơn (O(1) lookup)
    entity_texts_lower = {e['text'].lower(): e for e in all_entities}
    entity_list = list(entity_texts_lower.keys())
    
    # Chỉ giữ entities có độ dài >= 2 để tránh match sai
    entity_list = [e for e in entity_list if len(e) >= 2]
    
    print(f"    Xử lý TẤT CẢ {len(records)} records (không bỏ sót)")
    
    for i, record in enumerate(records, 1):
        if i % 100 == 0:
            print(f"    Đang xử lý: {i}/{len(records)} records...")
        
        text = record.get('text', '')
        if not text or len(text) < 100:
            continue
        
        text_lower = text.lower()
        
        # TỐI ƯU: Chỉ tìm entities thực sự xuất hiện trong text
        # Thay vì check từng entity, dùng set intersection
        relevant_entities = []
        
        for ent_lower in entity_list:
            # Chỉ check entities có khả năng xuất hiện (dài >= 2)
            if ent_lower in text_lower:
                relevant_entities.append(entity_texts_lower[ent_lower])
        
        if len(relevant_entities) < 2:
            continue
        
        # TỐI ƯU: Nếu quá nhiều entities, chia nhỏ text thành chunks
        # và chỉ xử lý các entities gần nhau
        if len(relevant_entities) > 50:
            # Chia text thành chunks 2000 ký tự, overlap 200
            chunks = []
            chunk_size = 2000
            overlap = 200
            for start in range(0, len(text), chunk_size - overlap):
                chunks.append(text[start:start + chunk_size])
            
            for chunk in chunks:
                chunk_lower = chunk.lower()
                chunk_entities = [e for e in relevant_entities 
                                 if e['text'].lower() in chunk_lower]
                if len(chunk_entities) >= 2:
                    rels = extractor.extract_relationships(chunk, chunk_entities)
                    text_relationships.extend(rels)
        else:
            # Xử lý bình thường
            rels = extractor.extract_relationships(text, relevant_entities)
            text_relationships.extend(rels)
    
    print(f"    ✓ Trích xuất {len(text_relationships)} quan hệ từ văn bản")
    all_relationships.extend(text_relationships)
    
    # Bước 5: Áp dụng source boost cho confidence
    print("  📌 Bước 5: Áp dụng source boost cho confidence...")
    boosted_count = 0
    for rel in all_relationships:
        source = rel.get('source', '')
        target = rel.get('target', '')
        if source and target:
            boost = calculate_source_boost(source, target, cooccurrence_map)
            if boost > 0:
                old_conf = rel.get('confidence', 0.7)
                rel['confidence'] = min(old_conf + boost, 0.98)
                rel['source_boost'] = boost
                boosted_count += 1
    print(f"    ✓ Đã boost confidence cho {boosted_count} quan hệ dựa trên co-occurrence")
    
    # Loại bỏ duplicates và quan hệ trùng với graph gốc
    print("\n📊 Đang gộp và loại bỏ trùng lặp...")
    
    # Tạo set các quan hệ gốc từ graph (để so sánh)
    # CHUẨN HÓA TÊN để loại bỏ suffix như "(ca sĩ)", "(nhóm nhạc)"
    # BAO GỒM source_type và target_type để phân biệt các node cùng tên nhưng khác loại
    existing_relationships = set()
    for edge in graph_edges:
        if isinstance(edge, dict):
            source_raw = edge.get('source', '').strip()
            target_raw = edge.get('target', '').strip()
            rel_type = edge.get('type', '').strip()
            source_type = edge.get('source_type', '').strip()
            target_type = edge.get('target_type', '').strip()

            if not (source_raw and target_raw and rel_type):
                continue

            # Nếu trong edge gốc chưa có source_type/target_type, suy ra từ graph_nodes
            if not source_type:
                node_src = graph_nodes.get(source_raw, {})
                source_type = node_src.get('label', '')
            if not target_type:
                node_tgt = graph_nodes.get(target_raw, {})
                target_type = node_tgt.get('label', '')

            # Chuẩn hóa tên để loại bỏ suffix và khoảng trắng thừa
            source = normalize_node_name(source_raw).lower()
            target = normalize_node_name(target_raw).lower()
            # Bao gồm source_type và target_type để phân biệt node cùng tên nhưng khác loại
            existing_relationships.add((source, target, rel_type, source_type, target_type))
    
    print(f"  ✓ Đã load {len(existing_relationships)} quan hệ từ graph gốc")
    
    # Loại bỏ duplicates và quan hệ trùng với graph gốc
    seen = set()
    unique_relationships = []
    duplicates_with_graph = 0
    
    for rel in all_relationships:
        # Chuẩn hóa tên để loại bỏ suffix và khoảng trắng thừa trước khi check trùng
        # normalize_node_name đã xử lý khoảng trắng, chỉ cần lower()
        source_raw = rel.get('source', '').strip()
        target_raw = rel.get('target', '').strip()
        source = normalize_node_name(source_raw).lower()
        target = normalize_node_name(target_raw).lower()
        rel_type = rel.get('type', '').strip()
        source_type = rel.get('source_type', '').strip()
        target_type = rel.get('target_type', '').strip()
        
        # Key bao gồm cả source_type và target_type để phân biệt node cùng tên nhưng khác loại
        # Ví dụ: "Big Bang (nhóm nhạc)" (Group) và "Big Bang (album)" (Album) là 2 node khác nhau
        key = (source, target, rel_type, source_type, target_type)
        
        # Kiểm tra trùng với graph gốc (đã normalize theo cùng cách)
        if key in existing_relationships:
            duplicates_with_graph += 1
            continue
        
        # Kiểm tra trùng trong batch hiện tại
        if key not in seen:
            seen.add(key)
            unique_relationships.append(rel)
    
    print(f"  ✓ Loại bỏ {duplicates_with_graph} quan hệ trùng với graph gốc")
    print(f"  ✓ Sau khi gộp: {len(unique_relationships)} quan hệ duy nhất (MỚI)")
    
    # Thống kê
    print("\n📊 Thống kê theo loại quan hệ:")
    rel_stats = defaultdict(int)
    for rel in unique_relationships:
        rel_stats[rel['type']] += 1
    
    for rel_type, count in sorted(rel_stats.items(), key=lambda x: -x[1]):
        print(f"  • {rel_type}: {count}")
    
    # Thống kê node nào có nhiều quan hệ bất thường
    print("\n📊 Top node có nhiều quan hệ (cả source lẫn target):")
    node_degree = defaultdict(int)
    for rel in unique_relationships:
        src = rel.get('source', '')
        tgt = rel.get('target', '')
        if src:
            node_degree[src] += 1
        if tgt:
            node_degree[tgt] += 1
    # In top 20 node có nhiều quan hệ nhất
    for node, deg in sorted(node_degree.items(), key=lambda x: -x[1])[:20]:
        print(f"  • {node}: {deg} quan hệ")
    
    # =====================================================
    # BƯỚC CUỐI: LỚP LỌC CUỐI CÙNG BẰNG SLM (SMALL LANGUAGE MODEL)
    # =====================================================
    print("\n🤖 Bước cuối: Lọc cuối cùng bằng Small Language Model (SLM)...")
    
    # Import SLM nếu có
    SLM_AVAILABLE = False
    slm_validator = None
    try:
        from chatbot.small_llm import get_llm
        try:
            slm_validator = get_llm("qwen2-0.5b")
            SLM_AVAILABLE = True
            print("  ✅ SLM đã sẵn sàng để validation")
        except Exception as e:
            print(f"  ⚠️  Không thể load SLM: {e}")
            print("  ⚠️  Bỏ qua lớp lọc SLM, chỉ dùng rule-based filtering")
            SLM_AVAILABLE = False
    except ImportError:
        print("  ⚠️  Module small_llm không khả dụng. Bỏ qua lớp lọc SLM.")
        SLM_AVAILABLE = False
    
    def validate_relationship_with_slm(rel, slm):
        """
        Validate relationship bằng Small Language Model
        
        Args:
            rel: Relationship dictionary
            slm: SmallLLM instance
        
        Returns:
            Tuple (is_valid, confidence_adjustment, reason)
        """
        if not slm:
            return True, 0.0, "SLM không khả dụng"
        
        source = rel.get('source', '')
        target = rel.get('target', '')
        rel_type = rel.get('type', '')
        context = rel.get('context', '')[:500]  # Giới hạn context
        
        # Tạo prompt validation
        validation_prompt = f"""Bạn là chuyên gia về K-pop (nhạc Hàn Quốc). Hãy đánh giá xem quan hệ sau có hợp lệ không.

QUAN HỆ CẦN KIỂM TRA:
- Source: "{source}"
- Target: "{target}"
- Loại quan hệ: {rel_type}
- Context: {context}

YÊU CẦU:
1. Nếu quan hệ {rel_type} giữa "{source}" và "{target}" là HỢP LỆ trong K-pop -> trả lời "VALID"
2. Nếu quan hệ KHÔNG hợp lệ (ví dụ: sai loại, không liên quan, không đúng context...) -> trả lời "INVALID"
3. Nếu không chắc chắn -> trả lời "UNCERTAIN"

Chỉ trả lời một từ: VALID, INVALID, hoặc UNCERTAIN."""

        try:
            response = slm.generate(
                validation_prompt,
                context="",
                max_new_tokens=20,
                temperature=0.1  # Low temperature để có kết quả nhất quán
            )
            
            response_upper = response.strip().upper()
            
            if "VALID" in response_upper:
                return True, 0.05, "SLM xác nhận hợp lệ"  # Tăng confidence nhẹ
            elif "INVALID" in response_upper:
                return False, -0.2, "SLM xác nhận không hợp lệ"  # Giảm confidence đáng kể
            else:
                # UNCERTAIN hoặc không rõ ràng
                return True, -0.05, "SLM không chắc chắn"  # Giảm confidence nhẹ
        except Exception as e:
            # Nếu SLM lỗi, giữ nguyên relationship (fallback)
            return True, 0.0, f"SLM error: {str(e)[:50]}"
    
    # Áp dụng SLM validation
    if SLM_AVAILABLE and slm_validator:
        print("  🔍 Đang validate relationships bằng SLM...")
        slm_validated_rels = []
        slm_removed_rels = 0
        slm_adjusted_rels = 0
        
        # Chỉ validate các relationships có confidence >= 0.75 (để tiết kiệm thời gian)
        rels_to_validate = [r for r in unique_relationships if r.get('confidence', 0) >= 0.75]
        print(f"    Validating {len(rels_to_validate)} relationships (confidence >= 0.75)...")
        
        for i, rel in enumerate(rels_to_validate, 1):
            if i % 50 == 0:
                print(f"    Đã validate: {i}/{len(rels_to_validate)}...")
            
            is_valid, conf_adj, reason = validate_relationship_with_slm(rel, slm_validator)
            
            if is_valid:
                # Điều chỉnh confidence
                new_confidence = max(0.0, min(1.0, rel.get('confidence', 0.7) + conf_adj))
                rel['confidence'] = new_confidence
                if conf_adj != 0:
                    slm_adjusted_rels += 1
                slm_validated_rels.append(rel)
            else:
                slm_removed_rels += 1
                if i <= 10:  # In 10 relationships đầu bị loại
                    print(f"      ❌ Removed: {rel['source']} → {rel['target']} ({rel['type']}) - {reason}")
        
        # Giữ lại các relationships có confidence < 0.75 (không validate bằng SLM)
        low_conf_rels = [r for r in unique_relationships if r.get('confidence', 0) < 0.75]
        unique_relationships = slm_validated_rels + low_conf_rels
        
        print(f"  ✓ SLM validation: Giữ {len(slm_validated_rels)} relationships, loại {slm_removed_rels} relationships")
        print(f"  ✓ Điều chỉnh confidence cho {slm_adjusted_rels} relationships")
    else:
        print("  ⚠️  Bỏ qua SLM validation (SLM không khả dụng)")
    
    # Lưu kết quả
    output = {
        'metadata': {
            'description': 'Mối quan hệ giữa các thực thể K-pop được trích xuất (ĐÃ LOẠI BỎ TRÙNG VỚI GRAPH GỐC)',
            'processed_at': datetime.now().isoformat(),
            'total_relationships': len(unique_relationships),
            'relationships_by_type': dict(rel_stats),
            'extraction_methods': ['infobox', 'subunit_detection', 'rule-based'],
            'confidence_boost': 'source_cooccurrence',
            'duplicates_removed': duplicates_with_graph,
            'existing_relationships_in_graph': len(existing_relationships),
        },
        'relationships': unique_relationships,
    }
    
    output_file = 'data/kpop_relationships_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Đã lưu {len(unique_relationships)} quan hệ vào {output_file}")
    
    # Hiển thị một số ví dụ
    print("\n📋 Một số ví dụ quan hệ được trích xuất:")
    examples_by_type = defaultdict(list)
    for rel in unique_relationships:
        if len(examples_by_type[rel['type']]) < 3:
            examples_by_type[rel['type']].append(rel)
    
    for rel_type, examples in examples_by_type.items():
        print(f"\n  [{rel_type}]")
        for ex in examples:
            print(f"    • {ex['source']} ({ex['source_type']}) → {ex['target']} ({ex['target_type']})")
    
    print("\n" + "=" * 70)
    print("HOÀN TẤT TRÍCH XUẤT QUAN HỆ")
    print("=" * 70)


if __name__ == '__main__':
    main()

