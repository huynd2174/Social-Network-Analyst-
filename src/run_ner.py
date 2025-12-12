# -*- coding: utf-8 -*-
"""
MÔ HÌNH NHẬN DẠNG THỰC THỂ K-POP (TÍCH HỢP BỘ LỌC)
===================================================
1. Nhận dạng entities từ văn bản Wikipedia
2. Lọc theo context K-pop
3. Loại bỏ entities không hợp lệ
"""
import sys
import io
import json
import re
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, Set

# Parse command line arguments
parser = argparse.ArgumentParser(description='K-pop NER: Rule-based và ML-based')
parser.add_argument('--rule-only', '--no-ml', action='store_true',
                    help='Chỉ chạy rule-based NER, không dùng ML-based')
args = parser.parse_args()

# Import ML-based NER module
if args.rule_only:
    ML_NER_AVAILABLE = False
    print("⚠️  Chế độ --rule-only: Chỉ sử dụng rule-based NER")
else:
    try:
        from ml_ner import extract_ml_entities, get_ner_model
        ML_NER_AVAILABLE = True
    except ImportError:
        ML_NER_AVAILABLE = False
        print("⚠️  ml_ner module không khả dụng. Chỉ sử dụng rule-based NER.")

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
if args.rule_only:
    print("MÔ HÌNH NHẬN DẠNG THỰC THỂ K-POP (RULE-BASED ONLY)")
else:
    print("MÔ HÌNH NHẬN DẠNG THỰC THỂ K-POP (HYBRID: RULE-BASED + ML)")
print("=" * 70)

# Khởi tạo ML model nếu có
if ML_NER_AVAILABLE:
    print("\n🤖 Đang khởi tạo ML-based NER model...")
    try:
        ml_model = get_ner_model()
        if ml_model and ml_model.available:
            print("  ✓ ML model đã sẵn sàng")
        else:
            print("  ⚠️  ML model không khả dụng, chỉ sử dụng rule-based")
    except Exception as e:
        print(f"  ⚠️  Lỗi khởi tạo ML model: {e}")
        print("  → Chỉ sử dụng rule-based NER")
else:
    print("\n⚠️  ML-based NER không khả dụng, chỉ sử dụng rule-based")

# =====================================================
# TỪ KHÓA K-POP (để kiểm tra context)
# =====================================================
KPOP_KEYWORDS = {
    # Thuật ngữ K-pop
    'k-pop', 'kpop', 'k pop', 'idol', 'idols', 'thần tượng',
    'debut', 'ra mắt', 'comeback', 'trở lại', 'fandom', 'fan',
    'trainee', 'thực tập sinh', 'agency', 'entertainment',
    'mv', 'music video', 'teaser', 'concept', 'mini album', 'ep',
    'title track', 'ca khúc chủ đề', 'bảng xếp hạng', 'chart',
    'melon', 'gaon', 'billboard', 'inkigayo', 'music bank', 'm countdown',
    'daesang', 'bonsang', 'rookie', 'tân binh', 'world tour',
    # Quốc gia
    'hàn quốc', 'korea', 'korean', 'seoul', 'nam hàn',
    # Vai trò
    'nhóm nhạc', 'ca sĩ', 'rapper', 'dancer', 'vocal', 'main vocal',
    'lead vocal', 'sub vocal', 'main dancer', 'lead dancer',
    'main rapper', 'leader', 'trưởng nhóm', 'maknae', 'visual', 'center',
    # Công ty
    'sm entertainment', 'jyp entertainment', 'yg entertainment', 'hybe',
    'cube entertainment', 'starship', 'pledis', 'fnc', 'woollim',
    'rbw', 'wm entertainment', 'dsp media', 'mbk', 'jellyfish',
    'big hit', 'source music', 'kq entertainment', 'ist entertainment',
    # Nhóm nhạc nổi tiếng
    'bts', 'blackpink', 'twice', 'exo', 'nct', 'aespa', 'ive', 'newjeans',
    'stray kids', 'seventeen', 'txt', 'enhypen', 'le sserafim', 'itzy',
    'red velvet', 'girls generation', 'snsd', 'super junior', 'shinee',
    'got7', 'monsta x', 'ateez', 'the boyz', 'treasure', 'bigbang',
    '2ne1', 'wonder girls', 'f(x)', 'mamamoo', 'gfriend', 'apink',
    'oh my girl', 'loona', 'fromis_9', 'wjsn', 'everglow', 'dreamcatcher',
}

# =====================================================
# TỪ KHÔNG HỢP LỆ (CHUNG CHUNG, KHÔNG PHẢI TÊN RIÊNG)
# =====================================================
INVALID_WORDS = {
    # Tiếng Anh chung
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'was', 'are', 'were',
    'has', 'have', 'had', 'been', 'to', 'for', 'of', 'in', 'on', 'at',
    'by', 'with', 'about', 'as', 'this', 'that', 'these', 'those',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'it', 'he', 'she',
    
    # Tiếng Việt chung
    'của', 'là', 'và', 'với', 'trong', 'có', 'được', 'từ', 'này', 'đó',
    'năm', 'tháng', 'ngày', 'sau', 'trước', 'cũng', 'như', 'khi', 'nếu',
    'bài', 'hát', 'ca', 'khúc', 'album', 'single', 'ep', 'mv',
    
    # Từ bị nhận nhầm thường gặp
    'aideul', 'n nay', 'ch', 'hottest rookies', 'i-land', 'who am i',
    'version', 'ver', 'remix', 'inst', 'instrumental', 'acoustic',
    'live', 'repackage', 'repack', 'special', 'deluxe', 'limited',
    
    # Thuật ngữ K-pop (không phải tên riêng)
    'k-pop', 'kpop', 'k pop', 'idol', 'idols', 'chart', 'charts',
    'gaon', 'oricon', 'billboard', 'melon', 'hanteo',
    'sales', 'vol', 'vol.', 'mr', 'mr.', 'ms', 'ms.',
    'producer', 'school', 'corp', 'corp.', 'inc', 'inc.',
    'lands no', 'earns madison beer', 'k-pop big bang',
    
    # Viết tắt ngắn vô nghĩa (1-2 ký tự)
    'al', 'ba', 'be', 'bo', 'bu', 'don', 'dr', 'el', 'fi', 'fo',
    'ga', 'go', 'ha', 'he', 'hi', 'ho', 'hu', 'i.o', 'h.o', 'fin',
    'ja', 'ji', 'jo', 'ju', 'ka', 'ki', 'ko', 'ku', 'la', 'le',
    'li', 'lo', 'lu', 'ma', 'me', 'mi', 'mo', 'mu', 'na', 'ne',
    'ni', 'no', 'nu', 'pa', 'pe', 'pi', 'po', 'pu', 'ra', 're',
    'ri', 'ro', 'ru', 'sa', 'se', 'si', 'so', 'su', 'ta', 'te',
    'ti', 'to', 'tu', 'va', 've', 'vi', 'vo', 'vu', 'wa', 'we',
    'wi', 'wo', 'wu', 'xa', 'xe', 'xi', 'xo', 'xu', 'ya', 'ye',
    'yi', 'yo', 'yu', 'za', 'ze', 'zi', 'zo', 'zu',
    
    # Suffix công ty (không phải nhóm nhạc)
    'n.v', 'n.v.', 'inc', 'inc.', 'ltd', 'ltd.', 'corp', 'corp.',
    'llc', 'llc.', 'co', 'co.', 'plc', 'plc.',
    
    # Từ chung khác
    'always', 'back', 'best', 'big', 'new', 'old', 'good', 'bad',
    'first', 'last', 'next', 'top', 'hit', 'hot', 'cool', 'nice',
    'love', 'like', 'want', 'need', 'know', 'think', 'feel',
    'day', 'night', 'time', 'year', 'week', 'month',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'beautiful', 'because of you', 'bo peep',
    
    # Từ tổng quát về media/technology (không phải tên nghệ sĩ/album/bài hát)
    'video', 'audio', 'music', 'clip', 'film', 'movie', 'photo', 'picture',
    'image', 'graphic', 'media', 'content', 'file', 'download', 'stream',
    'playback', 'recording', 'broadcast', 'television', 'tv', 'radio',
    
    # Chương trình thực tế/Show (không phải nghệ sĩ)
    'contest', 'season', 'episode', 'show', 'program', 'programme',
    'dictation contest', 'singing contest', 'dance contest',
    'audition', 'survival', 'competition', 'challenge',
    'talk tv', 'idol room', 'idol world', 'idol room', 'idol world',
    'team b', 'team a', 'team c', 'team d', 'team 8',  # Các team chung chung
    'mbc ep', 'radio', 'school class', 'idol maknae rebellion',
    'ost', 'producer idol producer', 'new storm',
    'to the beautiful you',  # Phim
    'hits mr',  # Node sai
    'idol intern king', 'idol maknae rebellion',  # Chương trình có chữ Idol
    'intern king', 'maknae rebellion',  # Tên chương trình (không cần chữ idol ở đầu)
    'debut countdown',  # Chương trình đếm ngược
    'dream team',  # Chương trình Let's Go Dream Team
}

# =====================================================
# ĐỊA DANH (KHÔNG PHẢI NGHỆ SĨ/NHÓM)
# =====================================================
LOCATION_NAMES = {
    'seoul', 'san francisco', 'busan', 'tokyo', 'osaka',
    'new york', 'los angeles', 'london', 'paris', 'berlin',
    'sydney', 'melbourne', 'bangkok', 'singapore', 'hong kong',
    'taipei', 'beijing', 'shanghai', 'mumbai', 'delhi',
    'manila', 'jakarta', 'kuala lumpur', 'ho chi minh',
    # Quận/huyện Hàn Quốc thường xuất hiện trong phần nơi sinh
    'dongdaemun-gu', 'dongdaemun gu',
}

# =====================================================
# TỪ KHÓA QUỐC GIA KHÔNG PHẢI HÀN QUỐC
# =====================================================
NON_KOREAN_COUNTRIES = {
    # Tên quốc gia tiếng Anh
    'malaysia', 'malaysian', 'thailand', 'thai', 'vietnam', 'vietnamese',
    'indonesia', 'indonesian', 'philippines', 'filipino', 'filipina',
    'singapore', 'singaporean', 'china', 'chinese', 'taiwan', 'taiwanese',
    'japan', 'japanese', 'india', 'indian', 'usa', 'american', 'america',
    'uk', 'british', 'england', 'english', 'australia', 'australian',
    'canada', 'canadian', 'france', 'french', 'germany', 'german',
    'brazil', 'brazilian', 'mexico', 'mexican', 'spain', 'spanish',
    'italy', 'italian', 'russia', 'russian', 'hong kong',
    'puerto rico', 'puerto rican',
    
    # Tên quốc gia tiếng Việt  
    'mỹ', 'nhật bản', 'trung quốc', 'đài loan', 'thái lan', 'malaysia',
    'indonesia', 'philippines', 'singapore', 'ấn độ', 'úc', 'anh',
    'pháp', 'đức', 'ý', 'nga', 'brazil', 'canada',
}

# =====================================================
# TỪ KHÓA CHỈ CHƯƠNG TRÌNH/SHOW (KHÔNG PHẢI NGHỆ SĨ)
# =====================================================
SHOW_KEYWORDS = {
    'contest', 'season', 'episode', 'show', 'program', 'programme',
    'audition', 'survival', 'competition', 'challenge', 'festival',
    'awards', 'award', 'ceremony', 'gala', 'concert tour',
    'dictation', 'singing', 'dance', 'talent', 'reality',
    'championship', 'tournament', 'battle', 'game', 'quiz',
    'talk tv', 'idol room', 'idol world', 'room', 'world',
    'tv', 'television', 'broadcast', 'variety',
    'radio', 'school class', 'idol maknae rebellion',
    'mbc ep', 'ep 347', 'ep ', ' ep',  # Pattern chương trình truyền hình
    'ost', 'producer idol producer',  # OST và Producer
    'intern king', 'maknae rebellion',  # Tên chương trình (không cần chữ idol ở đầu)
}

# =====================================================
# BLACKLIST CA SĨ NƯỚC NGOÀI (KHÔNG PHẢI K-POP)
# =====================================================
FOREIGN_ARTIST_BLACKLIST = {
    # Ca sĩ Việt Nam
    'thu minh', 'mỹ tâm', 'hồng nhung', 'thanh lam', 'hà trần',
    'đàm vĩnh hưng', 'lam trường', 'đan trường', 'sơn tùng m-tp',
    'soobin hoàng sơn', 'sơn tùng', 'đức phúc', 'minh hằng',
    'hương tràm', 'hoa minzy', 'minh hằng', 'chi pu',
    
    # Ca sĩ Mỹ/Quốc tế
    'nicki minaj', 'cardi b', 'ariana grande', 'taylor swift',
    'beyoncé', 'rihanna', 'lady gaga', 'katy perry', 'selena gomez',
    'justin bieber', 'ed sheeran', 'bruno mars', 'the weeknd',
    'drake', 'post malone', 'billie eilish', 'dua lipa',
    'adele', 'shakira', 'jennifer lopez', 'madonna',
    'mariah carey', 'arnold', 'lionel richie',
    'britney spears', 'hilary duff', 'michael jackson',
    
    # Ca sĩ Nhật Bản
    'utada hikaru', 'ayumi hamasaki', 'namie amuro', 'boa',  # BoA là K-pop nhưng cần kiểm tra context
    
    # Ca sĩ Trung Quốc
    'wang lee hom', 'jay chou', 'jolin tsai', 'g.e.m',
    
    # Ca sĩ Thái Lan
    'lisa',  # Cần kiểm tra context (có thể là Lisa của BLACKPINK)
    
    # Ca sĩ Malaysia
    'mizz nina', 'yuna',
}

# =====================================================
# TỪ THỪA CẦN LOẠI BỎ Ở CUỐI TÊN
# =====================================================
SUFFIX_WORDS_TO_REMOVE = {
    'rapping', 'singing', 'dancing', 'performing', 'performer',
    'singer', 'rapper', 'dancer', 'idol', 'artist', 'vocalist',
    'producer', 'composer', 'songwriter', 'musician',
    'ca sĩ', 'nghệ sĩ', 'thần tượng', 'rapper', 'dancer',
}

# =====================================================
# THỂ LOẠI NHẠC (KHÔNG PHẢI NGHỆ SĨ)
# =====================================================
MUSIC_GENRES = {
    'hip-hop', 'hip hop', 'hiphop', 'rap', 'r&b', 'rnb',
    'pop', 'rock', 'jazz', 'blues', 'country', 'folk',
    'electronic', 'edm', 'house', 'techno', 'trance',
    'classical', 'opera', 'reggae', 'salsa', 'latin',
    'k-pop', 'kpop', 'j-pop', 'jpop', 'c-pop', 'cpop',
    'ballad', 'dance', 'trot', 'indie', 'alternative',
    'metal', 'punk', 'grunge', 'soul', 'funk', 'disco',
    'gospel', 'christian', 'gospel', 'world music',
}

# =====================================================
# TÊN NHÓM NHẠC K-POP ĐÃ BIẾT (để phát hiện pattern "Group + Member")
# =====================================================
KNOWN_KPOP_GROUPS = {
    'exo', 'girls generation', "girls' generation", 'snsd',
    'bts', 'blackpink', 'twice', 'nct', 'aespa', 'ive',
    'newjeans', 'stray kids', 'seventeen', 'txt', 'enhypen',
    'le sserafim', 'itzy', 'red velvet', 'super junior',
    'shinee', 'got7', 'monsta x', 'ateez', 'the boyz',
    'treasure', 'bigbang', '2ne1', 'wonder girls', 'f(x)',
    'mamamoo', 'gfriend', 'apink', 'oh my girl', 'loona',
    'fromis_9', 'wjsn', 'everglow', 'dreamcatcher',
    'block b', 't-ara', 'kara', 'sistar', 'miss a',
    '4minute', '2pm', '2am', 'shinee', 'infinite',
    'beast', 'highlight', 'b1a4', 'cnblue', 'ftisland',
    'kara', 'after school', 'orange caramel', 'rainbow',
    'nine muses', 'girls day', 'aoa', 'exid', 'crayon pop',
    'ladies code', 'bestie', 'stellar', 'sonamoo',
    # Thêm các nhóm nhạc đã biết
    'one day', 'onetwo', 'pentagon', 'rania', 'sm rookies',
    'seeya', 'shinhwa', 'the ark', 'vixx', 'wanna one',
    'up10tion', 'bonus baby',
    'hello venus', 'cosmic girls',
    # Bổ sung thêm các nhóm mới để tránh pattern "Group + Member" bị giữ làm Artist
    'x1',
}

# =====================================================
# TỪ KHÓA XÁC ĐỊNH LÀ NGHỆ SĨ ÂM NHẠC (Artist phải có)
# =====================================================
MUSIC_ROLE_KEYWORDS = {
    'ca sĩ', 'nghệ sĩ', 'rapper', 'dancer', 'idol', 'thần tượng',
    'vocalist', 'main vocal', 'lead vocal', 'sub vocal',
    'main rapper', 'lead rapper', 'main dancer', 'lead dancer',
    'thành viên', 'cựu thành viên', 'leader', 'trưởng nhóm', 'maknae',
    'visual', 'center', 'all-rounder', 'producer', 'nhà sản xuất',
    'sáng tác', 'viết nhạc', 'composer', 'songwriter',
}

# =====================================================
# TỪ KHÓA LOẠI TRỪ (KHÔNG PHẢI NGHỆ SĨ ÂM NHẠC)
# =====================================================
EXCLUDE_KEYWORDS = {
    'diễn viên', 'actor', 'actress', 'đạo diễn', 'director',
    'nhà văn', 'tác giả', 'writer', 'author', 'tiểu thuyết',
    'mc', 'người dẫn chương trình', 'host', '司会',
    'vận động viên', 'cầu thủ', 'athlete', 'player', 'football',
    'chính trị gia', 'politician', 'tổng thống', 'president', 'bộ trưởng',
    'doanh nhân', 'businessman', 'ceo', 'giám đốc',
    'giáo sư', 'professor', 'bác sĩ', 'doctor', 'luật sư',
    'youtuber', 'streamer', 'influencer', 'tiktoker',
    'người mẫu', 'model', 'siêu mẫu',
}

# =====================================================
# CÔNG TY K-POP ĐÃ BIẾT
# =====================================================
KNOWN_COMPANIES = {
    'SM Entertainment', 'JYP Entertainment', 'YG Entertainment', 'HYBE',
    'Cube Entertainment', 'Starship Entertainment', 'Pledis Entertainment',
    'FNC Entertainment', 'Woollim Entertainment', 'RBW Entertainment',
    'WM Entertainment', 'DSP Media', 'MBK Entertainment',
    'Jellyfish Entertainment', 'Stone Music Entertainment',
    'Kakao Entertainment', 'CJ ENM', 'Big Hit Entertainment',
    'Source Music', 'KQ Entertainment', 'IST Entertainment',
    'Fantagio', 'Brand New Music', 'P Nation', 'AOMG',
    'H1GHR MUSIC', 'Antenna', 'TOP Media', 'Mystic Story',
    'ADOR', 'Belift Lab', 'Play M Entertainment',
}

# =====================================================
# HỌ HÀN QUỐC
# =====================================================
KOREAN_SURNAMES = {
    'Kim', 'Lee', 'Park', 'Choi', 'Jung', 'Jang', 'Cho', 'Kang', 'Yoon',
    'Shin', 'Han', 'Oh', 'Seo', 'Kwon', 'Hwang', 'Ahn', 'Song', 'Jeon',
    'Moon', 'Yang', 'Hong', 'Bae', 'Baek', 'Lim', 'Im', 'Ryu', 'Yoo',
    'Nam', 'Sim', 'Ha', 'Woo', 'Ji', 'Min', 'Cha', 'Jo', 'Noh', 'Ko',
}

# =====================================================
# HÀM CHUẨN HÓA TÊN (PHẢI ĐỊNH NGHĨA TRƯỚC KHI SỬ DỤNG)
# =====================================================
def normalize_for_comparison(name: str) -> str:
    """
    Chuẩn hóa tên để so sánh (loại bỏ khoảng trắng, dấu gạch nối, dấu gạch dưới, lowercase)
    ĐỒNG BỘ HÓA VỚI merge_and_import_neo4j.py để đảm bảo cùng cách chuẩn hóa
    
    Xử lý các trường hợp:
    - "Ahn Ji-young" vs "Ahn Ji young" -> cùng một node
    - "Miyeon" vs "Miyeon (ca sĩ)" -> cùng một node
    - "Cho Mi-yeon" vs "Miyeon" -> có thể match nếu dùng substring matching
    """
    if not name:
        return ""
    
    # Dùng clean_text để chuẩn hóa cơ bản (loại bỏ suffix, normalize khoảng trắng)
    normalized = clean_text(name)
    
    # Loại bỏ khoảng trắng, dấu gạch nối, dấu gạch dưới và lowercase để so sánh
    # Điều này giúp match "Ahn Ji-young" với "Ahn Ji young"
    normalized = normalized.lower().replace(' ', '').replace('-', '').replace('_', '')
    return normalized

def clean_text(text):
    """Làm sạch text và loại bỏ từ thừa ở đầu/cuối"""
    text = text.strip()
    
    # ============================================
    # LOẠI BỎ TIỀN TỐ "Kpop", "K-pop", "K pop" Ở ĐẦU
    # ============================================
    # Pattern: "Kpop BTS" -> "BTS", "K-pop Blackpink" -> "Blackpink"
    text = re.sub(r'^(?:k[\s\-]?pop|kpop|k-pop|k\s+pop)\s+', '', text, flags=re.IGNORECASE)
    
    # Xử lý dấu ngoặc đơn chưa đóng (ví dụ: "Euiwoong (Lew" -> "Euiwoong Lew")
    # Tìm các pattern có dấu mở ngoặc nhưng không có dấu đóng ngoặc
    if '(' in text and text.count('(') > text.count(')'):
        # Có dấu mở ngoặc nhưng không đóng -> chuyển phần trong ngoặc thành text bình thường
        # Pattern: "Name (Incomplete" -> "Name Incomplete"
        # Tìm vị trí dấu mở ngoặc cuối cùng không có dấu đóng
        last_open = text.rfind('(')
        if last_open != -1:
            # Lấy phần trước dấu mở ngoặc và phần sau (bỏ dấu mở ngoặc)
            before = text[:last_open].strip()
            after = text[last_open+1:].strip()
            # Gộp lại với khoảng trắng
            text = f"{before} {after}".strip()
    
    # Loại bỏ các pattern trong ngoặc đơn ở cuối (như "(ca sĩ)", "(nhóm nhạc)")
    # NHƯNG giữ lại nếu là (album), (bài hát), (EP) - vì đó là thông tin quan trọng
    text = re.sub(r'\s*\([^)]*(?:ca sĩ|nhóm nhạc|ban nhạc|nghệ sĩ|singer|group|band)[^)]*\)\s*$', '', text, flags=re.IGNORECASE)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    # Chuẩn hóa dấu gạch nối giữa chữ cái thành khoảng trắng (Ahn Ji-young -> Ahn Ji young)
    text = re.sub(r'(?<=\w)-(?!\s)(?=\w)', ' ', text)
    # Loại bỏ ký tự thừa ở đầu/cuối
    text = text.strip('.,;:!?"\'-()[]{}')
    
    # Loại bỏ từ thừa ở cuối tên (như "rapping", "singing", "dancing")
    words = text.split()
    if len(words) > 1:
        last_word = words[-1].lower()
        if last_word in SUFFIX_WORDS_TO_REMOVE:
            text = ' '.join(words[:-1])
    
    return text

# =====================================================
# LOAD DỮ LIỆU
# =====================================================
print("\n📂 Đang load dữ liệu...")
with open('data/enrichment_text_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

records = data.get('data', [])
print(f"✓ Đã load {len(records)} records")

# Tạo mapping node_id -> text (lowercase) để kiểm tra context
node_texts = {}
# QUAN TRỌNG: Lưu cả type để check trùng theo cả tên VÀ type
# existing_lower: Dict[normalized_name] -> Set[type]
existing_lower: Dict[str, Set[str]] = defaultdict(set)
for record in records:
    node_id = record.get('node_id', '')
    node_name = record.get('node_name', '')
    node_label = record.get('node_label', 'Entity')  # Lấy label/type từ record
    text = record.get('text', '')
    node_texts[node_id] = text.lower()
    if node_name:
        # CHUẨN HÓA tên node gốc để check trùng
        # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
        # Loại bỏ khoảng trắng, dấu gạch nối, dấu gạch dưới để check trùng (Big Bang = BIGBANG = Big-Bang)
        key_normalized = normalize_for_comparison(node_name)
        existing_lower[key_normalized].add(node_label)

total_existing = sum(len(types) for types in existing_lower.values())
print(f"✓ Có {total_existing} entities trong đồ thị (theo tên và type)")

# =====================================================
# LOAD THÔNG TIN THÀNH VIÊN TỪ INFOBOX (ĐÃ CRAWL SẴN)
# =====================================================
try:
    with open('data/infobox_members.json', 'r', encoding='utf-8') as f:
        INFOBOX_MEMBERS = json.load(f)
except Exception:
    INFOBOX_MEMBERS = {"groups": {}, "artists": {}}


# =====================================================
# HÀM KIỂM TRA CONTEXT K-POP
# =====================================================
def has_kpop_context(source_nodes, min_keywords=3):
    """
    Kiểm tra entity có trong context K-pop không
    
    Cải thiện: Sử dụng trọng số cho từ khóa quan trọng
    
    Args:
        source_nodes: Danh sách node IDs nguồn
        min_keywords: Số từ khóa K-pop tối thiểu (mặc định 3)
    """
    if isinstance(source_nodes, str):
        source_nodes = [source_nodes]
    
    # Từ khóa quan trọng có trọng số cao hơn
    HIGH_WEIGHT_KEYWORDS = {
        'k-pop', 'kpop', 'k pop', 'idol', 'idols', 'thần tượng',
        'debut', 'ra mắt', 'comeback', 'trở lại',
        'nhóm nhạc', 'ca sĩ', 'hàn quốc', 'korea', 'korean',
        'sm entertainment', 'jyp entertainment', 'yg entertainment', 'hybe',
    }
    
    for source in source_nodes:
        text = node_texts.get(source, '')
        if text:
            text_lower = text.lower()
            # Đếm với trọng số: từ khóa quan trọng = 2 điểm, từ khóa thường = 1 điểm
            kpop_score = 0
            for kw in KPOP_KEYWORDS:
                if kw.lower() in text_lower:
                    if kw.lower() in HIGH_WEIGHT_KEYWORDS:
                        kpop_score += 2
                    else:
                        kpop_score += 1
            
            # Ngưỡng: ít nhất 3 điểm (tương đương 3 từ khóa thường hoặc 1.5 từ khóa quan trọng)
            if kpop_score >= min_keywords:
                return True
    return False

def is_music_artist(entity_text, source_nodes):
    """
    Kiểm tra xem entity có phải là nghệ sĩ âm nhạc không
    - Phải có từ khóa vai trò âm nhạc trong context gần
    - Không được có từ khóa loại trừ (diễn viên, MC, etc.)
    """
    if isinstance(source_nodes, str):
        source_nodes = [source_nodes]
    
    entity_lower = entity_text.lower()
    
    for source in source_nodes:
        full_text = node_texts.get(source, '')
        if not full_text:
            continue
        
        # Tìm vị trí entity trong text
        idx = full_text.find(entity_lower)
        if idx == -1:
            continue
        
        # Lấy context gần (200 ký tự xung quanh)
        start = max(0, idx - 100)
        end = min(len(full_text), idx + len(entity_text) + 100)
        context = full_text[start:end]
        
        # Kiểm tra có từ khóa loại trừ không
        has_exclude = any(kw in context for kw in EXCLUDE_KEYWORDS)
        if has_exclude:
            return False
        
        # Kiểm tra có từ khóa vai trò âm nhạc không
        has_music_role = any(kw in context for kw in MUSIC_ROLE_KEYWORDS)
        if has_music_role:
            return True
    
    # Nếu không tìm thấy context rõ ràng, kiểm tra toàn bộ text
    for source in source_nodes:
        full_text = node_texts.get(source, '')
        # Nếu có từ khóa loại trừ trong toàn bộ text -> loại
        if any(kw in full_text for kw in EXCLUDE_KEYWORDS):
            # Nhưng nếu có nhiều từ khóa âm nhạc hơn -> có thể là nghệ sĩ kiêm diễn viên
            music_count = sum(1 for kw in MUSIC_ROLE_KEYWORDS if kw in full_text)
            exclude_count = sum(1 for kw in EXCLUDE_KEYWORDS if kw in full_text)
            if music_count > exclude_count * 2:  # Từ khóa âm nhạc phải gấp đôi
                return True
            return False
    
    return False  # Mặc định không phải nghệ sĩ nếu không có context rõ ràng

def is_related_to_existing_nodes(entity_text, source_nodes, existing_names, min_mentioned=2):
    """
    Kiểm tra entity có liên quan đến các node hiện có trong mạng không
    - Xuất hiện cùng với các nghệ sĩ/nhóm nhạc đã có
    
    Cải thiện: Kiểm tra tốt hơn với normalize và partial matching
    """
    if isinstance(source_nodes, str):
        source_nodes = [source_nodes]
    
    # Normalize entity text để so sánh
    entity_normalized = normalize_for_comparison(entity_text) if 'normalize_for_comparison' in globals() else entity_text.lower()
    
    for source in source_nodes:
        # source_node chính là một node trong mạng
        source_normalized = normalize_for_comparison(source) if 'normalize_for_comparison' in globals() else source.lower()
        if source_normalized in existing_names:
            return True
        
        full_text = node_texts.get(source, '')
        if not full_text:
            continue
        
        full_text_lower = full_text.lower()
        
        # Kiểm tra có nhắc đến các node hiện có không
        # Cải thiện: Kiểm tra cả exact match và partial match (tên dài)
        mentioned_count = 0
        for name in existing_names:
            name_lower = name.lower()
            # Exact match
            if name_lower in full_text_lower:
                mentioned_count += 1
            # Partial match cho tên dài (ít nhất 4 ký tự)
            elif len(name_lower) >= 4 and name_lower in full_text_lower:
                mentioned_count += 0.5  # Partial match có trọng số thấp hơn
        
        if mentioned_count >= min_mentioned:  # Phải nhắc đến ít nhất min_mentioned node hiện có
            return True
    
    return False

def is_valid_entity(text, entity_type):
    """Kiểm tra entity có hợp lệ không"""
    # Độ dài cơ bản
    if not text or len(text) > 50:
        return False
    
    # Loại bỏ entities quá ngắn (trừ một số tên nghệ sĩ hợp lệ như RM, IU, CL)
    # Bổ sung thêm các tên ngắn hợp lệ từ infobox: DK, ZN, P.O, The8
    valid_short_names = {'rm', 'iu', 'cl', 'bm', 'jb', 'jj', 'jo', 'im', 'do', 'dk', 'zn', 'p.o', 'the8'}
    if len(text) < 3 and text.lower() not in valid_short_names:
        return False
    # Không chấp nhận nghệ sĩ chỉ 1 ký tự (tránh các tên bị cắt cụt như "B", "K")
    if entity_type == 'Artist' and len(text) == 1:
        return False
    
    # Kiểm tra từ không hợp lệ
    if text.lower() in INVALID_WORDS:
        return False
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # ============================================
    # LOẠI BỎ CA SĨ NƯỚC NGOÀI (BLACKLIST)
    # ============================================
    if text_lower in FOREIGN_ARTIST_BLACKLIST:
        return False
    # Kiểm tra tên có chứa tên trong blacklist không
    for blacklisted in FOREIGN_ARTIST_BLACKLIST:
        if blacklisted in text_lower or text_lower in blacklisted:
            return False
    
    # ============================================
    # LOẠI BỎ NGHỆ SĨ TỪ QUỐC GIA KHÁC (không phải Hàn Quốc)
    # ============================================
    for country in NON_KOREAN_COUNTRIES:
        if country in text_lower:
            return False
    # Kiểm tra từng từ có phải tên quốc gia không
    if any(w in NON_KOREAN_COUNTRIES for w in words):
        return False
    
    # ============================================
    # LOẠI BỎ CHƯƠNG TRÌNH/SHOW (không phải nghệ sĩ/nhóm)
    # ============================================
    # Nếu entity chứa từ khóa show/contest và entity_type là Artist/Group -> loại
    if entity_type in ['Artist', 'Group']:
        for show_kw in SHOW_KEYWORDS:
            if show_kw in text_lower:
                return False
    
    # Loại bỏ các pattern như "... Season X", "... Contest", "... Show"
    show_patterns = [
        r'season\s*\d+', r'episode\s*\d+', r'part\s*\d+',
        r'contest$', r'show$', r'program$', r'competition$',
        r'audition', r'survival', r'challenge$', r'festival$',
        r'awards?$', r'ceremony$', r'gala$',
        r'talk\s*tv', r'idol\s*room', r'idol\s*world',  # Chương trình thực tế
        r'^team\s+[a-z]$', r'^team\s+[a-z]\s*$',  # Team A, Team B, Team C...
        r'countdown$', r'debut\s+countdown',  # Chương trình đếm ngược
    ]
    for pattern in show_patterns:
        if re.search(pattern, text_lower):
            return False
    
    # Loại bỏ các node chung chung như "Team B", "Team A", "Team 8"
    if re.match(r'^team\s+[a-z]$', text_lower) or re.match(r'^team\s+\d+$', text_lower):
        return False
    if text_lower in ['team a', 'team b', 'team c', 'team d', 'team 8']:
        return False
    
    # Phải bắt đầu bằng chữ in hoa, số, hoặc ký tự đặc biệt
    if not re.match(r'^[A-Z0-9가-힣("\']', text):
        return False
    
    # Không chứa chỉ số hoặc ký tự đặc biệt
    if re.match(r'^[\d\.\-\s]+$', text):
        return False
    
    # ============================================
    # LOẠI BỎ THỂ LOẠI NHẠC (KHÔNG PHẢI NGHỆ SĨ)
    # ============================================
    if entity_type == 'Artist':
        if text_lower in MUSIC_GENRES:
            return False
        # Kiểm tra từng từ có phải thể loại nhạc không
        if any(w in MUSIC_GENRES for w in words):
            return False
    
    # ============================================
    # LOẠI BỎ TỪ TỔNG QUÁT VỀ MEDIA/TECHNOLOGY (KHÔNG PHẢI NGHỆ SĨ)
    # ============================================
    if entity_type == 'Artist':
        generic_media_words = {
            'video', 'audio', 'music', 'clip', 'film', 'movie', 'photo', 'picture',
            'image', 'graphic', 'media', 'content', 'file', 'download', 'stream',
            'playback', 'recording', 'broadcast', 'television', 'tv', 'radio',
            'track', 'album', 'single', 'ep', 'mv', 'teaser', 'trailer',
        }
        if text_lower in generic_media_words:
            return False
    
    # ============================================
    # LOẠI BỎ CÁC NHÓM NHẠC ĐÃ BIẾT (KHÔNG PHẢI ARTIST)
    # ============================================
    if entity_type == 'Artist':
        if text_lower in KNOWN_KPOP_GROUPS:
            return False
    
    # ============================================
    # LOẠI BỎ PATTERN "SOLO + TÊN NGHỆ SĨ"
    # ============================================
    if entity_type == 'Artist':
        # Loại bỏ pattern "Solo Somi Zion" (nên tách thành 2 nghệ sĩ riêng)
        if text_lower.startswith('solo '):
            return False
    
    # ============================================
    # LOẠI BỎ PATTERN "EP" HOẶC "EPISODE" TRONG TÊN
    # ============================================
    if entity_type == 'Artist':
        # Loại bỏ pattern như "MBC Ep 347", "UP10TION Ep"
        if re.search(r'\bep\s*\d+', text_lower) or re.search(r'\bepisode\s*\d+', text_lower):
            return False
        # Loại bỏ nếu kết thúc bằng " Ep" hoặc " Episode"
        if text_lower.endswith(' ep') or text_lower.endswith(' episode'):
            return False
    
    # ============================================
    # LOẠI BỎ PHIM
    # ============================================
    if entity_type == 'Artist':
        # Loại bỏ phim như "To The Beautiful You"
        if 'phim' in text_lower or 'film' in text_lower or 'movie' in text_lower:
            return False
        # Loại bỏ các phim đã biết
        if text_lower in ['to the beautiful you']:
            return False
    
    # ============================================
    # LOẠI BỎ CHƯƠNG TRÌNH RADIO
    # ============================================
    if entity_type == 'Artist':
        # Loại bỏ pattern như "Radio' The Boyz Younghoon"
        if text_lower.startswith("radio'") or text_lower.startswith("radio "):
            return False
        # Loại bỏ nếu chứa "radio" và tên nhóm
        for group in KNOWN_KPOP_GROUPS:
            if f"radio" in text_lower and group in text_lower:
                return False
    
    # ============================================
    # LOẠI BỎ PATTERN "ALBUM + NĂM + SỐ" HOẶC "ALBUM + SỐ"
    # ============================================
    if entity_type in ['Artist', 'Album', 'Song']:
        # Loại bỏ pattern như "Album 2011 05"
        if re.match(r'^album\s+\d{4}\s+\d+', text_lower):
            return False
        if re.match(r'^album\s+\d+', text_lower):
            return False
    
    # ============================================
    # LOẠI BỎ PATTERN "IDOL + TÊN CHƯƠNG TRÌNH" HOẶC CHỈ TÊN CHƯƠNG TRÌNH
    # ============================================
    if entity_type == 'Artist':
        # Loại bỏ pattern như "Idol Intern King", "Idol Maknae Rebellion"
        if text_lower.startswith('idol '):
            # Kiểm tra xem có phải chương trình không
            remaining = text_lower[5:].strip()  # Bỏ "idol "
            # Nếu phần còn lại có từ khóa chương trình -> loại bỏ
            show_keywords = ['intern', 'maknae', 'rebellion', 'king', 'show', 'program']
            if any(kw in remaining for kw in show_keywords):
                return False
        
        # Loại bỏ các tên chương trình ngay cả khi không có chữ "idol" ở đầu
        show_names = ['intern king', 'maknae rebellion']
        if text_lower in show_names:
            return False
        # Kiểm tra xem có chứa tên chương trình không
        for show_name in show_names:
            if show_name in text_lower:
                return False
    
    # ============================================
    # LOẠI BỎ ĐỊA DANH CHUNG CHUNG
    # ============================================
    if entity_type in ['Artist', 'Group']:
        # Loại bỏ địa danh như "Seoul", "San Francisco"
        if text_lower in LOCATION_NAMES:
            return False
        # Kiểm tra từng từ có phải địa danh không
        if any(w in LOCATION_NAMES for w in words):
            return False
        # Loại bỏ pattern địa danh Hàn Quốc dạng "X-gu", "X si", "X-do"
        if re.search(r'\b(?:gu|si|do)\b$', text_lower.replace('-', ' ')):
            return False
    
    # ============================================
    # LOẠI BỎ PATTERN "HITS MR" HOẶC TƯƠNG TỰ
    # ============================================
    if entity_type == 'Artist':
        # Loại bỏ pattern như "Hits Mr"
        if text_lower.startswith('hits ') or text_lower == 'hits mr':
            return False
    
    # ============================================
    # LOẠI BỎ TÊN BỊ CẮT CỤT TRÙNG VỚI NODE GỐC
    # ============================================
    if entity_type == 'Artist':
        # Kiểm tra xem có phải tên bị cắt cụt không (ví dụ: "Shin Hye" vs "Park Shin-hye")
        # CHUẨN HÓA entity text trước khi check
        normalized_entity = clean_text(text)
        normalized_entity_lower = normalized_entity.lower()
        # Nếu entity là phần cuối của một node hiện có -> loại bỏ
        for existing_name in existing_lower:
            # Nếu entity là phần cuối của tên hiện có (ít nhất 3 ký tự)
            if len(normalized_entity_lower) >= 3 and existing_name.endswith(normalized_entity_lower):
                # Kiểm tra xem có phải tên bị cắt cụt không (không phải trùng hoàn toàn)
                if existing_name != normalized_entity_lower and len(existing_name) > len(normalized_entity_lower):
                    # Có thể là tên bị cắt cụt -> loại bỏ
                    return False
    
    # ============================================
    # LOẠI BỎ PATTERN "TÊN NHÓM + TÊN THÀNH VIÊN"
    # ============================================
    if entity_type == 'Artist':
        # Kiểm tra xem có phải pattern "Group Name + Member Name" không
        # Ví dụ: "EXO Xiumin", "Girls' Generation Tiffany"
        for group_name in KNOWN_KPOP_GROUPS:
            if text_lower.startswith(group_name + ' '):
                # Có thể là "Group Name + Member Name"
                remaining = text_lower[len(group_name):].strip()
                if remaining and len(remaining) > 1:
                    # Nếu phần còn lại là tên thành viên -> loại bỏ
                    return False
    
    # ============================================
    # LOẠI BỎ TÊN BỊ CẮT CỤT (CHỈ CÓ 1 CHỮ CÁI CUỐI)
    # ============================================
    if entity_type == 'Artist':
        # Kiểm tra pattern như "Block B P" (chỉ có 1 chữ cái cuối)
        # Hoặc "Dani T-ara N4" (có thể là tên bị nhầm)
        words = text.split()
        if len(words) >= 2:
            last_word = words[-1]
            # Nếu từ cuối chỉ có 1 chữ cái hoặc 1 chữ cái + số -> có thể bị cắt cụt
            if len(last_word) == 1 or (len(last_word) == 2 and last_word[1].isdigit()):
                # Kiểm tra xem có phải tên nhóm không
                prefix = ' '.join(words[:-1]).lower()
                if prefix in KNOWN_KPOP_GROUPS:
                    return False
            # Kiểm tra pattern "Name Group N4" hoặc tương tự
            if len(words) >= 3:
                # Ví dụ: "Dani T-ara N4"
                if any(w.lower() in KNOWN_KPOP_GROUPS for w in words):
                    return False
    
    # Kiểm tra theo loại
    if entity_type == 'Artist':
        if len(words) > 4:
            return False
        if any(w in INVALID_WORDS for w in words):
            return False
        # Loại bỏ pattern X.Y (2 chữ cái + dấu chấm) như "T.O"
        if re.match(r'^[A-Z]\.[A-Z]\.?$', text):
            return False
        # Loại bỏ tên bị cắt từ tên nhóm, ví dụ "T ara" từ "T-ara"
        normalized = re.sub(r'[^a-z0-9]', '', text_lower)
        for group_name in KNOWN_KPOP_GROUPS:
            g_norm = re.sub(r'[^a-z0-9]', '', group_name)
            if normalized == g_norm and normalized != group_name:
                return False
        # Tên nghệ sĩ thường có ít nhất 3 ký tự (trừ ngoại lệ)
        if len(text) < 3 and text.lower() not in valid_short_names:
            return False
            
    elif entity_type == 'Group':
        # Loại bỏ prefix là thể loại nhạc đứng trước tên nhóm (ví dụ: "Indie OKDAL", "K-pop Big Bang")
        # Dùng MUSIC_GENRES để cắt bỏ 1 hoặc nhiều thể loại ở đầu, miễn là còn lại >= 1 từ
        original_text = text
        while True:
            lowered = text.lower()
            stripped = lowered.lstrip()
            if stripped != lowered:
                # Đồng bộ lại text nếu có khoảng trắng đầu
                text = text[len(text) - len(stripped):]
                lowered = stripped
            # Tìm genre prefix dài nhất khớp ở đầu
            genre_prefix = None
            for genre in sorted(MUSIC_GENRES, key=lambda g: -len(g)):
                if lowered.startswith(genre + ' ') and len(text.split()) > len(genre.split()):
                    genre_prefix = genre
                    break
            if not genre_prefix:
                break
            # Cắt bỏ genre prefix + khoảng trắng
            cut_len = len(genre_prefix)
            text = text[cut_len:].lstrip()
        text_lower = text.lower()
        words = text_lower.split()

        if len(text) > 30 or text.count(' ') > 5:
            return False
        # Tên nhóm thường có ít nhất 3 ký tự
        if len(text) < 3:
            return False
        # Không phải thuật ngữ K-pop
        kpop_terms = {'k-pop', 'kpop', 'idol', 'chart', 'gaon', 'billboard'}
        if text_lower in kpop_terms:
            return False
        
        # ============================================
        # LOẠI BỎ TÊN CÔNG TY (KHÔNG PHẢI NHÓM NHẠC)
        # ============================================
        company_names = {
            'warner bros', 'warner music', 'warner bros.', 'warner brothers',
            'sony music', 'sony entertainment', 'sony bmg',
            'universal music', 'universal music group', 'umg',
            'emi', 'emi music', 'capitol records',
            'atlantic records', 'columbia records', 'rca records',
            'interscope', 'def jam', 'republic records',
            'geffen records', 'virgin records', 'island records',
        }
        if text_lower in company_names:
            return False
        # Kiểm tra từng phần của tên có phải công ty không
        for company in company_names:
            if company in text_lower:
                return False
        
        # ============================================
        # LOẠI BỎ CÂU MÔ TẢ (KHÔNG PHẢI TÊN NHÓM)
        # ============================================
        # Các động từ thường có trong câu mô tả
        sentence_verbs = [
            'drops', 'releases', 'announces', 'reveals', 'launches',
            'taps', 'hires', 'appoints', 'names', 'promotes',
            'signs', 'debuts', 'debut', 'debuting',
            'performs', 'sings', 'dances',
            'returns', 'confirms', 'denies', 'shares', 'posts',
            'being', 'breezes', 'bringing', 'hits',
        ]
        for verb in sentence_verbs:
            if f' {verb} ' in text_lower or text_lower.startswith(f'{verb} '):
                return False
        
        # Loại bỏ câu bắt đầu bằng động từ (không phải tên nhóm)
        first_word = words[0] if words else ''
        if first_word in sentence_verbs:
            return False

        # Loại bỏ cụm từ tiếng Việt thông dụng (không phải tên riêng), ví dụ: "Sau khi", "Trước khi"
        # Nếu tất cả các từ đều nằm trong INVALID_WORDS (từ chức năng) thì không phải tên nhóm
        if len(words) >= 2 and all(w in INVALID_WORDS for w in words):
            return False
        
        # ============================================
        # LOẠI BỎ CÂU CÓ DẤU NHÁY MỞ KHÔNG ĐÓNG
        # ============================================
        # Ví dụ: "NewJeans drops 'Hype Boy" - có dấu ' mở nhưng không đóng
        if "'" in text and text.count("'") == 1:
            # Có 1 dấu nháy đơn - có thể là câu bị cắt cụt
            return False
        if '"' in text and text.count('"') == 1:
            # Có 1 dấu nháy kép - có thể là câu bị cắt cụt
            return False
        
        # ============================================
        # LOẠI BỎ SUFFIX CÔNG TY
        # ============================================
        company_suffixes = ['n.v', 'n.v.', 'inc', 'inc.', 'ltd', 'ltd.', 
                           'corp', 'corp.', 'llc', 'llc.', 'co.', 'plc']
        if text_lower in company_suffixes:
            return False
        # Loại bỏ nếu kết thúc bằng suffix công ty
        for suffix in company_suffixes:
            if text_lower.endswith(f' {suffix}'):
                return False
        
        # ============================================
        # LOẠI BỎ TÊN NGƯỜI (KHÔNG PHẢI NHÓM)
        # ============================================
        # Pattern "Taps David Blackman" hoặc "Firstname Lastname"
        # Nếu có từ "David", "Scott", "Michael"... có thể là tên người
        common_western_names = {
            'david', 'scott', 'michael', 'john', 'james', 'robert', 'william',
            'richard', 'joseph', 'thomas', 'chris', 'daniel', 'mark', 'paul',
            'steven', 'kevin', 'brian', 'george', 'edward', 'ronald', 'anthony',
        }
        if any(name in text_lower for name in common_western_names):
            # Có thể là tên người phương Tây, không phải nhóm K-pop
            return False
        
        # ============================================
        # LOẠI BỎ CÁC TÊN GROUP SAI / ROMANIZATION KỲ LẠ / CÂU VĂN
        # (TỐI ƯU CHO BỘ DỮ LIỆU HIỆN TẠI)
        # ============================================
        bad_group_texts = {
            # Nhóm nước ngoài / J-pop / non K-pop hoặc câu văn
            'a.k.b. forty-eight', 'akb48 breezes through d',
            'beatles', 'being in hiatus right now',
            'girl next door', 'girl next',
            'daisokaku matsuri', 'declares debut in 2025',
            'doping panda', 'drippin on first full album',
            'exo-cbx hits no', 'garfunkel. sg wannabe',
            'kard talk tour', 'kep1er to debut on january 3rd',
            'k-pop blackpink', 'k-pop m3', 'kpop bts',
            'los angeles. txt', 'mbc chorus',
            'mum48',
            
            # Romanization/phiên âm tiếng Hàn của nhóm đã có node chuẩn
            'aideul', 'akdong myujisyeon', 'aseuteuro',
            'beu-ah-geol', 'beureibeu geolseu',
            'bolbbalgan sachungi', 'bolbbalgan sachungi ',
            'hacheutuhacheu', 'hacheutuhacheu ',
            'pipeuti pipeuti', 'pipeuti pipeuti ',
            'geullaem ', 'reddo berubetto ',
            'aseuteuro ', 'aideul ',
            'tee -eks-tee',
            
            # Mảnh tên / từ chung chung / bị cắt cụt
            'btob (2012', 'berhad', 'bernad',
            'boram . t-ara', 'gen4', 'gb9 b',
            'honeydew', 'jebewon ', 'junsu',
            'labelle', 'lesserafim', 'mio', 'muses ',
            'ne1', 'next year', 'note', 'one ',
            'oh won bin', 'rd ', 'rglow ', 'record',
            'seung-hyun', 'shabet hay dalshabet',
            'take over the u', 'syupeo junieo', 'teurejeo', 'yeoja chingu',
            'ensiti', 'jebewon', 'reddo berubetto', 'shoo',
            # Viết tắt không đầy đủ / từ bị cắt
            'one', 'rd', 'rglow', 'muses', 'tpst',
            # Từ chung chung
            # Nhóm nhạc nước ngoài
            'the beatles', 'beatles',
            # Bổ sung các phiên âm / mảng tên sai mới phát hiện
            'k pop big bang',
            'a.k.b. forty eight', 'a.k.b. forty eight ',  # biến thể spacing
            'beu ah geol', 'beu ah geol ',
            'boram . t ara', 'boram . t-ara',
            # Soloist / nghệ sĩ không phải nhóm
            'g-dragon', 'g dragon',
            # Mảnh tên nhóm bị cắt cụt
            'f ve',  # từ "F-ve Dolls" nhưng chỉ còn "F ve"
            # Các thực thể không phải group trong đồ thị của bạn
            'indie okdal y', 'indie okdal',  # cụm "Indie OKDAL (Y.BIRD from Jellyfish...)"
            'jewelry 2001',                  # tên nhóm kèm năm debut -> không phải tên group riêng
            'produce 101', 'produce 48',     # show tuyển chọn, không phải nhóm nhạc
            'unchanging',                    # album "Unchanging", không phải nhóm
        }
        if text_lower.strip() in bad_group_texts:
            return False
        
        # Loại bỏ tên group có đính kèm năm 19xx/20xx (Jewelry 2001, Fin.K.L 1998, ...)
        # Trong mạng lưới của bạn, năm debut không phải một phần của tên node group
        if re.search(r'\b(19|20)\d{2}\b', text_lower):
            return False
        
        # Loại bỏ cụm có từ khóa mang tính mô tả, không phải tên riêng group
        if any(kw in text_lower for kw in ['indie okdal', ' y.bird', ' y bird ']):
            return False
        
        # ============================================
        # LOẠI BỎ GROUP BẮT ĐẦU BẰNG "K POP" / "K-POP" / "KPOP"
        # ============================================
        # Ví dụ: "K pop Big Bang", "K-pop BTS", "Kpop Blackpink"
        if re.match(r'^k[\s\-]?pop\s+', text_lower):
            return False
        
        # ============================================
        # LOẠI BỎ PHIÊN ÂM TIẾNG ANH CỦA TÊN NHÓM (A.K.B. Forty Eight, etc.)
        # ============================================
        # Pattern: Tên viết tắt có dấu chấm + từ tiếng Anh (Forty, Eight, etc.)
        english_number_words = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                                'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                                'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                                'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
                                'seventy', 'eighty', 'ninety', 'hundred', 'thousand'}
        words_list = text_lower.split()
        if any(w in english_number_words for w in words_list):
            # Có từ số tiếng Anh -> có thể là phiên âm như "A.K.B. Forty Eight"
            if '.' in text or len(words_list) >= 2:
                return False
        
        # ============================================
        # LOẠI BỎ PATTERN "TÊN . TÊN NHÓM" (Boram . T ara)
        # ============================================
        # Pattern: "Tên người . Tên nhóm" hoặc có dấu chấm lẻ giữa các từ
        if re.search(r'\s+\.\s+', text):
            # Có dấu chấm được bao quanh bởi khoảng trắng -> không phải tên nhóm hợp lệ
            return False
        
        # ============================================
        # LOẠI BỎ PHIÊN ÂM TIẾNG HÀN DẠNG "BEU AH GEOL" (viết hoa từng âm tiết)
        # ============================================
        # Pattern: Nhiều từ ngắn (2-4 ký tự), viết hoa đầu, có nguyên âm Hàn
        korean_syllable_vowels = ('eu', 'eo', 'ae', 'ui', 'eui', 'yeo', 'weo', 'oe', 'wo', 'wa', 'ya', 'ye', 'yo', 'yu')
        if len(words_list) >= 2:
            short_syllable_count = 0
            korean_vowel_count = 0
            for w in words_list:
                w_lower = w.lower()
                if len(w) <= 5:  # Âm tiết ngắn
                    short_syllable_count += 1
                if any(v in w_lower for v in korean_syllable_vowels):
                    korean_vowel_count += 1
            # Nếu hầu hết các từ đều ngắn và có nguyên âm Hàn -> phiên âm
            if short_syllable_count >= len(words_list) * 0.6 and korean_vowel_count >= 1:
                # Kiểm tra không phải nhóm K-pop thật
                if text_lower not in KNOWN_KPOP_GROUPS:
                    return False
        
        # Loại thêm các phiên âm dạng "tee -eks-tee", "dee -ei-en" (chỉ toàn chữ thường + dấu gạch)
        if re.search(r'\b[a-z]+\s*-\s*[a-z]+', text_lower):
            return False
        
        # ============================================
        # LOẠI BỎ PHIÊN ÂM TIẾNG HÀN (ROMAJA/LATINH HÓA)
        # ============================================
        # Pattern phổ biến của phiên âm Hàn Quốc:
        # - Kết thúc bằng -eo, -eu, -ae, -ui, -eun, -eon
        # - Có các cụm nguyên âm đặc trưng: eu, eo, ae, ui, eui
        # - Thường viết liền hoặc có khoảng cách giữa các âm tiết
        korean_romanization_patterns = [
            r'^[A-Z]?[a-z]*(?:eu|eo|ae|ui|eui)[a-z]*$',  # Một từ có nguyên âm Hàn
            r'^[A-Z]?[a-z]+(?:eo|eu)$',  # Kết thúc bằng -eo hoặc -eu
            r'^[A-Z]?[a-z]+(?:eun|eon|eul)$',  # Kết thúc bằng -eun, -eon, -eul
        ]
        # Các hậu tố phiên âm Hàn phổ biến
        korean_suffixes = ('eo', 'eu', 'eun', 'eon', 'eul', 'eung')
        # Nếu là một từ đơn (không có khoảng trắng) và kết thúc bằng suffix Hàn
        if ' ' not in text and text_lower.endswith(korean_suffixes):
            # Loại trừ các từ tiếng Anh hợp lệ
            english_exceptions = {'neo', 'stereo', 'romeo', 'video', 'cameo'}
            if text_lower not in english_exceptions:
                return False
        
        # Phát hiện pattern phiên âm 2+ âm tiết viết hoa đầu (Syupeo Junieo, Teurejeo)
        # Nếu có nhiều từ và mỗi từ đều có pattern nguyên âm Hàn
        words_in_text = text.split()
        if len(words_in_text) >= 1:
            korean_vowel_combos = ('eu', 'eo', 'ae', 'ui', 'eui', 'yeo', 'weo')
            romanization_word_count = 0
            for word in words_in_text:
                word_lower = word.lower()
                if any(combo in word_lower for combo in korean_vowel_combos):
                    romanization_word_count += 1
            # Nếu tất cả các từ đều có nguyên âm Hàn -> có thể là phiên âm
            if romanization_word_count == len(words_in_text) and len(words_in_text) <= 3:
                # Kiểm tra thêm: không phải các nhóm K-pop thực sự viết theo kiểu này
                known_valid = {'aespa', 'neo', 'exo'}  
                if text_lower not in known_valid:
                    return False
        
        # ============================================
        # LOẠI BỎ PATTERN "NHÓM + THÀNH VIÊN"
        # ============================================
        # Ví dụ: "Blackpink Jennie" không phải là tên nhóm
        for group_name in KNOWN_KPOP_GROUPS:
            if text_lower.startswith(group_name + ' '):
                # Có thể là "Group Name + Member Name"
                remaining = text_lower[len(group_name):].strip()
                if remaining and len(remaining) > 1:
                    return False
        
        # ============================================
        # LOẠI BỎ NHÓM NHẠC NƯỚC NGOÀI (KHÔNG PHẢI K-POP)
        # ============================================
        non_kpop_groups = {
            'chopstick brothers',  # Nhóm Trung Quốc
            'jonas brothers',      # Nhóm Mỹ
            'backstreet boys',     # Nhóm Mỹ
            'one direction',       # Nhóm Anh
            'westlife',            # Nhóm Ireland
            'nsync', "n'sync",     # Nhóm Mỹ
        }
        if text_lower in non_kpop_groups:
            return False
        
        # ============================================
        # LOẠI BỎ TÊN BỊ CẮT CỤT (VIẾT TẮT KHÔNG ĐẦY ĐỦ)
        # ============================================
        # Ví dụ: "S.E" (từ S.E.S.), "T.O" (từ T.O.P)
        # Pattern: 1-2 chữ cái + dấu chấm, nhưng không phải tên đầy đủ
        if re.match(r'^[A-Z]\.[A-Z]$', text) and len(text) == 3:
            # Kiểm tra xem có phải tên đầy đủ không
            valid_short_groups = {'s.e.s', 'h.o.t', 'n.r.g'}
            if text_lower not in valid_short_groups:
                return False
        # Loại bỏ pattern X.Y (2 chữ cái + 1 dấu chấm ở giữa)
        if re.match(r'^[A-Z]\.[A-Z]\.?$', text):
            return False
            
    elif entity_type in ['Album', 'Song']:
        if len(text) > 40:
            return False
        # Album/Song thường có ít nhất 4 ký tự (loại bỏ từ quá ngắn như "Act", "Again")
        if len(text) < 4:
            return False
        
        # ============================================
        # LOẠI BỎ THUẬT NGỮ CHUNG / TỪ LIÊN QUAN BẢNG XẾP HẠNG / KHÔNG PHẢI ALBUM
        # ============================================
        chart_terms = {
            'chart', 'gaon', 'oricon', 'billboard', 'sales', 'vol', 'mr',
            'cover', 'remix', 'intro', 'outro', 'interlude',
            # Tổ chức/bảng xếp hạng liên quan K-pop
            'miak',  # Music Industry Association of Korea
        }
        if text_lower in chart_terms:
            return False
        
        # Loại bỏ pattern "Chart + năm/số" như "Chart 2022", "Chart 20"
        if re.match(r'^chart\s*\d+', text_lower):
            return False
        
        # Loại bỏ pattern bắt đầu bằng "Top + số" (Top 40, Top 100...)
        if re.match(r'^top\s+\d+', text_lower):
            return False
        
        # Loại bỏ tên có cả "miak" và "kpop/k-pop/k pop" (MIAK K-pop chart)
        if 'miak' in text_lower and ('k pop' in text_lower or 'k-pop' in text_lower or 'kpop' in text_lower):
            return False

        # Loại bỏ các album tổng hợp/best-of chung chung (Best of, Best Selection, Best Album, Compilation)
        # ví dụ: "BEST OF CNBLUE", "Best Selection 2010", "Best of Album"
        compilation_phrases = [
            'best of ', ' best of', 'best selection', 'greatest hits',
            'best album', 'best single', 'best collection',
        ]
        if any(phrase in text_lower for phrase in compilation_phrases):
            # Tuy nhiên vẫn cho qua nếu tên quá cụ thể (có tên nhóm rõ ràng và bạn muốn giữ)
            # Ở đây ưu tiên an toàn: loại bỏ để tránh nhầm với danh mục/playlist/giải thưởng
            return False
        
        # Loại bỏ tên bị cắt cụt kiểu "U KISS cho" (cụm tiếng Việt "cho" ở cuối)
        if text_lower.endswith(' cho'):
            return False
        
        # Loại bỏ các cụm rõ ràng là mô tả J-pop / nhóm Nhật, không phải album K-pop
        jpop_keywords_in_album = ['akb48', 'morning musume', 'musume']
        if any(kw in text_lower for kw in jpop_keywords_in_album):
            return False
        
        # ============================================
        # LOẠI BỎ TỪ ĐƠN CHUNG CHUNG (KHÔNG ĐỦ ĐẶC TRƯNG ĐỂ LÀ TÊN ALBUM)
        # ============================================
        # CHÚ Ý: Một số từ như "Tonight", "Always", "Alive", "Blue" là tên album K-pop thật
        # Chúng đã được lọc bởi pattern matching context-aware, nên bỏ khỏi blacklist
        generic_single_words = {
            'act', 'again', 'chain', 'cover', 'dreaming', 'sorry', 'love', 'heart',
            'step', 'dance', 'night', 'day', 'fire', 'water', 'star', 'moon', 'sun',
            'world', 'life', 'time', 'dream', 'hope', 'light', 'dark',  # Bỏ: blue, red, black, white, pink
            'gold', 'silver', 'sweet', 'crazy', 'happy',
            'sad', 'bad', 'good', 'new', 'old', 'young', 'wild', 'free',  # Bỏ: alive
            'forever', 'never', 'maybe', 'baby', 'honey', 'angel', 'devil',  # Bỏ: always
            'hero', 'power', 'magic', 'fantasy', 'miracle', 'secret', 'mystery',
            'story', 'memory', 'moment', 'feeling', 'emotion', 'passion', 'desire',
            'title', 'song', 'track', 'album', 'single', 'debut', 'comeback',
            'returns', 'youth', 'access', 'wings',  # Bỏ: tonight, solar
            'solo', 'champion', 'crown',  # Các từ đã thêm trước đó
        }
        if text_lower in generic_single_words:
            return False
        
        # ============================================
        # CHO PHÉP CÁC TÊN ALBUM/SONG K-POP ĐÃ BIẾT (1 TỪ)
        # ============================================
        # Những tên album/bài hát K-pop nổi tiếng chỉ có 1 từ
        known_kpop_album_song_names = {
            # Big Bang albums
            'tonight', 'alive', 'always', 'remember', 'made',
            # BTS albums
            'wings', 'proof',
            # BLACKPINK songs/albums
            'pink', 'born',
            # Other common K-pop album/song names (1 word, viết hoa)
            'blue', 'red', 'noir', 'neon', 'fever', 'bloom', 
            'lilac', 'palette', 'yellow', 'violet',
            # Thêm các tên đặc biệt
            'solar',  # MAMAMOO member nhưng cũng là album name pattern
        }
        # Nếu là tên đã biết của K-pop, CHO PHÉP
        if text_lower in known_kpop_album_song_names:
            return True  # Bypass các filter còn lại
        
        # ============================================
        # LOẠI BỎ TÊN NGHỆ SĨ BỊ NHẦM LÀ ALBUM
        # ============================================
        # Một số tên nghệ sĩ K-pop có thể bị nhầm là album
        artist_names_not_album = {
            'solar', 'moonbyul', 'wheein', 'hwasa',  # MAMAMOO members
            'irene', 'seulgi', 'wendy', 'joy', 'yeri',  # Red Velvet members
            'taeyeon', 'tiffany', 'jessica', 'sunny', 'yoona', 'sooyoung', 'yuri', 'hyoyeon', 'seohyun',  # SNSD
        }
        if text_lower in artist_names_not_album:
            return False
        
        # ============================================
        # LOẠI BỎ PATTERN BỊ CẮT CỤT / CÂU VĂN
        # ============================================
        # Pattern "By Step", "Your Head Down" - bị cắt từ tên dài hơn
        truncated_patterns = [
            r'^by\s+',               # "By Step" từ "Step By Step"
            r'^your\s+',             # "Your Head Down" từ "Keep Your Head Down"
            r'^the\s+\w+$',          # "The End" quá ngắn (chỉ 2 từ)
            r'^my\s+\w+$',           # "My Love" quá ngắn
            r'^our\s+\w+$',          # "Our Story" quá ngắn
            r'\s+pt\.?$',            # Kết thúc bằng "Pt" hoặc "Pt." - bị cắt
        ]
        for pattern in truncated_patterns:
            if re.match(pattern, text_lower):
                return False
        # Loại bỏ nếu kết thúc bằng "Pt" (bị cắt từ "Pt. 1", "Pt. 2")
        if text_lower.endswith(' pt') or text_lower.endswith(' pt.'):
            return False
        
        # ============================================
        # LOẠI BỎ CÂU VĂN / MÔ TẢ (KHÔNG PHẢI TÊN ALBUM)
        # ============================================
        # Pattern có động từ hoặc cấu trúc câu
        sentence_indicators = [
            r"exceeds\s+\d+",        # "Fearless' exceeds 380"
            r"has\s+now\s+hit",      # "Has Now Hit No"
            r"hits?\s+no\.?\s*\d*",  # "Hits No 1"
            r"reaches?\s+\d+",       # "Reaches 100"
            r"sells?\s+\d+",         # "Sells 1 Million"
            r"debuts?\s+at",         # "Debuts At No"
            r"peaks?\s+at",          # "Peaks At No"
            r"chart\s*\d+",          # "Chart 2022"
            r"kor\s+down",           # "KOR Down"
            r"title\s+song",         # "Title Song"
            r"love\s+day\s+\d+",     # "Love Day 2012 Jung Eunji"
            r"miak\s+k-?pop",        # "MIAK K-pop"
            r"ranking[s]?\s+\w+\s+\d+",  # "Ranking February 19", "Rankings April 10"
            r"sales\s+chart",        # "Sales Chart", "Sales Chart as Tom Petty"
            r"as\s+tom\s+petty",     # "... as Tom Petty"
            r"already\s+at\s+\d+",   # "Already at 150"
            r"award\s+for",          # "Award for 5 Consecutive Years"
            r"consecutive\s+years",  # "... Consecutive Years"
            r"authentic.*takes",     # "BE 'authentic' but takes 'few risks"
            r"but\s+takes",          # "... but takes ..."
            r"few\s+risks",          # "... few risks"
            r"preview\s+released",   # "Beam of Prism' preview released"
            r"surpasses?\s+\d+",     # "Blue Hour' surpasses 300"
            r"chart\s*-\s*\w+",      # "Chart - Annual", "Chart - Week 13"
            r"chart\s*-\s*week",     # "Chart - Week XX"
            r"charts?\s*-\s*\w+",    # "Charts - July", "Charts - September"
            r"chart\s+dated",        # "Chart dated February 1"
            r"chart\s+for\s+week",   # "Chart for Week ending November 23"
            r"chart\s+from",         # "Chart from September 6-12"
            r"chart\s+in\s+\w+",     # "Chart in November"
            r"to\s+be\s+released",   # "Chat-shire' to be released on October 23"
            r"released\s+on\s+\w+",  # "... released on October 23"
            r"pre-?order\s+begins",  # "IM HERO' pre-order begins"
            r"kicks\s+off",          # "Kicks Off With 'Freal Luv' Video ft"
            r"in\s+sales\s+with",    # "King in Sales with 400"
            r"ranking\s+as\s+of",    # "Ranking as of November 20"
            r"ranking\s+for\s+\w+",  # "Ranking for February 21"
            r"ranking\s+on\s+\w+",   # "Ranking on January 30"
            r"ranks?\s+no",          # "Ranks No"
            r"on\s+march\s+\d+",     # "Ruby on March 7"
            r"on\s+\w+\s+\d+",       # "... on October 23"
            r"top-?\d+\s+uge",       # "Top-40 Uge 38"
            r"label\s+notes?\s+ref", # "Label Notes Ref"
            r"up-?and-?coming",      # "Up-and-coming girls..."
            r"kpop\s+week\s+\d+",    # "KPOP Week 25"
            r"chart\s+week\s+\d+",   # "Chart Week 24"
            r"sold\s+more\s+than",   # "DYE' Sold More Than 280"
            r"\s+sau\s+khi",         # "Dear Santa sau khi" - có từ tiếng Việt
            r"tracklist\s+\d+",      # "Hear Things tracklist 1"
            r"is\s+an?\s+\w+ing",    # "Is an Inviting"
            r"k-?pop\s+miak",        # "K-pop MIAK"
            r"releases?\s+today",     # "Producer releases today"
            r"releases?\s+tomorrow",  # "... releases tomorrow"
            r"releases?\s+on\s+\w+", # "... releases on ..."
            r"producer\s+releases?",  # "Producer releases ..."
            r"on\s+why",              # "... on Why New 'Holler' EP Represents..."
            r"represents?\s+their",    # "... Represents Their 'Mind, Body and Soul'"
            r"represents?\s+",        # "... Represents ..."
        ]
        for pattern in sentence_indicators:
            if re.search(pattern, text_lower):
                return False
        
        # ============================================
        # LOẠI BỎ CÂU VĂN BẮT ĐẦU BẰNG DANH TỪ + ĐỘNG TỪ
        # ============================================
        # Pattern: "Producer releases today", "Album drops tomorrow"
        sentence_starters = ['producer', 'album', 'single', 'ep', 'song', 'track']
        if text_lower.split()[0] in sentence_starters:
            # Kiểm tra xem có động từ không
            if any(verb in text_lower for verb in ['releases', 'release', 'drops', 'drop', 'comes', 'come', 'arrives', 'arrive']):
                return False
        
        # ============================================
        # LOẠI BỎ TÊN BỊ CẮT CỤT (KẾT THÚC BẰNG "VOL", "FIN", ETC.)
        # ============================================
        truncated_suffixes = [' vol', ' fin', ' pt', ' cmb', ' ver', ' o', ' d', " don", " don'"]
        for suffix in truncated_suffixes:
            if text_lower.endswith(suffix):
                return False
        
        # Loại bỏ pattern bị cắt cụt phổ biến
        # "I Don" từ "I Don't...", "As If It" từ "As If It's Your Last", "Yes I" từ "Yes I Am"
        truncated_patterns = [
            r"^i don$",              # "I Don" từ "I Don't..."
            r"^baby don$",           # "Baby don" từ "Baby don't..."
            r"^as if it$",           # "As If It" từ "As If It's Your Last"
            r"^yes i$",              # "Yes I" từ "Yes I Am"
            r"^coup d$",             # "Coup d" từ "Coup d'Etat"
            r"\w+ don$",             # Bất kỳ từ nào kết thúc bằng " don"
        ]
        for pattern in truncated_patterns:
            if re.match(pattern, text_lower):
                return False
        
        # ============================================
        # LOẠI BỎ PATTERN "VERSE + SỐ"
        # ============================================
        # "Verse 2" - không phải album, là phần của album
        if re.match(r'^verse\s+\d+$', text_lower):
            return False
        
        # ============================================
        # LOẠI BỎ TÊN CÓ DẤU NHÁY LẺ + TÊN TRANG WEB
        # ============================================
        # "Red Light' Allkpop" - có dấu nháy lẻ + tên trang web
        website_names = ['allkpop', 'soompi', 'koreaboo', 'billboard', 'genius']
        if "'" in text:
            # Có dấu nháy đơn
            for website in website_names:
                if website in text_lower:
                    return False
        
        # ============================================
        # LOẠI BỎ TÊN NỀN TẢNG / DỊCH VỤ
        # ============================================
        platform_names = {'itunes', 'spotify', 'melon', 'genie', 'bugs', 'flo'}
        if text_lower in platform_names:
            return False
        
        # ============================================
        # LOẠI BỎ VIẾT TẮT KHÔNG RÕ RÀNG
        # ============================================
        abbreviation_patterns = [
            r'^jpn\s+\w+$',          # "JPN Cmb"
            r'^kor\s+\w+$',          # "KOR ..."
            r'^eng\s+\w+$',          # "ENG ..."
        ]
        for pattern in abbreviation_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # ============================================
        # LOẠI BỎ TÊN THÀNH VIÊN K-POP BỊ NHẦM LÀ ALBUM
        # ============================================
        kpop_member_names = {
            'jeonghan', 'wonwoo', 'mingyu', 'seungkwan', 'vernon', 'dino',  # Seventeen
            'gigi', 'bella',  # Tên người nổi tiếng khác
            'minkyeung', 'nayoung', 'kyulkyung', 'eunwoo', 'roa', 'yuha', 'rena', 'kyla', 'sungyeon',  # Pristin
        }
        if text_lower in kpop_member_names:
            return False
        
        # ============================================
        # LOẠI BỎ TỪ CHUNG CHUNG KHÁC (chỉ những từ rõ ràng không phải album)
        # ============================================
        generic_album_words = {
            'group note', 'notes ref',
            'makestar',  # Tên nền tảng crowdfunding
        }
        if text_lower in generic_album_words:
            return False
        
        # ============================================
        # LOẠI BỎ PATTERN CÓ DẤU NHÁY LẺ + NĂM
        # ============================================
        # "CRUSH' 2014" - có dấu nháy lẻ
        if re.search(r"'\s*\d{4}$", text):
            return False
        
        # ============================================
        # LOẠI BỎ TÊN CHƯƠNG TRÌNH / LIVE
        # ============================================
        if 'countdown live' in text_lower or 'live concert' in text_lower:
            return False
        
        # ============================================
        # LOẠI BỎ THÔNG TIN CHART (ORICON + SỐ)
        # ============================================
        if re.search(r'^oricon\s+\d+', text_lower):
            return False
        
        # ============================================
        # LOẠI BỎ PATTERN "TÊN + SỐ THỨ HẠNG" (chart positions)
        # ============================================
        # Pattern như "Crayon 16 1", "DDARA 12 1", "Feel me 17 1"
        # Thường là: Tên bài + vị trí chart + tuần
        if re.search(r'\s+\d+\s+\d+$', text):
            return False
        # Pattern kết thúc bằng "số 1" hoặc "số số"
        if re.search(r'\s+\d{1,3}\s+1$', text):
            return False
        
        # ============================================
        # LOẠI BỎ PATTERN VIẾT TẮT CHART
        # ============================================
        chart_abbreviations = [
            r'^gaon\s+',             # "Gaon 151", "Gaon khi"
            r'^hq\s+',               # "HQ Down", "HQ Gaon TQ Baidu"
            r'\s+hq\s+',             # "... HQ ..."
            r'^tq\s+',               # "TQ Baidu"
            r'\s+tq\s+',             # "... TQ ..."
            r'baidu',                # "... Baidu"
        ]
        for pattern in chart_abbreviations:
            if re.search(pattern, text_lower):
                return False
        
        # ============================================
        # LOẠI BỎ TÊN CÔNG TY BỊ NHẦM LÀ ALBUM
        # ============================================
        company_patterns = [
            r's\.?m\.?\s+entertainment',  # "S.M Entertainment Co"
            r'entertainment\s+co',
            r'yg\s+entertainment',
            r'jyp\s+entertainment',
            r'hybe\s+',
        ]
        for pattern in company_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # ============================================
        # LOẠI BỎ TÊN NHÓM + TỪ LẺ
        # ============================================
        # Pattern "U-KISS cho", "BTS và", "EXO với"
        group_plus_word = [
            r'^u-kiss\s+\w{1,4}$',    # "U-KISS cho"
            r'^bts\s+\w{1,4}$',
            r'^exo\s+\w{1,4}$',
            r'^nct\s+\w{1,4}$',
        ]
        for pattern in group_plus_word:
            if re.search(pattern, text_lower):
                return False
        
        # ============================================
        # LOẠI BỎ PATTERN UNLOCK/EP LẪN LỘN
        # ============================================
        # "Unlock UNIQ EP Falling In Love" - nhiều album gộp lại
        if 'unlock' in text_lower and 'ep' in text_lower:
            return False
        if text.count(' ') >= 4 and ('EP' in text or 'Album' in text):
            # Quá nhiều từ và có EP/Album trong tên -> có thể là lỗi
            return False
        
        # ============================================
        # LOẠI BỎ TÊN CÓ CHỨA TÊN NGHỆ SĨ/NHÓM NHẠC LẪN LỘN
        # ============================================
        # Pattern "Album Name + Artist Name" như "Beep BTOB Yoojin"
        kpop_group_names_in_album = ['btob', 'exo', 'bts', 'nct', 'got7', 'ikon', 'winner', 'ateez', 'stray kids']
        for group in kpop_group_names_in_album:
            if group in text_lower and len(text.split()) >= 2:
                # Có tên nhóm trong tên album và có nhiều từ -> có thể là lỗi
                words_after_group = text_lower.split(group)[-1].strip()
                if words_after_group and len(words_after_group) > 2:
                    return False
        
        # ============================================
        # LOẠI BỎ TÊN NHÓM NHẠC BỊ NHẦM LÀ ALBUM
        # ============================================
        group_names_not_album = {
            'april', 'twice', 'blackpink', 'bts', 'exo', 'nct', 'red velvet',
            'mamamoo', 'itzy', 'aespa', 'ive', 'newjeans', 'le sserafim',
            'stayc', 'nmixx', 'kep1er', 'gidle', 'everglow', 'loona',
        }
        if text_lower in group_names_not_album:
            return False
        
        # ============================================
        # LOẠI BỎ TÊN NGƯỜI (KHÔNG PHẢI ALBUM)
        # ============================================
        # Pattern "Firstname Lastname" với tên phương Tây
        western_names = {
            'danny', 'chung', 'david', 'scott', 'michael', 'john', 'james', 
            'robert', 'william', 'richard', 'joseph', 'thomas', 'chris', 
            'daniel', 'mark', 'paul', 'steven', 'kevin', 'brian', 'george',
            'jung', 'eunji', 'kim', 'lee', 'park', 'choi', 'kang',
        }
        words_in_album = text_lower.split()
        # Nếu tất cả các từ đều là tên người -> không phải album
        if len(words_in_album) >= 2 and all(w in western_names for w in words_in_album):
            return False
        # Nếu có tên + năm -> có thể là lỗi
        if re.search(r'\b\d{4}\b.*[A-Z][a-z]+', text) or re.search(r'[A-Z][a-z]+.*\b\d{4}\b', text):
            # Có năm trong tên album -> kiểm tra thêm
            if any(name in text_lower for name in ['jung', 'eunji', 'kim', 'lee', 'park']):
                return False
        
        # ============================================
        # LOẠI BỎ TỪ VIẾT TẮT / THUẬT NGỮ ÂM NHẠC
        # ============================================
        music_abbreviations = {
            'all out', 'kor down', 'jpn', 'usa', 'uk', 'eng', 'chn', 'twn',
            'mv', 'pv', 'ost', 'bgm', 'inst', 'ver', 'version',
        }
        if text_lower in music_abbreviations:
            return False
            
    elif entity_type == 'Company':
        company_kw = ['entertainment', 'music', 'media', 'records', 'label']
        if not any(kw in text_lower for kw in company_kw):
            if len(text) > 20 or not text[0].isupper():
                return False
    
    return True

# =====================================================
# PATTERNS NER (MỞ RỘNG ĐỂ BẮT NHIỀU THỰC THỂ HƠN)
# =====================================================
patterns = {
    'Artist': [
        # Pattern cơ bản
        r'(?:ca sĩ|nghệ sĩ|rapper|idol|thần tượng|thành viên)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s+(?:là|sinh|đã|được|có)|\,|\.|$)',
        r'([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s+(?:là một|là)\s+(?:ca sĩ|nghệ sĩ|rapper|idol)',
        # Thành viên nhóm: "thành viên G-Dragon và T.O.P"
        r'thành viên\s+([A-Z][a-zA-Z0-9\-\.]+)(?:\s+và|\s*,)',
        # Solo artist: "G-Dragon phát hành album solo"
        r'([A-Z][a-zA-Z0-9\-\.]+)\s+phát hành\s+(?:album|EP|single)\s+solo',
        # "do X viết lời" - nhạc sĩ
        r'do\s+(?:chính\s+)?([A-Z][a-zA-Z0-9\-\.]+)\s+(?:viết|sáng tác|sản xuất)',
        # "X tham gia" - nghệ sĩ
        r'([A-Z][a-zA-Z0-9\-\.]+)\s+(?:tham gia|hợp tác|góp mặt|viết lời)',
        # "thành viên Verbal của M-Flo" pattern
        r'thành viên\s+([A-Z][a-zA-Z0-9\-\.]+)\s+của',
    ],
    'Group': [
        r'(?:nhóm nhạc|ban nhạc|group|boyband|girlgroup)\s+([A-Z][a-zA-Z0-9\s\-\'\.()]+?)(?:\s+(?:là|gồm|có|được|ra mắt)|\,|\.|$)',
        r'([A-Z][a-zA-Z0-9\s\-\'\.()]+?)\s+(?:là một|là)\s+(?:nhóm nhạc|ban nhạc)',
        # "nhóm X trở lại", "nhóm X phát hành"
        r'nhóm\s+([A-Z][a-zA-Z0-9\s\-\'\.()]+?)\s+(?:trở lại|phát hành|ra mắt|biểu diễn)',
        # "của nhóm nhạc nam Hàn Quốc Big Bang" - rất phổ biến trong Wikipedia
        r'của\s+nhóm\s+nhạc\s+(?:nam|nữ)?\s*(?:Hàn\s+Quốc|Hàn–Trung\s+Quốc)?\s*([A-Z][a-zA-Z0-9\s\-\'\.()]+?)(?:\s*[,\.]|\s+(?:được|do|là|bao gồm))',
        # "của ban nhạc Hàn Quốc Big Bang"
        r'của\s+ban\s+nhạc\s+(?:Hàn\s+Quốc)?\s*([A-Z][a-zA-Z0-9\s\-\'\.()]+?)(?:\s*[,\.]|\s+(?:được|do|là))',
        # "nhóm nhạc nam Hàn Quốc X" - ngay sau định nghĩa
        r'nhóm\s+nhạc\s+(?:nam|nữ)?\s*(?:Hàn\s+Quốc|Hàn–Trung\s+Quốc)?\s+([A-Z][a-zA-Z0-9\s\-\'\.()]+?)(?:\s*[,\.]|\s+(?:được|do|là|gồm|bao gồm|thành lập))',
        # "nhóm nhỏ X của" - subgroup
        r'nhóm\s+nhỏ\s+(?:chính\s+thức)?\s*([A-Z][a-zA-Z0-9\s\-\'\.()]+?)\s+của',
        # "bộ đôi X" - duo group
        r'bộ\s+đôi\s+([A-Z][a-zA-Z0-9\s\-\'\.()]+?)(?:\s*[,\.]|\s+(?:được|do|là|gồm))',
    ],
    'Album': [
        # === PATTERNS CƠ BẢN ===
        r'(?:album|mini[- ]?album|EP)\s+["\']?([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']?(?:\s+(?:là|được|phát hành)|\,|\.|$)',
        # Album với dấu ngoặc kép đặc biệt (Wikipedia thường dùng)
        r'(?:album|mini[- ]?album|EP)\s+["""]([A-Z][a-zA-Z0-9\s\-\'\.]+?)["""]',
        
        # === PATTERNS THEO NGỮ CẢNH TIẾNG VIỆT ===
        # "EP Always được phát hành vào năm 2007"
        r'EP\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s+(?:được phát hành|ra mắt|bán được)',
        # "mini album đầu tiên Always"
        r'mini album\s+(?:đầu tiên|thứ \w+|tiếp theo|mới)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:được|bán|ra|đạt|giành))',
        # "album đầu tay Since 2007"
        r'album\s+(?:đầu tay|đầu tiên|thứ \w+|tiếp theo|mới nhất|phiên bản đặc biệt)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:được|bán|ra|tổng hợp))',
        # "phát hành album Tonight"
        r'phát hành\s+(?:album|EP|mini album)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:vào|với|bao gồm))',
        # "ra mắt album Alive"
        r'ra mắt\s+(?:album|EP|mini album)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:vào|với|dưới))',
        # "trở lại với album Tonight"
        r'trở lại\s+(?:với|bằng|cùng)\s+(?:album|EP)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:vào|với))',
        # "album thành công nhất của mình, Alive"
        r'album\s+(?:thành công nhất|nổi tiếng nhất|hay nhất)[^,]*,\s*([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:được|là))',
        
        # === PATTERNS TIẾNG ANH (PHỔ BIẾN TRONG WIKIPEDIA TIẾNG VIỆT) ===
        # "album tiếng Nhật đầu tiên mang tên Big Bang"
        r'album\s+(?:tiếng\s+\w+)?\s*(?:đầu tiên|thứ \w+)?\s*(?:mang tên|có tên|tên là)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.])',
        # "album Remember, với ca khúc"
        r'album\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s*,\s*với\s+(?:ca khúc|bài hát)',
        # "EP Stand Up - kết hợp với"
        r'(?:EP|album)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s*-\s*(?:kết hợp|bao gồm|với)',
        
        # === PATTERNS MỚI - PHỔ BIẾN TRONG WIKIPEDIA ===
        # "là album phòng thu đầu tay của X" - bắt album từ đầu câu
        r'([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s+là\s+(?:album|mini-album|EP)\s+(?:phòng thu|studio)?\s*(?:đầu tay|đầu tiên|thứ \w+)',
        # "album X được phát hành" - album + tên + động từ
        r'album\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s+(?:được phát hành|ra mắt|phát hành|bán được)',
        # "từ album X" - trích từ album
        r'từ\s+(?:album|EP)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:phát hành|của))',
        # "trong album X" - bài hát trong album
        r'(?:trong|nằm trong)\s+(?:album|EP)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:phát hành|của))',
        # "phiên bản tiếng Nhật của X"
        r'phiên bản\s+tiếng\s+\w+\s+của\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.])',
        # "đĩa đơn trích từ album X"
        r'trích\s+từ\s+album\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)(?:\s*[,\.]|\s+(?:phát hành))',
    ],
    'Song': [
        # === PATTERNS CƠ BẢN ===
        # Dạng có dấu ngoặc kép chuẩn
        r'(?:bài hát|ca khúc|single|đĩa đơn)\s+["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']',
        # Dạng có dấu ngoặc kép đặc biệt (Wikipedia)
        r'(?:bài hát|ca khúc|single|đĩa đơn)\s+["""]([A-Z][a-zA-Z0-9\s\-\'\.]+?)["""]',
        # Ca khúc chủ đề
        r'ca khúc chủ đề\s+["\']?([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']?',
        # Dạng không dấu ngoặc kép + động từ
        r'(?:bài hát|ca khúc|single|đĩa đơn)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s+(?:được|do|của|ra mắt|phát hành|trong|là|đứng đầu|giành|trở thành)\b',
        # Dạng "có tên"/"mang tên"
        r'(?:bài hát|ca khúc|single|đĩa đơn)\s+(?:có tên|mang tên)\s+["\']?([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']?',
        
        # === PATTERNS THEO NGỮ CẢNH TIẾNG VIỆT ===
        # "đĩa đơn số một của họ là \"Lies\""
        r'đĩa đơn\s+(?:số một|đầu tiên|thứ \w+)[^"]*["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']',
        # "ca khúc hit đột phá đầu tiên của nhóm" - thường theo sau là tên bài
        r'ca khúc\s+(?:hit|nổi tiếng|đột phá)[^,]*,?\s*["\']?([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']?(?:\s*[,\.]|\s+(?:trở thành|đứng đầu|giành))',
        # "single tiếng Nhật đầu tiên \"My Heaven\""
        r'single\s+(?:tiếng\s+\w+)?\s*(?:đầu tiên|thứ \w+|mới)?\s*["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']',
        # "bài hát chủ đề \"Monster\""
        r'bài hát\s+chủ đề\s+["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']',
        # "ca khúc \"Lies\" (Tiếng Triều Tiên: ...)"
        r'ca khúc\s+["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\'](?:\s*\()',
        # "Bài hát \" Flower Road \" được phát hành" (có khoảng trắng trong ngoặc kép)
        r'[Bb]ài hát\s+["\"]\s*([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s*["\"]\s+(?:được|do|là|đứng)',
        
        # === PATTERNS DANH SÁCH CA KHÚC ===
        # "các ca khúc \"Lies\", \"Last Farewell\""
        r'(?:các\s+)?ca khúc\s+["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\'](?:\s*,|\s+và)',
        # "bao gồm các ca khúc \"We Belong Together\""
        r'bao gồm\s+(?:các\s+)?ca khúc\s+["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']',
        
        # === PATTERNS CHO HIT/SINGLE PHỔ BIẾN ===
        # "hit X của nhóm"
        r'hit\s+["\']?([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']?\s+(?:của|giúp|đưa)',
        # "single X đạt được"
        r'single\s+["\']?([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']?\s+(?:đạt được|đứng|vươn)',
        # "Cú hít \"Lies\" đã đưa Big Bang"
        r'[Cc]ú hít\s+["\']([A-Z][a-zA-Z0-9\s\-\'\.]+?)["\']',
        
        # === PATTERNS MỚI - ESCAPED QUOTES TRONG JSON ===
        # Pattern cho dấu ngoặc kép escaped: \"X\"
        r'(?:bài hát|ca khúc|single|đĩa đơn)\s+\\"([A-Z][a-zA-Z0-9\s\-\'\.]+?)\\"',
        # "đĩa đơn \"Blue\", \"Fantastic Baby\""
        r'đĩa đơn\s*,?\s*\\"([A-Z][a-zA-Z0-9\s\-\'\.]+?)\\"',
        # "với ca khúc \"X\""
        r'với\s+ca\s+khúc\s+\\"([A-Z][a-zA-Z0-9\s\-\'\.]+?)\\"',
        # "bài hát \"X\" của"
        r'bài\s+hát\s+\\"([A-Z][a-zA-Z0-9\s\-\'\.]+?)\\"\s+(?:của|trong|là)',
        # Pattern cho đĩa đơn chính
        r'đĩa\s+đơn\s+(?:chính|mới)?\s*(?:mang tên)?\s*\\"([A-Z][a-zA-Z0-9\s\-\'\.]+?)\\"',
        # "\" X \"là" pattern - tên bài ở đầu đoạn text
        r'\\"\s*([A-Z][a-zA-Z0-9\s\-\'\.]+?)\s*\\"\s*(?:là đĩa đơn|là ca khúc|là bài hát)',
    ],
    'Company': [
        r'(?:công ty|agency|label)\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?(?:Entertainment|Music|Media)?)',
        # "được thành lập bởi YG Entertainment"
        r'(?:được thành lập|thuộc|quản lý)\s+bởi\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?(?:Entertainment|Music|Media))',
        # "dưới sự dẫn dắt của YG Entertainment"
        r'(?:dưới sự|thuộc)\s+(?:dẫn dắt|quản lý)\s+(?:của\s+)?([A-Z][a-zA-Z0-9\s\-\'\.]+?(?:Entertainment|Music|Media))',
        # "thông qua hãng thu âm X Entertainment"
        r'(?:thông qua|bởi)\s+(?:hãng\s+thu\s+âm|công ty)?\s*([A-Z][a-zA-Z0-9\s\-\'\.]+?(?:Entertainment|Music|Media))',
        # "được X Entertainment phát hành"
        r'được\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?(?:Entertainment|Music|Media))\s+(?:phát hành|phân phối)',
        # "ký hợp đồng với X Entertainment"
        r'ký\s+hợp\s+đồng\s+với\s+([A-Z][a-zA-Z0-9\s\-\'\.]+?(?:Entertainment|Music|Media))',
    ],
}

# =====================================================
# TRÍCH XUẤT ENTITIES
# =====================================================
def extract_entities(text, entity_type, pattern_list):
    """Trích xuất entities bằng regex"""
    entities = []
    seen = set()
    
    for pattern in pattern_list:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                entity_text = match.group(1) if match.lastindex else match.group(0)
                entity_text = clean_text(entity_text)
                
                if not entity_text or entity_text.lower() in seen:
                    continue
                # CHUẨN HÓA entity text trước khi check với existing_lower
                # QUAN TRỌNG: Dùng CÙNG cách chuẩn hóa như khi tạo existing_lower
                # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
                entity_key = normalize_for_comparison(entity_text)
                # QUAN TRỌNG: Chỉ loại bỏ nếu cùng tên VÀ cùng type
                # Nếu trùng -> bỏ qua (không tạo node mới)
                if entity_key in existing_lower and entity_type in existing_lower[entity_key]:
                    continue
                if not is_valid_entity(entity_text, entity_type):
                    continue
                
                seen.add(entity_text.lower())
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'method': 'rule-based',
                    'confidence': 0.7
                })
        except:
            continue
    return entities

def extract_members_from_list(text):
    """Trích xuất thành viên từ pattern liệt kê như 'bao gồm X thành viên: A, B, C và D'"""
    entities = []
    seen = set()
    name_list_pattern = r'([A-Za-z\-\'\.\s,&/]+?)'
    
    role_keywords_vi = [
        'thành viên', 'các thành viên', 'thành viên gồm', 'các thành viên gồm',
        'cựu thành viên', 'thành viên hiện tại', 'thành viên cũ', 'thành viên mới',
        'ca sĩ', 'các ca sĩ', 'nghệ sĩ', 'các nghệ sĩ', 'rapper', 'các rapper',
        'idol', 'các idol', 'giọng ca', 'giọng hát', 'vocal', 'vocal line',
        'rap line', 'dance line', 'trưởng nhóm', 'leader', 'maknae', 'visual', 'center'
    ]
    
    role_keywords_en = [
        'member', 'members', 'current members', 'former members', 'original members',
        'new members', 'lineup', 'line-up', 'line up', 'singer', 'singers',
        'artist', 'artists', 'rapper', 'rappers', 'idol', 'idols',
        'vocalist', 'vocalists', 'dancer', 'dancers', 'dance line', 'rap line',
        'vocal line', 'leader', 'leaders', 'maknae'
    ]
    
    # Các pattern liệt kê thành viên - sử dụng greedy match để lấy đủ danh sách
    member_list_patterns = [
        # === TIẾNG VIỆT (cố định) ===
        # "bao gồm X thành viên: A, B, C và D"
        r'(?:bao gồm|gồm có|gồm)\s+\d+\s+thành viên\s*[:\s]\s*([A-Za-z\-\'\.\s,và]+?)(?:\s*,\s*họ|\s*\.|$)',
        # "thành viên: A, B, C và D"
        r'thành viên\s*:\s*([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        # "X thành viên: list"
        r'\d+\s+thành viên\s*:\s*([A-Za-z\-\'\.\s,và]+?)(?:\s*,\s*họ|\s*\.|$)',
        # "các thành viên A, B, C và D"
        r'các\s+thành viên\s+([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        # "thành viên gồm A, B, C"
        r'thành viên\s+gồm\s+([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        # "thành viên là A, B, C"
        r'thành viên\s+là\s+([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        # "nhóm có X người: A, B, C"
        r'nhóm\s+(?:có|gồm)\s+\d+\s+(?:người|thành viên)\s*[:\s]\s*([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        # "một số thành viên, bao gồm A, B, C"
        r'(?:một số|nhiều|vài)\s+thành viên\s*,?\s*(?:bao gồm|gồm|như|là)\s+([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        # "các ca sĩ gồm A, B, C"
        r'các\s+ca sĩ\s+(?:gồm|bao gồm|như|là)\s+([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        # "các nghệ sĩ như A, B, C"
        r'các\s+nghệ sĩ\s+(?:như|gồm|bao gồm|là)\s+([A-Za-z\-\'\.\s,và]+?)(?:\s*\.|$)',
        
        # === TIẾNG ANH (cố định) ===
        # "consists of X members: A, B, C and D"
        r'consists?\s+of\s+\d+\s+members?\s*[:\s]\s*([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "members: A, B, C and D"
        r'members?\s*:\s*([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "X members: A, B, C"
        r'\d+\s+members?\s*:\s*([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "the members are A, B, C"
        r'(?:the\s+)?members?\s+(?:are|include|including)\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "some members, including A, B, C"
        r'(?:some|several|many|various)\s+members?\s*,?\s*(?:including|such as|like)\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "singers such as A, B, C"
        r'singers?\s+(?:such as|like|including|include)\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "artists including A, B, C"
        r'artists?\s+(?:including|such as|like)\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "comprising A, B, C"
        r'comprising\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "composed of A, B, C"
        r'composed\s+of\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "formed by A, B, C"
        r'formed\s+by\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "featuring A, B, C"
        r'featuring\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "with members A, B, C"
        r'with\s+members?\s+([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "lineup: A, B, C" hoặc "line-up: A, B, C"
        r'line[\-\s]?up\s*:\s*([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        # "current members: A, B, C"
        r'(?:current|original|former)\s+members?\s*:\s*([A-Za-z\-\'\.\s,and]+?)(?:\s*\.|$)',
        
        # === PATTERN CHUNG ===
        # "(A, B, C, D)" - danh sách trong ngoặc sau tên nhóm
        r'\(\s*([A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+){2,})\s*\)',
    ]
    
    # Dynamic patterns cho các từ khóa vai trò (tiếng Việt)
    connectors_vi = r'(?:bao gồm|gồm|gồm có|gồm cả|bao gồm cả|bao gồm những|bao gồm các|gồm những|gồm các|là|là những|là các)'
    for kw in role_keywords_vi:
        kw_pattern = re.escape(kw)
        member_list_patterns.append(
            rf'{kw_pattern}\s+{connectors_vi}\s*{name_list_pattern}(?:\s*\.|$)'
        )
        member_list_patterns.append(
            rf'{kw_pattern}\s*[:\-]\s*{name_list_pattern}(?:\s*\.|$)'
        )
    
    # Dynamic patterns cho từ khóa tiếng Anh
    connectors_en = r'(?:include|includes|including|consist of|consists of|consisting of|are|were|feature|featuring|with|comprise|comprises|comprised of)'
    for kw in role_keywords_en:
        kw_pattern = re.escape(kw)
        member_list_patterns.append(
            rf'(?:the\s+)?{kw_pattern}\s+{connectors_en}\s*{name_list_pattern}(?:\s*\.|$)'
        )
        member_list_patterns.append(
            rf'(?:the\s+)?{kw_pattern}\s*[:\-]\s*{name_list_pattern}(?:\s*\.|$)'
        )
    
    for pattern in member_list_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                member_list_text = match.group(1)
                if not member_list_text:
                    continue
                
                # Tách các thành viên bằng dấu phẩy hoặc "và"/"and"/"&"
                # Thay các từ nối bằng dấu phẩy để dễ tách
                member_list_text = re.sub(r'\s+và\s+', ', ', member_list_text, flags=re.IGNORECASE)
                member_list_text = re.sub(r'\s+and\s+', ', ', member_list_text, flags=re.IGNORECASE)
                member_list_text = re.sub(r'\s*&\s*', ', ', member_list_text)
                member_list_text = re.sub(r'\s*;\s*', ', ', member_list_text)  # Dấu chấm phẩy
                member_list_text = re.sub(r'\s*/\s*', ', ', member_list_text)  # Dấu gạch chéo
                
                # Tách bằng dấu phẩy
                members = [m.strip() for m in member_list_text.split(',')]
                
                for member in members:
                    member = clean_text(member)
                    
                    # Bỏ qua nếu quá ngắn hoặc quá dài
                    if not member or len(member) < 1 or len(member) > 30:
                        continue
                    
                    # Bỏ qua nếu chứa số (trừ khi là tên như "2PM")
                    if re.search(r'\d', member) and not re.match(r'^[0-9][A-Za-z]+', member):
                        continue
                    
                    # Bỏ qua nếu là từ chung chung
                    if member.lower() in INVALID_WORDS:
                        continue
                    
                    # Bỏ qua nếu đã tồn tại trong graph gốc (existing_lower)
                    # CHUẨN HÓA member trước khi check
                    # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
                    member_key = normalize_for_comparison(member)
                    # QUAN TRỌNG: Chỉ loại bỏ nếu cùng tên VÀ cùng type (Artist)
                    if member_key in existing_lower and 'Artist' in existing_lower[member_key]:
                        continue
                    
                    if member.lower() in seen:
                        continue
                    
                    # Kiểm tra tính hợp lệ - nhưng vẫn nới lỏng cho tên thành viên
                    lower_member = member.lower()
                    # Whitelist tên ngắn hợp lệ (trùng với is_valid_entity)
                    # Bổ sung thêm các tên ngắn hợp lệ từ infobox: DK, ZN, P.O, The8
                    valid_short_names = {'rm', 'iu', 'cl', 'bm', 'jb', 'jj', 'jo', 'im', 'do', 'dk', 'zn', 'p.o', 'the8'}
                    
                    if len(member) <= 2:
                        # Chỉ cho phép nếu là tên ngắn hợp lệ trong whitelist
                        if lower_member not in valid_short_names:
                            continue
                    elif len(member) == 3:
                        # Tên 3 ký tự: vẫn phải bắt đầu bằng chữ hoa và qua is_valid_entity
                        if not member[0].isupper() and not member.isupper():
                            continue
                        if not is_valid_entity(member, 'Artist'):
                            continue
                    else:
                        if not is_valid_entity(member, 'Artist'):
                            continue
                    
                    seen.add(member.lower())
                    entities.append({
                        'text': member,
                        'type': 'Artist',
                        'method': 'rule-based',
                        'confidence': 0.8  # Confidence cao vì được liệt kê rõ ràng trong context thành viên
                    })
        except Exception as e:
            continue
    
    return entities

def extract_groups_from_list(text):
    """Trích xuất nhóm nhạc từ các câu liệt kê như:
    - 'các nhóm nhạc chính gồm TVXQ, Super Junior, ...'
    - 'đã từng quản lý các nhóm nhạc H.O.T, S.E.S., Shinhwa, ...'
    """
    entities = []
    seen = set()
    
    # Cho phép cả chữ, số, dấu chấm, ngoặc, dấu gạch, &, /
    name_list_pattern = r'([A-Za-z0-9\-\'\.\s&/()]+?)'
    
    group_list_patterns = [
        # === TIẾNG VIỆT ===
        # "các nhóm nhạc chính gồm TVXQ, Super Junior, ..."
        r'(?:các|những)\s+nhóm nhạc(?:\s+\w+)*\s+(?:gồm|bao gồm|là|như)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "các nhóm nhạc gồm TVXQ, Super Junior, ..."
        r'các\s+nhóm nhạc\s+(?:gồm|bao gồm|như|là)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "quản lý các nhóm nhạc TVXQ, Super Junior, ..."
        r'(?:đã\s+)?(?:từng\s+)?quản lý\s+(?:các\s+)?nhóm nhạc\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "các nhóm nhạc TVXQ, Super Junior, ..."
        r'các\s+nhóm nhạc\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "các nhóm nhạc: TVXQ, Super Junior, ..." (có dấu :)
        r'các\s+nhóm nhạc\s*:\s*' + name_list_pattern + r'(?:[.;]|$)',
        # "một số nhóm nhạc, bao gồm A, B, C"
        r'(?:một số|nhiều|vài)\s+nhóm nhạc\s*,?\s*(?:bao gồm|gồm|như|là)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "nhóm nhạc bao gồm A, B, C"
        r'nhóm nhạc\s+(?:bao gồm|gồm|như|là)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "các nhóm như A, B, C"
        r'(?:các|những)\s+nhóm\s+(?:như|bao gồm|gồm)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "bao gồm các nhóm A, B, C"
        r'bao gồm\s+(?:các|những)?\s*nhóm(?:\s*nhạc)?\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "gồm các nhóm nhạc A, B, C"
        r'gồm\s+(?:các|những)?\s*nhóm(?:\s*nhạc)?\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "các nhóm nhạc nam/nữ A, B, C"
        r'(?:các|những)\s+nhóm(?:\s*nhạc)?(?:\s+nam|\s+nữ)?\s+(?:gồm|bao gồm|như|là)\s+' + name_list_pattern + r'(?:[.;]|$)',
        
        # === TIẾNG ANH ===
        # "groups such as A, B, C"
        r'(?:idol\s+)?groups?\s+(?:such as|like|including|include)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "some groups, including A, B, C"
        r'(?:some|several|many|various)\s+groups?\s*,?\s*(?:including|such as|like)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "groups including A, B, C"
        r'groups?\s+(?:including|such as|like)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "managed groups A, B, C"
        r'(?:managed|manages|managing)\s+(?:the\s+)?groups?\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "former/current/active groups include A, B, C"
        r'(?:former|current|active)\s+groups?\s+(?:include|including|such as|like)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "K-pop groups such as A, B, C"
        r'(?:k-?pop|korean)\s+groups?\s+(?:such as|like|including|include)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "boy groups A, B, C"
        r'(?:boy|girl)\s+groups?\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "groups: TVXQ, Super Junior, ..." (có dấu :)
        r'(?:idol\s+)?groups?\s*:\s*' + name_list_pattern + r'(?:[.;]|$)',
    ]
    
    for pattern in group_list_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                group_list_text = match.group(1)
                if not group_list_text:
                    continue
                
                # Chuẩn hóa nối: 'và' / 'and' / '&'
                group_list_text = re.sub(r'\s+và\s+', ', ', group_list_text, flags=re.IGNORECASE)
                group_list_text = re.sub(r'\s+and\s+', ', ', group_list_text, flags=re.IGNORECASE)
                group_list_text = re.sub(r'\s*&\s*', ', ', group_list_text)
                group_list_text = re.sub(r'\s*;\s*', ', ', group_list_text)
                
                groups = [g.strip() for g in group_list_text.split(',')]
                
                for grp in groups:
                    grp = clean_text(grp)
                    if not grp:
                        continue
                    
                    # Bỏ các mảnh câu kiểu 'và đã từng quản lý'
                    low = grp.lower()
                    if any(kw in low for kw in ['quản lý', 'từng quản', 'đã từng', 'đã quản']):
                        continue
                    
                    if len(grp) < 2 or len(grp) > 40:
                        continue
                    
                    # Bỏ qua nếu đã có trong graph hoặc đã thấy
                    # CHUẨN HÓA group name trước khi check
                    # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
                    group_key = normalize_for_comparison(grp)
                    # QUAN TRỌNG: Chỉ loại bỏ nếu cùng tên VÀ cùng type (Group)
                    if (group_key in existing_lower and 'Group' in existing_lower[group_key]) or group_key in seen:
                        continue
                    
                    # Phải qua kiểm tra group hợp lệ
                    if not is_valid_entity(grp, 'Group'):
                        continue
                    
                    seen.add(group_key)
                    entities.append({
                        'text': grp,
                        'type': 'Group',
                        'method': 'rule-based',
                        'confidence': 0.8,
                    })
        except Exception:
            continue
    
    return entities

def extract_companies_from_list(text):
    """Trích xuất công ty từ các câu liệt kê, ví dụ:
    - 'các công ty giải trí Hàn Quốc là YG Entertainment, Pledis Entertainment và Starship Entertainment'
    - 'người từng làm việc với các công ty như JYP Entertainment, Woollim Entertainment, Sony Music Korea và Blockberry Creative'
    - 'các công ty bao gồm Jin-ah Entertainment, Eru Entertainment và YMC Entertainment'
    """
    entities = []
    seen = set()
    
    # Cho phép cả chữ, số, dấu chấm, ngoặc, dấu gạch, &, /
    name_list_pattern = r'([A-Za-z0-9\-\'\.\s&/()]+?)'
    
    company_list_patterns = [
        # === TIẾNG VIỆT ===
        # "các công ty giải trí Hàn Quốc là YG Entertainment, Pledis Entertainment..."
        r'các\s+công ty(?:\s+giải trí)?(?:\s+[A-Za-zÀ-ỹ]+)*\s+(?:là|gồm|bao gồm|như)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "các công ty: YG Entertainment, ..."
        r'các\s+công ty(?:\s+giải trí)?\s*:\s*' + name_list_pattern + r'(?:[.;]|$)',
        # "các công ty như JYP Entertainment, Woollim Entertainment..."
        r'các\s+công ty(?:\s+giải trí)?\s+(?:như|bao gồm|gồm)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "công ty ... bao gồm Jin-ah Entertainment, Eru Entertainment..."
        r'công ty(?:\s+giải trí)?(?:\s+[A-Za-zÀ-ỹ]+)*\s+bao gồm\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "công ty ... như JYP Entertainment, ..."
        r'công ty(?:\s+giải trí)?(?:\s+[A-Za-zÀ-ỹ]+)*\s+như\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "các công ty bao gồm Jin-ah Entertainment, ... "
        r'các\s+công ty(?:\s+giải trí)?\s+bao gồm\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "người từng làm việc với các công ty như JYP Entertainment, ..."
        r'các\s+công ty\s+như\s+' + name_list_pattern + r'(?:[.;]|$)',
        
        # === TIẾNG ANH ===
        # "companies such as JYP Entertainment, Woollim Entertainment..."
        r'companies?\s+(?:such as|like|including|include)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "entertainment companies such as YG Entertainment, ..."
        r'(?:entertainment\s+companies?|record\s+labels?|agencies)\s+(?:such as|like|including|include)\s+' + name_list_pattern + r'(?:[.;]|$)',
        # "labels: YG Entertainment, JYP Entertainment, ..."
        r'(?:labels?|companies?)\s*:\s*' + name_list_pattern + r'(?:[.;]|$)',
    ]
    
    for pattern in company_list_patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                company_list_text = match.group(1)
                if not company_list_text:
                    continue
                
                # Chuẩn hóa nối: 'và' / 'and' / '&' / ';' / '/'
                company_list_text = re.sub(r'\s+và\s+', ', ', company_list_text, flags=re.IGNORECASE)
                company_list_text = re.sub(r'\s+and\s+', ', ', company_list_text, flags=re.IGNORECASE)
                company_list_text = re.sub(r'\s*&\s*', ', ', company_list_text)
                company_list_text = re.sub(r'\s*;\s*', ', ', company_list_text)
                company_list_text = re.sub(r'\s*/\s*', ', ', company_list_text)
                
                # Tách theo dấu phẩy
                companies = [c.strip() for c in company_list_text.split(',')]
                
                for comp in companies:
                    comp = clean_text(comp)
                    if not comp:
                        continue
                    
                    low = comp.lower()
                    
                    # Bỏ các mảnh câu còn sót động từ/mô tả
                    if any(kw in low for kw in ['người', 'từng', 'làm việc', 'hợp tác', 'cùng', 'với']):
                        continue
                    
                    if len(comp) < 3 or len(comp) > 60:
                        continue
                    
                    # Bỏ qua nếu đã có trong graph hoặc đã thấy
                    # CHUẨN HÓA company name trước khi check
                    # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
                    company_key = normalize_for_comparison(comp)
                    # QUAN TRỌNG: Chỉ loại bỏ nếu cùng tên VÀ cùng type (Company)
                    if (company_key in existing_lower and 'Company' in existing_lower[company_key]) or company_key in seen:
                        continue
                    
                    # Phải qua kiểm tra company hợp lệ
                    if not is_valid_entity(comp, 'Company'):
                        continue
                    
                    seen.add(normalized_company.lower())
                    entities.append({
                        'text': comp,
                        'type': 'Company',
                        'method': 'rule-based',
                        'confidence': 0.85,
                    })
        except Exception:
            continue
    
    return entities


def extract_artists_from_infobox_groups():
    """
    Tạo các node Artist mới từ infobox members của các Group gốc.
    - Dùng dữ liệu đã crawl trong 'infobox_members.json' (INFOBOX_MEMBERS['groups'])
    - Các trường sử dụng: 'Current members', 'Past members', 'Thành viên', 'Cựu thành viên', etc.
    - Không trùng với node gốc (existing_lower) và các node mới khác
    """
    entities = []
    seen = set()

    groups = INFOBOX_MEMBERS.get('groups') or {}
    if not isinstance(groups, dict):
        return entities

    # ĐỒNG BỘ HÓA VỚI RE: Sử dụng cùng member keys và logic parse
    member_keys = [
        # Current members (giống RE)
        'Current members',
        'Thành viên',
        'Thành viên hiện tại',
        'Members',
        # Past members (giống RE)
        'Past members',
        'Cựu thành viên',
        'Former members',
        # Bổ sung thêm
        'Thành viên cũ',
    ]

    for group_name, data in groups.items():
        info = data.get('infobox') or {}
        if not isinstance(info, dict):
            continue

        for key in member_keys:
            raw = info.get(key)
            if not raw:
                continue

            # ĐỒNG BỘ HÓA VỚI RE: Parse giống _parse_member_list()
            # Tách theo dấu phẩy, dấu *, dấu • (giống RE)
            parts = re.split(r'[,*•]', raw)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # ĐỒNG BỘ HÓA VỚI RE: Loại bỏ [1], [2], etc. và (notes) trước khi parse
                part = re.sub(r'\[.*?\]', '', part)  # Loại bỏ [1], [2], etc.
                part = re.sub(r'\(.*?\)', '', part)  # Loại bỏ (notes)
                part = part.strip()
                
                if not part:
                    continue
                
                # Kiểm tra độ dài cơ bản (giống RE)
                if len(part) < 2 or len(part) > 40:
                    continue
                
                # Kiểm tra bắt đầu bằng chữ (giống RE)
                if not re.match(r'^[A-Za-z\u3131-\u318E\u4E00-\u9FFF]', part):
                    continue
                
                # ĐỒNG BỘ HÓA VỚI RE: Loại bỏ các từ chung chung (không phải tên thành viên)
                # Giống RE: filter các từ như "Thành viên", "Danh sách", etc.
                GENERIC_TERMS = {
                    'thành viên', 'members', 'member', 'cựu thành viên', 'former members',
                    'past members', 'current members', 'thành viên hiện tại', 'thành viên cũ',
                    'danh sách', 'danh sách thành viên', 'danh sách cựu thành viên',
                    'list', 'list of members', 'list of former members',
                    'current', 'former', 'past', 'cựu'
                }
                part_lower = part.lower()
                if part_lower in GENERIC_TERMS:
                    continue
                # Loại bỏ nếu chứa cụm từ chung chung (chỉ check các từ dài hơn 3 ký tự)
                if any(term in part_lower for term in GENERIC_TERMS if len(term) > 3):
                    continue
                
                # ĐỒNG BỘ HÓA VỚI RE: KHÔNG dùng clean_text() vì RE không dùng
                # RE trả về part trực tiếp, NER cũng phải giữ tên gốc như vậy
                # CHỈ dùng normalize_for_comparison để check trùng lặp (không thay đổi tên gốc)
                member = part  # Giữ tên gốc như RE
                if not member:
                    continue

                # ============================================
                # CHỈ CHECK TRÙNG LẶP (KHÔNG CÓ BỘ LỌC KHÁC)
                # ============================================
                # Giống RE: đã filter từ chung chung ở trên, chỉ check trùng lặp
                # Vì đã được verify từ infobox Wikipedia
                
                # QUAN TRỌNG: Dùng CÙNG cách chuẩn hóa như khi tạo existing_lower
                # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
                member_key = normalize_for_comparison(member)
                
                # Check trùng với node gốc (cùng tên VÀ cùng type Artist)
                if member_key in existing_lower and 'Artist' in existing_lower[member_key]:
                    continue
                
                # Check trùng trong danh sách infobox đã thêm
                if member_key in seen:
                    continue

                # Thêm vào seen và entities
                seen.add(member_key)
                entities.append({
                    'text': member,
                    'type': 'Artist',
                    'method': 'infobox_members',
                    'confidence': 0.9,
                    'source_node': group_name,
                })

    return entities


def extract_known_companies(text):
    """Trích xuất công ty đã biết"""
    entities = []
    text_lower = text.lower()
    for company in KNOWN_COMPANIES:
        # CHUẨN HÓA company name trước khi check
        # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
        company_key = normalize_for_comparison(company)
        # QUAN TRỌNG: Chỉ thêm nếu chưa có trong graph (cùng tên VÀ cùng type Company)
        if company.lower() in text_lower and not (company_key in existing_lower and 'Company' in existing_lower[company_key]):
            entities.append({
                'text': company,
                'type': 'Company',
                'method': 'known_list',
                'confidence': 0.95
            })
    return entities

# =====================================================
# XỬ LÝ CHÍNH
# =====================================================
print("\n📊 Bước 1: Nhận dạng thực thể...")
all_entities = []  # Rule-based entities
ml_all_entities = []  # ML-based entities (riêng biệt)

# Trích xuất Artist mới từ infobox members của Group gốc (nếu có file)
infobox_artists = extract_artists_from_infobox_groups()
if infobox_artists:
    print(f"  ✓ Trích xuất {len(infobox_artists)} artist từ infobox members (file infobox_members.json)")
    all_entities.extend(infobox_artists)

for i, record in enumerate(records, 1):
    if i % 200 == 0:
        print(f"  Đã xử lý: {i}/{len(records)} records...")
    
    text = record.get('text', '')
    node_id = record.get('node_id', '')
    
    # Trích xuất theo từng loại (RULE-BASED)
    for entity_type, pattern_list in patterns.items():
        found = extract_entities(text, entity_type, pattern_list)
        for ent in found:
            ent['source_node'] = node_id
            all_entities.append(ent)
    
    # Công ty đã biết
    for ent in extract_known_companies(text):
        ent['source_node'] = node_id
        all_entities.append(ent)
    
    # Trích xuất công ty từ các câu liệt kê
    for ent in extract_companies_from_list(text):
        ent['source_node'] = node_id
        all_entities.append(ent)
    
    # Trích xuất thành viên từ danh sách liệt kê
    for ent in extract_members_from_list(text):
        ent['source_node'] = node_id
        all_entities.append(ent)
    
    # Trích xuất nhóm nhạc từ các câu liệt kê nhóm
    for ent in extract_groups_from_list(text):
        ent['source_node'] = node_id
        all_entities.append(ent)
    
    # Trích xuất bằng ML model (ML-BASED) - LƯU RIÊNG
    if ML_NER_AVAILABLE:
        try:
            ml_entities = extract_ml_entities(text, node_id)
            if ml_entities:
                for ent in ml_entities:
                    # ÁP DỤNG is_valid_entity CHO ML ENTITIES (giống rule-based)
                    entity_text = ent.get('text', '')
                    entity_type = ent.get('type', '')
                    if not is_valid_entity(entity_text, entity_type):
                        # Bỏ qua entity không hợp lệ
                        continue
                    
                    # CHECK TRÙNG VỚI GRAPH GỐC (giống rule-based)
                    # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
                    entity_key = normalize_for_comparison(entity_text)
                    # QUAN TRỌNG: Chỉ loại bỏ nếu cùng tên VÀ cùng type
                    if entity_key in existing_lower and entity_type in existing_lower[entity_key]:
                        # Entity đã tồn tại trong graph gốc (cùng tên và type) -> bỏ qua
                        continue
                    
                    # KHÔNG CHECK TRÙNG VỚI RULE-BASED (sẽ lưu riêng)
                    ml_all_entities.append(ent)
        except Exception as e:
            # Chỉ in lỗi nếu debug (để tránh spam)
            # Nếu có lỗi, bỏ qua và tiếp tục với rule-based
            if i <= 5:  # Chỉ in lỗi cho 5 records đầu để debug
                print(f"  ⚠️  Lỗi ML NER ở record {i}: {type(e).__name__}")
            pass

rule_based_count = len(all_entities)
print(f"  ✓ Nhận dạng được {rule_based_count} entities thô (rule-based)")
if ML_NER_AVAILABLE:
    ml_count = len(ml_all_entities)
    print(f"  ✓ Nhận dạng được {ml_count} entities thô (ML-based)")

# =====================================================
# GỘP VÀ LOẠI BỎ TRÙNG LẶP (RULE-BASED)
# =====================================================
print("\n📊 Bước 2a: Gộp và loại bỏ trùng lặp (Rule-based)...")
unique_rule = {}

for ent in all_entities:
    # QUAN TRỌNG: Entities từ infobox KHÔNG được clean_text() vì phải giữ tên gốc như RE
    # Chỉ clean_text() cho các entities từ rule-based (text extraction)
    if ent.get('method') == 'infobox_members':
        # Giữ nguyên tên gốc, không clean_text()
        normalized_text = ent['text']
    else:
        # Chuẩn hóa text để tránh trùng do khác khoảng trắng / hoa thường
        normalized_text = clean_text(ent['text'])
    
    ent['text'] = normalized_text
    
    # Tạo key để gộp: Dùng normalize_for_comparison() để match các biến thể
    # (ví dụ: "Kwon Jung-yeol" vs "Kwon Jung yeol" -> cùng một node)
    # QUAN TRỌNG: Dùng cùng cách normalize như khi check với existing_lower
    merge_key = normalize_for_comparison(normalized_text)
    key = (merge_key, ent['type'])
    
    if key not in unique_rule:
        unique_rule[key] = {**ent, 'sources': [ent.get('source_node', '')]}
    else:
        # ƯU TIÊN NODE TỪ INFOBOX: Nếu trùng, giữ node từ infobox, loại node còn lại
        existing = unique_rule[key]
        existing_is_infobox = existing.get('method') == 'infobox_members'
        new_is_infobox = ent.get('method') == 'infobox_members'
        
        if new_is_infobox and not existing_is_infobox:
            # Node mới từ infobox, node existing không phải -> thay thế
            unique_rule[key] = {**ent, 'sources': [ent.get('source_node', '')]}
        elif existing_is_infobox and not new_is_infobox:
            # Node existing từ infobox, node mới không phải -> bỏ qua node mới (giữ existing)
            continue
        else:
            # Cả hai đều từ infobox hoặc cả hai đều không -> merge như bình thường
            source_node = ent.get('source_node', '')
            if source_node and source_node not in existing.get('sources', []):
                existing['sources'].append(source_node)
            # Giữ confidence cao nhất
            existing['confidence'] = max(existing.get('confidence', 0), ent.get('confidence', 0))

merged_rule_entities = list(unique_rule.values())
print(f"  ✓ Còn {len(merged_rule_entities)} entities (rule-based) sau khi gộp")

# =====================================================
# GỘP VÀ LOẠI BỎ TRÙNG LẶP (ML-BASED)
# =====================================================
merged_ml_entities = []
if ML_NER_AVAILABLE and ml_all_entities:
    print("\n📊 Bước 2b: Gộp và loại bỏ trùng lặp (ML-based)...")
    unique_ml = {}
    
    for ent in ml_all_entities:
        # Chuẩn hóa text để tránh trùng do khác khoảng trắng / hoa thường
        normalized_text = clean_text(ent['text'])
        ent['text'] = normalized_text
        
        # Tạo key để gộp: Dùng normalize_for_comparison() để match các biến thể
        # QUAN TRỌNG: Dùng cùng cách normalize như khi check với existing_lower
        merge_key = normalize_for_comparison(normalized_text)
        key = (merge_key, ent['type'])
        
        if key not in unique_ml:
            unique_ml[key] = {**ent, 'sources': [ent.get('source_node', '')]}
        else:
            # Gộp sources - chỉ merge nếu text hoàn toàn giống nhau (sau normalize)
            existing = unique_ml[key]
            source_node = ent.get('source_node', '')
            if source_node and source_node not in existing.get('sources', []):
                existing['sources'].append(source_node)
            # Giữ confidence cao nhất
            existing['confidence'] = max(existing.get('confidence', 0), ent.get('confidence', 0))
    
    merged_ml_entities = list(unique_ml.values())
    print(f"  ✓ Còn {len(merged_ml_entities)} entities (ML-based) sau khi gộp")

# =====================================================
# HÀM FIX TYPE SAI
# =====================================================
def fix_entity_type(entity_text, entity_type, sources):
    """
    Sửa type sai cho entity.
    Ví dụ: "Lee Su ji" bị nhầm là Group nhưng thực ra là Artist.
    """
    text_lower = entity_text.lower().strip()
    
    # ============================================
    # FIX: TÊN NGƯỜI BỊ NHẦM LÀ GROUP
    # ============================================
    if entity_type == 'Group':
        # Pattern tên người Hàn Quốc: "Họ Tên" (2-3 từ, bắt đầu bằng họ Hàn)
        korean_surnames_lower = {s.lower() for s in KOREAN_SURNAMES}
        words = entity_text.split()
        
        # Nếu có 2-3 từ và từ đầu là họ Hàn Quốc -> có thể là tên người
        if 2 <= len(words) <= 3:
            first_word = words[0].strip()
            if first_word.lower() in korean_surnames_lower:
                # Kiểm tra context: nếu có từ khóa nghệ sĩ/ca sĩ -> chuyển thành Artist
                for source in sources:
                    full_text = node_texts.get(source, '')
                    if full_text:
                        # Tìm vị trí entity trong text
                        idx = full_text.find(text_lower)
                        if idx != -1:
                            # Lấy context xung quanh (200 ký tự)
                            start = max(0, idx - 100)
                            end = min(len(full_text), idx + len(entity_text) + 100)
                            context = full_text[start:end].lower()
                            
                            # Nếu có từ khóa nghệ sĩ/ca sĩ/thành viên -> là Artist
                            artist_keywords = ['ca sĩ', 'nghệ sĩ', 'thành viên', 'singer', 'artist', 'member', 'idol']
                            if any(kw in context for kw in artist_keywords):
                                return 'Artist'
                            
                            # Nếu có từ khóa nhóm nhạc/group -> giữ Group
                            group_keywords = ['nhóm nhạc', 'ban nhạc', 'group', 'band']
                            if any(kw in context for kw in group_keywords):
                                return 'Group'
        
        # Danh sách tên người đã biết bị nhầm là Group
        known_artist_names = {
            'lee su ji', 'lee su-ji', 'leesuji',  # Ví dụ từ user
            # Có thể thêm các tên khác nếu phát hiện
        }
        if text_lower in known_artist_names:
            return 'Artist'
    
    return entity_type

# =====================================================
# LỌC THEO CONTEXT K-POP VÀ PHÙ HỢP VỚI MẠNG LƯỚI (RULE-BASED)
# =====================================================
print("\n📊 Bước 3a: Lọc theo context K-pop và phù hợp mạng lưới (Rule-based)...")
filtered_rule_entities = []
removed_count_rule = defaultdict(int)
removed_reason_rule = defaultdict(lambda: defaultdict(int))
fixed_type_count = defaultdict(int)

for ent in merged_rule_entities:
    sources = ent.get('sources', [ent.get('source_node', '')])
    entity_type = ent['type']
    entity_text = ent['text']
    
    # ============================================
    # FIX TYPE SAI TRƯỚC KHI FILTER
    # ============================================
    original_type = entity_type
    entity_type = fix_entity_type(entity_text, entity_type, sources)
    if entity_type != original_type:
        ent['type'] = entity_type
        fixed_type_count[f"{original_type}->{entity_type}"] += 1
        print(f"   🔧 Fixed type: '{entity_text}' {original_type} -> {entity_type}")
    
    # Safety filter bổ sung cho Group để loại bỏ các mảnh tên sai còn sót như "Indie OKDAL Y"
    if entity_type == 'Group':
        low = entity_text.lower()
        if any(kw in low for kw in ['indie okdal', 'f ve', 'girl next door', 'girl next']):
            removed_count_rule[entity_type] += 1
            removed_reason_rule[entity_type]['post_filter_bad_group'] += 1
            continue
        
        # Cải thiện: Validation với danh sách nhóm K-pop đã biết
        # Nếu không có trong danh sách nhóm đã biết và không có trong graph -> cần kiểm tra kỹ hơn
        entity_normalized = normalize_for_comparison(entity_text)
        is_known_group = (
            low in KNOWN_KPOP_GROUPS or
            entity_normalized in existing_lower and 'Group' in existing_lower.get(entity_normalized, set())
        )
        
        # Nếu không phải nhóm đã biết và không có trong graph -> giảm confidence
        if not is_known_group:
            # Kiểm tra xem có phải nhóm thật không bằng cách xem context
            # Nếu có từ khóa "nhóm nhạc", "group", "band" trong context -> có thể là nhóm thật
            has_group_context = False
            for source in sources:
                full_text = node_texts.get(source, '')
                if full_text:
                    full_text_lower = full_text.lower()
                    group_keywords = ['nhóm nhạc', 'ban nhạc', 'group', 'band', 'idol group']
                    if any(kw in full_text_lower for kw in group_keywords):
                        has_group_context = True
                        break
            
            # Nếu không có context về nhóm -> giảm confidence đáng kể
            if not has_group_context:
                ent['confidence'] = max(0.5, ent.get('confidence', 0.7) - 0.2)
    
    # Known list (công ty đã biết) -> luôn giữ
    if ent.get('method') == 'known_list':
        filtered_rule_entities.append(ent)
        continue
    
    # QUAN TRỌNG: Artists từ infobox -> luôn giữ (vì đã được verify từ infobox của group)
    # Không cần check context vì họ là thành viên của group đã có trong graph
    if ent.get('method') == 'infobox_members':
        # Kiểm tra xem source_node (group name) có trong graph không
        source_node = ent.get('source_node', '')
        if source_node:
            # Nếu group có trong graph -> artist hợp lệ
            # Dùng normalize_for_comparison để đảm bảo cùng cách chuẩn hóa với merge_and_import_neo4j.py
            source_normalized = normalize_for_comparison(source_node)
            # existing_lower là Dict[str, Set[str]] - check xem có Group type không
            if source_normalized in existing_lower and 'Group' in existing_lower[source_normalized]:
                filtered_rule_entities.append(ent)
                continue
            # Hoặc nếu group có trong danh sách groups từ infobox -> cũng hợp lệ
            groups_infobox = INFOBOX_MEMBERS.get('groups', {})
            if source_node in groups_infobox or any(g.lower() == source_node.lower() for g in groups_infobox.keys()):
                filtered_rule_entities.append(ent)
                continue
        
        # Nếu không có source_node, vẫn giữ (có thể là từ infobox nhưng không có group name)
        # Vì đã được verify từ infobox nên đáng tin cậy
        filtered_rule_entities.append(ent)
        continue
    
    # Kiểm tra 1: Phải có context K-pop
    if not has_kpop_context(sources):
        removed_count_rule[entity_type] += 1
        removed_reason_rule[entity_type]['no_kpop_context'] += 1
        continue
    
    # Kiểm tra 2a: Nếu entity được nhận dạng là Artist nhưng có "album thành viên" trong context -> loại bỏ (vì là album)
    if entity_type == 'Artist':
        is_album_context = False
        for source in sources:
            full_text = node_texts.get(source, '')
            if full_text and ('album thành viên' in full_text or 'album của thành viên' in full_text):
                entity_lower = entity_text.lower()
                idx = full_text.find(entity_lower)
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(full_text), idx + len(entity_text) + 50)
                    context = full_text[start:end]
                    if 'album' in context and 'thành viên' in context:
                        is_album_context = True
                        break
        if is_album_context:
            removed_count_rule[entity_type] += 1
            removed_reason_rule[entity_type]['is_album_not_artist'] += 1
            continue
    
    # Kiểm tra 2b: Artist phải là nghệ sĩ âm nhạc (không phải diễn viên, MC...)
    if entity_type == 'Artist':
        if not is_music_artist(entity_text, sources):
            removed_count_rule[entity_type] += 1
            removed_reason_rule[entity_type]['not_music_artist'] += 1
            continue
    
    # Kiểm tra 3: Phải liên quan đến mạng lưới hiện có
    if not is_related_to_existing_nodes(entity_text, sources, existing_lower):
        removed_count_rule[entity_type] += 1
        removed_reason_rule[entity_type]['not_related_to_network'] += 1
        continue
    
    # Cải thiện: Tính confidence dựa trên nhiều yếu tố
    num_sources = len(set(sources))
    base_confidence = ent.get('confidence', 0.7)
    
    # Boost từ số nguồn
    if num_sources >= 5:
        base_confidence = min(0.95, base_confidence + 0.2)
    elif num_sources >= 3:
        base_confidence = min(0.9, base_confidence + 0.15)
    elif num_sources >= 2:
        base_confidence = min(0.85, base_confidence + 0.1)
    
    # Boost từ việc có trong danh sách đã biết
    entity_normalized = normalize_for_comparison(entity_text)
    if entity_type == 'Group' and entity_text.lower() in KNOWN_KPOP_GROUPS:
        base_confidence = min(0.95, base_confidence + 0.1)
    elif entity_type == 'Company' and entity_text in KNOWN_COMPANIES:
        base_confidence = min(0.95, base_confidence + 0.1)
    elif entity_type == 'Artist':
        # Kiểm tra có trong graph không
        if entity_normalized in existing_lower and 'Artist' in existing_lower.get(entity_normalized, set()):
            base_confidence = min(0.9, base_confidence + 0.1)
    
    # Boost từ context quality (số từ khóa K-pop)
    kpop_score = 0
    for source in sources:
        text = node_texts.get(source, '')
        if text:
            text_lower = text.lower()
            kpop_score += sum(1 for kw in KPOP_KEYWORDS if kw.lower() in text_lower)
    
    if kpop_score >= 5:
        base_confidence = min(0.95, base_confidence + 0.05)
    elif kpop_score >= 3:
        base_confidence = min(0.9, base_confidence + 0.03)
    
    ent['confidence'] = base_confidence
    filtered_rule_entities.append(ent)

# =====================================================
# LỌC THEO CONTEXT K-POP VÀ PHÙ HỢP VỚI MẠNG LƯỚI (ML-BASED)
# =====================================================
filtered_ml_entities = []
removed_count_ml = defaultdict(int)
removed_reason_ml = defaultdict(lambda: defaultdict(int))
fixed_type_count_ml = defaultdict(int)

if ML_NER_AVAILABLE and merged_ml_entities:
    print("\n📊 Bước 3b: Lọc theo context K-pop và phù hợp mạng lưới (ML-based)...")
    
    for ent in merged_ml_entities:
        sources = ent.get('sources', [ent.get('source_node', '')])
        entity_type = ent['type']
        entity_text = ent['text']
        
        # ============================================
        # FIX TYPE SAI TRƯỚC KHI FILTER (giống rule-based)
        # ============================================
        original_type = entity_type
        entity_type = fix_entity_type(entity_text, entity_type, sources)
        if entity_type != original_type:
            ent['type'] = entity_type
            fixed_type_count_ml[f"{original_type}->{entity_type}"] += 1
            print(f"   🔧 Fixed type (ML): '{entity_text}' {original_type} -> {entity_type}")
        
        # Kiểm tra 1: Phải có context K-pop
        if not has_kpop_context(sources):
            removed_count_ml[entity_type] += 1
            removed_reason_ml[entity_type]['no_kpop_context'] += 1
            continue
        
        # Kiểm tra 2: Artist phải là nghệ sĩ âm nhạc
        if entity_type == 'Artist':
            if not is_music_artist(entity_text, sources):
                removed_count_ml[entity_type] += 1
                removed_reason_ml[entity_type]['not_music_artist'] += 1
                continue
        
        # Kiểm tra 3: Phải liên quan đến mạng lưới hiện có
        if not is_related_to_existing_nodes(entity_text, sources, existing_lower):
            removed_count_ml[entity_type] += 1
            removed_reason_ml[entity_type]['not_related_to_network'] += 1
            continue
        
        # Kiểm tra 4: Loại bỏ entities có confidence quá thấp (< 0.65)
        if ent.get('confidence', 0) < 0.65:
            removed_count_ml[entity_type] += 1
            removed_reason_ml[entity_type]['ml_low_confidence'] += 1
            continue
        
        # Cải thiện: Tính confidence dựa trên nhiều yếu tố (giống rule-based)
        num_sources = len(set(sources))
        base_confidence = ent.get('confidence', 0.65)
        
        # Boost từ số nguồn
        if num_sources >= 5:
            base_confidence = min(0.95, base_confidence + 0.2)
        elif num_sources >= 3:
            base_confidence = min(0.9, base_confidence + 0.15)
        elif num_sources >= 2:
            base_confidence = min(0.85, base_confidence + 0.1)
        
        # Boost từ việc có trong danh sách đã biết
        entity_normalized = normalize_for_comparison(entity_text)
        if entity_type == 'Group' and entity_text.lower() in KNOWN_KPOP_GROUPS:
            base_confidence = min(0.95, base_confidence + 0.1)
        elif entity_type == 'Company' and entity_text in KNOWN_COMPANIES:
            base_confidence = min(0.95, base_confidence + 0.1)
        elif entity_type == 'Artist':
            # Kiểm tra có trong graph không
            if entity_normalized in existing_lower and 'Artist' in existing_lower.get(entity_normalized, set()):
                base_confidence = min(0.9, base_confidence + 0.1)
        
        # Boost từ context quality (số từ khóa K-pop)
        kpop_score = 0
        for source in sources:
            text = node_texts.get(source, '')
            if text:
                text_lower = text.lower()
                kpop_score += sum(1 for kw in KPOP_KEYWORDS if kw.lower() in text_lower)
        
        if kpop_score >= 5:
            base_confidence = min(0.95, base_confidence + 0.05)
        elif kpop_score >= 3:
            base_confidence = min(0.9, base_confidence + 0.03)
        
        ent['confidence'] = base_confidence
        filtered_ml_entities.append(ent)

# =====================================================
# BƯỚC 4: CHUẨN HÓA & GỘP LẠI LẦN CUỐI (RULE-BASED)
# =====================================================
final_unique_rule = {}
for ent in filtered_rule_entities:
    # QUAN TRỌNG: Entities từ infobox KHÔNG được clean_text() vì phải giữ tên gốc như RE
    if ent.get('method') == 'infobox_members':
        norm_text = ent['text']  # Giữ nguyên tên gốc
    else:
        norm_text = clean_text(ent['text'])
    ent['text'] = norm_text
    
    # Dùng normalize_for_comparison() để match các biến thể
    merge_key = normalize_for_comparison(norm_text)
    key = (merge_key, ent['type'])
    
    if key not in final_unique_rule:
        final_unique_rule[key] = {**ent}
    else:
        # ƯU TIÊN NODE TỪ INFOBOX: Nếu trùng, giữ node từ infobox, loại node còn lại
        existing = final_unique_rule[key]
        existing_is_infobox = existing.get('method') == 'infobox_members'
        new_is_infobox = ent.get('method') == 'infobox_members'
        
        if new_is_infobox and not existing_is_infobox:
            # Node mới từ infobox, node existing không phải -> thay thế
            final_unique_rule[key] = {**ent}
        elif existing_is_infobox and not new_is_infobox:
            # Node existing từ infobox, node mới không phải -> bỏ qua node mới (giữ existing)
            continue
        else:
            # Cả hai đều từ infobox hoặc cả hai đều không -> merge như bình thường
            existing_sources = set(existing.get('sources', []))
            new_sources = set(ent.get('sources', []))
            existing['sources'] = list(existing_sources | new_sources)
            existing['confidence'] = max(existing.get('confidence', 0), ent.get('confidence', 0))

filtered_rule_entities = list(final_unique_rule.values())
print(f"  ✓ Còn {len(filtered_rule_entities)} entities (rule-based) sau khi lọc")

# =====================================================
# BƯỚC 4: CHUẨN HÓA & GỘP LẠI LẦN CUỐI (ML-BASED)
# =====================================================
final_unique_ml = {}
for ent in filtered_ml_entities:
    norm_text = clean_text(ent['text'])
    ent['text'] = norm_text
    
    # Dùng normalize_for_comparison() để match các biến thể
    merge_key = normalize_for_comparison(norm_text)
    key = (merge_key, ent['type'])
    
    if key not in final_unique_ml:
        final_unique_ml[key] = {**ent}
    else:
        existing = final_unique_ml[key]
        existing_sources = set(existing.get('sources', []))
        new_sources = set(ent.get('sources', []))
        existing['sources'] = list(existing_sources | new_sources)
        existing['confidence'] = max(existing.get('confidence', 0), ent.get('confidence', 0))

filtered_ml_entities = list(final_unique_ml.values())
if ML_NER_AVAILABLE:
    print(f"  ✓ Còn {len(filtered_ml_entities)} entities (ML-based) sau khi lọc")

# =====================================================
# BƯỚC 5: LỚP LỌC CUỐI CÙNG BẰNG SLM (SMALL LANGUAGE MODEL)
# =====================================================
print("\n🤖 Bước 5: Lọc cuối cùng bằng Small Language Model (SLM)...")

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

def validate_entity_with_slm(entity_text, entity_type, sources, slm):
    """
    Validate entity bằng Small Language Model
    
    Args:
        entity_text: Tên entity
        entity_type: Loại entity (Artist, Group, Company, Album, Song)
        sources: Danh sách source nodes
        slm: SmallLLM instance
    
    Returns:
        Tuple (is_valid, confidence_adjustment, reason)
    """
    if not slm:
        return True, 0.0, "SLM không khả dụng"
    
    # Lấy context từ sources
    context_parts = []
    for source in sources[:3]:  # Chỉ lấy 3 sources đầu để tránh context quá dài
        text = node_texts.get(source, '')
        if text:
            # Lấy đoạn text xung quanh entity (200 ký tự)
            entity_lower = entity_text.lower()
            idx = text.lower().find(entity_lower)
            if idx != -1:
                start = max(0, idx - 100)
                end = min(len(text), idx + len(entity_text) + 100)
                context_parts.append(text[start:end])
            else:
                context_parts.append(text[:200])  # Lấy 200 ký tự đầu
    
    context = "\n".join(context_parts[:2])  # Chỉ lấy 2 đoạn context đầu
    
    # Tạo prompt validation
    validation_prompt = f"""Bạn là chuyên gia về K-pop (nhạc Hàn Quốc). Hãy đánh giá xem thực thể sau có hợp lệ không.

THỰC THỂ CẦN KIỂM TRA:
- Tên: "{entity_text}"
- Loại: {entity_type}
- Context: {context[:500]}

YÊU CẦU:
1. Nếu là {entity_type} trong K-pop (nhạc Hàn Quốc) -> trả lời "VALID"
2. Nếu KHÔNG phải {entity_type} trong K-pop (ví dụ: nghệ sĩ nước ngoài, chương trình TV, địa danh, câu văn...) -> trả lời "INVALID"
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
        # Nếu SLM lỗi, giữ nguyên entity (fallback)
        return True, 0.0, f"SLM error: {str(e)[:50]}"

# Áp dụng SLM validation cho rule-based entities
if SLM_AVAILABLE and slm_validator:
    print("  🔍 Đang validate rule-based entities bằng SLM...")
    slm_validated_rule = []
    slm_removed_rule = 0
    slm_adjusted_rule = 0
    
    # Chỉ validate các entities có confidence >= 0.7 (để tiết kiệm thời gian)
    entities_to_validate = [e for e in filtered_rule_entities if e.get('confidence', 0) >= 0.7]
    print(f"    Validating {len(entities_to_validate)} entities (confidence >= 0.7)...")
    
    for i, ent in enumerate(entities_to_validate, 1):
        if i % 50 == 0:
            print(f"    Đã validate: {i}/{len(entities_to_validate)}...")
        
        is_valid, conf_adj, reason = validate_entity_with_slm(
            ent['text'],
            ent['type'],
            ent.get('sources', [ent.get('source_node', '')]),
            slm_validator
        )
        
        if is_valid:
            # Điều chỉnh confidence
            new_confidence = max(0.0, min(1.0, ent.get('confidence', 0.7) + conf_adj))
            ent['confidence'] = new_confidence
            if conf_adj != 0:
                slm_adjusted_rule += 1
            slm_validated_rule.append(ent)
        else:
            slm_removed_rule += 1
            if i <= 10:  # In 10 entities đầu bị loại
                print(f"      ❌ Removed: '{ent['text']}' ({ent['type']}) - {reason}")
    
    # Giữ lại các entities có confidence < 0.7 (không validate bằng SLM)
    low_conf_entities = [e for e in filtered_rule_entities if e.get('confidence', 0) < 0.7]
    filtered_rule_entities = slm_validated_rule + low_conf_entities
    
    print(f"  ✓ SLM validation: Giữ {len(slm_validated_rule)} entities, loại {slm_removed_rule} entities")
    print(f"  ✓ Điều chỉnh confidence cho {slm_adjusted_rule} entities")
else:
    print("  ⚠️  Bỏ qua SLM validation (SLM không khả dụng)")

# Áp dụng SLM validation cho ML-based entities
if ML_NER_AVAILABLE and SLM_AVAILABLE and slm_validator:
    print("  🔍 Đang validate ML-based entities bằng SLM...")
    slm_validated_ml = []
    slm_removed_ml = 0
    slm_adjusted_ml = 0
    
    # Chỉ validate các entities có confidence >= 0.7
    entities_to_validate_ml = [e for e in filtered_ml_entities if e.get('confidence', 0) >= 0.7]
    print(f"    Validating {len(entities_to_validate_ml)} entities (confidence >= 0.7)...")
    
    for i, ent in enumerate(entities_to_validate_ml, 1):
        if i % 50 == 0:
            print(f"    Đã validate: {i}/{len(entities_to_validate_ml)}...")
        
        is_valid, conf_adj, reason = validate_entity_with_slm(
            ent['text'],
            ent['type'],
            ent.get('sources', [ent.get('source_node', '')]),
            slm_validator
        )
        
        if is_valid:
            new_confidence = max(0.0, min(1.0, ent.get('confidence', 0.65) + conf_adj))
            ent['confidence'] = new_confidence
            if conf_adj != 0:
                slm_adjusted_ml += 1
            slm_validated_ml.append(ent)
        else:
            slm_removed_ml += 1
            if i <= 10:
                print(f"      ❌ Removed: '{ent['text']}' ({ent['type']}) - {reason}")
    
    low_conf_entities_ml = [e for e in filtered_ml_entities if e.get('confidence', 0) < 0.7]
    filtered_ml_entities = slm_validated_ml + low_conf_entities_ml
    
    print(f"  ✓ SLM validation: Giữ {len(slm_validated_ml)} entities, loại {slm_removed_ml} entities")
    print(f"  ✓ Điều chỉnh confidence cho {slm_adjusted_ml} entities")

# Sắp xếp theo confidence giảm dần (sau khi SLM validation)
filtered_rule_entities.sort(key=lambda x: (-x['confidence'], x['type'], x['text']))
if ML_NER_AVAILABLE:
    filtered_ml_entities.sort(key=lambda x: (-x['confidence'], x['type'], x['text']))

# Đếm theo type
counts_rule = defaultdict(int)
for ent in filtered_rule_entities:
    counts_rule[ent['type']] += 1

counts_ml = defaultdict(int)
if ML_NER_AVAILABLE:
    for ent in filtered_ml_entities:
        counts_ml[ent['type']] += 1

# =====================================================
# LƯU KẾT QUẢ (RULE-BASED)
# =====================================================
output_rule = {
    'metadata': {
        'description': 'Thực thể K-pop được nhận dạng và lọc (Rule-based)',
        'processed_at': datetime.now().isoformat(),
        'total_records': len(records),
        'raw_entities': len(all_entities),
        'merged_entities': len(merged_rule_entities),
        'final_entities': len(filtered_rule_entities),
        'entities_by_type': dict(counts_rule),
        'filter_criteria': [
            'Phải có context K-pop (>=3 từ khóa K-pop trong văn bản nguồn)',
            'Artist: Phải có từ khóa vai trò âm nhạc (ca sĩ, rapper, thành viên...)',
            'Artist: Loại trừ diễn viên, MC, vận động viên, nhà văn...',
            'Phải liên quan đến ít nhất 1 node hiện có trong mạng lưới',
            'Tên phải bắt đầu bằng chữ in hoa hoặc số',
            'Không chứa từ chung chung'
        ]
    },
    'entities': filtered_rule_entities
}

with open('data/kpop_ner_result.json', 'w', encoding='utf-8') as f:
    json.dump(output_rule, f, ensure_ascii=False, indent=2)

# =====================================================
# LƯU KẾT QUẢ (ML-BASED)
# =====================================================
if ML_NER_AVAILABLE:
    output_ml = {
        'metadata': {
            'description': 'Thực thể K-pop được nhận dạng và lọc (ML-based)',
            'processed_at': datetime.now().isoformat(),
            'total_records': len(records),
            'raw_entities': len(ml_all_entities),
            'merged_entities': len(merged_ml_entities),
            'final_entities': len(filtered_ml_entities),
            'entities_by_type': dict(counts_ml),
            'ml_model': 'NlpHUST/ner-vietnamese-electra-base',
            'filter_criteria': [
                'Phải có context K-pop (>=3 từ khóa K-pop trong văn bản nguồn)',
                'Artist: Phải có từ khóa vai trò âm nhạc (ca sĩ, rapper, thành viên...)',
                'Artist: Loại trừ diễn viên, MC, vận động viên, nhà văn...',
                'Phải liên quan đến ít nhất 1 node hiện có trong mạng lưới',
                'Confidence >= 0.65',
                'Tên phải bắt đầu bằng chữ in hoa hoặc số',
                'Không chứa từ chung chung'
            ]
        },
        'entities': filtered_ml_entities
    }
    
    with open('data/kpop_ner_ml_result.json', 'w', encoding='utf-8') as f:
        json.dump(output_ml, f, ensure_ascii=False, indent=2)

# =====================================================
# IN KẾT QUẢ
# =====================================================
print("\n" + "=" * 70)
print("KẾT QUẢ NHẬN DẠNG THỰC THỂ K-POP")
print("=" * 70)
print(f"✓ Đã lưu: kpop_ner_result.json (Rule-based)")
if ML_NER_AVAILABLE:
    print(f"✓ Đã lưu: kpop_ner_ml_result.json (ML-based)")

print(f"\n📊 THỐNG KÊ RULE-BASED:")
print(f"   Records xử lý: {len(records)}")
print(f"   Entities thô: {len(all_entities)}")
print(f"   Sau khi gộp: {len(merged_rule_entities)}")
print(f"   Sau khi lọc K-pop: {len(filtered_rule_entities)}")

print(f"\n   Phân loại cuối cùng (Rule-based):")
for t in ['Company', 'Group', 'Artist', 'Album', 'Song']:
    print(f"     - {t}: {counts_rule.get(t, 0)}")

print(f"\n   Số entities đã sửa type (Rule-based):")
if fixed_type_count:
    for fix_type, count in fixed_type_count.items():
        print(f"     - {fix_type}: {count}")
else:
    print(f"     - Không có")

print(f"\n   Số entities bị loại (Rule-based):")
for t in ['Company', 'Group', 'Artist', 'Album', 'Song']:
    total_removed = removed_count_rule.get(t, 0)
    if total_removed > 0:
        reasons = removed_reason_rule.get(t, {})
        print(f"     - {t}: {total_removed}")
        for reason, count in reasons.items():
            reason_text = {
                'no_kpop_context': 'Thiếu context K-pop',
                'not_music_artist': 'Không phải nghệ sĩ âm nhạc',
                'not_related_to_network': 'Không liên quan mạng lưới'
            }.get(reason, reason)
            print(f"         + {reason_text}: {count}")

if ML_NER_AVAILABLE:
    print(f"\n📊 THỐNG KÊ ML-BASED:")
    print(f"   Entities thô: {len(ml_all_entities)}")
    print(f"   Sau khi gộp: {len(merged_ml_entities)}")
    print(f"   Sau khi lọc K-pop: {len(filtered_ml_entities)}")
    
    print(f"\n   Phân loại cuối cùng (ML-based):")
    for t in ['Company', 'Group', 'Artist', 'Album', 'Song']:
        print(f"     - {t}: {counts_ml.get(t, 0)}")
    
    print(f"\n   Số entities đã sửa type (ML-based):")
    if fixed_type_count_ml:
        for fix_type, count in fixed_type_count_ml.items():
            print(f"     - {fix_type}: {count}")
    else:
        print(f"     - Không có")
    
    print(f"\n   Số entities bị loại (ML-based):")
    for t in ['Company', 'Group', 'Artist', 'Album', 'Song']:
        total_removed = removed_count_ml.get(t, 0)
        if total_removed > 0:
            reasons = removed_reason_ml.get(t, {})
            print(f"     - {t}: {total_removed}")
            for reason, count in reasons.items():
                reason_text = {
                    'no_kpop_context': 'Thiếu context K-pop',
                    'not_music_artist': 'Không phải nghệ sĩ âm nhạc',
                    'not_related_to_network': 'Không liên quan mạng lưới',
                    'ml_low_confidence': 'Confidence < 0.65'
                }.get(reason, reason)
                print(f"         + {reason_text}: {count}")

# Hiển thị top entities
print(f"\n📝 TOP ENTITIES THEO ĐỘ TIN CẬY (Rule-based):")
for t in ['Company', 'Group', 'Artist', 'Album', 'Song']:
    items = [e for e in filtered_rule_entities if e['type'] == t][:10]
    if items:
        print(f"\n   {t} (top 10):")
        for i, e in enumerate(items, 1):
            src = len(set(e.get('sources', [])))
            print(f"     {i}. {e['text']} (conf: {e['confidence']:.2f}, {src} nguồn)")

if ML_NER_AVAILABLE and filtered_ml_entities:
    print(f"\n📝 TOP ENTITIES THEO ĐỘ TIN CẬY (ML-based):")
    for t in ['Company', 'Group', 'Artist', 'Album', 'Song']:
        items = [e for e in filtered_ml_entities if e['type'] == t][:10]
        if items:
            print(f"\n   {t} (top 10):")
            for i, e in enumerate(items, 1):
                src = len(set(e.get('sources', [])))
                print(f"     {i}. {e['text']} (conf: {e['confidence']:.2f}, {src} nguồn)")

print("\n✅ HOÀN TẤT!")
