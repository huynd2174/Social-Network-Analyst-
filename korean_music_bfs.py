# -*- coding: utf-8 -*-
import sys
import io
import time
import json
import argparse
from collections import deque
from typing import Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
import re
from neo4j import GraphDatabase

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
	try:
		# Re-wrap stdout/stderr to enforce UTF-8 regardless of console code page
		sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
		sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
	except Exception:
		pass


class WikipediaBFScraper:
	def __init__(self, request_timeout_seconds: int = 10, request_delay_seconds: float = 0.2):
		self.base_url: str = "https://vi.wikipedia.org/wiki/"
		self.session: requests.Session = requests.Session()
		self.session.headers.update({
			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-requests'
		})
		self.request_timeout_seconds: int = request_timeout_seconds
		self.request_delay_seconds: float = request_delay_seconds

		self.nodes: Dict[str, Dict[str, Any]] = {}
		self.edges: List[Dict[str, Any]] = []
		self.pending_edges: List[Dict[str, Any]] = []
		# Chỉ mục dùng để loại bỏ node trùng theo chữ ký infobox chuẩn hóa
		self.node_signature_index: Dict[str, str] = {}
		# Bản đồ alias: tiêu đề bị trùng -> tiêu đề gốc
		self.title_alias: Dict[str, str] = {}

		# Lưu danh sách seed titles để cộng điểm cho album/song của các hạt giống
		self.seed_artists: List[str] = []
		
		# Tạo các node Genre, Instrument, Company từ infobox
		self.genre_nodes: Dict[str, Dict[str, Any]] = {}
		self.instrument_nodes: Dict[str, Dict[str, Any]] = {}
		self.company_nodes: Dict[str, Dict[str, Any]] = {}
		self.occupation_nodes: Dict[str, Dict[str, Any]] = {}
		# Blacklist các "thể loại" không phải thể loại âm nhạc
		self.non_music_genres = {
			'vlog', 'giải trí', 'giai tri', 'đời sống', 'doi song', 'phim lãng mạn',
			'phim lang man', 'chính kịch', 'chinh kich', 'nhảy', 'nhay', 'âm nhạc',
			'am nhac', 'nhạc điện tử', 'nhac dien tu', 'điện tử', 'dien tu', 'nhạc kịch', 'nhac kich'
		}

		# Vietnamese markers for Korean music entities
		self.topic_keywords: List[str] = ['ca sĩ', 'nhạc sĩ', 'nhóm nhạc', 'k-pop', 'album', 'bài hát', 'ost']
		self.korean_markers: List[str] = ['hàn quốc', 'k-pop', 'hangul', 'seoul']

		# Đã xóa relation_patterns - không dùng nữa
		# Logic xác định quan hệ giờ dựa trên kiểm tra Infobox chính xác trong _determine_precise_relation()

	def save_to_neo4j(self, uri: str, user: str, password: str, database: str = None, batch_size: int = 1000, create_constraints: bool = False) -> None:
		"""Export current graph to Neo4j using MERGE on nodes and relationships.

		Nodes are merged by property id = node_key. Labels are applied based on node['label'].
		Edges are merged by (source id, type, target id). Minimal props are attached.
		"""
		driver = GraphDatabase.driver(uri, auth=(user, password))
		try:
			def run_write(tx, query, parameters=None):
				return tx.run(query, parameters or {})

			# Prepare nodes grouped by label to avoid dynamic label issues
			label_to_nodes: Dict[str, List[Dict[str, Any]] ] = {}
			for node_id, data in self.nodes.items():
				label = data.get('label', 'Entity')
				name = data.get('title') or node_id
				props = {'id': node_id, 'name': name}
				if 'url' in data:
					props['url'] = data['url']
				# merge additional properties if present
				for k, v in (data.get('properties') or {}).items():
					if k not in props:
						props[k] = v
				label_to_nodes.setdefault(label, []).append({'id': node_id, 'props': props})

			# Prepare relationships
			relationships: List[Dict[str, Any]] = []
			for e in self.edges:
				src = e.get('source')
				tgt = e.get('target')
				typ = e.get('type') or 'RELATED_TO'
				if not src or not tgt:
					continue
				relationships.append({
					'sourceId': src,
					'targetId': tgt,
					'type': typ,
					'props': {'text': e.get('text', '')}
				})

			# Cypher templates
			node_query_tpl = lambda label: f"""
			UNWIND $batch AS n
			MERGE (x:`{label}` {{id: n.id}})
			SET x += n.props
			"""

			rel_query_tpl = lambda rel_type: f"""
			UNWIND $batch AS r
			MATCH (s {{id: r.sourceId}}), (t {{id: r.targetId}})
			MERGE (s)-[e:`{rel_type}`]->(t)
			SET e += r.props
			"""

			# Open session
			with driver.session(database=database) if database else driver.session() as session:
				# Optionally create unique constraints per label on id
				if create_constraints:
					constraints = [
						"CREATE CONSTRAINT artist_id IF NOT EXISTS FOR (n:Artist) REQUIRE n.id IS UNIQUE",
						"CREATE CONSTRAINT group_id IF NOT EXISTS FOR (n:Group) REQUIRE n.id IS UNIQUE",
						"CREATE CONSTRAINT album_id IF NOT EXISTS FOR (n:Album) REQUIRE n.id IS UNIQUE",
						"CREATE CONSTRAINT song_id IF NOT EXISTS FOR (n:Song) REQUIRE n.id IS UNIQUE",
						"CREATE CONSTRAINT genre_id IF NOT EXISTS FOR (n:Genre) REQUIRE n.id IS UNIQUE",
						"CREATE CONSTRAINT inst_id IF NOT EXISTS FOR (n:Instrument) REQUIRE n.id IS UNIQUE",
						"CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
						"CREATE CONSTRAINT occ_id IF NOT EXISTS FOR (n:Occupation) REQUIRE n.id IS UNIQUE"
					]
					for q in constraints:
						session.execute_write(run_write, q)

				# Upsert nodes per label in batches
				for label, items in label_to_nodes.items():
					for i in range(0, len(items), batch_size):
						batch = items[i:i+batch_size]
						session.execute_write(run_write, node_query_tpl(label), {'batch': batch})

				# Group relationships by type to avoid dynamic type in Cypher
				type_to_rels: Dict[str, List[Dict[str, Any]]] = {}
				for r in relationships:
					type_to_rels.setdefault(r['type'], []).append(r)
				for rel_type, rels in type_to_rels.items():
					for i in range(0, len(rels), batch_size):
						batch = rels[i:i+batch_size]
						session.execute_write(run_write, rel_query_tpl(rel_type), {'batch': batch})
		finally:
			driver.close()

	def _create_genre_node(self, genre_name: str) -> str:
		"""Tạo node Genre từ tên thể loại (đã chuẩn hóa)."""
		normalized_genre = self._normalize_genre_name(genre_name)
		# Bỏ qua các thể loại không phải âm nhạc
		if not normalized_genre or normalized_genre.strip().lower() in self.non_music_genres:
			return ""
		# Ban list: không tạo node cho K-pop
		if normalized_genre.strip().lower() in {'k-pop','kpop','k pop'} or normalized_genre == 'K-pop':
			return ""
		genre_key = f"Genre_{normalized_genre}"
		if genre_key not in self.genre_nodes:
			self.genre_nodes[genre_key] = {
				"title": normalized_genre,
				"label": "Genre",
				"properties": {
					"name": normalized_genre,
					"url": f"https://vi.wikipedia.org/wiki/{normalized_genre.replace(' ', '_')}"
				}
			}
		return genre_key
	
	def _create_instrument_node(self, instrument_name: str) -> str:
		"""Tạo node Instrument từ tên nhạc cụ"""
		normalized_instrument = self._normalize_instrument_name(instrument_name)
		# Bỏ qua nếu không hợp lệ hoặc nằm trong ban list
		if not normalized_instrument:
			return ""
		if normalized_instrument.strip().lower() in {'vocals'} or normalized_instrument == 'Vocals':
			return ""
		instrument_key = f"Instrument_{normalized_instrument}"
		if instrument_key not in self.instrument_nodes:
			self.instrument_nodes[instrument_key] = {
				"title": normalized_instrument,
				"label": "Instrument", 
				"properties": {
					"name": normalized_instrument,
					"url": f"https://vi.wikipedia.org/wiki/{normalized_instrument.replace(' ', '_')}"
				}
			}
		return instrument_key
    
	def _normalize_instrument_name(self, instrument_name: str) -> str:
		"""Chuẩn hóa tên nhạc cụ: hợp nhất Vocals, sửa dính chuỗi, chuẩn Title Case, loại genre."""
		original = instrument_name or ''
		instrument_lower = original.lower().strip()
		
		# Loại riêng 'Thanh nhạc' khỏi Nhạc cụ
		if 'thanh nhạc' in instrument_lower or 'thanh nhac' in instrument_lower:
			return ''
		
		# Loại các mục không phải nhạc cụ (genre/phong cách)
		invalid = {'vocal trance', 'pop rap'}
		if instrument_lower in invalid:
			return ''
		
		# Fix dính chuỗi đặc thù
		if 'keyboard' in instrument_lower and ('dương cầm' in original or 'duong cam' in instrument_lower):
			return 'Piano'
		if instrument_lower in {'ghi-ta','ghita','ghitar'} or 'ghi-ta' in instrument_lower or 'ghita' in instrument_lower:
			return 'Guitar'
		if 'dương cầm' in instrument_lower or 'duong cam' in instrument_lower:
			return 'Piano'
		
		# Gộp các từ liên quan đến giọng hát (trừ 'Thanh nhạc' đã loại ở trên)
		vocal_keywords = ['hát', 'giọng hát', 'vocals', 'vocal', 'singing', 'voice']
		for keyword in vocal_keywords:
			if keyword in instrument_lower:
				return 'Vocals'
		
		# Chuẩn Title Case cơ bản; giữ một số mapping đơn giản
		mapping = {
			'piano': 'Piano', 'flute': 'Flute', 'guitar': 'Guitar', 'organ': 'Organ',
			'synth': 'Synth'
		}
		if instrument_lower in mapping:
			return mapping[instrument_lower]
		
		# Title Case
		words = [w.capitalize() for w in re.sub(r'\s+', ' ', original.strip()).split(' ')]
		return ' '.join(words)

	def _normalize_genre_name(self, genre_name: str) -> str:
		"""Chuẩn hóa tên thể loại - gộp các biến thể và chuẩn hóa chữ hoa"""
		genre_lower = genre_name.lower().strip()
		# Loại riêng K-pop khỏi node Thể loại
		if genre_lower in ['k-pop', 'kpop', 'k pop']:
			return ''
		# Gộp 'Nhạc Dance' -> 'Dance'
		if genre_lower in ['nhạc dance', 'nhac dance']:
			return 'Dance'
		# Gộp 'Nhạc Rock Điện Tử' -> 'Electronic Rock'
		if genre_lower in ['nhạc rock điện tử', 'nhac rock dien tu', 'electronic rock']:
			return 'Electronic Rock'
		# Sửa một số biến thể có gạch
		if genre_lower in ['house-pop', 'house pop']:
			return 'House'
		# Pbr&b -> R&B
		if genre_lower in ['pbr&b', 'pbrnb', 'pbr b']:
			return 'R&B'
		
		# Chuẩn hóa hip hop
		if genre_lower in ['hip-hop', 'hip hop', 'hiphop']:
			return "Hip hop"
		
		# Chuẩn hóa R&B
		if genre_lower in ['r&b', 'rnb', 'r and b', 'rhythm and blues']:
			return "R&B"
		
		# Chuẩn hóa K-pop
		if genre_lower in ['k-pop', 'kpop', 'k pop']:
			return "K-pop"
		
		# Chuẩn hóa J-pop
		if genre_lower in ['j-pop', 'jpop', 'j pop']:
			return "J-pop"
		
		# Chuẩn hóa Synthpop / Synth-pop
		if genre_lower in ['synthpop', 'synth-pop', 'synth pop']:
			return "Synthpop"
		
		# Chuẩn hóa Pop
		if genre_lower in ['pop music', 'popular music']:
			return "Pop"
		
		# Chuẩn hóa Ballad
		if genre_lower in ['ballad music', 'slow song']:
			return "Ballad"
		
		# Chuẩn hóa Rock
		if genre_lower in ['rock music', 'rock and roll']:
			return "Rock"
		
		# Chuẩn hóa Dance-pop
		if genre_lower in ['dance-pop', 'dance pop', 'dancepop']:
			return "Dance-pop"
		
		# Chuẩn hóa Electropop
		if genre_lower in ['electropop', 'electro pop', 'electro-pop']:
			return "Electropop"
		
		# Chuẩn hóa Bubblegum pop
		if genre_lower in ['bubblegum pop', 'bubblegum-pop', 'bubblegumpop']:
			return "Bubblegum pop"
		
		# Chuẩn hóa Trap
		if genre_lower in ['trap music', 'trap']:
			return "Trap"
		
		# Chuẩn hóa Electronic
		if genre_lower in ['electronic music', 'electronic']:
			return "Electronic"
		
		# Chuẩn hóa chữ hoa đầu cho các thể loại khác
		# Tách thành các từ và viết hoa chữ cái đầu mỗi từ
		words = genre_name.strip().split()
		normalized_words = []
		for word in words:
			if word:
				# Giữ nguyên các từ đặc biệt như "&", "-"
				if word in ['&', '-', '/']:
					normalized_words.append(word)
				else:
					normalized_words.append(word.capitalize())
		
		return ' '.join(normalized_words)
    
	def _create_company_node(self, company_name: str) -> str:
		"""Tạo node Company từ tên công ty"""
		# Chuẩn hóa tên công ty để tránh trùng lặp
		normalized_company = self._normalize_company_name(company_name)
		# Bỏ qua nếu tên không hợp lệ/blacklist
		if not normalized_company:
			return ""
		company_key = f"Company_{normalized_company}"
		if company_key not in self.company_nodes:
			self.company_nodes[company_key] = {
				"title": normalized_company,
				"label": "Company",
				"properties": {
					"name": normalized_company,
					"url": f"https://vi.wikipedia.org/wiki/{normalized_company.replace(' ', '_')}"
				}
			}
		return company_key

	def _create_occupation_node(self, occupation_name: str) -> str:
		"""Tạo node Occupation từ tên nghề nghiệp (đã chuẩn hóa)."""
		normalized = self._normalize_occupation_name(occupation_name)
		if not normalized:
			return ""
		key = f"Occupation_{normalized}"
		if key not in self.occupation_nodes:
			self.occupation_nodes[key] = {
				"title": normalized,
				"label": "Occupation",
				"properties": {
					"name": normalized,
					"url": f"https://vi.wikipedia.org/wiki/{normalized.replace(' ', '_')}"
				}
			}
		return key

	def _normalize_occupation_name(self, occupation_name: str) -> str:
		"""Chuẩn hóa nghề nghiệp; bỏ 'Ca sĩ' và tương đương."""
		name = re.sub(r"\s+", " ", (occupation_name or '').strip())
		lower = name.lower()
		# Bỏ các biến thể của Ca sĩ
		if lower in { 'ca sĩ', 'ca si', 'singer', 'vocalist' }:
			return ''
		# Chuẩn hóa cụm chứa 'nhạc sĩ' hoặc 'sáng tác'/'songwriter' về 'Nhạc sĩ'
		if 'nhạc sĩ' in lower or 'nhac si' in lower:
			return 'Nhạc sĩ'
		if 'sáng tác' in lower or 'viet loi' in lower or 'viết lời' in lower or 'songwriter' in lower:
			return 'Nhạc sĩ'
		# 'composer' hoặc 'nhà soạn nhạc'
		if 'nhà soạn nhạc' in lower or 'nha soan nhac' in lower or 'composer' in lower:
			return 'Nhà soạn nhạc'
		# 'producer' hoặc 'nhà sản xuất'
		if 'nhà sản xuất' in lower or 'nha san xuat' in lower or 'producer' in lower:
			return 'Nhà sản xuất'
		# 'diễn viên'
		if 'diễn viên' in lower or 'dien vien' in lower or 'actor' in lower or 'actress' in lower:
			return 'Diễn viên'
		# 'rapper'
		if 'rapper' in lower:
			return 'Rapper'
		# Map một số nghề phổ biến về dạng Title Case chuẩn
		mapping = {
			'nhạc sĩ': 'Nhạc sĩ',
			'nhac si': 'Nhạc sĩ',
			'diễn viên': 'Diễn viên',
			'dien vien': 'Diễn viên',
			'rapper': 'Rapper',
			'nhà sản xuất': 'Nhà sản xuất',
			'nha san xuat': 'Nhà sản xuất',
			'nhà soạn nhạc': 'Nhà soạn nhạc',
			'nha soan nhac': 'Nhà soạn nhạc'
		}
		if lower in mapping:
			return mapping[lower]
		# Title Case mặc định
		words = [w.capitalize() for w in re.sub(r'\s+', ' ', name).split(' ') if w]
		return ' '.join(words)

	def _normalize_company_name(self, company_name: str) -> str:
		"""Chuẩn hóa tên công ty/label để loại trùng logic."""
		name = company_name.strip()
		lower = name.lower()
		# Blacklist các tên địa lý/quốc gia không phải công ty
		country_like = {'hàn quốc','han quoc','south korea','korea','nhật bản','nhat ban','japan','việt nam','viet nam','vietnam','trung quốc','trung quoc','china'}
		if lower in country_like:
			return ""
		# Chuẩn hóa một số biến thể phổ biến của công ty Hàn Quốc
		# 1) SM Entertainment
		if any(x in lower for x in ["s.m entertainment", "s.m. entertainment", "sm ent.", "sm ent", "sm entertainment", "sm town", "smtown"]):
			return "SM Entertainment"
		# 2) YG Entertainment
		if any(x in lower for x in ["yg ent.", "yg ent", "yg entertainment"]):
			return "YG Entertainment"
		# 3) JYP Entertainment
		if any(x in lower for x in ["jyp ent.", "jyp ent", "j.y.p", "jyp entertainment"]):
			return "JYP Entertainment"
		# 4) HYBE / Big Hit / BIGHIT Music
		if any(x in lower for x in ["big hit", "bighit", "bighit music", "hybe"]):
			# Dùng HYBE làm dạng chuẩn hiện tại
			return "HYBE"
		# 5) CUBE Entertainment
		if any(x in lower for x in ["cube ent.", "cube ent", "cube entertainment"]):
			return "Cube Entertainment"
		# 6) Starship Entertainment
		if "starship" in lower:
			return "Starship Entertainment"
		# 7) Pledis Entertainment
		if "pledis" in lower:
			return "Pledis Entertainment"
		# 8) FNC Entertainment
		if any(x in lower for x in ["fnc ent", "fnc entertainment"]):
			return "FNC Entertainment"
		# 9) RBW
		if any(x in lower for x in ["rbw ent", "rbw entertainment", "rbw"]):
			return "RBW"
		# 10) Kakao Entertainment
		if any(x in lower for x in ["kakao m", "kakao entertainment", "loen entertainment", "loen"]):
			return "Kakao Entertainment"
		# 11) The Black Label
		if "the black label" in lower:
			return "The Black Label"
		# 12) Brand New Music
		if "brand new music" in lower:
			return "Brand New Music"
		# 13) DSP Media
		if any(x in lower for x in ["dsp media", "dsp ent"]) or lower == 'dsp':
			return "DSP Media"
		# 14) WM Entertainment
		if any(x in lower for x in ["wm ent", "wm entertainment"]):
			return "WM Entertainment"
		# 15) TOP Media
		if any(x in lower for x in ["top media", "top ent"]):
			return "TOP Media"
		# 16) Swing Entertainment
		if "swing entertainment" in lower:
			return "Swing Entertainment"
		# 17) P Nation
		if "p nation" in lower:
			return "P Nation"
		# 17.5) Sửa lỗi chính tả phổ biến
		if 'waner music korea' in lower:
			return 'Warner Music Korea'
		
		# 18) Interscope Records / Interscope
		if "interscope" in lower:
			if "records" in lower:
				return "Interscope Records"
			else:
				return "Interscope Records"  # Chuẩn hóa về dạng đầy đủ
		
		# 19) Virgin Music / Virgin
		if "virgin" in lower:
			if "music" in lower:
				return "Virgin Music"
			else:
				return "Virgin Music"  # Chuẩn hóa về dạng đầy đủ

		# 19.5) Columbia Records / Columbia
		if 'columbia' in lower:
			return 'Columbia Records'

		# 19.6) RCA Records / RCA
		if lower in {'rca', 'r.c.a'} or 'rca records' in lower:
			return 'RCA Records'
		
		# 20) Universal variants
		if 'universal' in lower:
			if 'music group' in lower:
				return 'Universal Music Group'
			if 'music japan' in lower or 'japan' in lower:
				return 'Universal Music Japan'
			return 'Universal Music'

		# 21) Sony Music variants
		if 'sony music' in lower:
			if 'japan' in lower:
				return 'Sony Music Japan'
			if 'taiwan' in lower:
				return 'Sony Music Taiwan'
			return 'Sony Music'

		# 22) Warner variants
		if 'warner music group' in lower:
			return 'Warner Music Group'
		if 'warner music japan' in lower or ('warner' in lower and 'japan' in lower):
			return 'Warner Music Japan'
		if 'warner' in lower and 'music' in lower:
			return 'Warner Music'
		if lower == 'warner':
			return 'Warner Music'

		# 23) Stone / Stone Music
		if 'stone music' in lower:
			return 'Stone Music'
		if lower == 'stone':
			return 'Stone Music'

		# 24) Genie / Genie Music
		if 'genie music' in lower:
			return 'Genie Music'
		if lower == 'genie':
			return 'Genie Music'

		# 25) Kakao unify
		if 'kakao' in lower:
			return 'Kakao Entertainment'

		# 26) JYP unify
		if lower in {'jyp','j.y.p','jyp ent','jyp ent.'} or 'jyp entertainment' in lower:
			return 'JYP Entertainment'

		# 27) The Black Label variants
		if 'theblacklabel' in lower.replace(' ','') or 'the black label' in lower:
			return 'The Black Label'

		# Chuẩn hóa chung: viết hoa tên riêng và giữ chữ "Entertainment"/"Records" chuẩn
		# Loại bỏ khoảng trắng thừa
		clean = re.sub(r"\s+", " ", name).strip()
		# Viết hoa chữ cái đầu mỗi từ, nhưng giữ các từ viết tắt (SM, YG, JYP, RBW)
		words = [w.upper() if w.upper() in {"SM", "YG", "JYP", "RBW"} else w.capitalize() for w in clean.split(" ")]
		return " ".join(words)

	@staticmethod
	def _has_hangul(text: str) -> bool:
		"""Detect presence of Hangul characters to infer Korean context."""
		if not text:
			return False
		return re.search(r"[\uac00-\ud7af]", text) is not None

	def _calculate_quality_score(self, node_title: str, node_data: Dict[str, Any]) -> float:
		"""Calculate quality score for a node (used during crawling)."""
		score = 0.0
		label = node_data.get('label', 'Entity')
		title_lower = node_title.lower()
        
		# Get infobox for checking
		infobox = node_data.get('infobox', {})
		infobox_text = ' '.join(str(v) for v in infobox.values())
		infobox_lower = infobox_text.lower()
        
		# BLACKLIST: Loại bỏ generic entities không liên quan
		blacklist_patterns = [
			'(họ)', '(họ người', '(địa danh)', '(thành phố)', '(tỉnh)', 
			'(quận)', '(huyện)', '(quốc gia)', '(lục địa)', '(châu lục)',
			'bảng xếp hạng', 'danh sách', 'thể loại:', 'wikipedia:',
			'gaon chart', 'billboard', 'melon chart','bùi',
			# TV shows and reality programs
			'chương trình truyền hình', 'reality show', 'game show', 
			'show âm nhạc', 'cuộc thi', 'tập phim', 'phim truyền hình',
			'running man', 'produce 101', 'unpretty rapstar', 
			'show me the money', 'king of mask singer', 'inkigayo',
			'music bank', 'music core', 'the show', 'show champion',
			# Films and movies
			'(phim)', '(film)', '(movie)', 'phim điện ảnh', 'phim lẻ',
			'phim truyền hình hàn quốc', '(phim truyền hình hàn quốc)', 'bộ phim truyền hình', 'web drama', 'sitcom',
			'romance drama', 'rom-com', 'romantic drama', 'k-drama', 'tv series', 'series', 'episode',
			'đạo diễn', 'nhà sản xuất phim', 'phát hành phim',
			'thời lượng phim', 'khởi chiếu', 'doanh thu phim',
			# Common VN titles to avoid misclassification as artists
			'cô nàng đáng yêu', 'cô nàng đẹp trai', 'vườn sao băng', 'người thừa kế', 'thư ký kim'
			# Famous movies/series (not music)
			'trò chơi con mực', 'squid game', 'parasite', 'train to busan',
			'my love from the star', 'descendants of the sun', 'crash landing',
			'goblin', 'signal', 'reply 1988', 'sky castle',
			# Places / landmarks / markets (địa danh, chợ, khu phố, ga tàu...)
			'chợ', 'market', 'chợ dongdaemun', 'dongdaemun market', 'namdaemun market', 'chợ namdaemun',
			'myeongdong', 'insadong', 'itaewon', 'hongdae', 'gangnam district', 'gangnam',
			'cầu', 'sông', 'núi', 'vườn quốc gia', 'bảo tàng', 'cung điện', 'di tích', 'đền', 'chùa', 'thánh đường',
			'sân vận động', 'sân bay', 'cảng', 'bến cảng', 'bến xe', 'quảng trường',
			'ga', 'station', 'subway', 'tàu điện ngầm', 'metro', 'line 1', 'line 2', 'line 3', 'line 4',
			'teheran-ro', 'đường teheran',
			# Exhibition/Convention centers
			'trung tâm triển lãm', 'trung tâm hội nghị', 'trung tâm triển lãm quốc tế',
			'exhibition center', 'convention center',
			'korea international exhibition center', 'kintex', 'coex', 'coex mall', 'coex convention', 'bexco',
			# Government/public buildings and institutions
			'tòa thị chính', 'city hall', 'tòa thị chính seoul', 'seoul city hall',
			'chính phủ', 'ủy ban', 'hội đồng', 'ủy ban nhân dân', 'seoul metropolitan government',
			# Administrative divisions/types
			'thành phố trực thuộc trung ương', 'thành phố tự trị đặc biệt', 'thành phố đặc biệt',
			'thành phố đô thị', 'đô thị đặc biệt', 'đô thị trực thuộc trung ương',
			'đơn vị hành chính', 'đơn vị hành chính cấp', 'tỉnh (hàn quốc)', 'tỉnh của hàn quốc',
			'đô thị', 'thành phố của hàn quốc'
			# Chinese/Taiwanese artists and music
			'trung quốc', 'đài loan', 'hồng kông', 'c-pop', 'cpop',
			'mandarin', 'cantonese', 'tfboys', 'jay chou', 'jolin tsai',
			'(ca sĩ trung quốc)', '(nhạc sĩ trung quốc)', '(nhóm nhạc trung quốc)',
			'(ca sĩ đài loan)', '(ca sĩ hồng kông)',
			# Universities and educational institutions
			'trường đại học', 'đại học', 'university', 'college', 
			'học viện', 'viện đại học', 'trường cao đẳng',
			'seoul national university', 'yonsei university', 'korea university',
			'hanyang university', 'sungkyunkwan university',
			# Generic album/song units and charts
			'đơn vị album', 'album chart', 'ifpi', 'riia','mnet media','nhạc trot',
			# Awards and prizes
			'giải mama', 'mnet asian music awards', 'giải thưởng', 'giải thưởng âm nhạc','genie music awards'
			'golden disc awards', 'melon music awards', 'mma (giải)', 'soribada best k-music awards',
			'asian artist awards', 'gaon chart music awards', 'billboard music awards', 'mtv', 'grammy', 'brit awards', 'mama awards','awards','award',
			# Broadcasting networks – ví dụ: Hệ thống Phát sóng Seoul (SBS)
			'hệ thống phát sóng seoul', 'seoul broadcasting system',
			# Broadcasting/Media/News/TV Channels (nhiều dạng)
			'sbs', 'kbs', 'mbc', 'ebs', 'jtbc', 'cable tv',
			'system broadcast', 'broadcast system', 'broadcasting system', 'television', 'truyền hình', 'truyền thanh', 'truyền thông',
			'đài truyền hình', 'đài phát thanh', 'đài phát sóng', 'đài quốc gia', 'trạm phát sóng',
			'korean broadcasting system', 'seoul broadcasting system', 'munhwa broadcasting corporation', 'educational broadcasting system',
			'sbs tv', 'kbs2', 'kbs1', 'sbs fun', 'sbs sports', 'mbc music', 'arirang', 'ytn', 'channel a', 'tv chosun',
			'mnet', 'onstyle', 'sohae broadcasting', 'dong-a broadcasting', 'busan broadcasting', 'jeonju broadcasting',
			'broadcasting-tower', 'hanminjok broadcasting', 'chosun broadcasting',
			'asia broadcasting', 'asia tv', 'bnt news', 'newsis', 'yonhap news', 'news1 korea', 'chosun ilbo', 'dong-a ilbo', 'hankyoreh',
			'osense', 'dispatch', 'allkpop', 'soompi', 'naver news', 'naver tv', 'daum tv', 'v live',
			# Western film/OST specific titles to exclude
			'another cinderella story', 'another cinderella story (soundtrack)',
			'a cinderella story', 'cinderella', 'cinderella story', 'nhạc phim cinderella',
			'nữ hoàng băng giá', 'frozen', 'let it go', 'bài hát của disney', 'disney soundtrack',
			# Traditional forms / show titles often misclassified
			'hát kể pansori', 'pansori',
			'hãy cười lên nào', 'smile, you', 'smile you',
			# Western specific songs (e.g., Alanis Morissette)
			'ironic (bài hát)', 'thank u (bài hát)', 'uninvited (bài hát)',
			'ironic', 'thank u', 'uninvited', 'alanis morissette','a-ha','alicia keys',
			# Reality/competition TV shows
			"korea's got talent", 'korea got talent', 'got talent',
			"korea's next top model", 'koreas next top model', 'next top model', 'top model',
			"belgium's next top model",
			# Specific show titles
			"song ji-hyo's beauty view", 'beauty view',
			# Politicians and political figures
			'park chung-hee', 'tổng thống hàn quốc', 'chính trị gia hàn quốc'
			,
			# Korean conglomerates/chaebols and large corporations (exclude as non-music)
			'hyundai group', 'hyundai motor', 'hyundai', 'kia', 'samsung', 'samsung group',
			'lg', 'lotte', 'shinsegae', 'hanwha', 'sk group', 'sk telecom', 'posco',
			'kt', 'korea telecom', 'naver', 'naver corp', 'kakao', 'kakao corp', 'cj group',
			'cj entertainment', 'doosan', 'hitejinro', 'hyosung', 'korean air', 'asiana airlines', 'hyundai mobis',
			# Additional non-music entities from user feedback
			'kim quan già da', 'kim kwan giada', 'chí tri vương', 'triều tiên túc tông',
			'vương', 'quốc vương', 'vua triều tiên', 'vua joseon',
			'khu vực an ninh chung', 'joint security area', 'jsa',
			'jeong (họ triều tiên)', 'họ triều tiên', '(họ triều tiên)',
			'hưng phu truyện', 'tiểu thuyết', 'truyện', 'novel',
			'bảo tàng quốc gia hàn quốc', 'bảo tàng', 'national museum of korea',
			'công viên olympic', 'olympic park', 'seoul olympic park',
			'doosan encyclopedia', 'doosan wiki', 'bách khoa doosan',
			'chợ bangsan', 'chợ cá hợp tác xã busan', 'chợ daesong', 'chợ eonyang', 'chợ garak', 'chợ gukje', 'chợ gwangjang', 'chợ gyeongdong', 'chợ jagalchi', 'chợ namdaemun', 'chợ seomun', 'chợ seongdong', 'chợ sinjeong', 'chợ suam', 'chợ taehwa', 'chợ thủy sản noryangjin', 'chợ trung tâm ulsan', 'chợ trời hwanghak-dong',
			'ẩm thực triều tiên', 'ẩm thực hàn quốc', 'korean cuisine', 'banchan', 'kimchi',
			'yeongnam', 'honam', 'chungcheong', 'gangwon', 'gyeongsang', 'jeolla', 'jeju-do',
			'trường trung học', 'trường trung học apgujeong', 'high school', 'apgujeong high school',
			'bốn con hổ châu á', 'thống nhất triều tiên',
			# Additional from latest feedback
			'hoàng phủ', 'ho chong',
			'gimbap', 'dubu kimchi', 'doenjang', 'dokkaebi', 'deoksugung', 'dasik',
			'ga gyeongju', 'ga sinn am', 'ga sinnam', 'ga ulsan', 'ga chợ seomun',
			'faker', 'esports', 'league of legends', 'liên minh huyền thoại', 't1 esports'
			,
			# New feedback batch
			'cư đăng vương', 'quân chủ', 'vương triều',
			'công viên yeouido', 'yeouido park', 'yeouido',
			'cách mạng 19 tháng 4', 'april revolution', '4.19 revolution', 'revolution hàn quốc'
			'chính sách ánh dương', 'sunshine policy'
		]
        
		for pattern in blacklist_patterns:
			if pattern in title_lower:
				return 0.0  # Loại bỏ ngay
		
		# BLACKLIST: Kiểm tra infobox để phát hiện phim
		# Nếu có các trường đặc trưng của phim trong infobox
		film_fields = ['Đạo diễn', 'Director', 'Diễn viên', 'Cast', 'Actors', 'Nhà sản xuất', 'Producer', 
					  'Khởi chiếu', 'Release date', 'Ngày khởi chiếu', 'Thời lượng', 'Running time',
					  'Doanh thu', 'Box office', 'Thể loại phim', 'Film genre', 'Loại phim']
		
		film_indicators = ['phim điện ảnh', 'phim lẻ', 'phim truyền hình', 'drama', 'series',
						  'đạo diễn', 'diễn viên chính', 'ngày khởi chiếu', 'doanh thu phim',
						  'thời lượng phim', 'phát hành phim', 'biên kịch', 'screenwriter', 'kịch bản']
		
		has_film_field = False
		for field in film_fields:
			if field in infobox:
				# Kiểm tra xem trường này có chứa dấu hiệu phim không
				field_value = str(infobox[field]).lower()
				if any(indicator in field_value for indicator in film_indicators):
					has_film_field = True
					break
				# Nếu là trường "Đạo diễn" hoặc "Diễn viên" thì chắc chắn là phim
				if field in ['Đạo diễn', 'Director', 'Diễn viên', 'Cast', 'Actors']:
					has_film_field = True
					break
		
		if has_film_field:
			return 0.0  # Loại bỏ phim ngay lập tức
        
		# WHITELIST: Known K-pop artists/groups (bonus cao)
		known_kpop_entities = [
			'bts', 'blackpink', 'exo', 'twice', 'red velvet', 'nct', 
			'seventeen', 'stray kids', 'itzy', 'aespa', 'newjeans',
			'big bang', "girls' generation", 'super junior', '2ne1',
			'got7', 'iu', 'psy', 'taeyeon', 'jennie', 'rosé', 'jisoo',
			'lisa', 'jimin', 'jungkook', 'suga', 'rm', 'jin', 'j-hope'
		]
        
		# Check if title mentions known K-pop entity
		has_kpop_entity = False
		for entity in known_kpop_entities:
			if entity in title_lower:
				score += 20  # Big bonus for known entities
				has_kpop_entity = True
				break
        
		# Base score for correct label types (không tính Company/Award)
		if label in ('Artist', 'Group', 'Album', 'Song'):
			score += 10
        
		# Check ONLY infobox for K-pop artist (for Albums/Songs) - more strict
		if label in ('Album', 'Song') and not has_kpop_entity:
			# Check specific fields only (not full page to avoid false positives)
			artist_fields = ['Nghệ sĩ', 'Ca sĩ', 'Nhóm nhạc', 'Artist', 'Performer']
			for field in artist_fields:
				if field in infobox:
					field_value = str(infobox[field]).lower()
                    
					# Đặc biệt cộng điểm cho album/song của SEED ARTISTS
					for seed_artist in self.seed_artists:
						seed_lower = seed_artist.lower().replace('(nhóm nhạc)', '').replace('(ca sĩ)', '').strip()
						if seed_lower in field_value:
							score += 30  # Bonus cao cho seed artist work
							has_kpop_entity = True
							break
                    
					if has_kpop_entity:
						break
                    
					# Check known K-pop entities
					for entity in known_kpop_entities:
						if entity in field_value:
							score += 20  # Strong bonus for confirmed K-pop artist work
							has_kpop_entity = True
							break
					if has_kpop_entity:
						break

			# Cộng điểm nếu album/bài hát thuộc các hãng/label Hàn Quốc nổi tiếng
			if not has_kpop_entity:
				label_fields = ['Hãng đĩa', 'Label', 'Công ty quản lý', 'Agency']
				known_korean_labels = [
					'sm entertainment', 'yg entertainment', 'jyp entertainment',
					'big hit', 'hybe', 'bighit music', 'source music', 'ador',
					'pledis', 'cube entertainment', 'starship entertainment', 'rbw',
					'fnc entertainment', 'woollim entertainment', 'kakao m', 'kakao entertainment',
					'loen entertainment', 'stone music', 'cj e&m', '1thek', 'brand new music',
					'dsp media', 'mbk entertainment', 'wm entertainment', 'top media',
					'swing entertainment', 'the black label', 'p nation','mnet media',
				]
				for field in label_fields:
					if field in infobox:
						field_value = str(infobox[field]).lower()
						if any(lbl in field_value for lbl in known_korean_labels):
							score += 18  # Bonus mạnh cho album/bài hát thuộc các label Hàn Quốc
							break
		
		# CỘNG ĐIỂM cho Album/Song có "Được thực hiện bởi" trùng với nghệ sĩ Hàn Quốc nổi tiếng
		if label in ('Album', 'Song'):
			performed_by_fields = ['Được thực hiện bởi', 'Performed by', 'Artist', 'Performer']
			for field in performed_by_fields:
				if field in infobox:
					performed_by_value = str(infobox[field]).lower()
					
					# Danh sách nghệ sĩ/nhóm nhạc Hàn Quốc nổi tiếng (mở rộng)
					famous_korean_artists = [
						# Groups
						'bts', 'blackpink', 'exo', 'twice', 'red velvet', 'nct',
						'seventeen', 'stray kids', 'itzy', 'aespa', 'newjeans',
						'big bang', "girls' generation", 'super junior', '2ne1',
						'got7', 'monsta x', 'stray kids', 'the boyz', 'ateez',
						'enhypen', 'txt', 'treasure', 'ive', 'le sserafim',
						'gidle', 'aespa', 'stayc', 'nmixx', 'kep1er',
						'kara', 'wonder girls', 'girls day', 'apink', 'sistar',
						'ailee', 'sunmi', 'chungha', 'boa', 'taeyeon',
						# Solo Artists
						'iu', 'psy', 'jennie', 'rosé', 'jisoo', 'lisa',
						'jimin', 'jungkook', 'suga', 'rm', 'jin', 'j-hope',
						'v', 'g-dragon', 'taeyang', 'daesung', 'taemin',
						'kai', 'baekhyun', 'chen', 'd.o.', 'chanyeol',
						'jihyo', 'nayeon', 'jeongyeon', 'momo', 'sana',
						'dahyun', 'chaeyoung', 'tzuyu', 'irene', 'seulgi',
						'wendy', 'joy', 'yeri', 'hyuna', 'hani',
						'moonbyul', 'wheein', 'hwasa', 'yerin', 'eunha',
						'cha eun-woo', 'minho', 'onew', 'key', 'shinee',
						'snsd', 'soshi', 'f(x)', 'rv', 'nct', 'seventeen'
					]
					
					# Kiểm tra từng nghệ sĩ trong danh sách
					for artist in famous_korean_artists:
						# Chuẩn hóa tên nghệ sĩ để so sánh (loại bỏ dấu, lowercase)
						artist_normalized = artist.lower().strip()
						
						# So khớp với word boundary để tránh false positive
						# Ví dụ: "twice" không match "fortwice"
						pattern = r'\b' + re.escape(artist_normalized) + r'\b'
						if re.search(pattern, performed_by_value, re.IGNORECASE):
							score += 25  # Bonus cao cho album/song của nghệ sĩ Hàn Quốc nổi tiếng
							has_kpop_entity = True
							break
					
					# Nếu không tìm thấy trong danh sách, kiểm tra với seed artists
					if not has_kpop_entity:
						for seed_artist in self.seed_artists:
							seed_normalized = seed_artist.lower().replace('(nhóm nhạc)', '').replace('(ca sĩ)', '').strip()
							pattern = r'\b' + re.escape(seed_normalized) + r'\b'
							if re.search(pattern, performed_by_value, re.IGNORECASE):
								score += 20  # Bonus cho seed artist
								has_kpop_entity = True
								break
					
					# Kiểm tra với known_kpop_entities
					if not has_kpop_entity:
						for entity in known_kpop_entities:
							pattern = r'\b' + re.escape(entity) + r'\b'
							if re.search(pattern, performed_by_value, re.IGNORECASE):
								score += 18  # Bonus cho known K-pop entities
								has_kpop_entity = True
								break
					
					if has_kpop_entity:
						break
					# Nếu vẫn không có dấu hiệu K-pop: loại ngay nếu performer là nghệ sĩ phương Tây
					western_block = {
						# Pop divas and stars
						'adele','beyoncé','beyonce','taylor swift','pink','the temptations','bobby day','westlife','katy perry','lady gaga','rihanna','madonna','whitney houston','shakira',
						'christina aguilera','mariah carey','dua lipa','ariana grande','selena gomez','miley cyrus','demi lovato','kesha',
						# Male pop and R&B
						'ed sheeran','justin bieber','bruno mars','charlie puth','john legend','the weeknd','harry styles','zayn','niall horan','janet jackson',
						# Bands
						'coldplay','maroon 5','imagine dragons','one direction','backstreet boys','nsync','onerepublic','tears for fears','u2','queen','the rolling stones','the mamas & the papas','fifth harmony',
						'abba','bee gees','fleetwood mac','the police','depeche mode','duran duran','spice girls','abba','bon jovi','metallica','nirvana','radiohead','muse','green day','linkin park','red hot chili peppers','arctic monkeys','oasis','blur','the cure','the smiths','evanescence','paramore','fall out boy','panic! at the disco','thirty seconds to mars','the chainsmokers','lmfao',
						# Hip-hop
						'kanye west','ye','jay-z','drake','eminem','post malone','travis scott','nicki minaj','cardi b','pitbull','flo rida','lil wayne','future',
					# Legacy
				'michael jackson','the beatles','elton john','prince','david bowie','phil collins','sting','rod stewart','eric clapton','don mclean','bing crosby','charlie chaplin','diana ross','lionel richie','jacques brel',
						# Latin
						'enrique iglesias','ricky martin','bad bunny','j balvin','maluma','anitta','karol g','becky g','shakira','rosalía','alanis morissette','alicia keys',
				# Others / J-pop non-targets
				'britney spears','camila cabello','avril lavigne','sia','p!nk','ellie goulding','rita ora','zara larsson','lorde','halsey','bebe rexha','nogizaka46','arashi','jay chou','jolin tsai','jj lin','the white stripes','damage','artists for haiti','bon iver','mika'
					}
					if not has_kpop_entity and any(w in performed_by_value for w in western_block):
						return 0.0
        
		# BLACKLIST: Western/Latin/Other non-Korean artists - DIRECT REJECTION
		western_artists = [
			# Pop/Rock artists
			'adele', 'beyoncé', 'beyonce', 'taylor swift', 'the temptations','ed sheeran','westlife','bobby day',
			'eminem', 'coldplay', 'maroon 5', 'rihanna', 'lady gaga','janet jackson'
			'ariana grande', 'justin bieber', 'selena gomez', 'katy perry',
			'bruno mars', 'the beatles', 'michael jackson', 'madonna',
			'drake', 'kanye west', 'jay-z', 'whitney houston',
			'u2', 'queen', 'the rolling stones', 'abba', 'bee gees', 'fleetwood mac', 'the police', 'depeche mode', 'duran duran', 'spice girls',
			'bon jovi', 'metallica', 'nirvana', 'radiohead', 'muse', 'green day', 'linkin park', 'red hot chili peppers', 'arctic monkeys', 'oasis', 'blur', 'the cure', 'the smiths', 'evanescence', 'paramore', 'fall out boy', 'panic! at the disco', 'thirty seconds to mars', 'the chainsmokers', 'lmfao', 'tears for fears', 'the mamas & the papas', 'fifth harmony',
			# Latin artists
			'bad bunny', 'j balvin', 'maluma', 'shakira', 'ricky martin',
			'enrique iglesias', 'daddy yankee', 'ozuna', 'karol g', 'anitta', 'rosalía', 'becky g',
			# Other international artists  
			'doja cat', 'billie eilish', 'olivia rodrigo', 'post malone',
			'the weeknd', 'harry styles', 'dualipa', 'camila cabello',
			'shawn mendes', 'charlie puth', 'ed sheeran', 'james arthur', 'sia', 'p!nk', 'ellie goulding', 'rita ora', 'zara larsson', 'lorde', 'halsey', 'bebe rexha',
			# Rap/Hip-hop
			'travis scott', 'migos', 'cardi b', 'nicki minaj', 'lil nas x',
			'jack harlow', 'megan thee stallion', 'doja cat', 'pitbull', 'flo rida', 'lil wayne', 'future',
			# Country
			'taylor swift', 'kacey musgraves', 'luke combs', 'morgan wallen','janet jackson'
			# Additional Western artists
			'kesha', 'james fauntleroy', 'backstreet boys', 'nsync',
			'onerepublic', 'imagine dragons', 'maroon 5', 'coldplay', 'nogizaka46', 'arashi', 'jay chou', 'jolin tsai', 'jj lin'
		]
        
		# DIRECT REJECTION: Check title first
		for artist in western_artists:
			if artist in title_lower:
				return 0.0  # Reject immediately
        
		# Penalty cho nghệ sĩ có quốc tịch/nơi sinh không phải Hàn Quốc
		nationality_fields = ['Quốc tịch', 'Quốc gia', 'Nguồn gốc', 'Sinh', 'Nơi sinh']
		non_korean_countries = [
			'mỹ', 'hoa kỳ', 'usa', 'united states', 'america',
			'puerto rico', 'canada', 'australia', 'new zealand',
			'anh', 'united kingdom', 'uk', 'england',
			'tây ban nha', 'spain', 'mexico', 'colombia', 'argentina',
			'brazil', 'italy', 'france', 'germany', 'pháp', 'đức',
			'thụy điển', 'sweden', 'norway', 'denmark', 'finland'
		]
        
		for field in nationality_fields:
			if field in infobox:
				field_value = str(infobox[field]).lower()
				for country in non_korean_countries:
					if country in field_value and 'hàn quốc' not in field_value:
						# Trừ điểm mạnh cho nghệ sĩ không phải Hàn Quốc
						score -= 25
        
		# Check for "của [Western artist]" pattern
		western_artist_patterns = [
			'của adele', 'của beyoncé', 'của taylor swift', 'của ed sheeran',
			'của eminem', 'của coldplay', 'của maroon 5', 'của rihanna',
			'của selena gomez', 'của ariana grande', 'của justin bieber',
			'của bad bunny', 'của doja cat', 'của billie eilish'
		]
        
		for pattern in western_artist_patterns:
			if pattern in title_lower:
				return 0.0  # Reject immediately
        
		# Strong penalty for Chinese/Taiwanese/HK markers
		chinese_indicators = ['c-pop', 'cpop', 'mandopop', 'cantopop', 'trung quốc', 'đài loan', 'hồng kông']
		if any(indicator in infobox_lower or indicator in title_lower for indicator in chinese_indicators):
			score -= 25
        
		# Check for Hangul in title or infobox
		has_hangul = self._has_hangul(node_title) or self._has_hangul(infobox_text)
		if has_hangul:
			score += 15
        
		# Check for Korean language indicators in infobox
		language_fields = ['Ngôn ngữ', 'Ngôn ngữ gốc', 'Ngôn ngữ âm nhạc', 'Language']
		for field in language_fields:
			if field in infobox:
				field_value = str(infobox[field]).lower()
				if 'tiếng hàn' in field_value or 'korean' in field_value or 'hangul' in field_value:
					score += 16
					break
				if any(lang in field_value for lang in ['tiếng trung', 'tiếng nhật', 'chinese', 'japanese', 'mandarin', 'cantonese']):
					score -= 15
        
		# Check for Korean nationality/origin in infobox
		korean_fields = ['Quốc tịch', 'Quốc gia', 'Nguồn gốc', 'Nơi sinh', 'Quê quán']
		for field in korean_fields:
			if field in infobox:
				field_value = str(infobox[field]).lower()
				if 'hàn quốc' in field_value or 'korea' in field_value:
					score += 10
					break
        
		# Check for K-pop related keywords in infobox
		if 'k-pop' in infobox_lower or 'kpop' in infobox_lower:
			score += 12
        
		# Entertainment companies (strong indicator)
		entertainment_companies = [
			'sm entertainment', 'yg entertainment', 'jyp entertainment', 
			'big hit', 'hybe', 'cube entertainment', 'starship entertainment',
			'pledis', 'fantagio', 'rbw', 'fnc entertainment'
		]
        
		for company in entertainment_companies:
			if company in infobox_lower:
				score += 10
				break
        
		# Check for music-related fields in infobox
		music_indicators = ['ca sĩ', 'nhạc sĩ', 'nhóm nhạc', 'album', 'đĩa đơn', 
						   'nghệ sĩ', 'rapper', 'producer', 'idol']
        
		for indicator in music_indicators:
			if indicator in infobox_lower:
				score += 5
				break
        
		# Check for Seoul/Korean cities
		korean_cities = ['seoul', 'busan', 'incheon', 'daegu', 'gwangju']
		if any(city in infobox_lower for city in korean_cities):
			score += 3
        
		# TẠM THỜI BỎ QUA - CRITICAL: For Artist/Group label, require STRONG Korean indicators
		# if label in ('Artist', 'Group') and not has_kpop_entity:
		# ...
        
		# TẠM THỜI BỎ QUA - Penalty for non-Korean indicators
		# if not has_kpop_entity:
		#   non_korean_markers = ['mỹ', 'anh', 'pháp', 'đức', 'nhật bản', 'trung quốc', 'việt nam', 'hoa kỳ', 'đài loan', 'hồng kông']
		#   for marker in non_korean_markers:
		#       if marker in infobox_lower:
		#           for field in korean_fields:
		#               if field in infobox and marker in str(infobox[field]).lower():
		#                   score -= 30  # Increased penalty
		#                   break
        
		# Additional penalty for TV show indicators
		tv_show_indicators = ['chương trình truyền hình', 'reality show', 'game show', 'cuộc thi']
		if any(indicator in title_lower for indicator in tv_show_indicators):
			score -= 30
        
		# 4. LOẠI BỎ HOÀN TOÀN các bài hát/album không thuộc mạng K-pop
		if label in ('Song', 'Album'):
			# Danh sách các nghệ sĩ không thuộc K-pop
			non_kpop_artists = [
				'taylor swift', 'adele', 'ed sheeran', 'justin bieber', 'ariana grande',
				'billie eilish', 'dua lipa', 'harry styles', 'olivia rodrigo', 'the weeknd',
				'alanis morissette', 'don mclean', 'janet jackson','alicia keys',
				'drake', 'kanye west', 'travis scott', 'post malone', 'lil nas x',
				'doja cat', 'megan thee stallion', 'cardi b', 'nicki minaj', 'rihanna',
				'beyonce', 'lady gaga', 'katy perry', 'miley cyrus', 'selena gomez',
				'bruno mars', 'john legend', 'alicia keys', 'mariah carey', 'whitney houston',
				'avril lavigne', 'david guetta', 'coldplay', 'maroon 5', 'eminem',
				'linkin park', 'green day', 'blink-182', 'sum 41', 'simple plan',
				'pink', 'christina aguilera', 'britney spears', 'nsync', 'backstreet boys',
				'one direction', '5 seconds of summer', 'imagine dragons', 'onerepublic',
				'fall out boy', 'panic at the disco', 'my chemical romance',
				'sia', 'sia furler', 'sia kate', 'sia kate isobel', 'sia furler',
				'cheap thrills', 'chandelier', 'elastic heart', 'titanium', 'wild ones',
				'clap your hands', 'the greatest', 'unstoppable', 'move your body',
				'helium', 'snowman', 'santa\'s coming for us', 'candy cane lane',
				'underneath the christmas lights', 'puppet on a string', 'bird set free',
				'alive', 'burn the pages', 'eye of the needle', 'fire meet gasoline',
				'free the animal', 'golden', 'hostage', 'house on fire', 'i go to sleep',
				'karma', 'let me love you', 'midnight decisions', 'never give up',
				'one million bullets', 'reaper', 'sweet design', 'the girl you lost',
				'this is acting', 'thunderclouds', 'together', 'trip the light',
				'unicorn', 'what i did for love', 'you\'re never fully dressed',
				'zombie', 'diamonds', 'we are young', 'somebody that i used to know',
				'call me maybe', 'gangnam style', 'harlem shake', 'thrift shop',
				'get lucky', 'blurred lines', 'roar', 'dark horse', 'shake it off',
				'uptown funk', 'see you again', 'lean on', 'love me like you do',
				'sugar', 'blank space', 'bad blood', 'style', 'wildest dreams',
				'out of the woods', 'new romantics', 'i don\'t wanna live forever',
				'look what you made me do', 'ready for it', 'end game', 'delicate',
				'gorgeous', 'getaway car', 'king of my heart', 'dancing with our hands tied',
				'dress', 'this is why we can\'t have nice things', 'call it what you want',
				'new year\'s day', 'me!', 'you need to calm down', 'lover', 'the man',
				'the archer', 'i think he knows', 'miss americana', 'paper rings',
				'cornelia street', 'death by a thousand cuts', 'london boy', 'soon you\'ll get better',
				'false god', 'you need to calm down', 'afterglow', 'it\'s nice to have a friend',
				'daylight', 'willow', 'champagne problems', 'gold rush', '\'tis the damn season',
				'tolerate it', 'no body no crime', 'happiness', 'dorothea', 'coney island',
				'ivy', 'cowboy like me', 'long story short', 'marjorie', 'closure',
				'evermore', 'right where you left me', 'it\'s time to go', 'anti-hero',
				'snow on the beach', 'you\'re on your own kid', 'midnight rain',
				'question...?', 'vigilante shit', 'bejeweled', 'labyrinth', 'karma',
				'sweet nothing', 'mastermind', 'the great war', 'bigger than the whole sky',
				'paris', 'high infidelity', 'glitch', 'would\'ve could\'ve should\'ve',
				'dear reader', 'hits different', 'you\'re losing me', 'slut!',
				'say don\'t go', 'now that we don\'t talk', 'suburban legends',
				'is it over now?', 'fortnight', 'the tortured poets department',
				'my boy only breaks his favorite toys', 'down bad', 'so long london',
				'but daddy i love him', 'fresh out the slammer', 'florida!!!',
				'guilty as sin?', 'who\'s afraid of little old me?', 'i can fix him',
				'loml', 'i can do it with a broken heart', 'the smallest man who ever lived',
				'the alchemy', 'clara bow', 'thanK you aIMee', 'i hate it here',
				'thanK you aIMee', 'i hate it here', 'thanK you aIMee', 'i hate it here'
			]
            
			# Kiểm tra trong title
			for artist in non_kpop_artists:
				if f' {artist} ' in f' {title_lower} ':
					# Kiểm tra xem có phải collaboration không (có cả nghệ sĩ K-pop và Western)
					has_kpop_artist = False
					kpop_artists = ['bts', 'blackpink', 'exo', 'twice', 'red velvet', 'nct', 'seventeen', 'stray kids', 'itzy', 'aespa', 'newjeans', 'big bang', "girls' generation", 'super junior', '2ne1', 'got7', 'psy', 'taeyeon', 'jennie', 'rosé', 'jisoo', 'jimin', 'jungkook', 'suga', 'j-hope']
					for kpop_artist in kpop_artists:
						if f' {kpop_artist} ' in f' {title_lower} ':
							has_kpop_artist = True
							break
                    
					# Nếu có cả K-pop và Western artist, đây là collaboration - không loại bỏ
					if has_kpop_artist:
						continue
					else:
						return 0.0  # LOẠI BỎ HOÀN TOÀN nếu chỉ có Western artist
            
			# Kiểm tra trong các trường infobox cụ thể
			artist_fields = ['Nghệ sĩ', 'Ca sĩ', 'Nhóm nhạc', 'Artist', 'Performer', 'Sản xuất', 'Producer']
			for field in artist_fields:
				if field in infobox:
					field_value = str(infobox[field]).lower()
					for artist in non_kpop_artists:
						if artist in field_value:
							# Kiểm tra xem có phải collaboration không (có cả nghệ sĩ K-pop và Western)
							has_kpop_artist = False
							kpop_artists = ['bts', 'blackpink', 'exo', 'twice', 'red velvet', 'nct', 'seventeen', 'stray kids', 'itzy', 'aespa', 'newjeans', 'big bang', "girls' generation", 'super junior', '2ne1', 'got7', 'psy', 'taeyeon', 'jennie', 'rosé', 'jisoo', 'jimin', 'jungkook', 'suga', 'j-hope']
							for kpop_artist in kpop_artists:
								if kpop_artist in field_value:
									has_kpop_artist = True
									break
                            
							# Nếu có cả K-pop và Western artist, đây là collaboration - không loại bỏ
							if has_kpop_artist:
								continue
							else:
								return 0.0  # LOẠI BỎ HOÀN TOÀN nếu chỉ có Western artist
            
			# Trừ điểm nhẹ cho Western labels (không loại bỏ hoàn toàn)
			western_labels = [
				'republic', 'atlantic', 'columbia', 'sony music', 'universal music',
				'warner music', 'interscope', 'capitol', 'emi', 'virgin',
				'def jam', 'island', 'geffen', 'arista', 'epic', 'rca',
				'elektra', 'asylum', 'atlantic records', 'columbia records'
			]
			label_fields = ['Hãng đĩa', 'Label', 'Công ty quản lý', 'Agency']
			for field in label_fields:
				if field in infobox:
					field_value = str(infobox[field]).lower()
					for label in western_labels:
						if label in field_value:
							score -= 10.0  # Trừ điểm nhẹ cho Western labels
							break
            
			# Trừ điểm cho phòng thu tại các thành phố không phải Hàn Quốc
			non_korean_cities = [
				'new york', 'los angeles', 'london', 'paris', 'berlin', 'tokyo',
				'sydney', 'toronto', 'vancouver', 'miami', 'nashville', 'atlanta',
				'chicago', 'boston', 'philadelphia', 'detroit', 'houston', 'dallas',
				'seattle', 'san francisco', 'las vegas', 'orlando', 'montreal',
				'melbourne', 'brisbane', 'perth', 'adelaide', 'auckland',
				'madrid', 'barcelona', 'rome', 'milan', 'amsterdam', 'brussels',
				'zurich', 'vienna', 'stockholm', 'oslo', 'copenhagen', 'helsinki',
				'moscow', 'prague', 'budapest', 'warsaw', 'bucharest', 'sofia',
				'athens', 'lisbon', 'dublin', 'glasgow', 'manchester', 'birmingham',
				'edinburgh', 'cardiff', 'belfast', 'singapore', 'hong kong',
				'taipei', 'bangkok', 'jakarta', 'kuala lumpur', 'manila',
				'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad',
				'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
				'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri',
				'patna', 'vadodara', 'ludhiana', 'agra', 'nashik', 'faridabad',
				'meerut', 'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar',
				'aurangabad', 'noida', 'howrah', 'ranchi', 'gwalior', 'jabalpur',
				'coimbatore', 'vijayawada', 'jodhpur', 'madurai', 'raipur',
				'kota', 'chandigarh', 'guwahati', 'solapur', 'hubli', 'tiruchirappalli',
				'bareilly', 'morang', 'mysore', 'tiruppur', 'gurgaon', 'aligarh',
				'jalandhar', 'bhubaneswar', 'salem', 'warangal', 'guntur',
				'bihta', 'amravati', 'noida', 'jamshedpur', 'bhilai', 'cuttack',
				'firozabad', 'kochi', 'bhavnagar', 'dehradun', 'durgapur',
				'asansol', 'rourkela', 'nanded', 'kolhapur', 'ajmer', 'akola',
				'gulbarga', 'jamnagar', 'ujjain', 'loni', 'siliguri', 'jhansi',
				'ulhasnagar', 'nellore', 'jammu', 'sangli', 'mira', 'belgaum',
				'mangalore', 'ambattur', 'tirunelveli', 'malegaon', 'gaya',
				'jalgaon', 'udaipur', 'maheshtala', 'davanagere', 'kozhikode',
				'kurnool', 'rajpur', 'rajahmundry', 'bokaro', 'south',
				'bellary', 'patiala', 'gopalpur', 'agartala', 'bhagalpur',
				'muzaffarnagar', 'bhatpara', 'pali', 'satna', 'mizoram',
				'bihar', 'bhiwandi', 'parbhani', 'shimla', 'berhampur',
				'ranchi', 'kadapa', 'karnal', 'bathinda', 'raichur', 'puducherry',
				'karnataka', 'tamil nadu', 'andhra pradesh', 'west bengal',
				'maharashtra', 'gujarat', 'rajasthan', 'uttar pradesh',
				'madhya pradesh', 'kerala', 'odisha', 'punjab', 'haryana',
				'himachal pradesh', 'jammu and kashmir', 'assam', 'tripura',
				'meghalaya', 'manipur', 'nagaland', 'arunachal pradesh',
				'sikkim', 'mizoram', 'goa', 'chhattisgarh', 'jharkhand',
				'uttarakhand', 'telangana', 'andaman and nicobar islands',
				'lakshadweep', 'dadra and nagar haveli', 'daman and diu',
				'chandigarh', 'delhi', 'puducherry'
			]
            
			studio_fields = ['Phòng thu', 'Studio', 'Thu âm', 'Recording']
			for field in studio_fields:
				if field in infobox:
					field_value = str(infobox[field]).lower()
                    
					# Cộng điểm cho phòng thu tại Hàn Quốc
					korean_studios = [
						'seoul', 'busan', 'incheon', 'daegu', 'gwangju', 'daejeon',
						'ulsan', 'suwon', 'changwon', 'goyang', 'yongin', 'seongnam',
						'bucheon', 'ansan', 'anyang', 'jeonju', 'cheonan', 'naju',
						'cheongju', 'gimhae', 'jeju', 'hàn quốc', 'korea', 'korean',
						'sm studio', 'yg studio', 'jyp studio', 'hybe studio', 'bighit studio',
						'cube studio', 'starship studio', 'pledis studio', 'rbw studio',
						'fnc studio', 'woollim studio', 'kakao studio', 'stone studio',
						'cj studio', 'brand new studio', 'dsp studio', 'mbk studio',
						'wm studio', 'top studio', 'swing studio', 'black label studio',
						'p nation studio', 'source studio', 'ador studio'
					]
                    
					for studio in korean_studios:
						if studio in field_value:
							score += 12.0  # Cộng điểm cho phòng thu tại Hàn Quốc
							break
                    
					# Trừ điểm cho phòng thu không phải Hàn Quốc
					for city in non_korean_cities:
						if city in field_value:
							score -= 15.0  # Trừ điểm cho phòng thu không phải Hàn Quốc
							break
            
			# Cộng điểm cho các yếu tố liên quan đến Hàn Quốc
			korean_indicators = [
				'hàn quốc', 'korea', 'korean', 'seoul', 'busan', 'incheon',
				'k-pop', 'kpop', 'hangul', '한국', '한국어', 'korean music',
				'korean entertainment', 'korean pop', 'korean hip hop',
				'korean r&b', 'korean ballad', 'korean rock', 'korean indie',
				'korean trot', 'korean folk', 'korean classical', 'korean jazz',
				'korean electronic', 'korean dance', 'korean rap', 'korean soul',
				'korean reggae', 'korean country', 'korean blues', 'korean gospel',
				'korean new age', 'korean world', 'korean experimental',
				'korean ambient', 'korean techno', 'korean house', 'korean trance',
				'korean drum and bass', 'korean dubstep', 'korean trap',
				'korean lo-fi', 'korean vaporwave', 'korean synthwave',
				'korean future bass', 'korean progressive', 'korean psytrance',
				'korean hardstyle', 'korean gabber', 'korean breakcore',
				'korean idm', 'korean glitch', 'korean noise', 'korean drone',
				'korean post-rock', 'korean post-punk', 'korean shoegaze',
				'korean dream pop', 'korean chillwave', 'korean witch house',
				'korean cloud rap', 'korean emo rap', 'korean soundcloud rap',
				'korean underground', 'korean alternative', 'korean indie pop',
				'korean indie rock', 'korean indie folk', 'korean indie electronic',
				'korean bedroom pop', 'korean diy', 'korean cassette culture',
				'korean tape', 'korean vinyl', 'korean cd', 'korean digital',
				'korean streaming', 'korean youtube', 'korean soundcloud',
				'korean bandcamp', 'korean spotify', 'korean apple music',
				'korean melon', 'korean genie', 'korean bugs', 'korean flo',
				'korean vibe', 'korean soribada', 'korean mnet', 'korean music bank',
				'korean music core', 'korean inkigayo', 'korean show champion',
				'korean the show', 'korean simply k-pop', 'korean after school club',
				'korean arirang tv', 'korean kbs world', 'korean sbs fune',
				'korean mbc every1', 'korean tvn', 'korean jtbc', 'korean channel a',
				'korean mbn', 'korean sbs', 'korean kbs', 'korean mbc',
				'korean ebs', 'korean cjb', 'korean tbc', 'korean kbc',
				'korean jtbc', 'korean channel a', 'korean mbn', 'korean sbs',
				'korean kbs', 'korean mbc', 'korean ebs', 'korean cjb',
				'korean tbc', 'korean kbc', 'korean jtbc', 'korean channel a',
				'korean mbn', 'korean sbs', 'korean kbs', 'korean mbc',
				'korean ebs', 'korean cjb', 'korean tbc', 'korean kbc'
			]
            
			# Kiểm tra trong tất cả các trường infobox
			for field, value in infobox.items():
				field_value = str(value).lower()
				for indicator in korean_indicators:
					if indicator in field_value:
						score += 8.0  # Cộng điểm cho yếu tố liên quan đến Hàn Quốc
						break
        
		# 5. KIỂM TRA DẤU HIỆU HÀN QUỐC cho Artist và Group (trừ 30 điểm nếu không có dấu hiệu)
		if label in ('Artist', 'Group'):
			# Kiểm tra các dấu hiệu Hàn Quốc cho nghệ sĩ/nhóm nhạc
			korean_signals = []
            
			# Dấu hiệu 1: Có chữ Hangul trong tên hoặc các trường infobox cụ thể
			if self._has_hangul(node_title):
				korean_signals.append("hangul_title")
            
			# Kiểm tra Hangul trong các trường infobox cụ thể
			hangul_fields = ['Tên thật', 'Real name', 'Hangul', 'Hanja', 'Romaja', 'McCune–Reischauer', 'Hepburn', 'Kunrei-shiki', 'Phiên âm', 'Chuyển tự']
			for field in hangul_fields:
				if field in infobox:
					field_value = str(infobox[field])
					if self._has_hangul(field_value):
						korean_signals.append("hangul_infobox")
						break
            
			# Dấu hiệu 2: Có từ khóa Hàn Quốc trong các trường infobox cụ thể
			korean_keywords = ['hàn quốc', 'korea', 'korean', 'k-pop', 'kpop', 'seoul', 'tiếng hàn']
			keyword_fields = ['Quốc tịch', 'Nationality', 'Nơi sinh', 'Birthplace', 'Năm hoạt động', 'Active years', 'Thể loại', 'Genre', 'Ngôn ngữ', 'Language', 'Sinh', 'Birth']
			for field in keyword_fields:
				if field in infobox:
					field_value = str(infobox[field]).lower()
					for keyword in korean_keywords:
						if keyword in field_value:
							korean_signals.append("korean_keyword")
							break
					if "korean_keyword" in korean_signals:
						break
            
			# Dấu hiệu 3: Có công ty quản lý K-pop trong các trường infobox cụ thể
			kpop_companies = [
				'sm entertainment', 'yg entertainment', 'jyp entertainment',
				'big hit', 'hybe', 'bighit music', 'source music', 'ador',
				'pledis', 'cube entertainment', 'starship entertainment', 'rbw',
				'fnc entertainment', 'woollim entertainment', 'kakao m', 'kakao entertainment',
				'loen entertainment', 'stone music', 'cj e&m', '1thek', 'brand new music',
				'dsp media', 'mbk entertainment', 'wm entertainment', 'top media',
				'swing entertainment', 'the black label', 'p nation','s.m entertainment'
			]
			company_fields = ['Hãng đĩa', 'Label', 'Công ty quản lý', 'Agency', 'Company', 'Nhãn đĩa', 'Record label']
			for field in company_fields:
				if field in infobox:
					field_value = str(infobox[field]).lower()
					for company in kpop_companies:
						if company in field_value:
							korean_signals.append("kpop_company")
							break
					if "kpop_company" in korean_signals:
						break
            
			# Dấu hiệu 4: Có thành viên K-pop trong infobox (cho Group)
			if label == 'Group':
				kpop_members = [
					'bts', 'blackpink', ' exo', 'twice', 'red velvet',
					'seventeen', 'stray kids', 'itzy', 'aespa', 'newjeans',
					'big bang', "girls' generation", 'super junior', '2ne1',
					'got7', 'psy', 'taeyeon', 'jennie', 'rosé', 'jisoo',
					'jimin', 'jungkook', 'suga', 'j-hope',
					'g-dragon', 'taeyang', 'daesung',
					'seungri', 'minho', 'onew', 'taemin', 'jonghyun',
					'yang hyun-suk', 'lee soo-man',
					'hyuna', 'sunmi', 'hani', 'moonbyul', 'wheein', 'hwasa',
					'yerin', 'eunha', 'sinb', 'umji', 'yuju', 'sowon',
					'jihyo', 'nayeon', 'jeongyeon', 'dahyun', 'chaeyoung', 'tzuyu',
					'seulgi', 'yeri',
					'ningning',
					'hanni', 'haerin', 'hyein',
					'bang chan', 'lee know', 'changbin', 'hyunjin', 'seungmin',
					'yeji', 'ryujin', 'chaeryeong', 'yuna',
					'wonyoung', 'yujin', 'gaeul', 'leeseo',
					'chaewon', 'yunjin', 'kazuha', 'eunchae',
					'heeseung', 'sunghoon', 'sunoo', 'jungwon',
					'soobin', 'yeonjun', 'beomgyu', 'taehyun', 'hueningkai',
					'hongjoong', 'seonghwa', 'yunho', 'yeosang', 'mingi', 'wooyoung', 'jongho',
					'jaehyun', 'taeil', 'taeyong', 'yuta', 'doyoung', 'winwin', 'jungwoo', 'xiaojun', 'hendery', 'renjun', 'jeno', 'haechan', 'jaemin', 'yangyang', 'shotaro', 'sungchan', 'chenle', 'jisung',
					'seungkwan', 'hoshi', 'woozi', 'mingyu', 'wonwoo', 'scoups', 'jeonghan'
				]
				member_fields = ['Thành viên', 'Members', 'Cựu thành viên', 'Former Members']
				for field in member_fields:
					if field in infobox:
						field_value = str(infobox[field]).lower()
						for member in kpop_members:
							if member in field_value:
								korean_signals.append("kpop_member")
								break
						if "kpop_member" in korean_signals:
							break
			
			# CỘNG ĐIỂM cho Artist/Group có dấu hiệu Hàn Quốc
			if korean_signals:
				# Cộng điểm cơ bản khi có bất kỳ dấu hiệu nào
				score += 15.0
				
				# Cộng điểm thêm cho từng loại dấu hiệu cụ thể
				if "hangul_title" in korean_signals:
					score += 12.0  # Bonus cao cho có Hangul trong title
				
				if "hangul_infobox" in korean_signals:
					score += 10.0  # Bonus cho có Hangul trong infobox
				
				if "korean_keyword" in korean_signals:
					score += 10.0  # Bonus cho từ khóa Hàn Quốc
				
				if "kpop_company" in korean_signals:
					score += 15.0  # Bonus cao cho công ty K-pop
				
				if "kpop_member" in korean_signals:
					score += 8.0   # Bonus cho thành viên K-pop (chỉ Group)
				
				# Bonus đặc biệt nếu có nhiều dấu hiệu (chứng tỏ rõ ràng là K-pop)
				if len(korean_signals) >= 2:
					score += 5.0   # Bonus nếu có từ 2 dấu hiệu trở lên
				if len(korean_signals) >= 3:
					score += 5.0   # Bonus thêm nếu có từ 3 dấu hiệu trở lên
            
			# Quyết định: Nếu không có dấu hiệu Hàn Quốc nào, trừ 30 điểm
			if not korean_signals:
				score -= 30.0
        
		# Đặc biệt kiểm tra cho các bài hát/album có 0 điểm
		if label in ('Song', 'Album') and score == 0.0:
				# Kiểm tra tiêu đề có chứa tên nghệ sĩ/nhóm nhạc Hàn Quốc không gây nhầm lẫn
				unambiguous_korean_artists = [
					'bts', 'blackpink', ' exo', 'twice', 'red velvet',
					'seventeen', 'stray kids', 'itzy', 'aespa', 'newjeans',
					'big bang', "girls' generation", 'super junior', '2ne1',
					'got7', 'psy', 'taeyeon', 'jennie', 'rosé', 'jisoo',
					'jimin', 'jungkook', 'suga', 'j-hope',
					'g-dragon', 'taeyang', 'daesung',
					'seungri', 'minho', 'onew', 'taemin', 'jonghyun',
					'yang hyun-suk', 'lee soo-man',
					'hyuna', 'sunmi', 'hani', 'moonbyul', 'wheein', 'hwasa',
					'yerin', 'eunha', 'sinb', 'umji', 'yuju', 'sowon',
					'jihyo', 'nayeon', 'jeongyeon', 'dahyun', 'chaeyoung', 'tzuyu',
					'seulgi', 'yeri',
					'ningning',
					'hanni', 'haerin', 'hyein',
					'bang chan', 'lee know', 'changbin', 'hyunjin', 'seungmin',
					'yeji', 'ryujin', 'chaeryeong', 'yuna',
					'wonyoung', 'yujin', 'gaeul', 'leeseo',
					'chaewon', 'yunjin', 'kazuha', 'eunchae',
					'heeseung', 'sunghoon', 'sunoo', 'jungwon',
					'soobin', 'yeonjun', 'beomgyu', 'taehyun', 'hueningkai',
					'hongjoong', 'seonghwa', 'yunho', 'yeosang', 'mingi', 'wooyoung', 'jongho',
					'jaehyun', 'taeil', 'taeyong', 'yuta', 'doyoung', 'winwin', 'jungwoo', 'xiaojun', 'hendery', 'renjun', 'jeno', 'haechan', 'jaemin', 'yangyang', 'shotaro', 'sungchan', 'chenle', 'jisung',
					'seungkwan', 'hoshi', 'woozi', 'mingyu', 'wonwoo', 'scoups', 'jeonghan'
				]
                
				# Kiểm tra tiêu đề có chứa tên nghệ sĩ/nhóm nhạc Hàn Quốc (chấp nhận dấu câu/ngoặc quanh tên)
				import re as _re
				title_has_korean_artist = False
				for artist in unambiguous_korean_artists:
					pattern = _re.compile(rf"(?<!\\w){_re.escape(artist)}(?!\\w)", _re.IGNORECASE)
					if pattern.search(title_lower):
						title_has_korean_artist = True
						break
                
				# Nếu tiêu đề có tên nghệ sĩ/nhóm nhạc Hàn Quốc, giữ lại
				if title_has_korean_artist:
					score += 20.0  # Cộng đủ điểm để vượt ngưỡng
					return score
				else:
					# Nếu không có, tiếp tục kiểm tra các dấu hiệu khác
					pass
            
        
		return score

	def get_page_soup(self, title: str) -> BeautifulSoup | None:
		"""Fetch and parse a Vietnamese Wikipedia page."""
		try:
			url = self.base_url + title.replace(" ", "_")
			response = self.session.get(url, timeout=self.request_timeout_seconds)
			response.raise_for_status()
			return BeautifulSoup(response.content, 'html.parser')
		except requests.exceptions.RequestException as e:
			print(f"  ✗ Lỗi mạng khi truy cập '{title}': {e}")
		return None

	def _compute_node_signature(self, label: str, infobox: Dict[str, str]) -> str:
		"""Tạo chữ ký duy nhất của node dựa trên label và nội dung infobox đã chuẩn hóa."""
		# Dùng mọi cặp key->value đã chuẩn hóa trong infobox (đã có normalize cho Genre/Company)
		# Sắp xếp theo key để ổn định, lowercase để so sánh bền vững
		parts = [label.strip().lower()]
		for k in sorted(infobox.keys()):
			v = str(infobox[k]).strip().lower()
			# Chuẩn hóa khoảng trắng và dấu phẩy
			v = re.sub(r"\s+", " ", v)
			v = re.sub(r",\s*", ", ", v)
			parts.append(f"{k.lower()}={v}")
		return "|".join(parts)

	def extract_infobox(self, soup: BeautifulSoup) -> Dict[str, str]:
		"""Extract key-value pairs from the right-side infobox table with smart separator."""
		infobox_data: Dict[str, str] = {}
		infobox_table = soup.find('table', class_='infobox')
		if infobox_table:
			for row in infobox_table.find_all('tr'):
				header = row.find('th')
				data = row.find('td')
				if header and data:
					key = header.get_text(strip=True)
					# Chuẩn hoá khoảng trắng không ngắt (NBSP) trong tiêu đề trường
					key = key.replace('\xa0', ' ')
                    
					# Ưu tiên: Với các trường Website/URL, chỉ lấy các liên kết (<a>)
					if key in ['Website', 'URL', 'Trang web']:
						anchor_texts: list[str] = []
						for a in data.find_all('a', href=True):
							# Bỏ qua các liên kết chú thích nằm trong <sup> hoặc dạng [1], [2]
							if a.find_parent('sup') is not None:
								continue
							text = a.get_text(strip=True)
							if re.fullmatch(r"\[\d+\]", text or ''):
								continue
							href = a.get('href', '')
							if 'cite_note' in href or '#cite_note' in href:
								continue
							candidate = text or href
							if candidate:
								anchor_texts.append(candidate)
						# Loại trùng và nối bằng dấu phẩy
						seen_w = set()
						unique_webs = []
						for t in anchor_texts:
							if t not in seen_w:
								seen_w.add(t)
								unique_webs.append(t)
						infobox_data[key] = ', '.join(unique_webs)
						continue

					# Ưu tiên: Với các trường công ty/label, chỉ lấy các mục có liên kết (<a>)
					if key in ['Hãng đĩa', 'Công ty quản lý', 'Hãng đĩa/Label', 'Label', 'Agency']:
						anchor_texts = []
						for a in data.find_all('a', href=True):
							# Bỏ qua các liên kết chú thích dạng [1], [2] hoặc nằm trong <sup>
							if a.find_parent('sup') is not None:
								continue
							text = a.get_text(strip=True)
							if re.fullmatch(r"\[\d+\]", text or ''):
								continue
							href = a.get('href', '')
							if 'cite_note' in href or '#cite_note' in href:
								continue
							if text:
								anchor_texts.append(text)
						# Chuẩn hóa tên công ty, loại trùng theo tên chuẩn hóa và nối bằng dấu phẩy
						seen = set()
						unique_texts = []
						for t in anchor_texts:
							norm = self._normalize_company_name(t)
							if norm not in seen:
								seen.add(norm)
								unique_texts.append(norm)
						infobox_data[key] = ', '.join(unique_texts)
						continue

		# Ưu tiên: Với các trường thể loại, chỉ lấy các mục có liên kết (<a>)
					if key in ['Thể loại', 'Genre', 'Genres']:
						genre_texts = []
						for a in data.find_all('a', href=True):
							# Bỏ qua các liên kết chú thích dạng [1], [2] hoặc nằm trong <sup>
							if a.find_parent('sup') is not None:
								continue
							text = a.get_text(strip=True)
							if re.fullmatch(r"\[\d+\]", text or ''):
								continue
							href = a.get('href', '')
							if 'cite_note' in href or '#cite_note' in href:
								continue
							if text:
								# Chuẩn hóa thể loại ngay từ đầu
								normalized_text = self._normalize_genre_name(text)
								# Bỏ qua non-music genres
								if normalized_text and normalized_text.strip().lower() not in self.non_music_genres:
									genre_texts.append(normalized_text)
						# Loại trùng và nối bằng dấu phẩy
						seen_g = set()
						unique_genres = []
						for t in genre_texts:
							if t not in seen_g:
								seen_g.add(t)
								unique_genres.append(t)
						# Fallback: nếu không có link nào, lấy text thuần và tách theo dấu phẩy/chấm phẩy
						if not unique_genres:
							raw = data.get_text(separator=' ', strip=True)
							# Chuẩn hóa các dấu phân tách: '·', '•', '・' -> ','
							raw = raw.replace('·', ',').replace('•', ',').replace('・', ',')
							# Tách riêng trường hợp "Tropical House Deep House" -> "Tropical House, Deep House"
							raw = re.sub(r"(?i)\b(Tropical\s+House)\s+(Deep\s+House)\b", r"\1, \2", raw)
							# bỏ chú thích [1], [2]
							raw = re.sub(r"\[\s*\d+\s*\]", '', raw)
							# tách theo , ; /
							parts = re.split(r"[,;/]", raw)
							for p in parts:
								name = p.strip()
								if name and name.lower() not in ('âm nhạc'):
										# Chuẩn hóa thể loại ngay từ đầu
										normalized_name = self._normalize_genre_name(name)
										if normalized_name and normalized_name.strip().lower() not in self.non_music_genres and normalized_name not in seen_g:
											seen_g.add(normalized_name)
											unique_genres.append(normalized_name)
					# Fallback: nếu là Album/Song và tiêu đề có dạng "(album của X)/(bài hát của X)" thì set "Được thực hiện bởi"
					if key in ['Mô tả album', 'description'] and not infobox_data.get('Được thực hiện bởi'):
						title_l = title.lower()
						m = re.search(r"\((?:album|bài hát) của ([^)]+)\)", title_l)
						if m:
							perf = m.group(1).strip()
							infobox_data['Được thực hiện bởi'] = perf
						infobox_data[key] = ', '.join(unique_genres)
						continue

					# Ưu tiên: Với các trường nhạc cụ, lấy cả <a> và text, tách theo các dấu phân tách phổ biến
					if key in ['Nhạc cụ', 'Instrument', 'Instruments']:
						cand: list[str] = []
						# Thu thập từ anchor
						for a in data.find_all('a', href=True):
							if a.find_parent('sup') is not None:
								continue
							text = a.get_text(strip=True)
							if re.fullmatch(r"\[\d+\]", text or ''):
								continue
							if text:
								cand.append(text)
						# Thu thập từ raw text
						raw = data.get_text(separator=' ', strip=True)
						# Chuẩn hóa dấu phân tách: chấm giữa, bullet, dot, dấu gạch…
						raw = raw.replace('·', ',').replace('•', ',').replace('・', ',').replace(' - ', ', ')
						# Loại chú thích [1]
						raw = re.sub(r"\[\s*\d+\s*\]", '', raw)
						for part in re.split(r"[,/;]", raw):
							p = part.strip()
							if p:
								cand.append(p)
						# Chuẩn hóa, loại trùng và loại giá trị không hợp lệ
						seen_i = set()
						unique_i = []
						for item in cand:
							norm = self._normalize_instrument_name(item)
							if not norm:
								continue
							key_norm = norm.lower()
							if key_norm not in seen_i:
								seen_i.add(key_norm)
								unique_i.append(norm)
						infobox_data[key] = ', '.join(unique_i)
						continue

					# Ưu tiên: Với các trường thành viên/cựu thành viên, chỉ lấy tên có liên kết (<a>)
					if key in ['Thành viên', 'Cựu thành viên', 'Members', 'Former Members']:
						member_texts = []
						for a in data.find_all('a', href=True):
							# Bỏ qua liên kết chú thích hoặc nằm trong <sup>
							if a.find_parent('sup') is not None:
								continue
							text = a.get_text(strip=True)
							if re.fullmatch(r"\[\d+\]", text or ''):
								continue
							href = a.get('href', '')
							if 'cite_note' in href or '#cite_note' in href:
								continue
							if text:
								# Loại bỏ ký hiệu như †, *, • ở đầu/cuối
								clean = re.sub(r'^[\u2020\u2021\*•\u00B7\u2022\s]+|[\u2020\u2021\*•\u00B7\u2022\s]+$', '', text)
								if clean:
									member_texts.append(clean)

						seen_m = set()
						unique_members = []
						for t in member_texts:
							if t not in seen_m:
								seen_m.add(t)
								unique_members.append(t)
						infobox_data[key] = ', '.join(unique_members)
						continue

					# Ưu tiên: Với trường Sáng tác (composer/lyricist), chỉ lấy các mục có liên kết (<a>)
					if key in ['Sáng tác', 'Nhạc sĩ', 'Tác giả', 'Composer', 'Lyricist', 'Writers']:
						writer_texts = []
						for a in data.find_all('a', href=True):
							if a.find_parent('sup') is not None:
								continue
							text = a.get_text(strip=True)
							if re.fullmatch(r"\[\d+\]", text or ''):
								continue
							href = a.get('href', '')
							if 'cite_note' in href or '#cite_note' in href:
								continue
							if text:
								writer_texts.append(text.strip('"'))

						seen_writers = set()
						unique_writers = []
						for t in writer_texts:
							if t not in seen_writers:
								seen_writers.add(t)
								unique_writers.append(t)
						infobox_data[key] = ', '.join(unique_writers)
						continue

					# Ưu tiên: Với trường Sản xuất, chỉ lấy các mục có liên kết (<a>)
					if key in ['Sản xuất', 'Producer', 'Production']:
						producer_texts = []
						for a in data.find_all('a', href=True):
							if a.find_parent('sup') is not None:
								continue
							text = a.get_text(strip=True)
							if re.fullmatch(r"\[\d+\]", text or ''):
								continue
							href = a.get('href', '')
							if 'cite_note' in href or '#cite_note' in href:
								continue
							if text:
								producer_texts.append(text.strip('"'))

						seen_producers = set()
						unique_producers = []
						for t in producer_texts:
							if t not in seen_producers:
								seen_producers.add(t)
								unique_producers.append(t)
						infobox_data[key] = ', '.join(unique_producers)
						continue
                    
					# Chiến lược: Thêm separator chỉ cho <a> tags
					# Clone td để không ảnh hưởng DOM gốc
					from copy import copy
					data_copy = copy(data)
                    
					# Thêm marker sau mỗi </a> tag để tách
					for link in data_copy.find_all('a'):
						link.insert_after('|||SEPARATOR|||')
                    
					# Lấy text với separator
					value = data_copy.get_text(strip=True)
                    
					# Thay separator bằng dấu phẩy
					value = value.replace('|||SEPARATOR|||', ', ')

					# Loại bỏ ký hiệu chú thích dạng [1], [2], và các nhãn như [cần dẫn nguồn], [citation needed]
					value = re.sub(r"\[\s*\d+\s*\]", '', value)
					value = re.sub(r"\[\s*citation needed\s*\]", '', value, flags=re.IGNORECASE)
					value = re.sub(r"\[\s*cần dẫn nguồn\s*\]", '', value, flags=re.IGNORECASE)
                    
					# Cleanup các ký tự thừa
					value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
					value = re.sub(r',\s*,+', ', ', value)  # Remove double commas
                    
					# Xử lý các trường đặc biệt
					# 1. Hangul, Hanja, Kanji, Hiragana -> KHÔNG tách
					if key in ['Hangul', 'Hanja', 'Kanji', 'Hiragana', 'Romaja quốc ngữ', 'McCune–Reischauer', 'Hepburn', 'Kunrei-shiki']:
						value = value.replace(', ', '')  # Gộp lại
                    
					# 2. Phiên âm, Chuyển tự -> Loại bỏ phần đầu trùng với tên trường
					elif key in ['Phiên âm', 'Chuyển tự']:
						# Loại bỏ phần đầu trùng với tên trường
						value = value.replace(key + ', ', '', 1)  # Chỉ thay thế lần đầu
						value = value.replace(key, '', 1)  # Loại bỏ nếu không có dấu phẩy
						# Sau khi loại bỏ phần đầu, áp dụng camelCase splitting để tách các từ dính
						value = re.sub(r'([a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ])([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ])', r'\1, \2', value)
						value = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1, \2', value)
                    
					# 2. Website, URL -> đã xử lý ở trên bằng anchor-only
					elif key in ['Website', 'URL', 'Trang web']:
						pass
                    
					# 2.5. Hãng đĩa -> Tách các công ty dính liền
					elif key in ['Hãng đĩa', 'Công ty quản lý']:
						# Tách các công ty dính liền: "Label V (Trung Quốc)SM True (Thái Lan)SM IPP (Việt Nam)" -> "Label V (Trung Quốc), SM True (Thái Lan), SM IPP (Việt Nam)"
						value = re.sub(r'\)\s*([A-Z][A-Za-z\s]+)', r'), \1', value)
                    
					# 3. Năm hoạt động, Phát hành, Sinh -> cleanup ngoặc
					elif key in ['Năm hoạt động', 'Phát hành', 'Sinh']:
						value = re.sub(r',\s*\(', ' (', value)
						value = re.sub(r',\s*\)', ')', value)
						value = re.sub(r'\(\s*,', '(', value)
						value = re.sub(r'(\d{4})\s*\(', r'\1 (', value)  # "2016(2016)" -> "2016 (2016)"
						# Loại bỏ cụm "(xem lịch sử phát hành)" với hoặc không có khoảng trắng
						value = re.sub(r"\s*\(xem\s*lịch\s*sử\s*phát\s*hành\)\s*", '', value, flags=re.IGNORECASE)
                        
						# Xử lý riêng cho từng trường
						if key == 'Sinh':
							# Tách tên và ngày sinh: "Lee Dong-min30 tháng 3" -> "Lee Dong-min; 30 tháng 3"
							value = re.sub(r'([a-zA-ZÀ-ỹ]+)(\d+)', r'\1; \2', value)
							# Tách tuổi và địa điểm: "28 tuổi)Gunpo" -> "28 tuổi); Gunpo"
							value = re.sub(r'(\d+ tuổi)\)([A-Za-zÀ-ỹ])', r'\1); \2', value)
                        
						elif key == 'Năm hoạt động':
							# Loại bỏ năm lặp: "2013 (2013)–nay" -> "2013–nay"
							value = re.sub(r'(\d{4})\s*\(\1\)', r'\1', value)
							# Tách khoảng thời gian dính: "2007–20172022–nay" -> "2007–2017, 2022–nay"
							value = re.sub(r'(\d{4})–(\d{4})(\d{4})–', r'\1–\2, \3–', value)
                    
					# 4. Thời lượng -> format time
					elif key in ['Thời lượng']:
						# Chuẩn hoá dấu ':'
						value = re.sub(r',\s*:\s*,', ':', value)
						value = re.sub(r'(\d+)\s*,\s*:\s*,\s*(\d+)', r'\1:\2', value)
						value = re.sub(r'(\d+)\s*:\s*(\d+)', r'\1:\2', value)  # Ensure no space in time
						# Hỗ trợ mm:ss và hh:mm:ss
						TIME = r'(?:\d+:\d{2}(?::\d{2})?)'
						# Thêm space trước ngoặc ngay sau time: "48:15(" -> "48:15 ("
						value = re.sub(rf'({TIME})\s*\(', r'\1 (', value)
						# Chèn dấu phẩy giữa ")" và time kế tiếp: ")1:06:18" -> "), 1:06:18"
						value = re.sub(rf'\)\s*({TIME})', r'), \1', value)
						# Chèn dấu phẩy giữa hai times dính nhau: "30:4345:52" -> "30:43, 45:52"
						value = re.sub(rf'({TIME})({TIME})', r'\1, \2', value)
                    
                    
					# Loại bỏ comma quanh dấu ngoặc tròn và vuông (áp dụng chung cho mọi trường)
					value = re.sub(r',\s*\(', ' (', value)
					value = re.sub(r'\(\s*,', '(', value)
                    
					# Loại bỏ comma quanh dấu ngoặc vuông
					value = re.sub(r',\s*\[', ' [', value)
					value = re.sub(r'\]\s*,', '], ', value)
					value = re.sub(r'\[\s*,', '[', value)
					value = re.sub(r',\s*\]', ']', value)
                    
					# Tách camelCase cho tất cả trường (trừ các trường đặc biệt)
					if key not in ['Hangul', 'Hanja', 'Kanji', 'Hiragana', 'Website', 'URL', 'Phiên âm', 'Chuyển tự', 'Thể loại', 'Genre', 'Genres']:
						# Tách chữ thường + chữ hoa (camelCase)
						value = re.sub(r'([a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ])([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ])', r'\1, \2', value)
						# Tách chữ hoa liền (như "MCBTS" -> "MC, BTS")
						value = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1, \2', value)
                    
					# Loại bỏ trailing comma
					value = re.sub(r',\s*$', '', value)
                    
					# Loại bỏ leading comma
					value = re.sub(r'^\s*,\s*', '', value)
                    
					# Cleanup double spaces
					value = re.sub(r'\s{2,}', ' ', value)
                    
					infobox_data[key] = value.strip()
                    
		# Thu thập description cho Album và Song
		description = self.extract_album_description_only(soup)
		if description:
			infobox_data['description'] = description
            
		# Thu thập thông tin "được thực hiện bởi" từ các link sau chữ "của"
		performed_by = self.extract_performed_by_from_description(soup)
		if performed_by:
			infobox_data['Được thực hiện bởi'] = performed_by
            
		# Thu thập thông tin mô tả album cho Song (nếu có "từ album")
		album_info = self.extract_album_description(soup)
		if album_info:
			infobox_data['Mô tả album'] = album_info['description_text']
			if album_info['album_name']:
				infobox_data['Tên album'] = album_info['album_name']
            
		return infobox_data

	def extract_album_description_only(self, soup: BeautifulSoup) -> str:
		"""
		Chỉ trích xuất album_description từ raw HTML
		"""
		# Tìm thẻ có class "infobox-header description"
		description_header = soup.find('th', class_='infobox-header description')
        
		if not description_header:
			return ""
        
		# Lấy text trực tiếp
		raw_text = description_header.get_text()
		clean_text = re.sub(r'\s+', ' ', raw_text).strip()
        
		return clean_text
    
	def extract_performed_by_from_description(self, soup: BeautifulSoup) -> str:
		"""
		Trích xuất thông tin "được thực hiện bởi" từ các link sau chữ "của" trong description.
		Nếu có nhiều hơn 1 liên kết sau chữ "của" thì nối bằng dấu phẩy.
		"""
		# Tìm thẻ có class "infobox-header description"
		description_header = soup.find('th', class_='infobox-header description')
        
		if not description_header:
			return ""
        
		# Tìm tất cả các link trong description
		links = description_header.find_all('a', href=True)
        
		# Tìm link sau chữ "của"
		description_text = description_header.get_text()
		if 'của' in description_text:
			# Vị trí bắt đầu của cụm sau chữ "của"
			of_index = description_text.find('của')
			collected: List[str] = []
			for link in links:
				text = link.get_text(strip=True)
				if not text:
					continue
				# Kiểm tra vị trí xuất hiện đầu tiên của text trong mô tả
				pos = description_text.find(text)
				if pos > of_index:
					collected.append(text)
			# Nối các link sau chữ "của"
			if collected:
				# Loại trùng theo thứ tự
				seen = set()
				ordered_unique = []
				for t in collected:
					if t not in seen:
						seen.add(t)
						ordered_unique.append(t)
				return ', '.join(ordered_unique)
		
		return ""
    
	def extract_album_description(self, soup: BeautifulSoup) -> dict:
		"""
		Trích xuất thông tin mô tả album từ raw HTML
		Trả về: {"description_text": "...", "album_name": "...", "links": [...]}
		"""
		# Tìm tất cả các thẻ có class "infobox-header description"
		description_headers = soup.find_all('th', class_='infobox-header description')
        
		for header in description_headers:
			# Lấy text thuần túy
			text = header.get_text(strip=True)
            
			# Kiểm tra xem có chứa "từ album", "từalbum", "từ EP", "từep" không
			if any(keyword in text for keyword in ["từ album", "từalbum", "từ EP", "từep"]):
				# Làm sạch text
				clean_text = re.sub(r'\s+', ' ', text).strip()
                
				# Trích xuất các link trong description
				links = []
				for link in header.find_all('a'):
					links.append({
						"text": link.get_text(strip=True),
						"href": link.get('href', ''),
						"title": link.get('title', '')
					})
                
				# Trích xuất tên album từ links (loại trừ "album", "EP", "ep")
				album_names = []
				for link in links:
					link_text = link['text']
					if link_text and link_text.lower() not in ['album', 'ep', 'ep']:
						album_names.append(link_text)
                
				# Nối các tên album bằng dấu phẩy
				album_name = ', '.join(album_names) if album_names else None
                
				return {
					"description_text": clean_text,
					"album_name": album_name,
					"links": links
				}
        
		return None
    
	# Đã xóa classify_relation_by_labels() và classify_relation() - không dùng nữa
	# Thay bằng _determine_precise_relation() với logic kiểm tra Infobox chính xác

	def classify_label(self, title: str, infobox: Dict[str, str], page_text_lower: str) -> str:
		"""Classify strictly into one of 4 labels: Artist, Group, Album, Song using scoring."""
		infobox_text = ' '.join(infobox.values()).lower()
		title_lower = title.lower()
		description_lower = str(infobox.get('description', '')).lower()

		# Nếu tiêu đề rõ ràng là công ty/agency/entertainment → gán nhãn Company ngay
		company_title_markers = ['entertainment', 'music and live', 'division', 'company', 'công ty', 'agency', 'co., ltd', 'corp', 'inc.', 'holding', 'subsidiary']
		if any(k in title_lower for k in company_title_markers):
			return 'Company'

		# Initialize scores
		scores = {
			'Artist': 0.0,
			'Group': 0.0,
			'Album': 0.0,
			'Song': 0.0,
		}

		# Company detection by content: if strong company markers and weak music-person markers -> Company
		company_markers = ['entertainment', 'music and live', 'division', 'agency', 'company', 'công ty', 'trụ sở', 'thành lập', 'người sáng lập', 'founder', 'headquarters', 'parent', 'subsidiary', 'chủ sở hữu', 'công ty mẹ', 'dịch vụ']
		person_markers = ['ca sĩ', 'nhạc sĩ', 'diễn viên', 'rapper']
		if any(k in title_lower for k in company_markers) or (any(k in infobox_text for k in company_markers) and not any(k in infobox_text for k in person_markers)):
			return 'Company'

		# Title-based cues
		if '(bài hát' in title_lower or '(song)' in title_lower:
			scores['Song'] += 50
		if '(album' in title_lower or 'album' in title_lower:
			scores['Album'] += 45
		if '(ep)' in title_lower:
			scores['Album'] += 35
		if '(nhóm nhạc)' in title_lower or '(ban nhạc)' in title_lower or '(group)' in title_lower:
			scores['Group'] += 40
		if '(ca sĩ)' in title_lower or '(singer)' in title_lower or '(rapper)' in title_lower:
			scores['Artist'] += 40

		# Infobox field presence
		if 'Thành viên' in infobox or 'Members' in infobox or 'Cựu thành viên' in infobox:
			scores['Group'] += 35
		if 'Nghề nghiệp' in infobox:
			occupation = str(infobox.get('Nghề nghiệp', '')).lower()
			if any(k in occupation for k in ['ca sĩ', 'nhạc sĩ', 'rapper', 'nghệ sĩ']):
				scores['Artist'] += 25
				if 'thành viên' in occupation or 'Cựu thành viên' in occupation:
					scores['Group'] += 10
		if any(f in infobox for f in ['Nghệ sĩ', 'Ca sĩ', 'Nhóm nhạc', 'Artist', 'Performer']):
			# Could be Album or Song, disambiguate by other signals below
			scores['Album'] += 15
			scores['Song'] += 20
		if any(f in infobox for f in ['Thời lượng', 'Running time']):
			scores['Song'] += 20
			scores['Album'] += 10
		if any(f in infobox for f in ['Phát hành', 'Release date']):
			scores['Album'] += 20
			scores['Song'] += 15
		if any(f in infobox for f in ['Hãng đĩa', 'Label', 'Công ty quản lý', 'Agency']):
			scores['Album'] += 12
			scores['Song'] += 10

		# Description-based cues (infobox['description'] extracted earlier)
		if any(k in description_lower for k in ['album phòng thu', 'album tuyển tập', 'mini album', 'ep']):
			scores['Album'] += 35
		if any(k in description_lower for k in ['đĩa đơn', 'single', 'bài hát']):
			scores['Song'] += 35

		# Album title pattern "(album của X)" – cộng điểm mạnh cho album của nghệ sĩ K-pop
		if 'album của' in title_lower:
			# Ưu tiên seed artists
			for seed_artist in self.seed_artists:
				seed_lower = seed_artist.lower().replace('(nhóm nhạc)', '').replace('(ca sĩ)', '').strip()
				if seed_lower and seed_lower in title_lower:
					scores['Album'] += 30
					break
			# Known K-pop entities
			for entity in ['bts', 'blackpink', 'exo', 'twice', 'red velvet', 'nct', 'seventeen', 'itzy', 'aespa', 'newjeans', 'big bang', 'rosé', 'rose', 'iu', 'psy']:
				if entity in title_lower:
					scores['Album'] += 25
					break

		# Avoid misclassification for organizations/charts
		if any(k in title_lower for k in ['company', 'công ty', 'official charts', 'chart', 'charts', 'entertainment', 'agency']):
			# Penalize song/album significantly
			scores['Song'] -= 40
			scores['Album'] -= 30

		# Strong Korean music context can lightly favor Artist/Group over misc
		if any(k in infobox_text for k in ['k-pop', 'kpop', 'ca sĩ', 'nhóm nhạc']):
			scores['Artist'] += 5
			scores['Group'] += 5

		# Final selection with tie-breakers
		# Prefer Artist/Group when tied with Album/Song to keep entities as people/groups
		ordered_labels = ['Artist', 'Group', 'Album', 'Song']
		best_label = max(scores.items(), key=lambda x: (x[1], -ordered_labels.index(x[0])))[0]
		return best_label

	@staticmethod
	def _is_allowed_label(label: str) -> bool:
		return label in ('Artist', 'Group', 'Album', 'Song', 'Genre', 'Instrument', 'Occupation', 'Company')

	@staticmethod
	def _is_allowed_edge(source_label: str, target_label: str, relation_type: str) -> bool:
		"""Check if edge type is allowed between source and target labels."""
		# Artist -> Album (PRODUCED_ALBUM)
		if source_label == 'Artist' and target_label == 'Album':
			return relation_type in ('PRODUCED_ALBUM', 'RELATED_TO')
		# Artist -> Song (PRODUCED_SONG)
		if source_label == 'Artist' and target_label == 'Song':
			return relation_type in ('PRODUCED_SONG', 'WROTE', 'RELATED_TO')
		# Artist -> Occupation (HAS_OCCUPATION)
		if source_label == 'Artist' and target_label == 'Occupation':
			return relation_type in ('HAS_OCCUPATION', 'RELATED_TO')
		# Artist/Group to Album (RELEASED)
		if source_label in ('Artist', 'Group') and target_label == 'Album':
			return relation_type in ('RELEASED', 'RELATED_TO')
		
		# Artist/Group to Song (SINGS)
		if source_label in ('Artist', 'Group') and target_label == 'Song':
			return relation_type in ('SINGS', 'RELATED_TO')
		
		# Album to Song (CONTAINS)
		if source_label == 'Album' and target_label == 'Song':
			return relation_type in ('CONTAINS', 'RELATED_TO')
		
		# Song to Album (PART_OF_ALBUM)
		if source_label == 'Song' and target_label == 'Album':
			return relation_type in ('PART_OF_ALBUM', 'RELATED_TO')
		
		# Artist to Group (MEMBER_OF)
		if source_label == 'Artist' and target_label == 'Group':
			return relation_type in ('MEMBER_OF', 'RELATED_TO')
		
		# Artist/Group to Company (MANAGED_BY)
		if source_label in ('Artist', 'Group') and target_label == 'Company':
			return relation_type in ('MANAGED_BY', 'RELATED_TO')
		
		# Artist/Group/Album/Song to Award (WON_AWARD)
		if source_label in ('Artist', 'Group', 'Album', 'Song') and target_label == 'Award':
			return relation_type in ('WON_AWARD', 'RELATED_TO')
		
		# Group to Group (SUB_UNIT_OF, COLLABORATED_WITH)
		if source_label == 'Group' and target_label == 'Group':
			return relation_type in ('SUB_UNIT_OF', 'COLLABORATED_WITH', 'RELATED_TO')
		
		# Artist to Artist (COLLABORATED_WITH)
		if source_label == 'Artist' and target_label == 'Artist':
			return relation_type in ('COLLABORATED_WITH', 'RELATED_TO')
		
		# Artist/Group to Genre (IS_GENRE)
		if source_label in ('Artist', 'Group') and target_label == 'Genre':
			return relation_type in ('IS_GENRE', 'RELATED_TO')
		
		# Artist to Instrument (PLAYS)
		if source_label == 'Artist' and target_label == 'Instrument':
			return relation_type in ('PLAYS', 'RELATED_TO')
		
		return False

	def is_relevant_page(self, title: str, soup: BeautifulSoup, is_seed: bool = False) -> bool:
		"""Accept only Korean-scope music entities (artist/group/album/song/company/award)."""
		# Always keep seeds regardless of checks
		if is_seed:
			return True
		full_text = soup.get_text(separator=' ')
		text_lower = full_text.lower()
		title_lower_only = title.lower()
		# Hard-ban specific non-targets
		if 'fifth harmony' in title_lower_only:
			return False
		infobox = self.extract_infobox(soup)
		infobox_text = ' '.join(infobox.values()).lower()

		# Loại trang về Họ (surname)
		surname_markers = ['họ người', 'họ (họ người)', 'là một họ', 'họ tiếng', 'họ trong tiếng']
		if any(m in text_lower for m in surname_markers) and not any(k in text_lower for k in ['ca sĩ','nhạc sĩ','nhóm nhạc','album','bài hát']):
			return False

		# Quick label guess
		label = self.classify_label(unquote(title), infobox, text_lower)

		# Korean markers
		has_hangul = self._has_hangul(unquote(title)) or self._has_hangul(full_text) or self._has_hangul(' '.join(infobox.values()))
		is_korean_marker = ('hàn quốc' in text_lower) or ('hàn quốc' in infobox_text) or ('k-pop' in text_lower)
		is_korean_national = any('hàn quốc' in infobox.get(kf, '').lower() for kf in ['Quốc tịch', 'Quốc gia', 'Nguồn gốc', 'Nơi sinh', 'Quê quán'])
		is_korean = has_hangul or is_korean_marker or is_korean_national

		# Music/Entertainment marker
		is_music = any(k in infobox_text for k in ['ca sĩ', 'nhạc sĩ', 'nhóm nhạc', 'album', 'bài hát', 'đĩa đơn', 'entertainment']) or \
			any(k in text_lower for k in ['ca sĩ', 'nhạc sĩ', 'nhóm nhạc', 'album', 'k-pop'])

		# Exclude generic pages
		exclude_markers = ['wikipedia:', 'tập tin:', 'đặc biệt:', 'chủ đề:', 'thể loại:']
		title_lower = title.lower()
		if any(m in title_lower for m in exclude_markers):
			return False

		# Apply rules per label
		if label == 'Artist':
			# BẮT BUỘC có trường nghề nghiệp và phải thuộc âm nhạc
			occupation_fields = ['Nghề nghiệp', 'Occupation', 'Nghề', 'Professions']
			music_occu_keywords = [
				'ca sĩ', 'ca si', 'singer', 'vocalist', 'rapper', 'nhạc sĩ', 'nhac si',
				'nhà sản xuất âm nhạc', 'producer', 'music producer', 'composer', 'songwriter',
				'idol', 'thần tượng', 'dancer'
			]
			has_occupation_field = any(f in infobox for f in occupation_fields)
			if not has_occupation_field:
				return False
			has_occupation_music = False
			for f in occupation_fields:
				if f in infobox:
					val = str(infobox[f]).lower()
					if any(k in val for k in music_occu_keywords):
						has_occupation_music = True
						break
			if not has_occupation_music:
				return False

			# Loại các nhân vật lịch sử/hoàng gia không thuộc âm nhạc
			history_indicators_text = [
				'vua', 'hoàng đế', 'hoàng hậu', 'thái tử', 'vương', 'quý tộc',
				'triều đại', 'vương triều', 'triều joseon', 'triều goryeo', 'baekje', 'silla',
				'sinh', 'mất', 'trị vì', 'lên ngôi', 'miếu hiệu', 'thụy hiệu', 'tôn hiệu', 'hậu duệ', 'tổ tiên', 'phối ngẫu', 'con cái'
			]
			history_fields = [
				'Triều đại', 'Triều', 'Dynasty', 'Reign', 'Miếu hiệu', 'Temple name',
				'Thụy hiệu', 'Posthumous name', 'Tôn hiệu', 'Era name', 'Regnal name',
				'Phối ngẫu', 'Spouse', 'Hậu duệ', 'Issue', 'Con cái', 'Children',
				'Sinh', 'Birth', 'Mất', 'Death'
			]
			is_historical = any(ind in text_lower for ind in history_indicators_text)
			for hf in history_fields:
				if hf in infobox:
					is_historical = True
					break
			if is_historical and not has_occupation_music:
				return False

			# Loại các vận động viên/cầu thủ bóng đá không thuộc mạng lưới âm nhạc
			sports_indicators_text = ['cầu thủ bóng đá', 'bóng đá', 'footballer', 'soccer player']
			sports_fields = [
				'Đội hiện nay', 'Số áo', 'Vị trí', 'Câu lạc bộ', 'Đội tuyển quốc gia',
				'Sự nghiệp cầu thủ', 'Sự nghiệp trẻ', 'Ghi bàn', 'Bàn thắng', 'Bàn thua',
				'Chiều cao', 'Cân nặng', 'Huấn luyện viên', 'Youth career', 'Senior career',
				'Current team', 'Club number', 'Position'
			]
			is_sports = any(ind in text_lower for ind in sports_indicators_text)
			for sf in sports_fields:
				if sf in infobox:
					is_sports = True
					break
			# Loại tuyển thủ eSports
			esports_indicators_text = [
				'esports', 'thể thao điện tử', 'game thủ', 'tuyển thủ', 'pro gamer',
				'league of legends', 'liên minh huyền thoại', 'lck', 't1', 'faker',
				'dota 2', 'overwatch', 'starcraft', 'pubg', 'valorant'
			]
			esports_fields = [
				'Đội tuyển', 'Tổ chức', 'Tuyển thủ', 'Trò chơi', 'Game', 'Vai trò', 'Role',
				'Giải đấu', 'League', 'Vị trí thi đấu', 'Main role', 'Coach', 'Huấn luyện viên trưởng'
			]
			is_esports = any(ind in text_lower for ind in esports_indicators_text)
			for ef in esports_fields:
				if ef in infobox:
					is_esports = True
					break
			if is_sports or is_esports:
				return False
			return is_music and is_korean
		elif label == 'Group':
			return is_music and is_korean
		elif label in ('Album', 'Song'):
			# Loại các album/bài hát nước ngoài (OST phương Tây, không dấu hiệu Hàn)
			foreign_music_indicators = [
				'original motion picture soundtrack', 'soundtrack',
				'hollywood', 'disney', 'warner bros', 'universal music', 'atlantic records',
				'columbia records', 'republic records', 'sony music', 'united states', 'usa', 'anh', 'uk', 'canada', 'mỹ'
			]
			if any(x in title_lower_only for x in foreign_music_indicators) or any(x in infobox_text for x in foreign_music_indicators):
				if not is_korean:
					return False
			# Chấp nhận chỉ khi có dấu hiệu Hàn rõ ràng
			return is_music and (has_hangul or is_korean_marker)
		elif label == 'Company':
			return False  # Không thu thập Company như một node chính
		elif label == 'Award':
			return False  # Không thu thập Award như một node chính
		else:
			return False

	def is_excluded_title(self, title: str) -> bool:
		"""Filter out disambiguation, list, year, and overly generic titles early."""
		t = title.strip().lower()
		# Hard-ban specific titles
		if 'fifth harmony' in t:
			return True
		# Ban award pages by common markers
		award_markers = [
			'giải mama', 'giải thưởng', 'giải thưởng âm nhạc', 'mnet asian music awards',
			'golden disc awards', 'melon music awards', 'soribada best k-music awards',
			'asian artist awards', 'gaon chart music awards', 'billboard music awards',
			'grammy', 'mtv', 'brit awards', 'mama awards'
		]
		if any(m in t for m in award_markers):
			return True
		if t.startswith('danh sách ') or '(định hướng)' in t or 'định hướng' == t:
			return True
		# Loại các trang về họ (surname/family name)
		if t.startswith('họ ') or '(họ)' in t or 'họ người' in t or 'họ (họ người)' in t:
			return True
		# Exclude bare years or dates
		if t.isdigit() and (len(t) == 4 or len(t) == 2):
			return True
		# Exclude very short or generic words
		if len(t) < 2:
			return True
		return False

	def score_link(self, current_title: str, target_title: str, link_text: str, depth: int) -> float:
		"""Heuristic score for prioritizing links without fetching target pages."""
		score = 0.0
		lt = link_text.strip().lower()
		tt = target_title.strip().lower()

		# Boost if link text hints music entities
		for k in ['album', 'bài hát', 'đĩa đơn', 'ca sĩ', 'nhạc sĩ', 'nhóm nhạc', 'ost']:
			if k in lt:
				score += 8
				break

		# Boost if title hints entity types
		if '(album' in tt:
			score += 10
		if '(bài hát' in tt:
			score += 9
		if '(nhóm nhạc' in tt:
			score += 9
		# Heavy penalty for company-like titles to avoid crawling companies
		company_title_markers = ['entertainment', 'music and live', 'division', 'company', 'agency', 'records', 'co.,', 'corp', 'inc.', 'limited', 'ltd', 'holding', 'subsidiary']
		if any(m in tt for m in company_title_markers):
			score -= 50

		# Korean markers in title
		for m in self.korean_markers:
			if m in tt:
				score += 4
				break

		# Penalize generic pages
		if self.is_excluded_title(target_title):
			score -= 20

		# Prefer closer depth lightly
		score -= depth * 0.5

		# Small penalty for very long titles (likely non-entity lists)
		if len(tt) > 60:
			score -= 4

		return score

	def scrape_with_bfs(self, seed_titles: List[str], max_nodes: int = 1000, top_k_per_node: int = 20) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
		"""Priority-based BFS using a simple heuristic score; defers edges until targets are accepted."""
		print("\n" + "=" * 70)
		print("Bắt đầu thu thập dữ liệu (BFS ưu tiên theo điểm liên quan)")
		print("=" * 70)

		# Lưu seed artists để cộng điểm cho album/song của họ
		self.seed_artists = seed_titles.copy()

		from heapq import heappush, heappop

		# Max-heap using negative scores
		pq: List[Tuple[float, str, int]] = []
		queued: set[str] = set()

		for t in seed_titles:
			# Rất cao priority cho seeds để xuất hiện đầu tiên
			heappush(pq, (-10000.0, t, 0))
			queued.add(t)

		# Ngưỡng chất lượng theo loại node
		threshold_artist_group = 15.0  # Artist/Group cần điểm cao hơn
		threshold_album_song = 15.0   # Album/Song: nhiều tên tiếng Anh nhưng vẫn K-pop
        
		# Set để lưu các hạt giống ban đầu
		seed_set = set(seed_titles)

		while pq and len(self.nodes) < max_nodes:
			neg_score, current_title, depth = heappop(pq)
            
			# Kiểm tra xem có phải hạt giống không
			is_seed = current_title in seed_set

			if self.is_excluded_title(current_title):
				if is_seed:
					print(f"  ✗ [SEED BỊ LOẠI] {current_title} - is_excluded_title")
				continue

			soup = self.get_page_soup(current_title)
			if not soup:
				if is_seed:
					print(f"  ✗ [SEED BỊ LOẠI] {current_title} - Không lấy được soup")
				continue

			if not self.is_relevant_page(current_title, soup, is_seed=is_seed):
				if is_seed:
					print(f"  ✗ [SEED BỊ LOẠI] {current_title} - is_relevant_page = False")
				continue

			infobox = self.extract_infobox(soup)
			page_text_lower = soup.get_text(separator=' ').lower()
			infobox_text = ' '.join(infobox.values()).lower()
			label = self.classify_label(current_title, infobox, page_text_lower)

			# LỌC NGAY: Chỉ chấp nhận node nếu label rõ ràng
			# NHƯNG: Với hạt giống, nếu label là Entity nhưng có từ khóa âm nhạc thì cũng chấp nhận
			if label not in ('Artist', 'Group', 'Album', 'Song'):
				if is_seed:
					# Kiểm tra xem có phải là music entity không
					has_music_keyword = any(k in infobox_text for k in ['ca sĩ', 'nhạc sĩ', 'nhóm nhạc', 'album', 'bài hát']) or \
										any(k in page_text_lower for k in ['ca sĩ', 'nhạc sĩ', 'nhóm nhạc', 'k-pop'])
					if has_music_keyword:
						# Thử phân loại lại dựa vào nội dung infobox
						if 'nhóm nhạc' in infobox_text or 'thành viên' in infobox_text or 'Cựu thành viên' in infobox_text:
							label = 'Group'
						elif 'ca sĩ' in infobox_text or 'nghề nghiệp' in infobox:
							label = 'Artist'
						else:
							# Mặc định là Group nếu không chắc
							label = 'Group'
						print(f"  ℹ [SEED RELABEL] {current_title}: Entity -> {label}")
					else:
						if is_seed:
							print(f"  ✗ [SEED BỊ LOẠI] {current_title} - label = {label}, không có từ khóa music")
						continue
				else:
					continue
            
			# Tạo node tạm để tính quality score
			temp_node = {
				'label': label,
				'infobox': infobox,
				'url': self.base_url + current_title.replace(" ", "_"),
				'depth': depth
			}
            
			# LỌC NGAY: Tính điểm chất lượng
			quality_score = self._calculate_quality_score(current_title, temp_node)
			# Nếu là Album/Song và tiêu đề có dạng "(album/bài hát của X)" mà điểm vẫn 0,
			# chỉ cộng điểm nếu X là nghệ sĩ Hàn nổi tiếng hoặc thuộc seed
			if quality_score == 0.0 and label in ('Album','Song'):
				title_lc = current_title.lower()
				m = re.search(r"\((?:album|bài hát) của ([^)]+)\)", title_lc)
				if m:
					perf = m.group(1).strip().lower()
					famous_korean_artists = {
						'bts','blackpink','exo','twice','red velvet','nct','seventeen','itzy','aespa','newjeans',
						'big bang',"girls' generation",'super junior','2ne1','iu','psy','taeyeon','rosé','jennie','jisoo','lisa'
					}
					seed_set = {s.lower().replace('(nhóm nhạc)','').replace('(ca sĩ)','').strip() for s in self.seed_artists}
					if any(name in perf for name in famous_korean_artists.union(seed_set)):
						quality_score = 20.0
            
			# Chọn ngưỡng phù hợp theo loại
			if label in ('Artist', 'Group'):
				quality_threshold = threshold_artist_group
			else:  # Album, Song
				quality_threshold = threshold_album_song
            
			# CHẤP NHẬN HẠT GIỐNG BẤT KỂ ĐIỂM (để đảm bảo có đầy đủ)
			if is_seed:
				# Bỏ qua kiểm tra điểm cho hạt giống
				# Kiểm tra trùng chữ ký trước khi thêm
				signature = self._compute_node_signature(label, infobox)
				if signature in self.node_signature_index:
					original = self.node_signature_index[signature]
					self.title_alias[current_title] = original
					print(f"  ↷ Bỏ qua node trùng (SEED): {current_title} ≡ {original}")
					# Không thêm node trùng
				else:
					self.nodes[current_title] = temp_node
					self.node_signature_index[signature] = current_title
					print(f"✓ [SEED] [{len(self.nodes)}/{max_nodes}] {current_title} ({label}, điểm: {quality_score:.1f}) - Hạt giống")
				# Tiếp tục xử lý (không skip)
			elif quality_score < quality_threshold:
				print(f"  ✗ Bỏ qua '{current_title}' ({label}, điểm: {quality_score:.1f} < {quality_threshold})")
				continue
			else:
				# Node đạt chuẩn - kiểm tra trùng chữ ký trước khi thêm
				signature = self._compute_node_signature(label, infobox)
				if signature in self.node_signature_index:
					original = self.node_signature_index[signature]
					self.title_alias[current_title] = original
					print(f"  ↷ Bỏ qua node trùng: {current_title} ≡ {original}")
					# Không thêm node trùng, nhưng vẫn duyệt link của trang này? Bỏ qua tiếp tục vòng lặp.
					continue
				else:
					self.nodes[current_title] = temp_node
					self.node_signature_index[signature] = current_title
					print(f"✓ [{len(self.nodes)}/{max_nodes}] {current_title} ({label}, điểm: {quality_score:.1f})")

			# Promote pending edges whose target is now accepted and allowed by labels
			if self.pending_edges:
				remaining: List[Dict[str, Any]] = []
				for pe in self.pending_edges:
					# Áp dụng alias nếu target đã được gộp vào node khác
					if pe['target'] in self.title_alias:
						pe['target'] = self.title_alias[pe['target']]
					if pe['source'] in self.title_alias:
						pe['source'] = self.title_alias[pe['source']]

					if pe['target'] == current_title and pe['source'] in self.nodes:
						s_label = self.nodes[pe['source']].get('label', 'Entity')
						t_label = self.nodes[current_title].get('label', 'Entity')
                        
						# Re-classify relation based on actual labels and Infobox (precise)
						source_title = pe['source']
						target_title = current_title
						link_text = pe.get('text', '')
                        
						# Get infobox data
						source_infobox = self.nodes[source_title].get('infobox', {})
						target_infobox = self.nodes[target_title].get('infobox', {})
                        
						# Classify relation với logic kiểm tra Infobox chính xác
						relation_type = self._determine_precise_relation(
							source_title, s_label, source_infobox,
							target_title, t_label, target_infobox,
							link_text
						)
                        
						# Chỉ thêm edge nếu relation type xác định được và hợp lệ
						if relation_type and self._is_allowed_label(s_label) and self._is_allowed_label(t_label):
							pe['type'] = relation_type
							self.edges.append(pe)
						else:
							remaining.append(pe)
				self.pending_edges = remaining

			content_div = soup.find('div', id='mw-content-text')
			if not content_div:
				continue

			candidates: List[Tuple[float, str, str]] = []  # (score, target_title, link_text)
			for link in content_div.find_all('a', href=True):
				href = link['href']
				if href.startswith('/wiki/') and ':' not in href and '#' not in href:
					target_title = href.replace('/wiki/', '').replace('_', ' ')
					# Decode percent-encoded Vietnamese characters
					target_title = unquote(target_title)
					if self.is_excluded_title(target_title):
						continue
					s = self.score_link(current_title, target_title, link.get_text(), depth)
					candidates.append((s, target_title, link.get_text(strip=True)))

			# Select top-K candidates
			candidates.sort(reverse=True, key=lambda x: x[0])
			selected = 0
			for s, target_title, link_text in candidates:
				if selected >= top_k_per_node:
					break
				if target_title not in queued and len(self.nodes) + len(pq) < max_nodes:
					queued.add(target_title)
					# Push with negative score for max-heap behavior
					heappush(pq, (-s, target_title, depth + 1))
					# Lưu pending edge (relation type sẽ được xác định sau khi target được chấp nhận)
					self.pending_edges.append({
						'source': current_title,
						'target': target_title,
						'type': 'PENDING',  # Sẽ được xác định lại bằng _determine_precise_relation()
						'text': link_text
					})
					selected += 1

			time.sleep(self.request_delay_seconds)

		print("\n✓ Thu thập dữ liệu hoàn tất!")
        
		# Tạo các node Genre, Instrument, Company từ infobox
		self._create_additional_nodes()
		
		# Clean up the network
		self.cleanup_network()
        
		# Tạo các cạnh quan hệ với Genre, Instrument, Company sau khi thu thập node xong
		self._create_additional_edges()
		
		return self.nodes, self.edges

	def _create_additional_nodes(self):
		"""Tạo các node Genre, Instrument, Company, Occupation từ thông tin infobox"""
		print("\n" + "=" * 70)
		print("TẠO CÁC NODE GENRE, INSTRUMENT, COMPANY, OCCUPATION")
		print("=" * 70)
        
		genre_count = 0
		instrument_count = 0
		company_count = 0
		occupation_count = 0
        
		# Tạo copy của keys để tránh lỗi "dictionary changed size during iteration"
		node_titles = list(self.nodes.keys())
        
		for node_title in node_titles:
			node_data = self.nodes[node_title]
			if 'infobox' not in node_data:
				continue
                
			infobox = node_data['infobox']
            
			# Tạo Genre nodes từ Thể loại
			if 'Thể loại' in infobox:
				genres_text = str(infobox['Thể loại'])
				# Tách các thể loại bằng dấu phẩy
				genres = [g.strip() for g in genres_text.split(',') if g.strip()]
				for genre in genres:
						if len(genre) > 2:  # Chỉ lấy thể loại có tên dài hơn 2 ký tự
							genre_key = self._create_genre_node(genre)
							if not genre_key:
								continue
						if genre_key not in self.nodes:
							self.nodes[genre_key] = self.genre_nodes[genre_key]
							genre_count += 1
            
			# Tạo Instrument nodes từ Nhạc cụ (chỉ cho Artist)
			if node_data['label'] == 'Artist' and 'Nhạc cụ' in infobox:
				instruments_text = str(infobox['Nhạc cụ'])
				# Tách các nhạc cụ bằng dấu phẩy
				instruments = [i.strip() for i in instruments_text.split(',') if i.strip()]
				for instrument in instruments:
					if len(instrument) > 2:  # Chỉ lấy nhạc cụ có tên dài hơn 2 ký tự
						# Chuẩn hóa tên nhạc cụ trước khi tạo node
						normalized_instrument = self._normalize_instrument_name(instrument)
						if not normalized_instrument:
							continue
						instrument_key = self._create_instrument_node(normalized_instrument)
						if not instrument_key:
							continue
						if instrument_key not in self.nodes:
							self.nodes[instrument_key] = self.instrument_nodes[instrument_key]
							instrument_count += 1
            
			# Tạo Company nodes từ Hãng đĩa, Công ty quản lý
			for field in ['Hãng đĩa', 'Công ty quản lý', 'Label', 'Agency']:
				if field in infobox:
					companies_text = str(infobox[field])
					# Tách các công ty bằng dấu phẩy
					companies = [c.strip() for c in companies_text.split(',') if c.strip()]
					for company in companies:
						if len(company) > 2:  # Chỉ lấy công ty có tên dài hơn 2 ký tự
							# Chuẩn hóa tên công ty để tránh trùng lặp
							n_company = self._normalize_company_name(company)
							company_key = self._create_company_node(n_company)
							if not company_key:
								continue
							if company_key not in self.nodes:
								self.nodes[company_key] = self.company_nodes[company_key]
								company_count += 1
        
		# Tạo Occupation nodes từ Nghề nghiệp (chỉ cho Artist)
		for node_title in node_titles:
			node_data = self.nodes[node_title]
			if node_data.get('label') != 'Artist' or 'infobox' not in node_data:
				continue
			info = node_data['infobox']
			if 'Nghề nghiệp' not in info:
				continue
			raw = str(info['Nghề nghiệp'])
			raw = raw.replace('·', ',').replace('•', ',').replace('・', ',')
			raw = re.sub(r"\[\s*\d+\s*\]", '', raw)
			items = [x.strip() for x in re.split(r",|;|/|\band\b|\bvà\b|\s[-–—]\s", raw, flags=re.IGNORECASE) if x.strip()]
			for occ in items:
				norm = self._normalize_occupation_name(occ)
				if not norm:
					continue
				key = self._create_occupation_node(norm)
				if key and key not in self.nodes:
					self.nodes[key] = self.occupation_nodes[key]
					occupation_count += 1

		print(f"✓ Đã tạo {genre_count} Genre nodes")
		print(f"✓ Đã tạo {instrument_count} Instrument nodes") 
		print(f"✓ Đã tạo {company_count} Company nodes")
		print(f"✓ Đã tạo {occupation_count} Occupation nodes")

	def _normalize_title_for_matching(self, title: str) -> str:
		"""Chuẩn hóa title để matching (loại bỏ phần trong ngoặc, lowercase)"""
		import re
		# Loại bỏ phần trong ngoặc: "Irene (ca sĩ)" -> "Irene"
		normalized = re.sub(r'\s*\([^)]*\)', '', title).strip()
		# Chuyển thành lowercase
		normalized = normalized.lower()
		return normalized
    
	def _create_additional_edges(self):
		"""Tạo các cạnh quan hệ với Genre, Instrument, Company và MEMBER_OF sau khi thu thập node xong"""
		print("\n" + "=" * 70)
		print("TẠO CÁC CẠNH QUAN HỆ BỔ SUNG")
		print("=" * 70)
        
		edge_count = 0
		member_of_count = 0
        
		# Trước tiên, tạo các cạnh MEMBER_OF
		for node_title, node_data in self.nodes.items():
			# Chỉ xử lý các node Group
			if node_data['label'] != 'Group':
				continue
            
			if 'infobox' not in node_data:
				continue
            
			infobox = node_data['infobox']
            
			# Kiểm tra các field chứa thành viên
			for field in ['Thành viên', 'Cựu thành viên', 'Members', 'Former Members']:
				if field not in infobox:
					continue
                
				members_text = str(infobox[field]).lower()
                
				# Duyệt qua tất cả các node Artist để tìm match
				for artist_title, artist_data in self.nodes.items():
					if artist_data['label'] != 'Artist':
						continue
                    
					# Chuẩn hóa tên artist để matching
					artist_normalized = self._normalize_title_for_matching(artist_title)
                    
					# Kiểm tra xem tên artist có trong danh sách thành viên không
					# Sử dụng word boundary để tránh false positive
					# Pattern matching với word boundary để tránh substring
					pattern = r'\b' + re.escape(artist_normalized) + r'\b'
					if re.search(pattern, members_text):
						# Tạo cạnh MEMBER_OF
						edge = {
							'source': artist_title,
							'target': node_title,
							'type': 'MEMBER_OF',
							'text': f"{artist_title} là thành viên của {node_title}"
						}
						self.edges.append(edge)
						member_of_count += 1
        
		print(f"✓ Đã tạo {member_of_count} cạnh MEMBER_OF")
		
		# Tạo các cạnh RELEASED: Artist/Group -> Album dựa trên các trường nghệ sĩ trong Album
		# Xây map tên artist/group đã chuẩn hóa
		artist_group_normalized = {}
		for node_title, node_data in self.nodes.items():
			if node_data['label'] in ('Artist', 'Group'):
				norm = self._normalize_title_for_matching(node_title)
				artist_group_normalized[norm] = node_title
		# BỎ: không tạo RELEASED từ trường nghệ sĩ trực tiếp trong infobox Album
		
		# Tạo các cạnh SINGS: Artist/Group -> Song dựa trên "Được thực hiện bởi"
		# Xây map tên artist/group đã chuẩn hóa để match nhanh
		artist_group_normalized = {}
		for node_title, node_data in self.nodes.items():
			if node_data['label'] in ('Artist', 'Group'):
				norm = self._normalize_title_for_matching(node_title)
				artist_group_normalized[norm] = node_title
		
		sings_count = 0
		for song_title, song_data in self.nodes.items():
			if song_data['label'] != 'Song' or 'infobox' not in song_data:
				continue
			infobox = song_data['infobox']
			performed = str(infobox.get('Được thực hiện bởi', '')).strip()
			if not performed:
				continue
			# Tách theo dấu phẩy
			names = [n.strip() for n in performed.split(',') if n.strip()]
			for name in names:
				norm_name = self._normalize_title_for_matching(name)
				if norm_name in artist_group_normalized:
					source_title = artist_group_normalized[norm_name]
					# Thêm edge nếu chưa có
					edge = {
						'source': source_title,
						'target': song_title,
						'type': 'SINGS',
						'text': f"{source_title} trình bày {song_title}"
					}
					self.edges.append(edge)
					sings_count += 1
		# Tạo RELEASED dựa trên Album['Được thực hiện bởi'] (nếu có)
		released_album_count = 0
		for album_title, album_data in self.nodes.items():
			if album_data['label'] != 'Album' or 'infobox' not in album_data:
				continue
			performed = str(album_data['infobox'].get('Được thực hiện bởi', '')).strip()
			if not performed:
				continue
			names = [n.strip() for n in performed.split(',') if n.strip()]
			for name in names:
				norm_name = self._normalize_title_for_matching(name)
				if norm_name in artist_group_normalized:
					source_title = artist_group_normalized[norm_name]
					edge = {
						'source': source_title,
						'target': album_title,
						'type': 'RELEASED',
						'text': f"{source_title} phát hành {album_title}"
					}
					self.edges.append(edge)
					released_album_count += 1
		print(f"✓ Đã tạo {released_album_count} cạnh RELEASED (Album: Được thực hiện bởi) và {sings_count} cạnh SINGS (Song)")

		# PRODUCED: Artist -> Album/Song dựa trên trường "Sản xuất" trong infobox
		artist_only_normalized = {}
		for node_title, node_data in self.nodes.items():
			if node_data['label'] == 'Artist':
				norm = self._normalize_title_for_matching(node_title)
				artist_only_normalized[norm] = node_title
		produced_album_count = 0
		produced_song_count = 0
		# Album producers
		for album_title, album_data in self.nodes.items():
			if album_data['label'] != 'Album' or 'infobox' not in album_data:
				continue
			info = album_data['infobox']
			prod = str(info.get('Sản xuất', '') or info.get('Nhà sản xuất', '') or info.get('Producer', '')).strip()
			if not prod:
				continue
			prod = prod.replace('·', ',').replace('•', ',').replace('・', ',')
			prod = re.sub(r"\[\s*\d+\s*\]", '', prod)
			names = re.split(r",|;|/|\band\b|\bvà\b", prod, flags=re.IGNORECASE)
			for name in [n.strip() for n in names if n.strip()]:
				norm_name = self._normalize_title_for_matching(name)
				if norm_name in artist_only_normalized:
					source_title = artist_only_normalized[norm_name]
					edge = {
						'source': source_title,
						'target': album_title,
						'type': 'PRODUCED_ALBUM',
						'text': f"{source_title} sản xuất {album_title}"
					}
					self.edges.append(edge)
					produced_album_count += 1
		# Song producers
		for song_title, song_data in self.nodes.items():
			if song_data['label'] != 'Song' or 'infobox' not in song_data:
				continue
			info = song_data['infobox']
			prod = str(info.get('Sản xuất', '') or info.get('Nhà sản xuất', '') or info.get('Producer', '')).strip()
			if not prod:
				continue
			prod = prod.replace('·', ',').replace('•', ',').replace('・', ',')
			prod = re.sub(r"\[\s*\d+\s*\]", '', prod)
			names = re.split(r",|;|/|\band\b|\bvà\b", prod, flags=re.IGNORECASE)
			for name in [n.strip() for n in names if n.strip()]:
				norm_name = self._normalize_title_for_matching(name)
				if norm_name in artist_only_normalized:
					source_title = artist_only_normalized[norm_name]
					edge = {
						'source': source_title,
						'target': song_title,
						'type': 'PRODUCED_SONG',
						'text': f"{source_title} sản xuất {song_title}"
					}
					self.edges.append(edge)
					produced_song_count += 1

		# WROTE: Artist -> Song dựa trên trường "Sáng tác" trong infobox
		wrote_song_count = 0
		for song_title, song_data in self.nodes.items():
			if song_data['label'] != 'Song' or 'infobox' not in song_data:
				continue
			info = song_data['infobox']
			wrote = ''
			for f in ['Sáng tác', 'Nhạc sĩ', 'Songwriter', 'Composer', 'Tác giả']:
				if f in info and info[f]:
					wrote = str(info[f]).strip()
					break
			if not wrote:
				continue
			wrote = wrote.replace('·', ',').replace('•', ',').replace('・', ',')
			wrote = re.sub(r"\[\s*\d+\s*\]", '', wrote)
			names = re.split(r",|;|/|\band\b|\bvà\b", wrote, flags=re.IGNORECASE)
			for name in [n.strip() for n in names if n.strip()]:
				norm_name = self._normalize_title_for_matching(name)
				if norm_name in artist_only_normalized:
					source_title = artist_only_normalized[norm_name]
					edge = {
						'source': source_title,
						'target': song_title,
						'type': 'WROTE',
						'text': f"{source_title} sáng tác {song_title}"
					}
					self.edges.append(edge)
					wrote_song_count += 1
		print(f"✓ Đã tạo {produced_album_count} cạnh PRODUCED_ALBUM, {produced_song_count} cạnh PRODUCED_SONG, {wrote_song_count} cạnh WROTE")
		
		# Tạo các cạnh CONTAINS: Album -> Song dựa trên Song['Tên album']
		# Xây map tên Album đã chuẩn hóa để match nhanh
		album_normalized = {}
		for node_title, node_data in self.nodes.items():
			if node_data['label'] == 'Album':
				norm = self._normalize_title_for_matching(node_title)
				album_normalized[norm] = node_title
		contains_count = 0
		for song_title, song_data in self.nodes.items():
			if song_data['label'] != 'Song' or 'infobox' not in song_data:
				continue
			album_name = str(song_data['infobox'].get('Tên album', '')).strip()
			if not album_name:
				continue
			candidates = [a.strip() for a in album_name.split(',') if a.strip()]
			for cand in candidates:
				norm_album = self._normalize_title_for_matching(cand)
				if norm_album in album_normalized:
					source_album = album_normalized[norm_album]
					edge = {
						'source': source_album,
						'target': song_title,
						'type': 'CONTAINS',
						'text': f"{source_album} chứa bài hát {song_title}"
					}
					self.edges.append(edge)
					contains_count += 1
		print(f"✓ Đã tạo {contains_count} cạnh CONTAINS (Album -> Song)")

		# HAS_OCCUPATION: Artist -> Occupation từ trường 'Nghề nghiệp'
		occupation_normalized = {}
		for node_title, node_data in self.nodes.items():
			if node_data['label'] == 'Occupation':
				norm = self._normalize_title_for_matching(node_title.replace('Occupation_', '')) if node_title.startswith('Occupation_') else self._normalize_title_for_matching(node_data.get('title',''))
				occupation_normalized[norm] = node_title
		occ_edge_count = 0
		for artist_title, artist_data in self.nodes.items():
			if artist_data['label'] != 'Artist' or 'infobox' not in artist_data:
				continue
			info = artist_data['infobox']
			if 'Nghề nghiệp' not in info:
				continue
			raw = str(info['Nghề nghiệp'])
			raw = raw.replace('·', ',').replace('•', ',').replace('・', ',')
			raw = re.sub(r"\[\s*\d+\s*\]", '', raw)
			items = [x.strip() for x in re.split(r",|;|/|\band\b|\bvà\b", raw, flags=re.IGNORECASE) if x.strip()]
			for occ in items:
				norm_occ = self._normalize_occupation_name(occ)
				if not norm_occ:
					continue
				key = f"Occupation_{norm_occ}"
				if key in self.nodes:
					edge = {
						'source': artist_title,
						'target': key,
						'type': 'HAS_OCCUPATION',
						'text': f"{artist_title} có nghề nghiệp {norm_occ}"
					}
					self.edges.append(edge)
					occ_edge_count += 1
		print(f"✓ Đã tạo {occ_edge_count} cạnh HAS_OCCUPATION (Artist -> Occupation)")
		
		# Tiếp tục tạo các cạnh Genre, Instrument, Company
		for node_title, node_data in self.nodes.items():
			if 'infobox' not in node_data or node_data['label'] not in ('Artist', 'Group'):
				continue
                
			infobox = node_data['infobox']
            
			# Tạo cạnh IS_GENRE: Artist/Group -> Genre
			if 'Thể loại' in infobox:
				genres_text = str(infobox['Thể loại'])
				genres = [g.strip() for g in genres_text.split(',') if g.strip()]
				for genre in genres:
					if len(genre) > 2:
						# Chuẩn hóa tên thể loại để khớp với key đã tạo
						normalized_genre = self._normalize_genre_name(genre)
						genre_key = f"Genre_{normalized_genre}"
						if genre_key in self.nodes:
							edge = {
								'source': node_title,
								'target': genre_key,
								'type': 'IS_GENRE',
								'text': f"{node_title} thuộc thể loại {normalized_genre}"
							}
							self.edges.append(edge)
							edge_count += 1
            
			# Tạo cạnh PLAYS: Artist -> Instrument (chỉ cho Artist)
			if node_data['label'] == 'Artist' and 'Nhạc cụ' in infobox:
				instruments_text = str(infobox['Nhạc cụ'])
				instruments = [i.strip() for i in instruments_text.split(',') if i.strip()]
				for instrument in instruments:
					if len(instrument) > 2:
						# Chuẩn hóa tên nhạc cụ trước khi tạo edge
						normalized_instrument = self._normalize_instrument_name(instrument)
						instrument_key = f"Instrument_{normalized_instrument}"
						if instrument_key in self.nodes:
							edge = {
								'source': node_title,
								'target': instrument_key,
								'type': 'PLAYS',
								'text': f"{node_title} có thể dùng {normalized_instrument}"
							}
							self.edges.append(edge)
							edge_count += 1
            
			# Tạo cạnh MANAGED_BY: Artist/Group -> Company
			for field in ['Hãng đĩa', 'Công ty quản lý', 'Label', 'Agency']:
				if field in infobox:
					companies_text = str(infobox[field])
					companies = [c.strip() for c in companies_text.split(',') if c.strip()]
					for company in companies:
						if len(company) > 2:
							# Chuẩn hóa để khớp đúng key đã tạo
							n_company = self._normalize_company_name(company)
							if not n_company:
								continue
							company_key = f"Company_{n_company}"
							if company_key in self.nodes:
								edge = {
									'source': node_title,
									'target': company_key,
									'type': 'MANAGED_BY',
									'text': f"{node_title} được quản lý bởi {n_company}"
								}
								self.edges.append(edge)
								edge_count += 1
        
		print(f"✓ Đã tạo {edge_count} cạnh quan hệ IS_GENRE, PLAYS, MANAGED_BY")
		print(f"\n✓ Tổng cộng: {member_of_count + edge_count} cạnh quan hệ mới được tạo")

	def calculate_node_quality_score(self, node_title: str) -> float:
		"""Calculate quality score for a node based on Korean music relevance."""
		if node_title not in self.nodes:
			return 0.0
        
		node = self.nodes[node_title]
		score = 0.0
        
		# Base score for correct label types
		if node['label'] in ('Artist', 'Group', 'Album', 'Song'):
			score += 10
        
		# Check for Hangul in title or infobox
		infobox_text = ' '.join(node['infobox'].values())
		if self._has_hangul(node_title) or self._has_hangul(infobox_text):
			score += 15
        
		# Check for Korean nationality/origin in infobox
		korean_fields = ['Quốc tịch', 'Quốc gia', 'Nguồn gốc', 'Nơi sinh', 'Quê quán']
		for field in korean_fields:
			if field in node['infobox']:
				field_value = node['infobox'][field].lower()
				if 'hàn quốc' in field_value or 'korea' in field_value:
					score += 10
					break
        
		# Check for K-pop related keywords in infobox
		infobox_lower = infobox_text.lower()
		if 'k-pop' in infobox_lower:
			score += 8
		if 'sm entertainment' in infobox_lower or 'yg entertainment' in infobox_lower or \
		   'jyp entertainment' in infobox_lower or 'big hit' in infobox_lower or \
		   'hybe' in infobox_lower:
			score += 7
        
		# Check for Seoul/Korean cities
		korean_cities = ['seoul', 'busan', 'incheon', 'daegu', 'gwangju']
		if any(city in infobox_lower for city in korean_cities):
			score += 5
        
		# Penalty for non-Korean indicators
		non_korean_markers = ['mỹ', 'anh', 'pháp', 'đức', 'nhật bản', 'trung quốc', 'việt nam']
		for marker in non_korean_markers:
			if marker in infobox_lower:
				# Check if it's not just mentioning these countries
				for field in korean_fields:
					if field in node['infobox'] and marker in node['infobox'][field].lower():
						score -= 15
						break
        
		return score

	def cleanup_network(self) -> None:
		"""TẠM THỜI BỎ QUA - Không cleanup sau khi crawl."""
		print("\n" + "=" * 70)
		print("HOÀN THIỆN MẠNG LƯỚI")
		print("=" * 70)
		print("(TẠM THỜI BỎ QUA - Chỉ lọc trong quá trình crawl)")
        
		# TẠM THỜI: Không thực hiện cleanup
		return
        
		initial_nodes = len(self.nodes)
		initial_edges = len(self.edges)

		# Stage 1: Build adjacency list and calculate degrees
		print("\n[Giai đoạn 1] Phân tích cấu trúc mạng lưới...")
		adjacency = {node: set() for node in self.nodes}
		for edge in self.edges:
			adjacency[edge['source']].add(edge['target'])
			adjacency[edge['target']].add(edge['source'])

		# Calculate degree for each node
		node_degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
        
		# Stage 2: Find connected components
		print("\n[Giai đoạn 2] Tìm thành phần liên thông...")
		visited = set()
		components = []

		def dfs(node, component):
			visited.add(node)
			component.add(node)
			for neighbor in adjacency[node]:
				if neighbor not in visited and neighbor in self.nodes:
					dfs(neighbor, component)

		for node in self.nodes:
			if node not in visited:
				component = set()
				dfs(node, component)
				components.append(component)

		# Sort components by size
		components.sort(key=len, reverse=True)
        
		print(f"Tìm thấy {len(components)} thành phần liên thông")
		if components:
			for i, comp in enumerate(components[:5]):  # Show top 5
				print(f"  - Thành phần {i+1}: {len(comp)} node")
        
		# Keep largest component
		main_component = components[0] if components else set()
		print(f"\nGiữ lại thành phần chính với {len(main_component)} node")

		# Remove nodes not in main component
		nodes_to_remove = set(self.nodes.keys()) - main_component
		for node in nodes_to_remove:
			del self.nodes[node]

		# Update edges
		self.edges = [edge for edge in self.edges 
					  if edge['source'] in self.nodes and edge['target'] in self.nodes]

		# Rebuild adjacency after removing small components
		adjacency = {node: set() for node in self.nodes}
		for edge in self.edges:
			adjacency[edge['source']].add(edge['target'])
			adjacency[edge['target']].add(edge['source'])
        
		# Stage 3: Remove low-degree isolated nodes
		print("\n[Giai đoạn 3] Loại bỏ node cô lập (độ liên kết thấp)...")
		min_degree = 1  # At least 1 connection
		isolated_nodes = []
        
		for node_title in list(self.nodes.keys()):
			degree = len(adjacency.get(node_title, set()))
			if degree < min_degree:
				isolated_nodes.append(node_title)
				print(f"  ✗ Loại bỏ '{node_title}' (độ liên kết: {degree})")
        
		for node_title in isolated_nodes:
			del self.nodes[node_title]
        
		print(f"Đã loại bỏ {len(isolated_nodes)} node cô lập")
        
		# Final edge cleanup
		self.edges = [edge for edge in self.edges 
					  if edge['source'] in self.nodes and edge['target'] in self.nodes]

		# Final statistics
		print("\n" + "=" * 70)
		print("KẾT QUẢ HOÀN THIỆN")
		print("=" * 70)

		removed_nodes = initial_nodes - len(self.nodes)
		removed_edges = initial_edges - len(self.edges)
		
		if removed_nodes > 0:
			print(f"Đã loại bỏ thêm {removed_nodes} node trong quá trình hoàn thiện")
			print(f"Đã loại bỏ thêm {removed_edges} cạnh")
		else:
			print(f"Không có node nào bị loại bỏ thêm")
        
		print(f"\nMạng lưới cuối cùng: {len(self.nodes)} nodes, {len(self.edges)} edges")
        
		# Label distribution
		print("\n--- Phân bố nhãn node ---")
		label_counts = {}
		for node in self.nodes.values():
			label = node['label']
			label_counts[label] = label_counts.get(label, 0) + 1
        
		for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
			percentage = (count / len(self.nodes) * 100) if len(self.nodes) > 0 else 0
			print(f"  {label}: {count} ({percentage:.1f}%)")
        
		# Degree statistics
		if self.nodes:
			adjacency = {node: set() for node in self.nodes}
			for edge in self.edges:
				adjacency[edge['source']].add(edge['target'])
				adjacency[edge['target']].add(edge['source'])
            
			degrees = [len(neighbors) for neighbors in adjacency.values()]
			if degrees:
				avg_degree = sum(degrees) / len(degrees)
				max_degree = max(degrees)
				min_degree = min(degrees)
                
				print("\n--- Thống kê độ liên kết ---")
				print(f"  Trung bình: {avg_degree:.2f}")
				print(f"  Cao nhất: {max_degree}")
				print(f"  Thấp nhất: {min_degree}")
                
				# Find most connected nodes
				node_degree_pairs = [(node, len(adjacency[node])) for node in self.nodes]
				node_degree_pairs.sort(key=lambda x: x[1], reverse=True)
                
				print("\n--- Top 10 node có nhiều kết nối nhất ---")
				for i, (node, degree) in enumerate(node_degree_pairs[:10], 1):
					label = self.nodes[node]['label']
					print(f"  {i}. {node} ({label}): {degree} kết nối")

	def validate_network_requirements(self) -> bool:
		"""Validate if network meets project requirements."""
		print("\n" + "=" * 70)
		print("KIỂM TRA YÊU CẦU DỰ ÁN")
		print("=" * 70)
        
		is_valid = True
        
		# Requirement 1: Minimum 1000 nodes
		print(f"\n1. Số lượng node tối thiểu 1000:")
		if len(self.nodes) >= 1000:
			print(f"   ✓ PASS - Có {len(self.nodes)} node")
		else:
			print(f"   ✗ FAIL - Chỉ có {len(self.nodes)} node (thiếu {1000 - len(self.nodes)})")
			is_valid = False
        
		# Requirement 2: Quality nodes (meaningful labels)
		print(f"\n2. Node phải có ý nghĩa rõ ràng:")
		music_nodes = sum(1 for n in self.nodes.values() 
						  if n['label'] in ('Artist', 'Group', 'Album', 'Song'))
		ratio = (music_nodes / len(self.nodes) * 100) if len(self.nodes) > 0 else 0
        
		if ratio >= 80:
			print(f"   ✓ PASS - {ratio:.1f}% node là Artist/Group/Album/Song")
		else:
			print(f"   ⚠ WARN - Chỉ {ratio:.1f}% node có nhãn âm nhạc rõ ràng")
        
		# Requirement 3: Meaningful edges
		print(f"\n3. Cạnh phải có ý nghĩa rõ ràng:")
		if len(self.edges) > 0:
			print(f"   ✓ PASS - Có {len(self.edges)} mối quan hệ")
            
			# Check edge type distribution
			edge_types = {}
			for edge in self.edges:
				edge_type = edge.get('type', 'RELATED_TO')
				edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
			print(f"\n   Phân bố loại quan hệ:")
			for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
				percentage = (count / len(self.edges) * 100) if len(self.edges) > 0 else 0
				print(f"     - {edge_type}: {count} ({percentage:.1f}%)")
		else:
			print(f"   ✗ FAIL - Không có cạnh nào")
			is_valid = False
        
		# Requirement 4: Network connectivity
		print(f"\n4. Mạng lưới phải liên thông:")
		if self.nodes:
			adjacency = {node: set() for node in self.nodes}
			for edge in self.edges:
				adjacency[edge['source']].add(edge['target'])
				adjacency[edge['target']].add(edge['source'])
            
			# Check if connected
			visited = set()
			def dfs(node):
				visited.add(node)
				for neighbor in adjacency[node]:
					if neighbor not in visited and neighbor in self.nodes:
						dfs(neighbor)
            
			start_node = next(iter(self.nodes))
			dfs(start_node)
            
			if len(visited) == len(self.nodes):
				print(f"   ✓ PASS - Mạng lưới hoàn toàn liên thông")
			else:
				print(f"   ⚠ WARN - Có {len(self.nodes) - len(visited)} node không liên thông")
        
		# Requirement 5: Korean music context
		print(f"\n5. Nội dung liên quan âm nhạc Hàn Quốc:")
		korean_nodes = 0
		for node_title in self.nodes:
			if self._has_hangul(node_title) or \
			   'infobox' in self.nodes[node_title] and self._has_hangul(' '.join(self.nodes[node_title]['infobox'].values())):
				korean_nodes += 1
        
		korean_ratio = (korean_nodes / len(self.nodes) * 100) if len(self.nodes) > 0 else 0
		if korean_ratio >= 70:
			print(f"   ✓ PASS - {korean_ratio:.1f}% node có chữ Hangul")
		else:
			print(f"   ⚠ WARN - Chỉ {korean_ratio:.1f}% node có chữ Hangul")
        
		print("\n" + "=" * 70)
		if is_valid:
			print("✓ MẠNG LƯỚI ĐẠT YÊU CẦU DỰ ÁN")
		else:
			print("✗ MẠNG LƯỚI CHƯA ĐẠT YÊU CẦU - CẦN THU THẬP THÊM DỮ LIỆU")
		print("=" * 70)
        
		return is_valid

	def export_network_statistics(self) -> Dict[str, Any]:
		"""Export detailed network statistics."""
		stats = {
			'basic': {
				'node_count': len(self.nodes),
				'edge_count': len(self.edges),
			},
			'labels': {},
			'edge_types': {},
			'connectivity': {},
			'quality': {}
		}
        
		# Label distribution
		for node in self.nodes.values():
			label = node['label']
			stats['labels'][label] = stats['labels'].get(label, 0) + 1
        
		# Edge type distribution
		for edge in self.edges:
			edge_type = edge.get('type', 'RELATED_TO')
			stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
		# Connectivity metrics
		if self.nodes:
			adjacency = {node: set() for node in self.nodes}
			for edge in self.edges:
				adjacency[edge['source']].add(edge['target'])
				adjacency[edge['target']].add(edge['source'])
            
			degrees = [len(neighbors) for neighbors in adjacency.values()]
			if degrees:
				stats['connectivity'] = {
					'avg_degree': sum(degrees) / len(degrees),
					'max_degree': max(degrees),
					'min_degree': min(degrees),
					'density': len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1) / 2) if len(self.nodes) > 1 else 0
				}
        
		# Quality metrics
		korean_nodes = sum(1 for node_title in self.nodes 
					   if self._has_hangul(node_title) or 
					   ('infobox' in self.nodes[node_title] and 
						self._has_hangul(' '.join(self.nodes[node_title]['infobox'].values()))))
        
		stats['quality'] = {
			'korean_nodes': korean_nodes,
			'korean_ratio': korean_nodes / len(self.nodes) if len(self.nodes) > 0 else 0,
			'music_nodes': sum(1 for n in self.nodes.values() 
							   if n['label'] in ('Artist', 'Group', 'Album', 'Song')),
			'music_ratio': sum(1 for n in self.nodes.values() 
							   if n['label'] in ('Artist', 'Group', 'Album', 'Song')) / len(self.nodes) if len(self.nodes) > 0 else 0
		}
        
		return stats

	def save_data(self, filename: str = 'korean_artists_graph_bfs.json') -> None:
		"""Save network data with detailed statistics."""
		# Export statistics
		stats = self.export_network_statistics()
        
		graph_data = {
			'nodes': self.nodes,
			'edges': self.edges,
			'statistics': stats,
			'metadata': {
				'description': 'Mạng lưới nghệ sĩ và ca sĩ Hàn Quốc từ Wikipedia tiếng Việt',
				'source': 'Vietnamese Wikipedia',
				'generation_date': time.strftime('%Y-%m-%d %H:%M:%S')
			}
		}
        
		with open(filename, 'w', encoding='utf-8') as f:
			json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
		print(f"\n✓ Dữ liệu đã được lưu vào tệp: {filename}")
		print(f"  - {len(self.nodes)} nodes")
		print(f"  - {len(self.edges)} edges")
		print(f"  - Kèm theo thống kê chi tiết")
    
	def _determine_precise_relation(self, source_title: str, source_label: str, source_infobox: dict,
									target_title: str, target_label: str, target_infobox: dict,
									link_text: str) -> str:
		"""
		Xác định relation type chính xác dựa trên Infobox.
		Trả về None nếu không tìm thấy quan hệ chính xác.
		"""
		# Chuẩn hóa title: loại bỏ phần trong ngoặc để so sánh linh hoạt hơn
		import re
		source_title_clean = re.sub(r'\s*\([^)]*\)', '', source_title).strip().lower()
		target_title_clean = re.sub(r'\s*\([^)]*\)', '', target_title).strip().lower()
		source_title_lower = source_title.lower()
		target_title_lower = target_title.lower()
        
		# 1. MEMBER_OF: Artist -> Group
		if source_label == 'Artist' and target_label == 'Group':
			# Kiểm tra trong Group infobox có member này không
			for field in ['Thành viên', 'Cựu thành viên', 'Members', 'Former Members']:
				if field in target_infobox:
					members_text = str(target_infobox[field]).lower()
					if source_title_clean in members_text or source_title_lower in members_text:
						return 'MEMBER_OF'
			# Kiểm tra trong Artist infobox có mention Group không
			for field in ['Nhóm nhạc', 'Nhóm', 'Group', 'Band']:
				if field in source_infobox:
					groups_text = str(source_infobox[field]).lower()
					if target_title_clean in groups_text or target_title_lower in groups_text:
						return 'MEMBER_OF'
        
		# 2. MANAGED_BY: Artist/Group -> Company
		elif source_label in ('Artist', 'Group') and target_label == 'Company':
			for field in ['Hãng đĩa', 'Công ty quản lý', 'Công ty', 'Label', 'Agency']:
				if field in source_infobox:
					company_text = str(source_infobox[field]).lower()
					if target_title_clean in company_text or target_title_lower in company_text:
						return 'MANAGED_BY'
        
		# 3. RELEASED: Artist/Group -> Album
		elif source_label in ('Artist', 'Group') and target_label == 'Album':
			# Loại bỏ logic RELEASED sớm; sẽ tạo sau khi thu thập node dựa trên
			# các trường nghệ sĩ trong Album tại _create_additional_edges
			return None
        
		# 4. SINGS: Artist/Group -> Song
		elif source_label in ('Artist', 'Group') and target_label == 'Song':
			# Loại bỏ logic SINGS sớm; sẽ tạo sau khi thu thập node dựa trên
			# trường "Được thực hiện bởi" trong _create_additional_edges
			return None
        
		# 5. CONTAINS: Album -> Song
		elif source_label == 'Album' and target_label == 'Song':
			# Kiểm tra trong Song infobox có mention Album này không
			for field in ['Album', 'from Album', 'in Album']:
				if field in target_infobox:
					album_text = str(target_infobox[field]).lower()
					if source_title_clean in album_text or source_title_lower in album_text:
						return 'CONTAINS'
        
		# 6. PART_OF_ALBUM: Song -> Album (ngược chiều CONTAINS)
		elif source_label == 'Song' and target_label == 'Album':
			for field in ['Album', 'from Album', 'in Album']:
				if field in source_infobox:
					album_text = str(source_infobox[field]).lower()
					if target_title_clean in album_text or target_title_lower in album_text:
						return 'PART_OF_ALBUM'
        
		# 7. WON_AWARD: Artist/Group/Album/Song -> Award
		elif source_label in ('Artist', 'Group', 'Album', 'Song') and target_label == 'Award':
			for field in ['Giải thưởng', 'Awards', 'Awards and nominations']:
				if field in source_infobox:
					awards_text = str(source_infobox[field]).lower()
					if target_title_clean in awards_text or target_title_lower in awards_text:
						return 'WON_AWARD'
        
		# 8. SUB_UNIT_OF: Group -> Group
		elif source_label == 'Group' and target_label == 'Group':
			for field in ['Nhóm gốc', 'Liên quan', 'Related Group', 'Parent Group']:
				if field in source_infobox:
					parent_text = str(source_infobox[field]).lower()
					if target_title_clean in parent_text or target_title_lower in parent_text:
						return 'SUB_UNIT_OF'
        
		# 9. COLLABORATED_WITH: Artist <-> Artist hoặc Group <-> Group
		elif source_label in ('Artist', 'Group') and target_label in ('Artist', 'Group'):
			# Kiểm tra trong infobox có mention collaboration
			for field in ['Hợp tác', 'Collaboration', 'Featuring', 'Ft.', 'Feat.']:
				if field in source_infobox:
					collab_text = str(source_infobox[field]).lower()
					if target_title_clean in collab_text or target_title_lower in collab_text:
						return 'COLLABORATED_WITH'
        
		# 10. IS_GENRE: Artist/Group -> Genre
		elif source_label in ('Artist', 'Group') and target_label == 'Genre':
			for field in ['Thể loại', 'Genre', 'Genres']:
				if field in source_infobox:
					genre_text = str(source_infobox[field]).lower()
					if target_title_clean in genre_text or target_title_lower in genre_text:
						return 'IS_GENRE'
        
		# 11. PLAYS: Artist -> Instrument
		elif source_label == 'Artist' and target_label == 'Instrument':
			for field in ['Nhạc cụ', 'Instrument', 'Instruments']:
				if field in source_infobox:
					instrument_text = str(source_infobox[field]).lower()
					if target_title_clean in instrument_text or target_title_lower in instrument_text:
						return 'PLAYS'
        
		# Không tìm thấy quan hệ chính xác
		return None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Korean Artists/Groups BFS Scraper (Wikipedia Vietnamese)')
	parser.add_argument('--seeds', nargs='*', default=[
		"BTS", "Blackpink", "Big Bang (nhóm nhạc)", "Girls' Generation", "EXO",
		"Red Velvet (nhóm nhạc)", "NCT (nhóm nhạc)", "Psy", "IU (ca sĩ)", "Cha Eun-woo",
		"TWICE", "SEVENTEEN", "Stray Kids", "NewJeans", "LE SSERAFIM",
		"ITZY", "IVE", "(G)I-dle", "TXT (ban nhạc)", "SHINee",
		"Super Junior", "2NE1", "Mamamoo", "Kara (nhóm nhạc Hàn Quốc)"
	], help='Danh sách hạt giống (tiêu đề trang Wikipedia tiếng Việt)')
	parser.add_argument('--max-nodes', type=int, default=1700, help='Số lượng node tối đa')
	parser.add_argument('--delay', type=float, default=0.2, help='Độ trễ giữa các request (giây)')
	parser.add_argument('--timeout', type=int, default=10, help='Timeout mỗi request (giây)')
	parser.add_argument('--top-k', type=int, default=30, help='Số liên kết ưu tiên tối đa mỗi node')
	parser.add_argument('--output', type=str, default='korean_artists_graph_bfs.json', help='Tên file JSON đầu ra')
	# Neo4j optional export flags
	parser.add_argument('--neo4j-uri', type=str, default=None, help='Neo4j Bolt URI, ví dụ bolt://localhost:7687')
	parser.add_argument('--neo4j-user', type=str, default=None, help='Neo4j username')
	parser.add_argument('--neo4j-pass', type=str, default=None, help='Neo4j password')
	parser.add_argument('--neo4j-db', type=str, default=None, help='Neo4j database name (optional)')
	parser.add_argument('--neo4j-batch', type=int, default=1000, help='Batch size khi ghi Neo4j')
	parser.add_argument('--neo4j-create-constraints', action='store_true', default=False, help='Tự động tạo UNIQUE CONSTRAINT theo id cho các label chính')
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print("=" * 70)
	print("CHƯƠNG TRÌNH XÂY DỰNG MẠNG LƯỚI NGHỆ SĨ HÀN QUỐC")
	print(f"Bắt đầu từ {len(args.seeds)} node hạt giống, mục tiêu {args.max_nodes} node.")
	print("=" * 70)

	scraper = WikipediaBFScraper(request_timeout_seconds=args.timeout, request_delay_seconds=args.delay)
	scraper.scrape_with_bfs(args.seeds, max_nodes=args.max_nodes, top_k_per_node=args.top_k)
    
	# Validate network requirements
	scraper.validate_network_requirements()
    
	# Save data
	scraper.save_data(args.output)

	# Optional: export to Neo4j if credentials provided
	if args.neo4j_uri and args.neo4j_user and args.neo4j_pass:
		print("\nBắt đầu export dữ liệu sang Neo4j...")
		scraper.save_to_neo4j(
			uri=args.neo4j_uri,
			user=args.neo4j_user,
			password=args.neo4j_pass,
			database=args.neo4j_db,
			batch_size=args.neo4j_batch,
			create_constraints=args.neo4j_create_constraints
		)
		print("✓ Export Neo4j hoàn tất.")


if __name__ == '__main__':
	main()
