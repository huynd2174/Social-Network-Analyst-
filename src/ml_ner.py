# -*- coding: utf-8 -*-
"""
ML-BASED NER MODULE
Sử dụng pre-trained Vietnamese NER models để bổ sung cho rule-based NER
"""
import sys
import io
import re
from typing import List, Dict, Optional

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Try to import transformers
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    # Không in lỗi để tránh spam, chỉ set flag
except Exception as e:
    # Xử lý các lỗi khác (như GenerationMixin, version conflict, etc.)
    TRANSFORMERS_AVAILABLE = False

# Try to import spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spacy không được cài đặt. Chạy: pip install spacy")

# Mapping từ labels của model sang labels của chúng ta
LABEL_MAPPING = {
    # PERSON -> có thể là Artist
    'PERSON': 'Artist',
    'PER': 'Artist',
    'B-PER': 'Artist',
    'I-PER': 'Artist',
    
    # ORG -> có thể là Group hoặc Company
    'ORG': 'Group',  # Mặc định là Group, sẽ được điều chỉnh sau
    'ORGANIZATION': 'Group',
    'B-ORG': 'Group',
    'I-ORG': 'Group',
    
    # LOC -> có thể là Company (nếu có "Entertainment", "Music"...)
    'LOC': None,  # Không map trực tiếp
    'LOCATION': None,
    'B-LOC': None,
    'I-LOC': None,
    
    # MISC -> có thể là Album/Song
    'MISC': None,
    'B-MISC': None,
    'I-MISC': None,
}

# Keywords để phân biệt Company vs Group
COMPANY_KEYWORDS = ['entertainment', 'music', 'media', 'label', 'agency', 'công ty', 'hãng']
GROUP_KEYWORDS = ['nhóm', 'nhóm nhạc', 'group', 'band', 'idol']

def clean_text(text: str) -> str:
    """Chuẩn hóa text entity"""
    if not text:
        return ""
    text = text.strip()
    # Loại bỏ ký tự đặc biệt ở đầu/cuối
    text = re.sub(r'^[.,;:!?"\'()\[\]{}]+|[.,;:!?"\'()\[\]{}]+$', '', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def classify_entity_type(text: str, original_label: str) -> Optional[str]:
    """
    Phân loại entity type dựa trên text và label gốc
    
    Args:
        text: Text của entity
        original_label: Label từ model (PERSON, ORG, etc.)
    
    Returns:
        Entity type phù hợp (Artist, Group, Company, Album, Song) hoặc None
    """
    text_lower = text.lower()
    
    # Normalize label để so sánh (case-insensitive)
    label_upper = original_label.upper() if original_label else ''
    
    # PERSON -> Artist
    if label_upper in ['PERSON', 'PER', 'B-PER', 'I-PER'] or 'PERSON' in label_upper:
        return 'Artist'
    
    # ORG -> cần phân biệt Group vs Company
    if label_upper in ['ORG', 'ORGANIZATION', 'B-ORG', 'I-ORG'] or 'ORG' in label_upper:
        # Nếu có keywords của Company -> Company
        if any(kw in text_lower for kw in COMPANY_KEYWORDS):
            return 'Company'
        # Nếu có keywords của Group -> Group
        if any(kw in text_lower for kw in GROUP_KEYWORDS):
            return 'Group'
        # Mặc định là Group (vì nhóm nhạc phổ biến hơn)
        return 'Group'
    
    # MISC -> có thể là Album/Song (cần context để phân biệt)
    if label_upper in ['MISC', 'B-MISC', 'I-MISC'] or 'MISC' in label_upper:
        # Heuristic: nếu có từ khóa album/song -> phân loại
        if any(kw in text_lower for kw in ['album', 'ep', 'mini-album']):
            return 'Album'
        if any(kw in text_lower for kw in ['song', 'bài hát', 'ca khúc', 'single']):
            return 'Song'
        # Không phân loại được -> None
        return None
    
    return None

class VietnameseNERModel:
    """Wrapper cho Vietnamese NER models"""
    
    def __init__(self, model_name: str = "NlpHUST/ner-vietnamese-electra-base"):
        """
        Khởi tạo model
        
        Args:
            model_name: Tên model trên HuggingFace
        """
        self.model_name = model_name
        self.ner_pipeline = None
        self.available = False
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  transformers không khả dụng. Bỏ qua ML-based NER.")
            return
        
        try:
            print(f"📥 Đang tải model {model_name}...")
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple"
            )
            self.available = True
            print(f"✓ Đã tải model thành công")
        except Exception as e:
            # In lỗi để debug
            print(f"⚠️  ML model không khả dụng: {type(e).__name__}: {str(e)[:100]}")
            print(f"   Chỉ sử dụng rule-based NER")
            self.available = False
    
    def extract_entities(self, text: str, source_node: str = "") -> List[Dict]:
        """
        Trích xuất entities từ text
        
        Args:
            text: Text cần trích xuất
            source_node: Node ID nguồn
        
        Returns:
            List các entities được trích xuất
        """
        if not self.available or not self.ner_pipeline:
            return []
        
        if not text or len(text.strip()) < 3:
            return []
        
        try:
            # Giới hạn độ dài text để tránh lỗi tensor size mismatch
            # Model thường có max_length = 512 tokens
            # Ước tính: 1 token ≈ 0.75 từ, 512 tokens ≈ 384 từ ≈ 2000 ký tự
            MAX_TEXT_LENGTH = 2000  # An toàn hơn
            
            if len(text) > MAX_TEXT_LENGTH:
                # Chia text thành các chunks nhỏ hơn
                chunks = []
                chunk_size = MAX_TEXT_LENGTH
                i = 0
                while i < len(text):
                    chunk = text[i:i + chunk_size]
                    # Cố gắng cắt ở khoảng trắng để tránh cắt giữa từ
                    if i + chunk_size < len(text):
                        last_space = chunk.rfind(' ')
                        if last_space > chunk_size * 0.8:  # Nếu có khoảng trắng gần cuối
                            chunk = chunk[:last_space]
                            i = i + last_space + 1
                        else:
                            i = i + chunk_size
                    else:
                        i = len(text)  # Chunk cuối cùng
                    chunks.append(chunk)
            else:
                chunks = [text]
            
            # Chạy NER trên từng chunk
            all_results = []
            for chunk in chunks:
                try:
                    # Pipeline không nhận truncation/max_length như parameter
                    # Nó tự động xử lý trong tokenizer
                    chunk_results = self.ner_pipeline(chunk)
                    if isinstance(chunk_results, list):
                        all_results.extend(chunk_results)
                    elif isinstance(chunk_results, dict):
                        all_results.append(chunk_results)
                except Exception as chunk_error:
                    # Bỏ qua chunk có lỗi, tiếp tục với chunk khác
                    # Không in lỗi để tránh spam
                    continue
            
            results = all_results
            
            # Debug: đếm số results trước khi filter
            total_results = len(results)
            
            entities = []
            filtered_count = 0
            for result in results:
                # Xử lý format khác nhau của kết quả
                if isinstance(result, dict):
                    entity_text = result.get('word', '') or result.get('entity', '')
                    label = result.get('entity_group', '') or result.get('label', '')
                    score = result.get('score', 0.7)
                else:
                    continue
                
                entity_text = str(entity_text).strip()
                if not entity_text:
                    continue
                
                # Chuẩn hóa text
                entity_text = clean_text(entity_text)
                if not entity_text or len(entity_text) < 2:
                    continue
                
                # Phân loại entity type
                entity_type = classify_entity_type(entity_text, label)
                if not entity_type:
                    # Bỏ qua nếu không phân loại được
                    filtered_count += 1
                    continue
                
                # Loại bỏ các entity quá ngắn hoặc không hợp lệ
                if len(entity_text) < 2 or len(entity_text) > 50:
                    continue
                
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'method': 'ml-based',
                    'confidence': min(0.95, score * 0.9),  # Giảm confidence một chút so với rule-based
                    'source_node': source_node,
                    'ml_label': label,  # Lưu label gốc từ model
                })
            
            # Debug info (chỉ in cho vài lần đầu)
            if len(entities) == 0 and total_results > 0:
                # Có results nhưng không có entities nào pass filter
                # Có thể do classify_entity_type quá strict
                pass
            
            return entities
            
        except Exception as e:
            # Không in lỗi để tránh spam (đã xử lý ở trên)
            return []


# Global model instance
_ner_model = None

def get_ner_model(model_name: str = "NlpHUST/ner-vietnamese-electra-base") -> Optional[VietnameseNERModel]:
    """
    Lấy instance của NER model (singleton pattern)
    
    Args:
        model_name: Tên model trên HuggingFace
    
    Returns:
        VietnameseNERModel instance hoặc None
    """
    global _ner_model
    if _ner_model is None:
        _ner_model = VietnameseNERModel(model_name)
    return _ner_model if _ner_model.available else None

def extract_ml_entities(text: str, source_node: str = "") -> List[Dict]:
    """
    Trích xuất entities bằng ML model
    
    Args:
        text: Text cần trích xuất
        source_node: Node ID nguồn
    
    Returns:
        List các entities được trích xuất
    """
    model = get_ner_model()
    if not model:
        return []
    return model.extract_entities(text, source_node)

