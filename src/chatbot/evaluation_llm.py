"""
Optional: Generate evaluation questions using ChatGPT/NotebookLM

This is an alternative/additional method to generate questions.
Can be used to supplement the rule-based generation.
"""

import os
import json
from typing import List, Dict, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Install with: pip install openai")


def generate_questions_with_chatgpt(
    knowledge_graph_info: Dict,
    num_questions: int = 500,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo"
) -> List[Dict]:
    """
    Generate evaluation questions using ChatGPT.
    
    Args:
        knowledge_graph_info: Info about knowledge graph (entities, relationships)
        num_questions: Number of questions to generate
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: Model to use
        
    Returns:
        List of generated questions
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed")
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
    
    openai.api_key = api_key
    
    # Prepare context about knowledge graph
    context = f"""
    Knowledge Graph Info:
    - Total entities: {knowledge_graph_info.get('total_entities', 0)}
    - Entity types: {', '.join(knowledge_graph_info.get('entity_types', []))}
    - Relationship types: {', '.join(knowledge_graph_info.get('relationship_types', []))}
    - Sample entities: {', '.join(knowledge_graph_info.get('sample_entities', [])[:10])}
    """
    
    prompt = f"""
    Bạn là chuyên gia tạo câu hỏi đánh giá cho chatbot về K-pop.
    
    {context}
    
    Tạo {num_questions} câu hỏi đánh giá với các yêu cầu:
    1. Câu hỏi phải dựa trên thông tin trong knowledge graph trên
    2. Câu hỏi phải yêu cầu multi-hop reasoning (1-hop, 2-hop, hoặc 3-hop)
    3. Các loại câu hỏi:
       - True/False: "Jungkook là thành viên của BTS." → Đúng/Sai
       - Yes/No: "Jungkook có phải thành viên của BTS không?" → Có/Không
       - Multiple Choice: "Jungkook thuộc công ty nào?" → A/B/C/D
    
    4. Phân bố:
       - 1-hop: 40% (direct relationships)
       - 2-hop: 40% (1 intermediate)
       - 3-hop: 20% (2 intermediates)
    
    Trả về JSON format:
    {{
        "questions": [
            {{
                "question": "...",
                "question_type": "true_false|yes_no|multiple_choice",
                "answer": "Đúng|Sai|Có|Không|A|B|C|D",
                "choices": ["A", "B", "C", "D"] (only for multiple_choice),
                "hops": 1|2|3,
                "explanation": "..."
            }}
        ]
    }}
    """
    
    questions = []
    batch_size = 50  # Generate in batches
    
    for i in range(0, num_questions, batch_size):
        current_batch = min(batch_size, num_questions - i)
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia tạo câu hỏi đánh giá về K-pop."},
                    {"role": "user", "content": prompt.replace(str(num_questions), str(current_batch))}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Note: ChatGPT might return markdown code blocks, need to extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            data = json.loads(content)
            questions.extend(data.get("questions", []))
            
            print(f"  Generated {len(questions)}/{num_questions} questions...")
            
        except Exception as e:
            print(f"  ⚠️ Error generating batch {i//batch_size + 1}: {e}")
            continue
    
    return questions


def generate_questions_with_notebooklm(
    knowledge_graph_info: Dict,
    num_questions: int = 500
) -> List[Dict]:
    """
    Generate questions using NotebookLM (Google's AI notebook).
    
    Note: NotebookLM doesn't have a public API yet, so this would require:
    1. Manual export from NotebookLM
    2. Or using Google's Gemini API as alternative
    
    Args:
        knowledge_graph_info: Info about knowledge graph
        num_questions: Number of questions
        
    Returns:
        List of questions
    """
    # NotebookLM doesn't have public API
    # Would need to:
    # 1. Upload knowledge graph info to NotebookLM
    # 2. Ask it to generate questions
    # 3. Export results
    
    print("⚠️ NotebookLM doesn't have public API yet.")
    print("   Alternative: Use Google Gemini API or manual export from NotebookLM")
    
    # Could use Gemini API as alternative
    try:
        import google.generativeai as genai
        # Similar implementation to ChatGPT
        pass
    except ImportError:
        print("   Install: pip install google-generativeai")
    
    return []


if __name__ == "__main__":
    # Example usage
    from .knowledge_graph import KpopKnowledgeGraph
    
    kg = KpopKnowledgeGraph()
    stats = kg.get_statistics()
    
    knowledge_graph_info = {
        "total_entities": stats['total_nodes'],
        "entity_types": list(stats['entity_types'].keys()),
        "relationship_types": list(stats['relationship_types'].keys()),
        "sample_entities": list(kg.get_entities_by_type("Group"))[:10]
    }
    
    # Generate with ChatGPT (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        print("🔄 Generating questions with ChatGPT...")
        questions = generate_questions_with_chatgpt(
            knowledge_graph_info,
            num_questions=200
        )
        print(f"✅ Generated {len(questions)} questions")
    else:
        print("⚠️ OPENAI_API_KEY not set. Skipping ChatGPT generation.")







