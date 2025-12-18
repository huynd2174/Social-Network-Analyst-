"""
Test script cho logic reasoning về năm (không cần LLM)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatbot.knowledge_graph import KpopKnowledgeGraph
from chatbot.multi_hop_reasoning import MultiHopReasoner

def test_year_reasoning():
    """Test logic reasoning về năm hoạt động"""
    print("="*60)
    print("🧪 Test Logic Reasoning Về Năm Hoạt Động")
    print("="*60)
    
    # Initialize KG và Reasoner
    print("\n🔄 Đang load Knowledge Graph...")
    kg = KpopKnowledgeGraph()
    reasoner = MultiHopReasoner(kg)
    print("✅ Knowledge Graph đã sẵn sàng!\n")
    
    # Test queries
    test_queries = [
        {
            "query": "năm hoạt động của nhóm nhạc có ca sĩ đã thể hiện bài hát Rockstar",
            "description": "3-hop: Song → Artist → Group → Year"
        },
        {
            "query": "năm hoạt động của nhóm nhạc đã thể hiện ca khúc Rockstar",
            "description": "2-hop: Song → Group → Year"
        },
        {
            "query": "năm hoạt động của BTS",
            "description": "1-hop: Group → Year"
        },
    ]
    
    for i, test in enumerate(test_queries, 1):
        query = test['query']
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['description']}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            result = reasoner.reason(
                query,
                start_entities=[],
                max_hops=3
            )
            
            print(f"\n🤖 Answer:")
            print(f"   {result.answer_text}")
            
            print(f"\n📊 Details:")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Reasoning type: {result.reasoning_type.value}")
            print(f"   Steps: {len(result.steps)}")
            
            if result.steps:
                print(f"\n🔍 Reasoning Steps:")
                for step in result.steps:
                    print(f"   Step {step.hop_number}: {step.explanation}")
            
            if result.answer_entities:
                print(f"\n📍 Answer Entities:")
                for entity in result.answer_entities[:5]:
                    print(f"   - {entity}")
            
            if result.explanation:
                print(f"\n💡 Full Explanation:")
                print(f"   {result.explanation}")
                
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_year_reasoning()






