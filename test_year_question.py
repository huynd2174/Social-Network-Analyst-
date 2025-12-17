"""
Test script cho câu hỏi về năm hoạt động
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatbot import KpopChatbot

def test_year_questions():
    """Test các câu hỏi về năm hoạt động"""
    print("="*60)
    print("🧪 Test Câu Hỏi Về Năm Hoạt Động")
    print("="*60)
    
    # Initialize chatbot
    print("\n🔄 Đang khởi tạo chatbot...")
    chatbot = KpopChatbot(
        llm_model="qwen2-0.5b",
        verbose=False
    )
    print("✅ Chatbot đã sẵn sàng!\n")
    
    # Test queries
    test_queries = [
        "năm hoạt động của nhóm nhạc có ca sĩ đã thể hiện bài hát Rockstar",
        "năm hoạt động của BTS",
        "năm hoạt động của nhóm nhạc đã thể hiện ca khúc Rockstar",
    ]
    
    session_id = chatbot.create_session()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print('='*60)
        
        try:
            result = chatbot.chat(
                query,
                session_id=session_id,
                use_multi_hop=True,
                max_hops=3,
                return_details=True
            )
            
            print(f"\n🤖 Response:")
            print(f"   {result['response']}")
            
            print(f"\n📊 Details:")
            print(f"   Entities found: {result.get('entities_found', 0)}")
            print(f"   Reasoning hops: {result.get('reasoning_hops', 0)}")
            
            if result.get('reasoning'):
                reasoning = result['reasoning']
                print(f"   Reasoning type: {reasoning.get('type', 'N/A')}")
                if reasoning.get('explanation'):
                    print(f"   Explanation: {reasoning['explanation']}")
            
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_year_questions()





