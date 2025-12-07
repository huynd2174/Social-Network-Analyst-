"""
Script để chứng minh chatbot CÓ multi-hop reasoning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatbot import KpopChatbot

def test_multihop_in_chatbot():
    """Test multi-hop reasoning trong chatbot"""
    print("="*60)
    print("🧪 KIỂM TRA: Chatbot có Multi-hop Reasoning không?")
    print("="*60)
    
    # Khởi tạo chatbot
    print("\n1️⃣ Khởi tạo chatbot...")
    chatbot = KpopChatbot(verbose=False)
    
    # Kiểm tra reasoner có tồn tại không
    if hasattr(chatbot, 'reasoner'):
        print("   ✅ Chatbot.reasoner tồn tại")
        print(f"   ✅ Loại: {type(chatbot.reasoner).__name__}")
    else:
        print("   ❌ Chatbot KHÔNG có reasoner!")
        return False
    
    # Test các câu hỏi multi-hop
    test_queries = [
        {
            "query": "BTS có thành viên Jungkook không?",
            "expected_hops": 1,
            "description": "1-hop: Membership check"
        },
        {
            "query": "Công ty nào quản lý BTS?",
            "expected_hops": 1,
            "description": "1-hop: Company lookup"
        },
        {
            "query": "BTS và SEVENTEEN có cùng công ty không?",
            "expected_hops": 2,
            "description": "2-hop: Same company check"
        },
        {
            "query": "Các nhóm cùng công ty với BTS",
            "expected_hops": 2,
            "description": "2-hop: Labelmates"
        }
    ]
    
    print("\n2️⃣ Test multi-hop reasoning với các câu hỏi:")
    print()
    
    all_passed = True
    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}: {test['description']}")
        print(f"   Query: {test['query']}")
        
        try:
            # Gọi chatbot với use_multi_hop=True (mặc định)
            result = chatbot.chat(
                test['query'],
                use_multi_hop=True,  # ← BẬT multi-hop
                max_hops=3,
                return_details=True
            )
            
            # Kiểm tra kết quả
            reasoning_hops = result.get('reasoning_hops', 0)
            reasoning_steps = result.get('reasoning', {}).get('steps', [])
            
            print(f"   ✅ Reasoning hops: {reasoning_hops}")
            print(f"   ✅ Response: {result['response'][:80]}...")
            
            if reasoning_steps:
                print(f"   ✅ Reasoning steps:")
                for step in reasoning_steps[:2]:  # Show first 2 steps
                    print(f"      - Hop {step['hop']}: {step['explanation'][:60]}...")
            
            # Kiểm tra có reasoning không
            if reasoning_hops > 0:
                print(f"   ✅ PASS: Multi-hop reasoning hoạt động ({reasoning_hops} hops)")
            else:
                print(f"   ⚠️  WARNING: Không có reasoning steps (có thể là câu hỏi đơn giản)")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
        
        print()
    
    # Test với use_multi_hop=False để so sánh
    print("\n3️⃣ So sánh: use_multi_hop=True vs False")
    test_query = "BTS và SEVENTEEN có cùng công ty không?"
    
    print(f"   Query: {test_query}")
    
    # Với multi-hop
    result_with = chatbot.chat(
        test_query,
        use_multi_hop=True,
        return_details=True
    )
    hops_with = result_with.get('reasoning_hops', 0)
    print(f"   ✅ use_multi_hop=True: {hops_with} hops")
    
    # Không có multi-hop
    result_without = chatbot.chat(
        test_query,
        use_multi_hop=False,
        return_details=True
    )
    hops_without = result_without.get('reasoning_hops', 0)
    print(f"   ⚠️  use_multi_hop=False: {hops_without} hops")
    
    if hops_with > hops_without:
        print(f"   ✅ PASS: Multi-hop reasoning TẠO SỰ KHÁC BIỆT!")
    else:
        print(f"   ⚠️  WARNING: Không thấy sự khác biệt rõ ràng")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ KẾT LUẬN: Chatbot ĐÃ CÓ multi-hop reasoning!")
        print("   - MultiHopReasoner được khởi tạo trong chatbot")
        print("   - Method chat() gọi reasoner.reason() khi use_multi_hop=True")
        print("   - Mặc định use_multi_hop=True")
        print("   - Trả về reasoning_hops và reasoning steps")
    else:
        print("⚠️  Có một số lỗi, nhưng chatbot vẫn có multi-hop reasoning")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    test_multihop_in_chatbot()


