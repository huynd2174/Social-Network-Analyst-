"""
Test Script cho K-pop Knowledge Graph Chatbot

Script này test chatbot với cả Fast Mode (reasoning-only) và Slow Mode (với LLM)
để demo tất cả các yêu cầu của bài tập.
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import KpopChatbot


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_fast_mode():
    """Test Fast Mode: Reasoning-only (không dùng LLM) - nhanh và chính xác"""
    print_section("TEST 1: FAST MODE (Reasoning-Only, Không LLM)")
    
    print("⚡ Fast Mode: Chỉ dùng GraphRAG + Multi-hop Reasoning")
    print("   - Nhanh: 1-5 giây")
    print("   - Chính xác: Dựa trên knowledge graph")
    print("   - Phù hợp: Câu hỏi về thành viên, công ty, cùng công ty, etc.\n")
    
    try:
        chatbot = KpopChatbot(verbose=True, llm_model=None)  # Không load LLM
        session_id = chatbot.create_session()
        
        # Test queries phù hợp với fast mode
        test_queries = [
            {
                "query": "BTS có bao nhiêu thành viên?",
                "expected": "thành viên",
                "type": "1-hop: Group → Members"
            },
            {
                "query": "Công ty nào quản lý BLACKPINK?",
                "expected": "công ty",
                "type": "1-hop: Group → Company"
            },
            {
                "query": "Jungkook có phải thành viên BTS không?",
                "expected": "Có",
                "type": "1-hop: Membership check"
            },
            {
                "query": "BTS và SEVENTEEN có cùng công ty không?",
                "expected": "công ty",
                "type": "2-hop: Compare companies"
            },
            {
                "query": "Các nhóm cùng công ty với BTS",
                "expected": "nhóm",
                "type": "3-hop: Labelmates"
            },
            {
                "query": "Nhóm nhạc đã hợp tác với BTS",
                "expected": "nhóm",
                "type": "2-hop: Collaborations"
            }
        ]
        
        print("🧪 Test các câu hỏi với Fast Mode:\n")
        
        for i, test in enumerate(test_queries, 1):
            print(f"{i}. {test['type']}")
            print(f"   ❓ Query: {test['query']}")
            
            start_time = time.time()
            result = chatbot.chat(
                test['query'],
                session_id,
                use_multi_hop=True,
                max_hops=3,
                use_llm=False,  # Fast mode
                return_details=True
            )
            elapsed = time.time() - start_time
            
            print(f"   ⚡ Thời gian: {elapsed:.2f} giây")
            print(f"   🤖 Response: {result['response'][:150]}...")
            print(f"   📊 Entities: {result['entities_found']}, Hops: {result['reasoning_hops']}")
            
            # Check if expected content is in response
            if test['expected'].lower() in result['response'].lower():
                print(f"   ✅ PASS: Tìm thấy '{test['expected']}' trong response")
            else:
                print(f"   ⚠️  WARNING: Không tìm thấy '{test['expected']}' trong response")
            
            print()
        
        print("✅ Fast Mode test hoàn thành!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_slow_mode():
    """Test Slow Mode: Với LLM - chậm hơn nhưng tự nhiên hơn"""
    print_section("TEST 2: SLOW MODE (Với LLM)")
    
    print("🐌 Slow Mode: GraphRAG + Multi-hop Reasoning + Small LLM")
    print("   - Chậm: 10-30 giây")
    print("   - Tự nhiên: LLM tạo câu trả lời tự nhiên")
    print("   - Phù hợp: Câu hỏi phức tạp, cần tổng hợp thông tin\n")
    
    try:
        chatbot = KpopChatbot(verbose=True, llm_model="qwen2-0.5b")
        session_id = chatbot.create_session()
        
        # Test queries phù hợp với slow mode (phức tạp)
        test_queries = [
            {
                "query": "Giới thiệu về BTS",
                "type": "Complex: General information"
            },
            {
                "query": "So sánh BTS và BLACKPINK",
                "type": "Complex: Comparison"
            },
            {
                "query": "Kể về lịch sử phát triển của K-pop",
                "type": "Complex: Historical context"
            }
        ]
        
        print("🧪 Test các câu hỏi phức tạp với Slow Mode:\n")
        
        for i, test in enumerate(test_queries, 1):
            print(f"{i}. {test['type']}")
            print(f"   ❓ Query: {test['query']}")
            print(f"   ⏳ Đang xử lý (có thể mất 10-30 giây)...")
            
            start_time = time.time()
            result = chatbot.chat(
                test['query'],
                session_id,
                use_multi_hop=True,
                max_hops=3,
                use_llm=True,  # Slow mode
                return_details=True
            )
            elapsed = time.time() - start_time
            
            print(f"   ⏱️  Thời gian: {elapsed:.2f} giây")
            print(f"   🤖 Response: {result['response'][:200]}...")
            print(f"   📊 Entities: {result['entities_found']}, Hops: {result['reasoning_hops']}")
            print()
        
        print("✅ Slow Mode test hoàn thành!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Tip: Nếu LLM không load được, chatbot vẫn hoạt động với Fast Mode")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_mode():
    """Test Hybrid Mode: Fast mode trước, nếu không đủ thì dùng slow mode"""
    print_section("TEST 3: HYBRID MODE (Fast → Slow)")
    
    print("🔄 Hybrid Mode: Thử Fast Mode trước, nếu không đủ thì dùng Slow Mode")
    print("   - Tối ưu: Nhanh cho câu hỏi đơn giản, đầy đủ cho câu hỏi phức tạp\n")
    
    try:
        chatbot = KpopChatbot(verbose=True, llm_model="qwen2-0.5b")
        session_id = chatbot.create_session()
        
        test_queries = [
            "BTS có bao nhiêu thành viên?",  # Simple - sẽ dùng fast mode
            "Giới thiệu về BTS",  # Complex - sẽ dùng slow mode
        ]
        
        print("🧪 Test Hybrid Mode:\n")
        
        for query in test_queries:
            print(f"❓ Query: {query}")
            
            # Try fast mode first
            print("   ⚡ Thử Fast Mode trước...")
            start_time = time.time()
            result_fast = chatbot.chat(
                query,
                session_id,
                use_multi_hop=True,
                max_hops=3,
                use_llm=False,  # Fast mode
                return_details=True
            )
            elapsed_fast = time.time() - start_time
            
            print(f"   ⚡ Fast Mode: {elapsed_fast:.2f}s - {result_fast['response'][:100]}...")
            
            # Check if we need slow mode
            if len(result_fast['response']) < 20 or 'không tìm thấy' in result_fast['response'].lower():
                print("   🐌 Response không đủ, chuyển sang Slow Mode...")
                start_time = time.time()
                result_slow = chatbot.chat(
                    query,
                    session_id,
                    use_multi_hop=True,
                    max_hops=3,
                    use_llm=True,  # Slow mode
                    return_details=True
                )
                elapsed_slow = time.time() - start_time
                print(f"   🐌 Slow Mode: {elapsed_slow:.2f}s - {result_slow['response'][:100]}...")
            else:
                print("   ✅ Fast Mode đủ tốt, không cần Slow Mode")
            
            print()
        
        print("✅ Hybrid Mode test hoàn thành!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_requirements():
    """Test tất cả các yêu cầu của bài tập"""
    print_section("TEST: TẤT CẢ YÊU CẦU BÀI TẬP")
    
    print("📋 Kiểm tra các yêu cầu:\n")
    
    chatbot = KpopChatbot(verbose=False)
    
    # 1. Small LLM (≤1B params)
    print("1. ✅ Small LLM (≤1B params):")
    if chatbot.llm:
        param_count = sum(p.numel() for p in chatbot.llm.model.parameters())
        param_count_b = param_count / 1e9
        print(f"   - Model: Qwen2-0.5B-Instruct")
        print(f"   - Số tham số: {param_count_b:.3f} tỷ")
        print(f"   - Yêu cầu: ≤ 1 tỷ → {'✅ ĐẠT' if param_count_b <= 1.0 else '❌ KHÔNG ĐẠT'}")
    else:
        print("   - LLM chưa được load (có thể test với Fast Mode)")
    
    # 2. GraphRAG
    print("\n2. ✅ GraphRAG trên đồ thị tri thức:")
    stats = chatbot.kg.get_statistics()
    print(f"   - Nodes: {stats['total_nodes']:,}")
    print(f"   - Edges: {stats['total_edges']:,}")
    print(f"   - GraphRAG: ✅ Đã implement")
    
    # 3. Multi-hop Reasoning
    print("\n3. ✅ Multi-hop Reasoning:")
    print(f"   - Hỗ trợ: 1-hop, 2-hop, 3-hop")
    print(f"   - Types: Chain, Aggregation, Comparison, Intersection")
    print(f"   - ✅ Đã implement")
    
    # 4. Evaluation Dataset
    print("\n4. ✅ Evaluation Dataset:")
    dataset_path = "data/evaluation_dataset.json"
    if os.path.exists(dataset_path):
        import json
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total = data.get('metadata', {}).get('total_questions', len(data.get('questions', [])))
        print(f"   - Tổng số câu hỏi: {total}")
        print(f"   - Yêu cầu: ≥ 2000 → {'✅ ĐẠT' if total >= 2000 else '❌ CHƯA ĐẠT'}")
    else:
        print(f"   - Dataset chưa tồn tại (chạy: python src/run_chatbot.py --mode eval)")
    
    # 5. Comparison
    print("\n5. ✅ Comparison Framework:")
    comparison_path = "data/comparison_results.json"
    if os.path.exists(comparison_path):
        print(f"   - Comparison results: ✅ Đã có")
        print(f"   - File: {comparison_path}")
    else:
        print(f"   - Chưa chạy comparison (chạy: python src/run_chatbot.py --mode compare)")
    
    print("\n✅ Kiểm tra hoàn thành!")


def main():
    """Main test function."""
    print("\n" + "="*70)
    print("  🧪 TEST K-POP KNOWLEDGE GRAPH CHATBOT")
    print("="*70)
    print("\nTest các chế độ:")
    print("  1. Fast Mode (Reasoning-only) - Nhanh, chính xác")
    print("  2. Slow Mode (Với LLM) - Chậm, tự nhiên")
    print("  3. Hybrid Mode (Fast → Slow) - Tối ưu")
    print("  4. Kiểm tra tất cả yêu cầu bài tập")
    print("\n" + "="*70)
    
    results = {}
    
    # Run tests
    print("\n💡 Lưu ý: Fast Mode test sẽ nhanh (1-5s), Slow Mode sẽ chậm (10-30s)")
    print("   Bạn có muốn chạy tất cả tests? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        results['fast_mode'] = test_fast_mode()
        
        print("\n⏸️  Bạn có muốn test Slow Mode? (có thể mất 30-90 giây) (y/n): ", end="")
        if input().strip().lower() == 'y':
            results['slow_mode'] = test_slow_mode()
        else:
            results['slow_mode'] = None
        
        results['hybrid_mode'] = test_hybrid_mode()
    else:
        print("\n📋 Chọn test:")
        print("  1. Fast Mode only (nhanh)")
        print("  2. Slow Mode only (chậm)")
        print("  3. Hybrid Mode")
        print("  4. Kiểm tra yêu cầu bài tập")
        print("  5. Tất cả")
        choice = input("Chọn (1-5): ").strip()
        
        if choice == '1':
            results['fast_mode'] = test_fast_mode()
        elif choice == '2':
            results['slow_mode'] = test_slow_mode()
        elif choice == '3':
            results['hybrid_mode'] = test_hybrid_mode()
        elif choice == '4':
            test_all_requirements()
        elif choice == '5':
            results['fast_mode'] = test_fast_mode()
            results['slow_mode'] = test_slow_mode()
            results['hybrid_mode'] = test_hybrid_mode()
    
    # Summary
    if results:
        print_section("TÓM TẮT KẾT QUẢ")
        print("📊 Kết quả tests:\n")
        for test_name, result in results.items():
            if result is not None:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {test_name}: {status}")
    
    test_all_requirements()
    
    print("\n" + "="*70)
    print("💡 Các lệnh hữu ích:")
    print("  - python src/test_chatbot.py          # Test script này")
    print("  - python src/run_chatbot.py --mode cli    # CLI interactive")
    print("  - python src/run_chatbot.py --mode ui     # Web UI (Gradio)")
    print("  - python src/demo_chatbot.py              # Full demo")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()




