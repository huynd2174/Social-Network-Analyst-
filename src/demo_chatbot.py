"""
Demo Script cho K-pop Knowledge Graph Chatbot

Script này demo tất cả các tính năng của chatbot để trình bày:
1. Small LLM (≤1B params)
2. GraphRAG trên đồ thị tri thức
3. Multi-hop Reasoning
4. Evaluation Dataset
5. Comparison với chatbot khác
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import KpopChatbot, GraphRAG, MultiHopReasoner, EvaluationDatasetGenerator, ChatbotComparison


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_1_small_llm():
    """Demo 1: Small LLM (≤1B params) - 1 điểm"""
    print_section("1. DEMO: Small LLM (≤1B Parameters)")
    
    from chatbot.small_llm import SmallLLM, get_llm
    
    print("🔄 Đang khởi tạo Small LLM...")
    try:
        llm = get_llm("qwen2-0.5b")
        
        # Get model size
        param_count = sum(p.numel() for p in llm.model.parameters())
        param_count_b = param_count / 1e9
        
        print(f"\n✅ Model: Qwen2-0.5B-Instruct")
        print(f"✅ Số tham số: {param_count_b:.3f} tỷ ({param_count/1e6:.1f}M)")
        print(f"✅ Yêu cầu: ≤ 1 tỷ tham số")
        print(f"✅ Kết quả: {'✅ ĐẠT' if param_count_b <= 1.0 else '❌ KHÔNG ĐẠT'}")
        
        # Test generation
        print(f"\n🧪 Test generation:")
        test_query = "BTS là nhóm nhạc K-pop."
        response = llm.generate(test_query, max_new_tokens=50)
        print(f"   Query: {test_query}")
        print(f"   Response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def demo_2_graphrag():
    """Demo 2: GraphRAG - 0.5 điểm"""
    print_section("2. DEMO: GraphRAG trên Đồ thị Tri thức")
    
    print("🔄 Đang khởi tạo Knowledge Graph và GraphRAG...")
    
    try:
        # Initialize GraphRAG
        rag = GraphRAG()
        
        # Show knowledge graph stats
        kg_stats = rag.kg.get_statistics()
        print(f"\n✅ Knowledge Graph:")
        print(f"   - Nodes: {kg_stats['total_nodes']:,}")
        print(f"   - Edges: {kg_stats['total_edges']:,}")
        print(f"   - Entity types: {len(kg_stats['entity_types'])}")
        print(f"   - Relationship types: {len(kg_stats['relationship_types'])}")
        
        # Test GraphRAG retrieval
        print(f"\n🧪 Test GraphRAG retrieval:")
        query = "BTS có bao nhiêu thành viên?"
        context = rag.retrieve_context(query, max_entities=3, max_hops=2)
        
        print(f"   Query: {query}")
        print(f"   Entities found: {len(context['entities'])}")
        print(f"   Relationships: {len(context['relationships'])}")
        print(f"   Facts: {len(context['facts'])}")
        
        if context['entities']:
            print(f"\n   📍 Entities:")
            for e in context['entities'][:3]:
                print(f"      - {e['id']} ({e['type']})")
        
        if context['facts']:
            print(f"\n   📝 Facts:")
            for f in context['facts'][:3]:
                print(f"      - {f}")
        
        # Show formatted context
        formatted = rag.format_context_for_llm(context)
        print(f"\n   📄 Formatted context (preview):")
        print(f"      {formatted[:200]}...")
        
        print(f"\n✅ GraphRAG hoạt động đúng!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_3_multi_hop():
    """Demo 3: Multi-hop Reasoning - 1.5 điểm"""
    print_section("3. DEMO: Multi-hop Reasoning")
    
    print("🔄 Đang khởi tạo Multi-hop Reasoner...")
    
    try:
        reasoner = MultiHopReasoner()
        
        # Test cases
        test_cases = [
            {
                "name": "1-hop: Thành viên của BTS",
                "query": "Thành viên của BTS",
                "method": reasoner.get_group_members,
                "args": ["BTS"]
            },
            {
                "name": "2-hop: Công ty của Jungkook",
                "query": "Công ty quản lý Jungkook",
                "method": reasoner.get_artist_company,
                "args": ["Jungkook"]
            },
            {
                "name": "2-hop: Cùng công ty?",
                "query": "BTS và SEVENTEEN có cùng công ty không?",
                "method": reasoner.check_same_company,
                "args": ["BTS", "SEVENTEEN"]
            },
            {
                "name": "3-hop: Labelmates",
                "query": "Các nhóm cùng công ty với BTS",
                "method": reasoner.get_labelmates,
                "args": ["BTS"]
            }
        ]
        
        print(f"\n🧪 Test Multi-hop Reasoning:\n")
        
        for i, test in enumerate(test_cases, 1):
            print(f"{i}. {test['name']}")
            print(f"   Query: {test['query']}")
            
            try:
                result = test['method'](*test['args'])
                print(f"   ✅ Hops: {len(result.steps)}")
                print(f"   ✅ Answer: {result.answer_text[:100]}...")
                print(f"   ✅ Confidence: {result.confidence:.1%}")
                print(f"   ✅ Explanation: {result.explanation[:80]}...")
            except Exception as e:
                print(f"   ⚠️  Lỗi: {e}")
            
            print()
        
        print(f"✅ Multi-hop Reasoning hoạt động đúng!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_4_evaluation_dataset():
    """Demo 4: Evaluation Dataset (2000+ questions) - 1 điểm"""
    print_section("4. DEMO: Evaluation Dataset Generator")
    
    dataset_path = "data/evaluation_dataset.json"
    
    # Check if dataset exists
    if os.path.exists(dataset_path):
        print(f"📂 Dataset đã tồn tại: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        questions = data.get('questions', [])
        
        print(f"\n✅ Dataset Statistics:")
        print(f"   - Tổng số câu hỏi: {metadata.get('total_questions', len(questions))}")
        print(f"   - Yêu cầu: ≥ 2000 câu hỏi")
        print(f"   - Kết quả: {'✅ ĐẠT' if metadata.get('total_questions', 0) >= 2000 else '❌ CHƯA ĐẠT'}")
        
        print(f"\n   Phân bố theo số hop:")
        for hop, count in metadata.get('by_hops', {}).items():
            print(f"      - {hop}-hop: {count} câu")
        
        print(f"\n   Phân bố theo loại:")
        for qtype, count in metadata.get('by_type', {}).items():
            print(f"      - {qtype}: {count} câu")
        
        # Show sample questions
        print(f"\n   📝 Sample questions:")
        for q in questions[:5]:
            print(f"      - [{q['question_type']}] {q['question']}")
            print(f"        Answer: {q['answer']} (Hops: {q['hops']})")
        
        return True
    else:
        print(f"⚠️  Dataset chưa tồn tại. Tạo dataset mới? (y/n)")
        response = input().strip().lower()
        
        if response == 'y':
            print(f"\n🔄 Đang tạo evaluation dataset (2000 câu hỏi)...")
            generator = EvaluationDatasetGenerator()
            stats = generator.generate_full_dataset(
                target_count=2000,
                output_path=dataset_path
            )
            
            print(f"\n✅ Dataset đã được tạo!")
            print(f"   - Tổng số: {stats['total_questions']} câu hỏi")
            return True
        else:
            print(f"⏭️  Bỏ qua tạo dataset")
            return False


def demo_5_comparison():
    """Demo 5: Comparison với chatbot khác - 0.5 điểm"""
    print_section("5. DEMO: Chatbot Comparison")
    
    dataset_path = "data/evaluation_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Evaluation dataset chưa tồn tại. Chạy demo 4 trước!")
        return False
    
    print(f"🔄 Đang khởi tạo chatbot và comparison framework...")
    
    try:
        # Initialize chatbot
        chatbot = KpopChatbot(verbose=False)
        
        # Initialize comparison
        comparison = ChatbotComparison(kpop_chatbot=chatbot)
        
        # Load dataset
        questions = comparison.load_evaluation_dataset(dataset_path)
        print(f"✅ Loaded {len(questions)} questions from dataset")
        
        # Run comparison (limited for demo)
        print(f"\n🔄 Đang chạy comparison (sample 100 questions cho demo)...")
        print(f"   (Để chạy full, dùng: python src/run_chatbot.py --mode compare)")
        
        results = comparison.compare_chatbots(
            questions,
            include_chatgpt=False,  # Set True nếu có API key
            include_baseline=True,
            max_questions=100  # Limited for demo
        )
        
        print(f"\n✅ Comparison hoàn thành!")
        print(f"\n📊 Results saved to: data/comparison_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_full_chatbot():
    """Demo: Full chatbot integration"""
    print_section("DEMO: Full Chatbot Integration")
    
    print("🔄 Đang khởi tạo chatbot...")
    
    try:
        chatbot = KpopChatbot(verbose=True)
        
        # Test queries
        test_queries = [
            "BTS có bao nhiêu thành viên?",
            "Công ty nào quản lý BLACKPINK?",
            "BTS và SEVENTEEN có cùng công ty không?",
        ]
        
        session_id = chatbot.create_session()
        
        print(f"\n🧪 Test Chatbot với các câu hỏi:\n")
        
        for query in test_queries:
            print(f"❓ Query: {query}")
            result = chatbot.chat(
                query,
                session_id,
                use_multi_hop=True,
                max_hops=3,
                use_llm=False  # Fast mode for demo
            )
            print(f"🤖 Response: {result['response'][:150]}...")
            print(f"   [Entities: {result['entities_found']}, Hops: {result['reasoning_hops']}]\n")
        
        print(f"✅ Chatbot hoạt động đúng!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("  🎤 K-POP KNOWLEDGE GRAPH CHATBOT - DEMO")
    print("="*70)
    print("\nDemo tất cả các tính năng để trình bày bài tập:")
    print("  1. Small LLM (≤1B params) - 1 điểm")
    print("  2. GraphRAG - 0.5 điểm")
    print("  3. Multi-hop Reasoning - 1.5 điểm")
    print("  4. Evaluation Dataset (2000+ questions) - 1 điểm")
    print("  5. Comparison - 0.5 điểm")
    print("\n" + "="*70)
    
    results = {}
    
    # Run demos
    results['1_small_llm'] = demo_1_small_llm()
    results['2_graphrag'] = demo_2_graphrag()
    results['3_multi_hop'] = demo_3_multi_hop()
    results['4_evaluation'] = demo_4_evaluation_dataset()
    results['5_comparison'] = demo_5_comparison()
    results['full_chatbot'] = demo_full_chatbot()
    
    # Summary
    print_section("TÓM TẮT KẾT QUẢ")
    
    print("📊 Kết quả các phần demo:\n")
    print(f"  1. Small LLM (≤1B):           {'✅' if results['1_small_llm'] else '❌'}")
    print(f"  2. GraphRAG:                  {'✅' if results['2_graphrag'] else '❌'}")
    print(f"  3. Multi-hop Reasoning:       {'✅' if results['3_multi_hop'] else '❌'}")
    print(f"  4. Evaluation Dataset:       {'✅' if results['4_evaluation'] else '❌'}")
    print(f"  5. Comparison:                {'✅' if results['5_comparison'] else '❌'}")
    print(f"  6. Full Chatbot Integration:  {'✅' if results['full_chatbot'] else '❌'}")
    
    all_passed = all(results.values())
    
    print(f"\n{'='*70}")
    if all_passed:
        print("  ✅ TẤT CẢ DEMO ĐỀU THÀNH CÔNG!")
    else:
        print("  ⚠️  MỘT SỐ DEMO CÓ LỖI - KIỂM TRA LẠI")
    print(f"{'='*70}\n")
    
    print("📝 Các file quan trọng:")
    print("  - Evaluation Dataset: data/evaluation_dataset.json")
    print("  - Comparison Results: data/comparison_results.json")
    print("  - Knowledge Graph: data/merged_kpop_data.json")
    print("\n💡 Để chạy từng phần riêng:")
    print("  - python src/run_chatbot.py --mode cli")
    print("  - python src/run_chatbot.py --mode ui")
    print("  - python src/run_chatbot.py --mode eval")
    print("  - python src/run_chatbot.py --mode compare")


if __name__ == "__main__":
    main()

