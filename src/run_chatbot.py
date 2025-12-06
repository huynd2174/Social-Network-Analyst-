"""
Main Runner Script for K-pop Knowledge Graph Chatbot

This script provides different modes to run the chatbot:
1. Interactive CLI mode
2. Web UI mode (Gradio)
3. Evaluation mode
4. Comparison mode

Usage:
    python src/run_chatbot.py --mode cli
    python src/run_chatbot.py --mode ui
    python src/run_chatbot.py --mode eval
    python src/run_chatbot.py --mode compare
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import KpopChatbot
from chatbot.evaluation import EvaluationDatasetGenerator
from chatbot.comparison import ChatbotComparison


def run_cli_mode():
    """Run interactive CLI chatbot."""
    print("\n" + "="*60)
    print("🎤 K-pop Knowledge Graph Chatbot - Interactive Mode")
    print("="*60)
    print("Nhập câu hỏi về K-pop hoặc gõ 'quit' để thoát.\n")
    print("💡 Tip: Dùng lệnh nhanh để tránh chờ LLM:")
    print("   - 'members BTS' hoặc 'BTS members'")
    print("   - 'company BLACKPINK'")
    print("   - 'same BTS SEVENTEEN'")
    print("")
    
    # Initialize
    chatbot = KpopChatbot(verbose=True)
    session_id = chatbot.create_session()
    
    print("\n✅ Sẵn sàng! Hãy đặt câu hỏi về K-pop.\n")
    print("⚠️  Lưu ý: Câu hỏi thường sẽ chậm (5-30 giây) vì LLM chạy trên CPU.")
    print("   Dùng lệnh đặc biệt để nhanh hơn!\n")
    
    while True:
        try:
            query = input("Bạn: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q', 'thoát']:
                print("\n👋 Tạm biệt!")
                break
                
            if query.lower() == 'help':
                print("""
📚 Các lệnh đặc biệt:
- 'members <group>': Xem thành viên nhóm
- 'company <group>': Xem công ty quản lý
- 'same <group1> <group2>': Kiểm tra cùng công ty
- 'path <entity1> <entity2>': Tìm đường đi
- 'stats': Xem thống kê
- 'quit': Thoát
                """)
                continue
                
            if query.lower() == 'stats':
                stats = chatbot.get_statistics()
                print(f"\n📊 Thống kê:")
                print(f"  - Nodes: {stats['knowledge_graph']['total_nodes']}")
                print(f"  - Edges: {stats['knowledge_graph']['total_edges']}")
                continue
                
            # Handle "members <group>" or "<group> members"
            if query.lower().startswith('members '):
                group = query[8:].strip()
                result = chatbot.get_group_members(group)
                print(f"\n🤖 {result['answer']}\n")
                continue
            elif query.lower().endswith(' members'):
                group = query[:-8].strip()
                result = chatbot.get_group_members(group)
                print(f"\n🤖 {result['answer']}\n")
                continue
                
            if query.lower().startswith('company '):
                group = query[8:].strip()
                result = chatbot.get_group_company(group)
                print(f"\n🤖 {result['answer']}\n")
                continue
                
            if query.lower().startswith('same '):
                parts = query[5:].strip().split()
                if len(parts) >= 2:
                    result = chatbot.check_same_company(parts[0], parts[1])
                    print(f"\n🤖 {result['answer']}\n")
                continue
                
            if query.lower().startswith('path '):
                parts = query[5:].strip().split()
                if len(parts) >= 2:
                    result = chatbot.find_path(parts[0], parts[1])
                    print(f"\n🤖 {result['description']}\n")
                continue
                
            # Normal chat - smart routing based on query type
            print("🔄 Đang xử lý...")
            
            # Check if it's a simple query that should use reasoning only
            simple_keywords = ['members', 'thành viên', 'member', 'company', 'công ty', 
                             'cùng công ty', 'same company', 'labelmate']
            is_simple = any(kw in query.lower() for kw in simple_keywords)
            
            if is_simple:
                # Simple queries: Use reasoning only (fast and accurate)
                result = chatbot.chat(
                    query, 
                    session_id, 
                    use_multi_hop=True,
                    return_details=True,
                    use_llm=False  # Skip LLM for simple queries
                )
            else:
                # Complex queries: Try reasoning first, then LLM if needed
                result = chatbot.chat(
                    query, 
                    session_id, 
                    use_multi_hop=True,
                    return_details=True,
                    use_llm=False  # Try reasoning first
                )
                
                # Only use LLM if reasoning didn't give good answer
                if not result['response'] or len(result['response']) < 20 or 'không tìm thấy' in result['response'].lower():
                    print("   (Đang dùng LLM cho câu hỏi phức tạp... có thể mất 10-30 giây)")
                    result = chatbot.chat(
                        query, 
                        session_id, 
                        use_multi_hop=True,
                        return_details=True,
                        use_llm=True  # Use LLM for complex queries
                    )
            
            print(f"\n🤖 {result['response']}")
            print(f"   [Entities: {result['entities_found']}, Hops: {result['reasoning_hops']}]\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}\n")


def run_ui_mode():
    """Run Gradio web UI."""
    from chatbot.app import main as run_app
    run_app()


def run_eval_mode(num_questions: int = 2000):
    """Generate evaluation dataset."""
    print("\n" + "="*60)
    print("📝 Evaluation Dataset Generator")
    print("="*60)
    
    generator = EvaluationDatasetGenerator()
    stats = generator.generate_full_dataset(
        target_count=num_questions,
        output_path="data/evaluation_dataset.json"
    )
    
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def run_compare_mode(max_questions: int = 500):
    """Run chatbot comparison."""
    print("\n" + "="*60)
    print("🔬 Chatbot Comparison Mode")
    print("="*60)
    
    # Initialize chatbot
    chatbot = KpopChatbot(verbose=True)
    
    # Check if dataset exists
    dataset_path = "data/evaluation_dataset.json"
    if not os.path.exists(dataset_path):
        print("\n📝 Generating evaluation dataset first...")
        generator = EvaluationDatasetGenerator()
        generator.generate_full_dataset(output_path=dataset_path)
        
    # Run comparison
    comparison = ChatbotComparison(kpop_chatbot=chatbot)
    questions = comparison.load_evaluation_dataset(dataset_path)
    
    results = comparison.compare_chatbots(
        questions,
        include_chatgpt=False,  # Set True if API key available
        include_baseline=True,
        max_questions=max_questions
    )
    
    print("\n✅ Comparison complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="K-pop Knowledge Graph Chatbot"
    )
    
    parser.add_argument(
        '--mode',
        choices=['cli', 'ui', 'eval', 'compare'],
        default='cli',
        help='Chế độ chạy: cli (command line), ui (web), eval (tạo dataset), compare (so sánh)'
    )
    
    parser.add_argument(
        '--num-questions',
        type=int,
        default=2000,
        help='Số câu hỏi cho eval mode (mặc định: 2000)'
    )
    
    parser.add_argument(
        '--max-compare',
        type=int,
        default=500,
        help='Số câu hỏi tối đa cho compare mode (mặc định: 500)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        run_cli_mode()
    elif args.mode == 'ui':
        run_ui_mode()
    elif args.mode == 'eval':
        run_eval_mode(args.num_questions)
    elif args.mode == 'compare':
        run_compare_mode(args.max_compare)


if __name__ == "__main__":
    main()

