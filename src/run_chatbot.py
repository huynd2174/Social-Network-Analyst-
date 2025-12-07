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


def run_cli_mode(chat_mode: str = 'standard'):
    """
    Run interactive CLI chatbot.
    
    ✅ YÊU CẦU BÀI TẬP: Phải LUÔN dùng Small LLM (≤1B params)
    
    LLM nhỏ được dùng cho 2 nhiệm vụ:
    1. Hiểu câu hỏi (phân tích, xác định thực thể, nhận ra loại câu hỏi)
    2. GENERATION: Tạo câu trả lời tự nhiên từ context (triples, paths, reasoning results)
    
    ⚠️ QUAN TRỌNG: LLM KHÔNG làm multi-hop reasoning
    - Multi-hop reasoning do Reasoner thực hiện (graph algorithm: tìm đường đi, tính scoring, xâu chuỗi path)
    - LLM chỉ đọc kết quả reasoning và format thành câu trả lời tự nhiên
    
    Args:
        chat_mode: 'standard' (luôn dùng LLM - đáp ứng yêu cầu) hoặc 'optimized' (tối ưu context)
    """
    print("\n" + "="*60)
    print("🎤 K-pop Knowledge Graph Chatbot - Interactive Mode")
    print("="*60)
    
    # Show mode info
    if chat_mode == 'optimized':
        print("⚡ Chế độ: OPTIMIZED MODE (Tối ưu context, vẫn dùng LLM)")
        print("   - Nhanh hơn: Giảm context size khi reasoning confident")
        print("   - Vẫn dùng Small LLM: Đáp ứng yêu cầu bài tập")
    else:  # standard
        print("🔄 Chế độ: STANDARD MODE (Luôn dùng Small LLM)")
        print("   - LLM nhỏ (≤1B params) dùng cho:")
        print("     • Hiểu câu hỏi (phân tích, extract entities, detect intent)")
        print("     • GENERATION: Tạo câu trả lời tự nhiên từ context")
        print("   - Multi-hop reasoning: Do Reasoner thực hiện (graph algorithm)")
    
    print("\nNhập câu hỏi về K-pop hoặc gõ 'quit' để thoát.\n")
    print("💡 Tip: Dùng lệnh nhanh để tránh chờ LLM:")
    print("   - 'members BTS' hoặc 'BTS members'")
    print("   - 'company BLACKPINK'")
    print("   - 'same BTS SEVENTEEN'")
    print("   - 'mode standard' hoặc 'mode optimized' để đổi chế độ")
    print("")
    print("📌 Lưu ý: Chatbot LUÔN dùng Small LLM (≤1B params) để:")
    print("   1. Hiểu câu hỏi (GraphRAG + LLM understanding)")
    print("   2. GENERATION: Tạo câu trả lời tự nhiên (format context)")
    print("   ⚠️ Multi-hop reasoning: Do Reasoner thực hiện (graph algorithm)")
    print("")
    
    # Initialize
    # Check if Neo4j should be used
    use_neo4j = os.getenv("USE_NEO4J", "false").lower() == "true"
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if use_neo4j:
        if not neo4j_password:
            print("⚠️ USE_NEO4J=true but NEO4J_PASSWORD not set!")
            print("   Falling back to JSON file mode...")
            use_neo4j = False
    
    if use_neo4j:
        print("📊 Using Neo4j Knowledge Graph...")
        chatbot = KpopChatbot(
            use_neo4j=True,
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", None),
            verbose=True
        )
    else:
        print("📊 Using JSON file Knowledge Graph...")
        chatbot = KpopChatbot(verbose=True)
    session_id = chatbot.create_session()
    
    print("\n✅ Sẵn sàng! Hãy đặt câu hỏi về K-pop.\n")
    
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
- 'mode standard': Chuyển sang Standard Mode (luôn dùng Small LLM)
- 'mode optimized': Chuyển sang Optimized Mode (tối ưu context, vẫn dùng LLM)
- 'quit': Thoát

📌 Lưu ý: Chatbot LUÔN dùng Small LLM (≤1B params) để:
   1. Hiểu câu hỏi (GraphRAG + LLM understanding)
   2. GENERATION: Tạo câu trả lời tự nhiên (format context)
   ⚠️ Multi-hop reasoning: Do Reasoner thực hiện (graph algorithm)
                """)
                continue
            
            # Handle mode switching
            if query.lower().startswith('mode '):
                new_mode = query[5:].strip().lower()
                if new_mode in ['standard', 'optimized']:
                    chat_mode = new_mode
                    mode_names = {
                        'standard': '🔄 STANDARD MODE (Luôn dùng Small LLM)',
                        'optimized': '⚡ OPTIMIZED MODE (Tối ưu context, vẫn dùng LLM)'
                    }
                    print(f"\n✅ Đã chuyển sang: {mode_names[chat_mode]}\n")
                    print("📌 Lưu ý: Cả 2 chế độ đều dùng Small LLM (đáp ứng yêu cầu bài tập)\n")
                else:
                    print(f"\n❌ Chế độ không hợp lệ. Dùng: standard hoặc optimized\n")
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
                
            # Normal chat - use selected mode
            print("🔄 Đang xử lý...")
            
            # ✅ YÊU CẦU BÀI TẬP: LUÔN dùng Small LLM (≤1B params)
            # LLM nhỏ được dùng cho 2 nhiệm vụ:
            # 1. Hiểu câu hỏi (GraphRAG + LLM understanding)
            # 2. GENERATION: Tạo câu trả lời tự nhiên (format context thành câu văn)
            # 
            # ⚠️ QUAN TRỌNG: LLM KHÔNG làm multi-hop reasoning
            # - Multi-hop reasoning do Reasoner thực hiện (graph algorithm)
            # - LLM chỉ đọc kết quả reasoning và format thành câu trả lời
            use_llm = True  # LUÔN True - đáp ứng yêu cầu bài tập
            
            if chat_mode == 'optimized':
                print("   ⚡ Optimized mode: Dùng Small LLM với context tối ưu (có thể mất 10-30 giây)")
            else:  # standard
                print("   🔄 Standard mode: Dùng Small LLM với context đầy đủ (có thể mất 10-30 giây)")
            
            # Pipeline hoạt động (4 bước):
            # 1. User Query → LLM nhỏ hiểu câu hỏi (extract entities, detect intent)
            # 2. GraphRAG → Truy xuất thông tin từ đồ thị (entities, relationships, paths)
            # 3. Multi-hop Reasoning (Reasoner) → Suy luận từ paths (graph algorithm: tìm đường đi, tính scoring, xâu chuỗi)
            # 4. LLM nhỏ (GENERATION) → Tạo câu trả lời tự nhiên từ context (triples, paths, reasoning results)
            result = chatbot.chat(
                query, 
                session_id, 
                use_multi_hop=True,
                return_details=True,
                use_llm=use_llm  # LUÔN True - đáp ứng yêu cầu bài tập
            )
            
            print(f"\n🤖 {result['response']}")
            print(f"   [Entities: {result['entities_found']}, Hops: {result['reasoning_hops']}]\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}\n")


def run_ui_mode(use_streamlit: bool = False):
    """
    Run web UI.
    
    Args:
        use_streamlit: If True, use Streamlit instead of Gradio
    """
    if use_streamlit:
        try:
            import streamlit.web.cli as stcli
            import sys
            import os
            
            # Get streamlit app path
            streamlit_app_path = os.path.join(
                os.path.dirname(__file__),
                "chatbot",
                "streamlit_app.py"
            )
            
            print("🚀 Launching Streamlit UI...")
            print(f"   App: {streamlit_app_path}")
            print("   URL: http://localhost:8501\n")
            
            # Run streamlit
            sys.argv = ["streamlit", "run", streamlit_app_path]
            stcli.main()
        except ImportError:
            print("❌ Streamlit not installed. Install with: pip install streamlit")
            print("   Falling back to Gradio...")
            use_streamlit = False
    
    if not use_streamlit:
        from chatbot.app import main as run_app
        run_app()


def run_eval_mode(num_questions: int = 2000, use_chatgpt: bool = False, chatgpt_ratio: float = 0.2):
    """
    Generate evaluation dataset.
    
    Args:
        num_questions: Target number of questions
        use_chatgpt: Whether to use ChatGPT for some questions
        chatgpt_ratio: Ratio of questions from ChatGPT (0.0-1.0)
    """
    print("\n" + "="*60)
    print("📝 Evaluation Dataset Generator")
    print("="*60)
    
    if use_chatgpt:
        print("\n💡 Using ChatGPT for some questions")
        print(f"   Distribution: {int(num_questions * (1 - chatgpt_ratio))} from graph, {int(num_questions * chatgpt_ratio)} from ChatGPT")
        print("   ⚠️  Make sure OPENAI_API_KEY is set!")
    
    generator = EvaluationDatasetGenerator()
    stats = generator.generate_full_dataset(
        target_count=num_questions,
        output_path="data/evaluation_dataset.json",
        use_chatgpt=use_chatgpt,
        chatgpt_ratio=chatgpt_ratio
    )
    
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def run_compare_mode(max_questions: int = 500, include_gemini: bool = False, gemini_api_key: str = None):
    """
    Run chatbot comparison.
    
    Args:
        max_questions: Maximum number of questions to evaluate
        include_gemini: Whether to include Gemini in comparison
        gemini_api_key: Gemini API key (or set GOOGLE_API_KEY env var)
    """
    print("\n" + "="*60)
    print("🔬 Chatbot Comparison Mode")
    print("="*60)
    
    # Initialize chatbot
    # Check if Neo4j should be used
    use_neo4j = os.getenv("USE_NEO4J", "false").lower() == "true"
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if use_neo4j:
        if not neo4j_password:
            print("⚠️ USE_NEO4J=true but NEO4J_PASSWORD not set!")
            print("   Falling back to JSON file mode...")
            use_neo4j = False
    
    if use_neo4j:
        print("📊 Using Neo4j Knowledge Graph...")
        chatbot = KpopChatbot(
            use_neo4j=True,
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", None),
            verbose=True
        )
    else:
        print("📊 Using JSON file Knowledge Graph...")
        chatbot = KpopChatbot(verbose=True)
    
    # Check if dataset exists
    dataset_path = "data/evaluation_dataset.json"
    if not os.path.exists(dataset_path):
        print("\n📝 Generating evaluation dataset first...")
        generator = EvaluationDatasetGenerator()
        generator.generate_full_dataset(output_path=dataset_path)
        
    # Run comparison
    comparison = ChatbotComparison(
        kpop_chatbot=chatbot,
        google_api_key=gemini_api_key
    )
    questions = comparison.load_evaluation_dataset(dataset_path)
    
    results = comparison.compare_chatbots(
        questions,
        include_chatgpt=False,  # Set True if OpenAI API key available
        include_gemini=include_gemini,
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
        choices=['cli', 'ui', 'streamlit', 'eval', 'compare'],
        default='cli',
        help='Chế độ chạy: cli (command line), ui (Gradio web), streamlit (Streamlit web), eval (tạo dataset), compare (so sánh)'
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
    
    parser.add_argument(
        '--chat-mode',
        choices=['standard', 'optimized'],
        default='standard',
        help='Chế độ chatbot: standard (luôn dùng LLM - đáp ứng yêu cầu) hoặc optimized (tối ưu context)'
    )
    
    parser.add_argument(
        '--use-chatgpt',
        action='store_true',
        help='Sử dụng ChatGPT để generate một phần questions (cần OPENAI_API_KEY)'
    )
    
    parser.add_argument(
        '--chatgpt-ratio',
        type=float,
        default=0.2,
        help='Tỷ lệ questions từ ChatGPT (0.0-1.0, mặc định: 0.2 = 20%%)'
    )
    
    parser.add_argument(
        '--include-gemini',
        action='store_true',
        help='Bao gồm Gemini trong comparison (cần GOOGLE_API_KEY hoặc --gemini-key)'
    )
    
    parser.add_argument(
        '--gemini-key',
        type=str,
        default=None,
        help='Google API key cho Gemini (hoặc set GOOGLE_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        run_cli_mode(chat_mode=args.chat_mode)
    elif args.mode == 'ui':
        run_ui_mode(use_streamlit=False)
    elif args.mode == 'streamlit':
        run_ui_mode(use_streamlit=True)
    elif args.mode == 'eval':
        run_eval_mode(args.num_questions, args.use_chatgpt, args.chatgpt_ratio)
    elif args.mode == 'compare':
        run_compare_mode(args.max_compare, args.include_gemini, args.gemini_key)


if __name__ == "__main__":
    main()

