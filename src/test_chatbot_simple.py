"""
Script đơn giản để test chatbot thuần

Chỉ test chatbot, không có evaluation, comparison, etc.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import KpopChatbot


def main():
    """Simple chatbot test."""
    print("\n" + "="*70)
    print("  🎤 TEST CHATBOT - Chế độ thuần chatbot")
    print("="*70)
    
    print("\nChọn chế độ chatbot:")
    print("  1. Fast Mode (Reasoning-only, nhanh, không LLM)")
    print("  2. Slow Mode (Với LLM, chậm, tự nhiên)")
    print("  3. Hybrid Mode (Tự động, khuyến nghị)")
    print("  4. Thoát")
    
    choice = input("\nChọn (1-4): ").strip()
    
    if choice == '1':
        chat_mode = 'fast'
        mode_name = "⚡ FAST MODE"
    elif choice == '2':
        chat_mode = 'slow'
        mode_name = "🐌 SLOW MODE"
    elif choice == '3':
        chat_mode = 'hybrid'
        mode_name = "🔄 HYBRID MODE"
    elif choice == '4':
        print("\n👋 Tạm biệt!")
        return
    else:
        print("\n❌ Lựa chọn không hợp lệ!")
        return
    
    print(f"\n{mode_name}")
    print("="*70)
    print("\n🔄 Đang khởi tạo chatbot...")
    
    try:
        chatbot = KpopChatbot(verbose=True)
        session_id = chatbot.create_session()
        
        print("\n✅ Sẵn sàng! Hãy đặt câu hỏi về K-pop.")
        print("   Gõ 'quit' để thoát, 'help' để xem hướng dẫn.\n")
        
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
📚 Hướng dẫn:
- Đặt câu hỏi về K-pop: "BTS có bao nhiêu thành viên?"
- Câu hỏi Yes/No: "Jungkook có phải thành viên BTS không?"
- So sánh: "BTS và SEVENTEEN có cùng công ty không?"
- 'quit': Thoát
                    """)
                    continue
                
                print("🔄 Đang xử lý...")
                
                # Determine LLM usage based on mode
                if chat_mode == 'fast':
                    use_llm = False
                elif chat_mode == 'slow':
                    use_llm = True
                    print("   (Đang dùng LLM... có thể mất 10-30 giây)")
                else:  # hybrid
                    # Smart routing
                    simple_keywords = ['members', 'thành viên', 'member', 'company', 'công ty', 
                                     'cùng công ty', 'same company', 'labelmate']
                    is_simple = any(kw in query.lower() for kw in simple_keywords)
                    use_llm = False if is_simple else False  # Try fast first
                
                # First attempt
                result = chatbot.chat(
                    query,
                    session_id,
                    use_multi_hop=True,
                    max_hops=3,
                    return_details=True,
                    use_llm=use_llm
                )
                
                # Hybrid mode: Fallback to LLM if needed
                if chat_mode == 'hybrid' and not use_llm:
                    if not result['response'] or len(result['response']) < 20 or 'không tìm thấy' in result['response'].lower():
                        print("   (Đang dùng LLM cho câu hỏi phức tạp... có thể mất 10-30 giây)")
                        result = chatbot.chat(
                            query,
                            session_id,
                            use_multi_hop=True,
                            max_hops=3,
                            return_details=True,
                            use_llm=True
                        )
                
                print(f"\n🤖 {result['response']}")
                print(f"   [Entities: {result['entities_found']}, Hops: {result['reasoning_hops']}]\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {e}\n")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"\n❌ Lỗi khởi tạo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




