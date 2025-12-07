"""
Gradio Web Interface for K-pop Knowledge Graph Chatbot

A beautiful, interactive web interface for the K-pop chatbot
with support for:
- Multi-turn conversations
- Multi-hop reasoning visualization
- Knowledge graph exploration
- Evaluation mode
"""

import json
import os
from typing import List, Tuple, Optional
from datetime import datetime

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ Gradio not installed. Run: pip install gradio")

from .chatbot import KpopChatbot
from .evaluation import EvaluationDatasetGenerator


# Global chatbot instance
chatbot = None


def initialize_chatbot(skip_llm: bool = False):
    """
    Initialize the chatbot.
    
    Args:
        skip_llm: If True, skip loading LLM (faster startup, graph-only mode)
    """
    global chatbot
    if chatbot is None:
        try:
            print("🔄 Initializing K-pop Chatbot...")
            print("   (Lần đầu khởi tạo có thể mất 30-60 giây...)")
            
            # Initialize without LLM first for fast startup
            chatbot = KpopChatbot(
                verbose=True,
                llm_model="qwen2-0.5b" if not skip_llm else None
            )
            print("✅ Chatbot initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            import traceback
            traceback.print_exc()
            
            # Try fallback without LLM
            try:
                print("🔄 Retrying without LLM...")
                chatbot = KpopChatbot(verbose=True, llm_model=None)
                print("✅ Chatbot initialized (graph-only mode)")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                raise
    return chatbot


def chat_response(
    message: str,
    history: List[List[str]],
    use_multihop: bool,
    max_hops: int
) -> Tuple[str, List[List[str]]]:
    """
    Process chat message and return response.
    
    Args:
        message: User's message
        history: Chat history
        use_multihop: Enable multi-hop reasoning
        max_hops: Maximum reasoning hops
        
    Returns:
        Tuple of (response, updated_history)
    """
    if not message.strip():
        return "", history
        
    try:
        bot = initialize_chatbot()
        
        # ✅ YÊU CẦU BÀI TẬP: Phải dùng Small LLM dựa trên đồ thị tri thức
        # LLM sẽ sử dụng context từ Knowledge Graph (GraphRAG) để trả lời
        use_llm = True  # Luôn dùng LLM để đáp ứng yêu cầu
        
        # Get response using Small LLM with Knowledge Graph context
        # Note: This may take 10-30 seconds, but UI will wait
        result = bot.chat(
            message,
            use_multi_hop=use_multihop,
            max_hops=max_hops,
            return_details=True,
            use_llm=use_llm  # Dùng Small LLM với context từ Knowledge Graph
        )
        
        response = result['response']
        
        # Add reasoning info if available
        if result.get('reasoning', {}).get('steps'):
            steps = result['reasoning']['steps']
            response += f"\n\n📊 *Suy luận {len(steps)}-hop*"
            
        # Update history
        history.append([message, response])
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"❌ Lỗi: {str(e)}\n\n💡 Vui lòng thử lại hoặc kiểm tra console để biết thêm chi tiết."
        history.append([message, error_msg])
        print(f"❌ Error in chat_response: {e}")
        import traceback
        traceback.print_exc()
    
    return "", history


def answer_question(
    question: str,
    question_type: str,
    choices: str
) -> str:
    """
    Answer a specific question.
    
    Args:
        question: The question
        question_type: Type of question
        choices: Comma-separated choices (for MC)
        
    Returns:
        Formatted answer
    """
    if not question.strip():
        return "Vui lòng nhập câu hỏi."
        
    try:
        bot = initialize_chatbot()
        
        if question_type == "Đúng/Sai" or question_type == "Có/Không":
            result = bot.answer_yes_no(question, return_details=True)
            answer = f"""
### Kết quả:
- **Câu trả lời**: {result['answer']}
- **Độ tin cậy**: {result['confidence']:.1%}
- **Giải thích**: {result.get('explanation', 'N/A')}
"""
        else:
            choice_list = [c.strip() for c in choices.split(',')]
            if len(choice_list) < 2:
                return "Vui lòng nhập ít nhất 2 đáp án, cách nhau bởi dấu phẩy."
                
            result = bot.answer_multiple_choice(question, choice_list, return_details=True)
            answer = f"""
### Kết quả:
- **Đáp án**: {result['selected_letter']}. {result['selected_choice']}
- **Độ tin cậy**: {result['confidence']:.1%}
"""
            
        return answer
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}\n\n💡 Vui lòng kiểm tra console để biết thêm chi tiết."
        print(f"❌ Error in answer_question: {e}")
        import traceback
        traceback.print_exc()
        return error_msg


def search_entity(entity_name: str) -> str:
    """Search for an entity in the knowledge graph."""
    if not entity_name.strip():
        return "Vui lòng nhập tên thực thể."
        
    try:
        bot = initialize_chatbot()
        
        results = bot.kg.search_entities(entity_name, limit=5)
        
        if not results:
            return f"Không tìm thấy kết quả cho '{entity_name}'"
            
        output = f"### Kết quả tìm kiếm cho '{entity_name}':\n\n"
        
        for r in results:
            entity_data = bot.kg.get_entity(r['id'])
            infobox = entity_data.get('infobox', {}) if entity_data else {}
            
            output += f"**{r['id']}** ({r['type']})\n"
            for key, value in list(infobox.items())[:3]:
                if value:
                    output += f"- {key}: {value}\n"
            output += "\n"
            
        return output
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        print(f"❌ Error in search_entity: {e}")
        import traceback
        traceback.print_exc()
        return error_msg


def get_group_info(group_name: str) -> str:
    """Get detailed information about a K-pop group."""
    if not group_name.strip():
        return "Vui lòng nhập tên nhóm nhạc."
        
    try:
        bot = initialize_chatbot()
        
        # Get group data
        group_data = bot.kg.get_entity(group_name)
        if not group_data:
            return f"Không tìm thấy nhóm '{group_name}'"
            
        # Get members
        members = bot.kg.get_group_members(group_name)
        
        # Get company
        company = bot.kg.get_group_company(group_name)
        
        # Get songs
        songs = bot.kg.get_group_songs(group_name)
        
        infobox = group_data.get('infobox', {})
        
        output = f"""
### {group_name}

**Loại**: {group_data.get('label', 'N/A')}

**Thông tin cơ bản**:
- Năm hoạt động: {infobox.get('Năm hoạt động', 'N/A')}
- Thể loại: {infobox.get('Thể loại', 'N/A')}
- Công ty: {company or infobox.get('Hãng đĩa', 'N/A')}

**Thành viên** ({len(members)}):
{', '.join(members) if members else 'N/A'}

**Bài hát** ({len(songs)}):
{', '.join(songs[:10]) if songs else 'N/A'}{'...' if len(songs) > 10 else ''}
"""
        
        return output
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        print(f"❌ Error in get_group_info: {e}")
        import traceback
        traceback.print_exc()
        return error_msg


def find_relationship(entity1: str, entity2: str) -> str:
    """Find relationship path between two entities."""
    if not entity1.strip() or not entity2.strip():
        return "Vui lòng nhập cả hai thực thể."
        
    try:
        bot = initialize_chatbot()
        
        result = bot.find_path(entity1, entity2)
        
        if result['path_found']:
            output = f"""
### Đường đi từ {entity1} đến {entity2}:

**Số bước**: {result['hops']} hop(s)

**Đường đi**: {result['description']}
"""
        else:
            output = f"Không tìm thấy đường đi từ '{entity1}' đến '{entity2}'."
            
        return output
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        print(f"❌ Error in find_relationship: {e}")
        import traceback
        traceback.print_exc()
        return error_msg


def get_statistics() -> str:
    """Get knowledge graph statistics."""
    try:
        bot = initialize_chatbot()
        stats = bot.get_statistics()
        
        kg_stats = stats['knowledge_graph']
        
        output = f"""
### 📊 Thống kê Đồ thị Tri thức

**Tổng quan**:
- Tổng số nodes: {kg_stats['total_nodes']:,}
- Tổng số edges: {kg_stats['total_edges']:,}
- Mật độ đồ thị: {kg_stats['density']:.4f}
- Bậc trung bình: {kg_stats['average_degree']:.2f}

**Phân bố theo loại thực thể**:
"""
        
        for entity_type, count in kg_stats['entity_types'].items():
            output += f"- {entity_type}: {count:,}\n"
            
        output += "\n**Phân bố theo loại quan hệ**:\n"
        
        for rel_type, count in list(kg_stats['relationship_types'].items())[:10]:
            output += f"- {rel_type}: {count:,}\n"
            
        output += f"""
**Trạng thái hệ thống**:
- LLM: {'✅ Hoạt động' if stats['llm_available'] else '❌ Không khả dụng'}
- Embeddings: {'✅ Hoạt động' if stats['embeddings_available'] else '❌ Không khả dụng'}
- Sessions hoạt động: {stats['active_sessions']}
"""
        
        return output
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        print(f"❌ Error in get_statistics: {e}")
        import traceback
        traceback.print_exc()
        return error_msg


def generate_evaluation_dataset(num_questions: int) -> str:
    """Generate evaluation dataset."""
    try:
        generator = EvaluationDatasetGenerator()
        stats = generator.generate_full_dataset(
            target_count=num_questions,
            output_path="data/evaluation_dataset.json"
        )
        
        return f"""
### ✅ Dataset đã được tạo!

- **Tổng số câu hỏi**: {stats['total_questions']}
- **Theo số hop**: {stats['by_hops']}
- **Theo loại câu hỏi**: {stats['by_type']}
- **Lưu tại**: data/evaluation_dataset.json
"""
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"


def create_ui():
    """Create Gradio UI."""
    if not GRADIO_AVAILABLE:
        print("❌ Gradio not available. Please install: pip install gradio")
        return None
        
    # Use minimal parameters for maximum compatibility with different Gradio versions
    # Create Blocks with no parameters (most compatible)
    with gr.Blocks() as app:
        gr.Markdown("""
        # 🎤 K-pop Knowledge Graph Chatbot
        
        Chatbot thông minh về K-pop sử dụng **đồ thị tri thức** và **suy luận multi-hop**.
        
        > 💡 *Powered by GraphRAG + Small LLM (Qwen2-0.5B)*
        > 
        > ⏳ **Lưu ý:** Các câu hỏi có thể mất 10-30 giây để xử lý. Vui lòng kiên nhẫn đợi, chương trình sẽ không bị dừng!
        """)
        
        with gr.Tabs():
            # Tab 1: Chat
            with gr.Tab("💬 Trò chuyện"):
                chatbot_ui = gr.Chatbot(
                    label="Chat",
                    height=400
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Hỏi về K-pop... (VD: BTS có bao nhiêu thành viên?)",
                        label="Câu hỏi"
                    )
                    submit_btn = gr.Button("Gửi 🚀")
                    
                with gr.Row():
                    use_multihop = gr.Checkbox(
                        label="Suy luận Multi-hop",
                        value=True
                    )
                    max_hops = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Max hops"
                    )
                    clear_btn = gr.Button("Xóa 🗑️")
                    
                gr.Markdown("""
                > 💡 **Gợi ý:** 
                > - Chatbot sử dụng chế độ nhanh (graph-only) trước, sau đó mới dùng LLM nếu cần.
                > - Để có câu trả lời nhanh nhất, hỏi về: thành viên, công ty, cùng công ty, nhóm nhạc...
                > - ⏳ **Lưu ý:** Câu hỏi có thể mất 10-30 giây để xử lý. Vui lòng đợi, UI sẽ không bị dừng!
                """)
                    
                # Event handlers - queue parameter may not be available in older Gradio versions
                # If queue is not supported, Gradio will still process requests, just without queuing
                try:
                    submit_btn.click(
                        chat_response,
                        inputs=[msg, chatbot_ui, use_multihop, max_hops],
                        outputs=[msg, chatbot_ui],
                        queue=True  # Enable queue for long-running tasks (if supported)
                    )
                    msg.submit(
                        chat_response,
                        inputs=[msg, chatbot_ui, use_multihop, max_hops],
                        outputs=[msg, chatbot_ui],
                        queue=True  # Enable queue for long-running tasks (if supported)
                    )
                except TypeError:
                    # Fallback for older Gradio versions without queue parameter
                    submit_btn.click(
                        chat_response,
                        inputs=[msg, chatbot_ui, use_multihop, max_hops],
                        outputs=[msg, chatbot_ui]
                    )
                    msg.submit(
                        chat_response,
                        inputs=[msg, chatbot_ui, use_multihop, max_hops],
                        outputs=[msg, chatbot_ui]
                    )
                clear_btn.click(lambda: (None, []), outputs=[msg, chatbot_ui])
                
            # Tab 2: Question Answering
            with gr.Tab("❓ Hỏi đáp"):
                gr.Markdown("### Trả lời câu hỏi Đúng/Sai, Có/Không, hoặc Trắc nghiệm")
                
                question_input = gr.Textbox(
                    label="Câu hỏi",
                    placeholder="VD: BTS thuộc công ty HYBE đúng không?"
                )
                
                question_type = gr.Radio(
                    choices=["Đúng/Sai", "Có/Không", "Trắc nghiệm"],
                    label="Loại câu hỏi",
                    value="Có/Không"
                )
                
                choices_input = gr.Textbox(
                    label="Đáp án (cho trắc nghiệm, cách nhau bởi dấu phẩy)",
                    placeholder="HYBE, SM Entertainment, JYP Entertainment, YG Entertainment",
                    visible=True
                )
                
                answer_btn = gr.Button("Trả lời")
                answer_output = gr.Markdown(label="Kết quả")
                
                answer_btn.click(
                    answer_question,
                    inputs=[question_input, question_type, choices_input],
                    outputs=answer_output
                )
                
            # Tab 3: Knowledge Graph Explorer
            with gr.Tab("🔍 Khám phá"):
                gr.Markdown("### Khám phá Đồ thị Tri thức K-pop")
                
                with gr.Row():
                    with gr.Column():
                        search_input = gr.Textbox(
                            label="Tìm thực thể",
                            placeholder="VD: BTS, BLACKPINK, Jungkook..."
                        )
                        search_btn = gr.Button("Tìm kiếm 🔍")
                        search_output = gr.Markdown()
                        
                    with gr.Column():
                        group_input = gr.Textbox(
                            label="Thông tin nhóm nhạc",
                            placeholder="VD: BTS"
                        )
                        group_btn = gr.Button("Xem chi tiết 📋")
                        group_output = gr.Markdown()
                        
                search_btn.click(search_entity, inputs=search_input, outputs=search_output)
                group_btn.click(get_group_info, inputs=group_input, outputs=group_output)
                
                gr.Markdown("### Tìm mối quan hệ")
                
                with gr.Row():
                    entity1_input = gr.Textbox(label="Thực thể 1", placeholder="VD: Jungkook")
                    entity2_input = gr.Textbox(label="Thực thể 2", placeholder="VD: HYBE")
                    
                path_btn = gr.Button("Tìm đường đi 🔗")
                path_output = gr.Markdown()
                
                path_btn.click(
                    find_relationship,
                    inputs=[entity1_input, entity2_input],
                    outputs=path_output
                )
                
            # Tab 4: Statistics
            with gr.Tab("📊 Thống kê"):
                stats_btn = gr.Button("Cập nhật thống kê 📈")
                stats_output = gr.Markdown()
                
                stats_btn.click(get_statistics, outputs=stats_output)
                
            # Tab 5: Evaluation
            with gr.Tab("📝 Đánh giá"):
                gr.Markdown("""
                ### Tạo Dataset Đánh giá
                
                Tạo tập dữ liệu câu hỏi để đánh giá chatbot với các loại:
                - Câu hỏi Đúng/Sai
                - Câu hỏi Có/Không
                - Câu hỏi Trắc nghiệm
                - Suy luận 1-hop, 2-hop, 3-hop
                """)
                
                num_questions = gr.Slider(
                    minimum=100,
                    maximum=5000,
                    value=2000,
                    step=100,
                    label="Số lượng câu hỏi"
                )
                
                generate_btn = gr.Button("Tạo Dataset 📝")
                generate_output = gr.Markdown()
                
                generate_btn.click(
                    generate_evaluation_dataset,
                    inputs=num_questions,
                    outputs=generate_output
                )
                
        gr.Markdown("""
        ---
        *Made with ❤️ for K-pop fans | Using GraphRAG + Multi-hop Reasoning*
        """)
        
    return app


def main():
    """Run the Gradio app."""
    if not GRADIO_AVAILABLE:
        print("❌ Gradio not available. Please install: pip install gradio")
        return
        
    # Pre-initialize chatbot
    initialize_chatbot()
    
    # Create and launch app
    app = create_ui()
    
    if app:
        print("\n🚀 Launching K-pop Chatbot UI...")
        print("💡 Lưu ý: Các câu hỏi có thể mất 10-30 giây để xử lý.")
        print("   UI sẽ hiển thị 'Đang xử lý...' trong lúc chờ.\n")
        
        # Try with max_threads, fallback if not supported
        try:
            app.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True,
                max_threads=10  # Allow multiple concurrent requests
            )
        except TypeError:
            # Fallback for older Gradio versions
            app.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False
            )


if __name__ == "__main__":
    main()

