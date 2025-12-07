"""
Streamlit Web Interface for K-pop Knowledge Graph Chatbot

Alternative to Gradio - simpler and lighter weight.
"""

import streamlit as st
from chatbot import KpopChatbot
from chatbot.evaluation import EvaluationDatasetGenerator

# Page config
st.set_page_config(
    page_title="K-pop Knowledge Graph Chatbot",
    page_icon="🎤",
    layout="wide"
)

# Initialize chatbot (cached)
@st.cache_resource
def get_chatbot():
    """Get chatbot instance (cached)."""
    return KpopChatbot(verbose=True)

# Title
st.title("🎤 K-pop Knowledge Graph Chatbot")
st.markdown("Chatbot thông minh về K-pop sử dụng **đồ thị tri thức** và **suy luận multi-hop**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    use_multihop = st.checkbox("Suy luận Multi-hop", value=True)
    max_hops = st.slider("Max hops", 1, 5, 3)
    use_llm = st.checkbox("Sử dụng LLM (chậm hơn)", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Thống kê")
    if st.button("Cập nhật"):
        chatbot = get_chatbot()
        stats = chatbot.get_statistics()
        st.json(stats)

# Main chat interface
st.header("💬 Trò chuyện")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Hỏi về K-pop... (VD: BTS có bao nhiêu thành viên?)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("⏳ Đang xử lý... (Có thể mất 10-30 giây)"):
            try:
                chatbot = get_chatbot()
                result = chatbot.chat(
                    prompt,
                    use_multi_hop=use_multihop,
                    max_hops=max_hops,
                    use_llm=use_llm,
                    return_details=True
                )
                
                response = result['response']
                
                # Add reasoning info
                if result.get('reasoning', {}).get('steps'):
                    steps = result['reasoning']['steps']
                    response += f"\n\n📊 *Suy luận {len(steps)}-hop*"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Lỗi: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Quick actions
st.markdown("---")
st.subheader("⚡ Lệnh nhanh")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Thống kê"):
        chatbot = get_chatbot()
        stats = chatbot.get_statistics()
        st.json(stats)

with col2:
    if st.button("🔄 Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()

with col3:
    if st.button("ℹ️ Hướng dẫn"):
        st.info("""
        **Các lệnh nhanh:**
        - `members BTS` - Xem thành viên
        - `company BLACKPINK` - Xem công ty
        - `same BTS SEVENTEEN` - Kiểm tra cùng công ty
        
        **Tips:**
        - Tắt LLM để nhanh hơn
        - Dùng lệnh nhanh cho câu trả lời tức thì
        """)




