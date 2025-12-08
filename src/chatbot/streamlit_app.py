"""
Streamlit Web Interface for K-pop Knowledge Graph Chatbot.

Mục tiêu: UI đơn giản, ít tuỳ chọn để tránh lỗi, vẫn cho phép bật/tắt multi-hop.
"""

import os
import sys

# Ensure project root and src are on sys.path so "from chatbot import ..." works when run via streamlit
CURR_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURR_DIR, "..", ".."))
for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
from chatbot import KpopChatbot

# Page config
st.set_page_config(
    page_title="K-pop Knowledge Graph Chatbot",
    page_icon="🎤",
    layout="wide"
)

# Initialize chatbot (cached) với catch lỗi rõ ràng
@st.cache_resource
def get_chatbot():
    """Get chatbot instance (cached)."""
    try:
        return KpopChatbot(verbose=False)
    except Exception as e:
        st.error(f"Không khởi tạo được chatbot: {e}")
        return None

# Title
st.title("🎤 K-pop Knowledge Graph Chatbot")
st.markdown("Chatbot thông minh về K-pop sử dụng **đồ thị tri thức** và **suy luận multi-hop**")

# Sidebar (gọn nhẹ)
with st.sidebar:
    st.header("⚙️ Chế độ")
    ui_mode = st.radio("Chọn chế độ UI", ["Đơn giản", "Nâng cao"], index=0)

    if ui_mode == "Đơn giản":
        use_multihop = True
        max_hops = 3
        use_llm = True  # luôn dùng LLM nhỏ cho understanding + generation
        st.caption("Đơn giản: Multi-hop ON, max_hops=3, LLM bật.")
    else:
        use_multihop = st.checkbox("Suy luận Multi-hop", value=True)
        max_hops = st.slider("Max hops", 1, 5, 3)
        use_llm = st.checkbox("Sử dụng LLM (chậm hơn)", value=True)

    st.markdown("---")
    st.markdown("### 📊 Thống kê")
    if st.button("Cập nhật"):
        chatbot = get_chatbot()
        if chatbot:
            stats = chatbot.get_statistics()
            st.json(stats)
        else:
            st.error("Chưa khởi tạo được chatbot.")

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
            chatbot = get_chatbot()
            if chatbot is None:
                error_msg = "❌ Chưa khởi tạo được chatbot. Kiểm tra lại model/weights."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                try:
                    result = chatbot.chat(
                        prompt,
                        use_multi_hop=use_multihop,
                        max_hops=max_hops,
                        use_llm=use_llm,
                        return_details=True
                    )
                    
                    response = result.get('response', 'Không có phản hồi.')
                    
                    # Add reasoning info
                    steps = result.get('reasoning', {}).get('steps') if result.get('reasoning') else []
                    if steps:
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
        - Chế độ Đơn giản: mặc định multi-hop + LLM nhỏ
        - Nếu lỗi model, kiểm tra lại đường dẫn checkpoint/weights
        """)





