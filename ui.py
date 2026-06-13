import streamlit as st
import uuid

from app.agent import run_agent
from app.session_store import reset_session

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="IT Runbook Agent",
    layout="wide"
)

st.title("🤖 IT Runbook Agent")

# =====================================================
# SESSION ID (multi-user)
# =====================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id

# =====================================================
# UI HEADER (SESSION + RESET)
# =====================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(f"🆔 **Session:** `{SESSION_ID}`")

with col2:
    if st.button("🔄 Reset"):
        reset_session(SESSION_ID)

        # reset UI history
        st.session_state.messages = []

        # reset semantic query nếu anh muốn làm sạch hoàn toàn
        if "semantic_query" in st.session_state:
            del st.session_state["semantic_query"]

        st.success("✅ Session đã được reset")
        st.rerun()

# =====================================================
# CHAT HISTORY
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"👤 **{msg['text']}**")
    else:
        st.markdown(f"🤖 {msg['text']}")

# =====================================================
# INPUT BOX
# =====================================================
st.markdown("---")

user_input = st.text_input("Nhập câu hỏi...")

col_send, col_clear = st.columns([1,1])

with col_send:
    send_clicked = st.button("📨 Gửi")

with col_clear:
    clear_clicked = st.button("🧹 Clear chat (UI only)")

# =====================================================
# CLEAR CHAT (KHÔNG RESET AGENT)
# =====================================================
if clear_clicked:
    st.session_state.messages = []
    st.rerun()

# =====================================================
# SEND MESSAGE
# =====================================================
if send_clicked and user_input.strip():

    # add user message
    st.session_state.messages.append({
        "role": "user",
        "text": user_input
    })

    # call agent
    answer = run_agent(SESSION_ID, user_input)

    # add bot message
    st.session_state.messages.append({
        "role": "assistant",
        "text": answer
    })

    st.rerun()