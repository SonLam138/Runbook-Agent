import streamlit as st
import uuid

from app.agent import run_agent, save_feedback
from app.session_store import reset_session

st.set_page_config(layout="wide")
st.title("🤖 IT Runbook Agent")

# ====================
# INIT
# ====================
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.current_chat = cid
    st.session_state.chats[cid] = {
        "session_id": str(uuid.uuid4()),
        "messages": []
    }

# ====================
# SIDEBAR
# ====================
with st.sidebar:
    st.header("💬 Chats")

    # ✅ NEW CHAT
    if st.button("➕ New Chat"):
        cid = str(uuid.uuid4())
        st.session_state.current_chat = cid
        st.session_state.chats[cid] = {
            "session_id": str(uuid.uuid4()),
            "messages": []
        }
        st.rerun()

    # ✅ CHAT LIST + DELETE
    for cid in list(st.session_state.chats.keys()):
        col1, col2 = st.columns([4,1])

        with col1:
            if st.button(f"Chat {cid[:6]}", key=cid):
                st.session_state.current_chat = cid
                st.rerun()

        with col2:
            if st.button("❌", key=f"del_{cid}"):
                del st.session_state.chats[cid]

                if st.session_state.current_chat == cid:
                    # chọn chat còn lại nếu có
                    if st.session_state.chats:
                        st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                    else:
                        new_id = str(uuid.uuid4())
                        st.session_state.current_chat = new_id
                        st.session_state.chats[new_id] = {
                            "session_id": str(uuid.uuid4()),
                            "messages": []
                        }

                st.rerun()

# ====================
# CURRENT CHAT
# ====================
chat = st.session_state.chats[st.session_state.current_chat]
SESSION_ID = chat["session_id"]
messages = chat["messages"]

# ====================
# RENDER CHAT
# ====================
for i, msg in enumerate(messages):
    if msg["role"] == "user":
        st.markdown(f"👤 **{msg['text']}**")

    else:
        st.markdown(f"🤖 {msg['text']}")

        # ✅ LIKE BUTTON cho runbook
        if msg.get("is_runbook"):

            col1, col2 = st.columns([1,8])

            with col1:
                if st.button("👍", key=f"like_{i}"):

                    # lấy query tương ứng
                    query = messages[i-1]["text"]

                    # save vào cache
                    save_feedback(query, msg["runbook"])

                    st.success("✅ Đã lưu feedback!")
                    st.rerun()

            with col2:
                st.caption("Runbook này có phù hợp không? Hãy bấm 👍 nếu đúng.")

# ====================
# INPUT
# ====================
st.markdown("---")

user_input = st.text_input("Nhập câu hỏi...")

if st.button("📨 Gửi") and user_input.strip():

    messages.append({"role": "user", "text": user_input})

    with st.spinner("🤖 Agent đang tìm kiếm runbook..."):
        answer = run_agent(SESSION_ID, user_input)

    # detect nếu là runbook (simple heuristic)
    is_runbook = "📘" in answer

    messages.append({
        "role": "assistant",
        "text": answer,
        "is_runbook": is_runbook,
        "runbook": answer
    })

    st.rerun()