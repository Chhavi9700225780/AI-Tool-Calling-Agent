import streamlit as st
import requests

CHAT_API_URL = "http://127.0.0.1:8000/chat"
HISTORY_API_URL = "http://127.0.0.1:8000/history"

st.set_page_config(page_title="LangChain Tool Agent", layout="centered")
st.title("🤖 LangChain Semantic Tool Agent")

# ================================
# Session Initialization
# ================================

if "session_id" not in st.session_state:
    st.session_state.session_id = "user1"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================================
# User Input
# ================================

user_input = st.text_input("Enter your message:")

# ================================
# Send Message
# ================================

if st.button("Send"):
    if user_input.strip():
        response = requests.post(
            CHAT_API_URL,
            json={
                "message": user_input,
                "session_id": st.session_state.session_id
            }
        )

        result = response.json()
        reply = result["reply"]

        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Agent", reply))

# ================================
# Display Chat
# ================================

for role, msg in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"🧑 **User:** {msg}")
    else:
        st.markdown(f"🤖 **Agent:** {msg}")

# ================================
# Show Full Stored History ✅
# ================================

if st.button("Show Chat History"):
    history_response = requests.post(
        HISTORY_API_URL,
        json={"session_id": st.session_state.session_id}
    )

    history_result = history_response.json()
    st.text_area(
        "📜 Full Chat History (From Memory)",
        history_result["history"],
        height=300
    )
