import streamlit as st
import requests
import uuid

# --- Configuration ---
# it uses the same port as your FastAPI backend (default is 8000)
API_URL = "http://localhost:8000/query" 
st.set_page_config(page_title="LangChain Tool Router Chat", layout="wide")

# Initialize session state for chat history and unique session ID
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def get_response_from_backend(query: str, session_id: str) -> str:
    """Sends a query to the FastAPI backend and returns the response."""
    try:
        response = requests.post(
            API_URL, 
            json={"query": query, "session_id": session_id},
            timeout=40 
        )
        response.raise_for_status() 
        return response.json().get("response", "Error: No response field in API output.")
    except requests.exceptions.RequestException as e:
        return f"Error communicating with the backend API: {e}. Is the FastAPI server running?"


# --- Streamlit UI ---

st.title("🧩 LangChain Router & Tool Demo")
st.caption(f"Backend Version: LangChain 0.1.20 | Session ID: {st.session_state.session_id[:8]}...")

# Display chat history
chat_container = st.container(height=400, border=True)
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question, or try a command (e.g., 'add mark', 'motivate me', 'i feel sad')"):
    # 1. Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # 2. Display the user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # 3. Get response from FastAPI backend
    with st.spinner("Processing request..."):
        ai_response = get_response_from_backend(prompt, st.session_state.session_id)
    
    # 4. Add AI response to history
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    
    # 5. Display the AI response
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            
# Instructions for testing
st.sidebar.markdown("### How to Use:")
st.sidebar.code("1. add alice marks 92 for math")
st.sidebar.code("2. get marks of alice for math")
st.sidebar.code("3. say something positive")
st.sidebar.code("4. i feel hopeless")
st.sidebar.code("5. What is the largest planet?")