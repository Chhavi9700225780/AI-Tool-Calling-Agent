import streamlit as st
import requests
import json
from typing import Dict, Any

# --- Configuration ---
#  FastAPI backend server is running at this address!
FASTAPI_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="Minimal Tool-Calling Agent",
    layout="centered",
    initial_sidebar_state="auto"
)

## 💬 API Interaction Function
def get_agent_response(user_query: str) -> Dict[str, Any]:
    """Sends the user query to the FastAPI backend and returns the JSON response."""
    try:
        # The FastAPI /chat endpoint expects a JSON payload with the 'user_input' key
        response = requests.post(
            FASTAPI_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"user_input": user_query}),
            timeout=10 # Set a timeout for the request
        )
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status() 
        
        return response.json()
    
    except requests.exceptions.ConnectionError:
        return {"error": "Connection Error", "message": f"Could not connect to FastAPI server at {FASTAPI_URL}. Is your server running?"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout Error", "message": "The request timed out. The server might be busy or slow."}
    except requests.exceptions.RequestException as e:
        return {"error": "API Request Error", "message": f"An error occurred: {e}"}

## 🖼️ Streamlit UI
st.title("🧠 Streamlit Agent Interface")
st.subheader("My AI Tool-Calling Agent")

st.markdown("""
This frontend communicates with my **FastAPI backend** running locally.
- **Backend:** `agent_backend.py` (it is running on port 8000)
- **Frontend:** `agent_frontend.py` (This is an frontend app)
""")

# --- User Input Area ---
user_input = st.text_area(
    "Enter your query:",
    placeholder="Example: Add mark 95 for Chris in History or Give me a positive prompt about resilience.",
    height=100
)

# --- Send Button ---
if st.button("Submit Query", type="primary") and user_input:
    # 1. Start the spinner while processing
    with st.spinner('Thinking and routing query...'):
        # 2. Get the response from the FastAPI server
        response_data = get_agent_response(user_input)

    # 3. Handle Errors
    if "error" in response_data:
        st.error(f"Error: {response_data['error']}")
        st.warning(response_data['message'])
    
    # 4. Display Successful Response
    else:
        st.success("Response Received!")
        
        # Display key information prominently
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tool Used", response_data.get("routed_tool", "N/A"))
        with col2:
            st.metric("Router Score", round(response_data.get("router_score", 0), 3))
        
        st.divider()

        # Display the tool's final output
        st.markdown("### Tool Output")
        tool_output = response_data.get("tool_output", {})
        
        # Special formatting for the student-marks tool's results list
        if response_data.get("routed_tool") == "student-marks" and tool_output.get("results"):
             st.json(tool_output)
        else:
             st.json(tool_output)

        # Display the full structured response for debugging
        st.divider()
        st.markdown("### Full JSON Payload")
        st.json(response_data)


# --- Initial Check for Server Status (Optional but helpful) ---
st.sidebar.title("Server Status")
try:
    # Check if the root endpoint is accessible
    check_response = requests.get("http://127.0.0.1:8000/", timeout=5)
    if check_response.status_code == 200:
        st.sidebar.success("✅ FastAPI Backend is Running!")
    else:
        st.sidebar.error(f"❌ Backend Responded with Status: {check_response.status_code}")
except requests.exceptions.RequestException:
    st.sidebar.error("❌ FastAPI Backend is OFFLINE.")
    st.sidebar.info("Please start your server: `uvicorn agent_backend:app --reload`")