import os
import re
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports (Pinned Versions: 0.1.x)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.memory import ConversationBufferMemory # This import works for 0.1.20

# --- Setup ---
load_dotenv()
app = FastAPI(title="LangChain ToolCalling Agent API", version="0.1.20-compatible")

# Configure CORS to allow the Streamlit frontend to access the API
# NOTE: If deploying, replace "*" with your Streamlit app's URL (e.g., "http://localhost:8501")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str
    session_id: str


# ============================================================
# 1. STATE MANAGEMENT (Global for simplicity, thread-safe access is limited)
# ============================================================

# In-memory dict to store student marks - using a global lock might be needed in production, 
# but for simple demonstration, this is fine.
marks_memory: Dict[str, Dict[str, int]] = {}

# Dictionary to hold separate memory buffers for each user/session
session_memories: Dict[str, ConversationBufferMemory] = {}

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    """Retrieves or creates a ConversationBufferMemory instance for a given session ID."""
    if session_id not in session_memories:
        # Initialize memory with your compatible ChatGoogleGenerativeAI LLM
        # Note: In 0.1.x, memory usually stores the history but doesn't handle the conversion for LLMs
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
    return session_memories[session_id]


# ============================================================
# 2. LLM CONFIGURATION
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # This setting is crucial for the older version to handle the system prompt
    convert_system_message_to_human=True 
)


# ============================================================
# 3. TOOL HANDLERS
# ============================================================

def positive_prompt_tool(request: str):
    return f"Stay strong! {request}"


def negative_prompt_tool(request: str):
    return f"Negative prompt: Avoid negativity like '{request}'"


def suicide_related_tool(request: str):
    return (
        "I'm really sorry you feel this way. "
        "Please reach out to someone you trust or a nearby helpline immediately. "
        "You matter, and there are people who care about you."
    )


def student_marks_tool(request: str):
    global marks_memory

    # ADD MARKS
    add_pattern = r"add\s+(\w+)\s+marks\s+(\d+)\s+for\s+(\w+)"
    match_add = re.search(add_pattern, request.lower())

    if match_add:
        name, score, subject = match_add.groups()
        score = int(score)

        if name not in marks_memory:
            marks_memory[name] = {}
        marks_memory[name][subject] = score

        return f"Added marks: {name.title()} - {subject}: {score}"

    # GET MARKS
    get_pattern = r"get\s+.*marks.*\b(\w+)\b.*for\s+(\w+)"
    match_get = re.search(get_pattern, request.lower())

    if match_get:
        name, subject = match_get.groups()
        if name in marks_memory and subject in marks_memory[name]:
            return f"{name.title()} scored {marks_memory[name][subject]} in {subject}."
        else:
            return f"No marks found for {name.title()} in {subject}."

    return "Use 'add alice marks 92 for math' or 'get marks of alice for math'."


# ============================================================
# 4. ROUTER CHAIN (The logic remains the same)
# ============================================================
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Classify the user's request into exactly ONE label:
       positive, negative, marks, suicide, default"""
     ),
    ("user", "{request}")
])

router_chain = router_prompt | llm | StrOutputParser()

# --- Conditions ---
def is_positive(x): return x["decision"].strip().lower() == "positive"
def is_negative(x): return x["decision"].strip().lower() == "negative"
def is_marks(x): return x["decision"].strip().lower() == "marks"
def is_suicide(x): return x["decision"].strip().lower() == "suicide"

# --- Branches ---
positive_branch = RunnablePassthrough.assign(
    output=lambda x: positive_prompt_tool(x.get("request"))
)

negative_branch = RunnablePassthrough.assign(
    output=lambda x: negative_prompt_tool(x.get("request"))
)

marks_branch = RunnablePassthrough.assign(
    output=lambda x: student_marks_tool(x.get("request"))
)

suicide_branch = RunnablePassthrough.assign(
    output=lambda x: suicide_related_tool(x.get("request"))
)

# The default branch now handles general questions by invoking the LLM directly
# The LLM invocation here bypasses the memory, which is handled in the FastAPI endpoint below.
default_branch = RunnablePassthrough.assign(
    output=lambda x: llm.invoke(x.get('request')).content
)


# --- Delegation Logic ---
delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch
)


# --- Coordinator ---
coordinator_agent = (
    RunnablePassthrough()
    .assign(decision=router_chain)
    | delegation_chain
    | (lambda x: x["output"])
)

# ============================================================
# 5. API ENDPOINT
# ============================================================

@app.post("/query")
async def handle_query(request: QueryRequest) -> Dict[str, str]:
    """
    Endpoint to receive a user query and process it through the LangChain router.
    Handles memory storage and retrieval.
    """
    user_query = request.query
    session_id = request.session_id
    
    # 1. Retrieve history for context (Note: The coordinator_agent above doesn't use history in this setup)
    # The current setup only uses memory for *saving* context, not retrieving it for the router prompt.
    # To fully integrate memory for conversation, the `router_prompt` and `default_branch` LLM call 
    # would need to be updated to inject history. For this implementation, we simply save it.
    
    try:
        # 2. Invoke the router/tool chain
        result = coordinator_agent.invoke({"request": user_query})

        # 3. Save conversation history
        session_memory = get_session_memory(session_id)
        session_memory.save_context({"human": user_query}, {"ai": result})
        
        return {"response": result}

    except Exception as e:
        return {"response": f"An error occurred during processing: {e}"}

@app.get("/")
def read_root():
    return {"message": "LangChain Router API is running."}

# if __name__ == "__main__":
#     # For local testing, run with: uvicorn fastapi_backend:app --reload
#     pass