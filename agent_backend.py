import os
import re
from typing import Dict, Any

from fastapi import FastAPI                          # For creating backend API
from fastapi.middleware.cors import CORSMiddleware  # To allow frontend requests
from pydantic import BaseModel                      # For request validation
from dotenv import load_dotenv                      # To load environment variables

# LangChain + Semantic Imports
from langchain_google_genai import ChatGoogleGenerativeAI   # Gemini LLM
from langchain_core.runnables import RunnableBranch, RunnablePassthrough  # For routing
from langchain.memory import ConversationBufferMemory      # For chat memory
from langchain_community.embeddings import HuggingFaceEmbeddings  # For local embeddings
from langchain_community.vectorstores import FAISS                  # For vector DB
from langchain.schema import Document                               # For intent docs

# --- Setup ---
load_dotenv()   # Loads API keys from .env file
app = FastAPI(title="LangChain ToolCalling Agent API", version="0.1.20-compatible")

# Allow Streamlit frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body format
class QueryRequest(BaseModel):
    query: str
    session_id: str


# ============================================================
# 1. STATE MANAGEMENT
# ============================================================

# Stores student marks in memory
marks_memory: Dict[str, Dict[str, int]] = {}

# Stores conversation memory for each user session
session_memories: Dict[str, ConversationBufferMemory] = {}

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    # Creates memory if session is new
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return session_memories[session_id]


# ============================================================
# 2. LLM CONFIGURATION (ONLY FOR DEFAULT CHAT)
# ============================================================

# Gemini model is only used for general queries
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)


# ============================================================
# 3. EMBEDDINGS + VECTOR DATABASE (SEMANTIC ROUTING)
# ============================================================

# Local embedding model to convert text into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Example texts for each intent
documents = [
    Document(
        page_content="positive motivation encouragement happy inspire",
        metadata={"label": "positive"}
    ),

    Document(
        page_content="negative sad angry complain frustrated upset",
        metadata={"label": "negative"}
    ),

    Document(
        page_content=(
            "add student marks get student marks "
            "add alice marks 92 for math "
            "get marks of alice for math"
        ),
        metadata={"label": "marks"}
    ),

    Document(
        page_content="suicide self harm kill myself depression hopeless",
        metadata={"label": "suicide"}
    ),

    Document(
        page_content="general question normal chat ask anything",
        metadata={"label": "default"}
    ),
]


# Store the intent vectors in FAISS
vector_db = FAISS.from_documents(documents, embeddings)

# Finds the closest intent using vector similarity
def semantic_router(user_query: str):
    result = vector_db.similarity_search_with_score(user_query, k=1)

    best_match = result[0]
    label = best_match[0].metadata["label"]
    score = best_match[1]

    # If similarity is weak, send to default LLM
    if score > 1.2:
        return "default"

    return label


# ============================================================
# 4. TOOL HANDLERS
# ============================================================

# Handles positive type responses
def positive_prompt_tool(request: str):
    return f"Stay strong! {request}"

# Handles negative type responses
def negative_prompt_tool(request: str):
    return f"Negative prompt: Avoid negativity like '{request}'"

# Handles sensitive suicide-related queries
def suicide_related_tool(request: str):
    return (
        "I'm really sorry you feel this way. "
        "Please reach out to someone you trust or a nearby helpline immediately. "
        "You matter, and there are people who care about you."
    )

# Adds and retrieves student marks
def student_marks_tool(request: str):
    global marks_memory

    # Pattern to ADD marks
    add_pattern = r"add\s+(\w+)\s+marks\s+(\d+)\s+for\s+(\w+)"
    match_add = re.search(add_pattern, request.lower())

    if match_add:
        name, score, subject = match_add.groups()
        score = int(score)

        if name not in marks_memory:
            marks_memory[name] = {}
        marks_memory[name][subject] = score

        return f"Added marks: {name.title()} - {subject}: {score}"

    # Pattern to GET marks
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
# 5. ROUTER CONDITIONS
# ============================================================

# These act like if–else checks
def is_positive(x): return x["decision"] == "positive"
def is_negative(x): return x["decision"] == "negative"
def is_marks(x): return x["decision"] == "marks"
def is_suicide(x): return x["decision"] == "suicide"


# ============================================================
# 6. TOOL BRANCHES
# ============================================================

# Connects each intent to its tool
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

# Default branch uses Gemini for normal questions
default_branch = RunnablePassthrough.assign(
    output=lambda x: llm.invoke(x.get('request')).content
)


# ============================================================
# 7. DELEGATION LOGIC
# ============================================================

# Decides which branch to execute
delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch
)


# ============================================================
# 8. COORDINATOR (SEMANTIC ROUTING)
# ============================================================

# Full flow: input → semantic router → tool → output
coordinator_agent = (
    RunnablePassthrough()
    .assign(decision=lambda x: semantic_router(x["request"]))
    | delegation_chain
    | (lambda x: x["output"])
)


# ============================================================
# 9. API ENDPOINT
# ============================================================

@app.post("/query")
async def handle_query(request: QueryRequest) -> Dict[str, str]:
    """
    Receives query from frontend,
    routes it using semantic similarity,
    and returns the response.
    """
    user_query = request.query
    session_id = request.session_id

    try:
        # Run router + tool chain
        result = coordinator_agent.invoke({"request": user_query})

        # Save conversation for the session
        session_memory = get_session_memory(session_id)
        session_memory.save_context({"human": user_query}, {"ai": result})

        return {"response": result}

    except Exception as e:
        return {"response": f"An error occurred: {e}"}


# Health check API
@app.get("/")
def read_root():
    return {"message": "LangChain Semantic Router API is running."}
