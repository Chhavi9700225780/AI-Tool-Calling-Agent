import json
import os
import time
from typing import Any, Dict, Callable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm # This is imported but not used, good practice is to remove it, but keeping it for now

# FastAPI Imports
from fastapi import FastAPI
from pydantic import BaseModel
# Removed: import nest_asyncio, uvicorn (will be imported later for __main__)

# --- CONFIGURATION (Global Settings) ---
MEMORY_FILE = "longterm_memory.json"
# The model that turns text into vectors for meaning comparison (Semantic Routing)
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", token=False)


# --- MEMORY AND UTILITY FUNCTIONS ---
def load_memory() -> List[Dict[str, Any]]:
    # Loads memories from the JSON file on disk.
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            # Handle empty file case
            content = f.read()
            if content:
                return json.loads(content)
            return []
    return []

def save_memory(memories: List[Dict[str, Any]]):
    # Writes the current memories back to the JSON file.
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)

def embed(texts: List[str]) -> np.ndarray:
    # Converts text into normalized numerical vectors (embeddings).
    vecs = EMBEDDER.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    # Normalize vectors to unit length
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Prevent division by zero for zero vectors (though rare)
    norms[norms == 0] = 1.0
    return vecs / norms

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Calculates how similar two vectors (and thus two texts) are.
    # Assumes inputs a and b are already normalized unit vectors.
    return float(np.dot(a, b))


# --- TOOL IMPLEMENTATIONS (What the Agent Can Do) ---

def positive_prompt_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Generates a simple uplifting prompt suggestion.
    prompt = payload.get("text", "")
    return {
        "tool": "positive-prompt",
        "generated": (f"Positive prompt for: {prompt}\nTry: \"Celebrate progress — what's one small win today?\"")
    }

def negative_prompt_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Provides advice on what to filter out of a prompt.
    prompt = payload.get("text", "")
    return {
        "tool": "negative-prompt",
        "generated": (f"Negative prompt for: {prompt}\nTry to remove: overly-specific negative descriptors.")
    }

def student_marks_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Stores or retrieves student marks from the long-term memory file.
    memories = load_memory()
    action = payload.get("action", "get")
    student = payload.get("student")
    subject = payload.get("subject")
    marks = payload.get("marks")

    if action == "add":
        # Logic to add a new mark record.
        if not student or subject is None or marks is None:
            return {"error": "Missing fields for add action", "details": "Requires 'student', 'subject', and 'marks' fields."}
        entry = {"type": "student_mark", "student": student, "subject": subject, "marks": marks, "timestamp": int(time.time()), "note": payload.get("note", "")}
        memories.append(entry)
        save_memory(memories)
        return {"status": "ok", "added": entry}
    else: # action == "get"
        # Logic to filter and retrieve matching marks.
        results = [m for m in memories if m.get("type") == "student_mark"]
        if student:
            results = [r for r in results if r.get("student", "").lower() == student.lower()]
        if subject:
            results = [r for r in results if r.get("subject", "").lower() == subject.lower()]
        return {"status": "ok", "results": results}

def suicide_detection_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Handles sensitive queries by detecting risk and providing safety resources.
    text = payload.get("text", "")
    LOWER = text.lower()
    danger_keywords = ["kill myself", "end my life", "suicide", "want to die", "i'm going to die"]
    is_high_risk = any(k in LOWER for k in danger_keywords)
    response = {"tool": "suicide-related", "detected": bool(is_high_risk)}
    if is_high_risk:
        response["message"] = ("I'm really sorry you're feeling this way. If you need immediate help, please contact the National Suicide Prevention Lifeline at 988 (US) or your local emergency services. There are people who want to support you.")
    else:
        response["message"] = ("I detected concerns about self-harm but it doesn't look like immediate high-risk. Please remember that resources are available if you ever need to talk to someone.")
    return response


# --- TOOL REGISTRY AND EMBEDDINGS ---
TOOLS: List[Dict[str, Any]] = [
    # Master list of all available tools and their descriptions.
    {"name": "positive-prompt", "description": "Generate short, uplifting prompts or positive writing prompts. Input: {'text': str}", "callable": positive_prompt_tool},
    {"name": "negative-prompt", "description": "Produce negative prompt filtering suggestions and 'what to avoid' guidance for prompts. Input: {'text': str}", "callable": negative_prompt_tool},
    {"name": "student-marks", "description": "Store or retrieve student marks. Actions: add/get. Input: {'action':'add'/'get','student':str,'subject':str,'marks':int}.", "callable": student_marks_tool},
    {"name": "suicide-related", "description": "Detect suicide/self-harm risk and return a safety-first response. Input: {'text': str}.", "callable": suicide_detection_tool}
]

TOOL_TEXTS = [t["description"] for t in TOOLS]
TOOL_EMBS = embed(TOOL_TEXTS) # Pre-calculate embeddings for faster routing.


# --- SEMANTIC ROUTER ---
def route_to_tool(user_text: str, top_k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
    # Compares the user's query to the tool descriptions to find the best match.
    q_emb = embed([user_text])[0]
    scores = [cosine_sim(q_emb, tvec) for tvec in TOOL_EMBS]
    ranked = sorted(list(zip(TOOLS, scores)), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def retrieve_memories(user_text: str, top_n: int = 3) -> List[Dict[str, Any]]:
    # Searches past memory entries for semantic similarity to the current query.
    memories = load_memory()
    if not memories:
        return []
    mem_texts = [json.dumps(m, ensure_ascii=False) for m in memories]
    mem_embs = embed(mem_texts)
    q_emb = embed([user_text])[0]
    sims = [(m, float(np.dot(q_emb, mem_embs[i]))) for i, m in enumerate(memories)]
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    return [m for m, s in sims_sorted[:top_n]]

# Removed: format_tool_output (not used in the final endpoint)

# ----------------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ----------------------------------------------------------------------------------

# Initialize the FastAPI app object.
app = FastAPI(title="Minimal Tool-Calling Agent API")

# Defines the expected input structure for the API (must have 'user_input' field).
class UserQuery(BaseModel):
    user_input: str

# ----------------------------------------------------------------------------------
# API ENDPOINT DEFINITION (The core logic that runs everything)
# ----------------------------------------------------------------------------------

@app.post("/chat")
async def chat_handler(query: UserQuery):
    # This function runs every time someone sends a POST request to /chat.
    user_input = query.user_input

    # 1) Use the semantic router to pick the right tool.
    ranked = route_to_tool(user_input, top_k=1)
    best_tool, score = ranked[0]

    # 2) Prepare the input data (payload) for the chosen tool.
    payload = {}

    # Simple text input for these tools.
    if best_tool["name"] in ("positive-prompt", "negative-prompt", "suicide-related"):
        payload = {"text": user_input}

    # Complex parsing for the student marks tool (to extract add/get, name, subject, marks).
    elif best_tool["name"] == "student-marks":
        tokens = user_input.lower().split()
        
        # --- ADD/STORE Logic ---
        if "add" in tokens or "store" in tokens:
            action = "add"
            try:
                # Basic attempts to extract marks (first digit)
                numbers = [int(t) for t in tokens if t.isdigit()]
                marks = numbers[0] if numbers else None
                
                # Basic attempts to extract student name (word after 'add' or 'store')
                idx = tokens.index("add") if "add" in tokens else tokens.index("store")
                student = tokens[idx + 1] if idx + 1 < len(tokens) else "unknown"
                
                # Basic attempts to extract subject (word before 'marks' or last non-number word)
                subject_tokens = [t for t in tokens if t not in ["add", "store", "marks"] and not t.isdigit()]
                subject = subject_tokens[-1] if subject_tokens else "unknown"
                
                payload = {"action": action, "student": student.capitalize(), "subject": subject.capitalize(), "marks": marks}
            except Exception:
                # Fallback on failure
                payload = {"action": "get", "student": None, "subject": None}

        # --- GET/RETRIEVE Logic ---
        else:
            action = "get"
            student = None
            subject = None
            
            # Simple parsing for student name
            if "marks for" in user_input.lower():
                parts = user_input.lower().split("marks for")
                # Look for the student name after "marks for"
                student_match = parts[1].strip().split()
                if student_match:
                    student = student_match[0].capitalize()
            
            payload = {"action": action, "student": student, "subject": subject}

    else:
        # Fallback payload
        payload = {"text": user_input}

    # 3) Run the selected tool's function.
    try:
        result = best_tool["callable"](payload)
    except Exception as e:
        # If the tool breaks, return a clear error message.
        return {"error": f"Tool execution error: {str(e)}", "tool_name": best_tool["name"], "payload_attempted": payload}

    # 4) Save the original query to memory for later retrieval.
    # Note: Only saving the query itself, not the result of the tool call.
    memories = load_memory()
    memories.append({
        "type": "user_query",
        "text": user_input,
        "timestamp": int(time.time()),
        "tool_used": best_tool["name"],
        "route_score": score
    })
    save_memory(memories)

    # 5) Return the final structured response.
    return {
        "user_query": user_input,
        "routed_tool": best_tool["name"],
        "router_score": round(score, 3),
        "tool_payload": payload,
        "tool_output": result
    }

# Add a root endpoint for a simple check
@app.get("/")
def read_root():
    # Check to see if the API is alive.
    return {"message": "Minimal Tool-Calling Agent API is running."}

# ----------------------------------------------------------------------------------
# LOCAL ENTRY POINT (Replaces Colab/Jupyter execution)
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    # This block allows you to run the file directly from your terminal:
    # uvicorn agent_backend:app --reload
    import uvicorn
    uvicorn.run(
        "agent_backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True, # Auto-reload on code changes (helpful during development)
        log_level="info"
    )

    #streamlit run agent_frontend.py
    #uvicorn agent_backend:app --reload