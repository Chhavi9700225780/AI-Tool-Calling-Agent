from fastapi import FastAPI
from pydantic import BaseModel
from main import chat, get_user_history

app = FastAPI(title="LangChain Tool Calling Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat_api(req: ChatRequest):
    reply = chat(req.message, req.session_id)
    return {"reply": reply}

@app.get("/history/{session_id}")
def history_api(session_id: str):
    return {
        "history": get_user_history(session_id)
    }
