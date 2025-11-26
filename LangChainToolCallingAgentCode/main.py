import re
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain.memory import ConversationTokenBufferMemory

from langchain.agents import Tool, initialize_agent, AgentType

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv 
import os 
load_dotenv()

# ============================================================
# 1. MAIN LLM
# ============================================================

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

# ============================================================
# 2. GLOBAL STORES
# ============================================================

memory_store: Dict[str, ConversationTokenBufferMemory] = {}
sessions: Dict[str, Dict[str, Any]] = {}
agent_store: Dict[str, Any] = {}

vector_store = None
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# 3. MEMORY HANDLING
# ============================================================

def get_memory(session_id: str) -> ConversationTokenBufferMemory:
    if session_id not in memory_store:
        memory_store[session_id] = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=3000,
            return_messages=True,
            memory_key="chat_history",
        )
    return memory_store[session_id]


def get_or_create_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {"marks": {}}
    return sessions[session_id]


def get_user_history(session_id: str) -> str:
    memory = get_memory(session_id)
    messages = memory.load_memory_variables({}).get("chat_history", [])

    if not messages:
        return "No chat history found."

    lines = []
    for msg in messages:
        role = "USER" if msg.type == "human" else "AI"
        lines.append(f"{role}: {msg.content}")

    return "\n".join(lines)

# ============================================================
# 4. VECTOR DATABASE
# ============================================================

def init_vector_db():
    global vector_store
    vector_store = FAISS.from_documents(
        [Document(page_content="system initialization", metadata={"intent": "system"})],
        embedding_model
    )

def store_in_vector_db(text: str, intent: str):
    doc = Document(
        page_content=text,
        metadata={"intent": intent}
    )
    vector_store.add_documents([doc])


def semantic_router(query: str) -> str:
    result = vector_store.similarity_search_with_score(query, k=1)

    if not result:
        return "generic"

    doc, score = result[0]
    intent = doc.metadata.get("intent", "generic")

    if score > 1.2:
        return "generic"

    return intent


init_vector_db()

# ============================================================
# 5. STUDENT MARKS TOOL
# ============================================================

def calculate_grade(score: int) -> str:
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    if score >= 40: return "E"
    return "F"


def student_marks_tool(text: str, session_id: str):
    marks = get_or_create_session(session_id)["marks"]

    pairs = re.findall(r"([A-Za-z]+)\s*[-:]?\s*(\d{1,3})", text)

    if not pairs:
        if not marks:
            return "No marks stored yet."

        lines = ["| Subject | Marks | Grade |", "|---|---|---|"]
        for s, m in marks.items():
            lines.append(f"| {s} | {m} | {calculate_grade(m)} |")
        return "\n".join(lines)

    for subject, score in pairs:
        marks[subject.title()] = int(score)

    avg = sum(marks.values()) / len(marks)
    return f"Marks updated. Current Average = {avg:.2f}%"

# ============================================================
# 6. TOOLS
# ============================================================

def positive_prompt_tool(text: str, session_id: str):
    prompt = f"User: {text}\nRespond positively in 2 sentences."
    return llm.invoke(prompt).content.strip()


def negative_prompt_tool(text: str, session_id: str):
    prompt = f"User: {text}\nRespond empathetically with one suggestion."
    return llm.invoke(prompt).content.strip()


def suicide_safety_tool(_: str, session_id: str):
    return (
        "I'm really sorry you're feeling this way.\n"
        "Please reach out to someone you trust or call a helpline immediately."
    )

# ============================================================
# 7. LANGCHAIN TOOL REGISTRATION
# ============================================================

def get_agent(session_id: str):

    if session_id in agent_store:
        return agent_store[session_id]

    memory = get_memory(session_id)

    tools = [
        Tool("PositiveResponse",
             lambda t: positive_prompt_tool(t, session_id),
             "For happy or motivational messages"),

        Tool("NegativeResponse",
             lambda t: negative_prompt_tool(t, session_id),
             "For sad or emotional messages"),

        Tool("StudentMarks",
             lambda t: student_marks_tool(t, session_id),
             "For storing and viewing marks"),

        Tool("SafetyTool",
             lambda t: suicide_safety_tool(t, session_id),
             "For suicide or self-harm situations"),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        memory=memory,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True
    )

    agent_store[session_id] = agent
    return agent

# ============================================================
# 8. MAIN CHAT ROUTER (WITH HISTORY SUPPORT )
# ============================================================

def chat(message: str, session_id: str = "default") -> str:

    if not message.strip():
        return "Please type something."

    #  History Command
    if message.lower() == "history":
        return get_user_history(session_id)

    # Save query to memory
    get_memory(session_id).save_context({"input": message}, {"output": ""})

    #  Save query to vector DB
    store_in_vector_db(message, "raw_user_input")

    #  Semantic routing
    intent = semantic_router(message)

    #  Tool execution
    if intent == "positive":
        reply = positive_prompt_tool(message, session_id)
    elif intent == "negative":
        reply = negative_prompt_tool(message, session_id)
    elif intent == "academic":
        reply = student_marks_tool(message, session_id)
    elif intent == "safety":
        reply = suicide_safety_tool(message, session_id)
    else:
        agent = get_agent(session_id)
        reply = agent.run(message)

    #  Save final response to memory
    get_memory(session_id).save_context({"input": message}, {"output": reply})

    #  Store response in vector DB
    store_in_vector_db(reply, intent)

    return reply

# ============================================================
# 9. RUN LOOP
# ============================================================

print("\n=== LangChain Semantic Tool Agent (With History) ===")

while True:
    q = input("\nUser: ")
    if q.lower() in ["exit", "quit"]:
        break

    ans = chat(q)
    print("AI:", ans)
