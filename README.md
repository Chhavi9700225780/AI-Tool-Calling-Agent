
# 🤖 LangChain Tool Calling Agent with Semantic Routing & Memory

This project is an intelligent multi-tool conversational AI system built using **LangChain**, **Groq LLM**, **FAISS Vector Database**, **FastAPI backend**, and **Streamlit frontend**.

It supports:
- ✅ Multi-turn conversations with memory
- ✅ Semantic routing using vector similarity
- ✅ Automatic tool selection via agent reasoning
- ✅ Student marks storage & retrieval
- ✅ Emotion-aware responses (positive / negative)
- ✅ Safety handling for sensitive queries
- ✅ API + Web UI integration

---
## Demo Video
https://drive.google.com/file/d/1Qrh_3ah9itbfgYXtPaEivJbiWM09syhM/view?usp=sharing

## 📁 Project Folder Structure

```

LangChainToolCallingAgentCode/
│
│
├── agent_backend.py           # FastAPI backend server
├── agent_frontend.py          # Streamlit frontend UI
├── main.py                    # Core AI agent logic (Brain of project)
├── check.py                   # Dependency / environment testing file
├── requirements.txt          # Project dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation

```

---

## 🧠 Overall Project Architecture

<img width="2496" height="1468" alt="autodraw 28_11_2025" src="https://github.com/user-attachments/assets/df971e5c-f4c3-4fc5-bbb6-0c36366c6ff0" />

---

## 🔄 Detailed Internal Workflow

### Step 1: User Sends a Message
User types a message in **Streamlit UI** → It is sent to FastAPI via `/chat` API.

![WhatsApp Image 2025-11-26 at 22 09 00_de5a47bb](https://github.com/user-attachments/assets/a13a0d79-9c5a-481b-9a9e-0bf09f7a94bd)

![WhatsApp Image 2025-11-26 at 22 08 08_e10aa8df](https://github.com/user-attachments/assets/171e741c-6fa2-4376-bb62-7c516e0916dc)

![WhatsApp Image 2025-11-26 at 22 08 42_0f462c70](https://github.com/user-attachments/assets/1520a027-63ac-41e6-8648-0793add6f924)

<img width="1893" height="809" alt="session-history" src="https://github.com/user-attachments/assets/a8ecbee5-e6e1-44a7-9e90-beea4ae9173a" />


### Step 2: Router Memory Stores the Message
```python
get_memory(session_id).save_context({"input": message}, {"output": ""})
````

This stores the user message for:

* Conversation continuity
* Follow-up question support
* Emotional context tracking
* `history` command support

✅ **Memory is for CONTEXT, not routing.**

---

### Step 3: Vector Database Stores the Message

```python
store_in_vector_db(message, "raw_user_input")
```

This stores the message as an embedding for:

* Semantic similarity
* Intent detection
* Repeated pattern learning

✅ **Vector DB is ONLY for semantic routing.**

---

### Step 4: Semantic Router Determines Intent

```python
intent = semantic_router(message)
```

* Converts user query into embeddings
* Compares with FAISS stored vectors
* Detects closest intent:

  * positive
  * negative
  * academic
  * safety
  * generic

---

### Step 5: Tool Selection & Execution

```python
if intent == "positive":
    reply = positive_prompt_tool(...)
elif intent == "negative":
    reply = negative_prompt_tool(...)
elif intent == "academic":
    reply = student_marks_tool(...)
elif intent == "safety":
    reply = suicide_safety_tool(...)
else:
    reply = agent.run(message)
```

* If intent matches a tool → that tool is executed directly
* Otherwise → LangChain **agent reasoning** selects the best tool automatically

---

### Step 6: LLM Generates Final Response

Groq LLM (`llama-3.3-70b-versatile`) generates the response using:

* Current user input
* Previous memory
* Tool output (if any)

---

### Step 7: Response Stored Again

```python
get_memory(session_id).save_context({"input": message}, {"output": reply})
store_in_vector_db(reply, intent)
```

* Final answer saved in memory
* Final answer stored in FAISS

---

### Step 8: Response Returned to Frontend

Streamlit displays:

* User message
* Agent reply
* Full chat history (if requested)

---

## 📂 File-wise Detailed Explanation

---

### ✅ `main.py` — Core AI Brain (MOST IMPORTANT)

This file contains:

* Groq LLM initialization
* Router memory setup
* FAISS vector database
* Semantic router
* Tool definitions
* Agent registration
* Main chat pipeline

It controls:

* ✅ Intent detection
* ✅ Tool execution
* ✅ Memory storage
* ✅ Vector DB updates
* ✅ Agent reasoning
* ✅ Final response generation

Without this file → **Project will not work.**

---

### ✅ `agent_backend.py` — FastAPI Backend

This file:

* Exposes `/chat` API
* Accepts user messages
* Calls `chat()` function from `main.py`
* Returns AI response as JSON

Acts as:

> **Bridge between frontend and AI brain**

---

### ✅ `agent_frontend.py` — Streamlit Frontend

This file:

* Creates UI for chat
* Sends requests to FastAPI
* Shows chat history
* Displays responses in real-time

Acts as:

> **User Interface of the AI system**

---

### ✅ `.env` — Environment Variables

Contains:

```
GROQ_API_KEY=your_api_key_here
```

Used securely to authenticate Groq LLM.

---

### ✅ `requirements.txt` — Dependencies

Contains:

* LangChain core libraries
* Groq SDK
* FAISS
* Sentence Transformers
* FastAPI
* Streamlit
* Torch
* dotenv

Used for:

```bash
pip install -r requirements.txt
```

---

### ✅ `check.py` — Environment Test File

Used to:

* Verify imports
* Verify LangChain installation
* Test memory availability

Not used in production.

---

### ✅ `venv/` — Virtual Environment

Isolated Python environment to:

* Avoid version conflicts
* Ensure stable execution

---

### ✅ `.gitignore`

Prevents pushing:

* `venv/`
* `.env`
* `__pycache__/`
  to GitHub.

---

## 🛠 Tools in the Project

| Tool Name        | Purpose                           |
| ---------------- | --------------------------------- |
| PositiveResponse | Motivation & happy replies        |
| NegativeResponse | Emotional & empathetic replies    |
| StudentMarks     | Stores & retrieves marks          |
| SafetyTool       | Handles suicide/sensitive queries |

The agent automatically decides which tool to call.

---

## 🧭 Key Concept Differences

| Component     | Role                       |
| ------------- | -------------------------- |
| Router Memory | Conversation context       |
| Vector DB     | Semantic intent detection  |
| Agent         | Tool selection & reasoning |
| Tools         | Task execution             |
| LLM           | Natural language response  |

---

## ▶️ How to Run the Project

### 1️⃣ Activate Virtual Environment

```bash
venv\Scripts\activate
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Backend (FastAPI)

```bash
uvicorn agent_backend:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

### 4️⃣ Run Frontend (Streamlit)

```bash
streamlit run agent_frontend.py
```

---

## 🧪 Example Queries

```
motivate me
i am feeling sad
store marks 93 in maths
get the marks in maths
history
```

---

## ✅ Final Summary

This project demonstrates:

* ✅ Intelligent tool-calling with LangChain
* ✅ Semantic routing with FAISS
* ✅ Multi-session memory handling
* ✅ Emotion-aware conversational AI
* ✅ Full-stack AI system (UI + API + LLM)

---

### 👩‍💻 Author

**Chhavi**
Final Year B.Tech | AI & DevOps Enthusiast

```


