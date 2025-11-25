
# **🧠 Tool-Calling AI Agent**

A lightweight AI system built using **FastAPI**, **Streamlit**, **Langchain** **Sentence Transformers** that can *understand user queries* and *route them to the correct tool* using **semantic similarity**.

It also stores **long-term memory** in a JSON file for retrieval and context awareness.

---

## 📂 **Folder Structure**

Based on your screenshot:

```
FAPI/
│
├── __pycache__/            # Auto-generated cache files
│
├── agent_backend.py        # FastAPI backend (main AI logic + semantic router)
├── agent_frontend.py       # Streamlit frontend UI
├── longterm_memory.json    # Local memory storage file
│
└── README.md               # Project documentation (this file)
```

**Screenshot reference:**


```

<img width="693" height="395" alt="image" src="https://github.com/user-attachments/assets/517894f3-726a-4412-a2a9-ac8e5cf2d269" />

```

---

# 🚀 **Project Purpose**

This project demonstrates how to build your own **AI Agent** that:

### ✔️ Understands user queries

### ✔️ Chooses the correct tool using semantic embeddings

### ✔️ Runs custom tools (positive prompts, negative prompts, student marks, suicide detection)

### ✔️ Stores long-term memory

### ✔️ Has a working frontend + backend system

---

# 🧩 **What Each File Does**

## 🔹 **1. agent_backend.py (FastAPI Backend)**

<img width="692" height="390" alt="image" src="https://github.com/user-attachments/assets/2e2da724-9606-4295-bacc-165f2fd406a1" />

This is the *brain* of the entire system.

### **Purpose**

* Exposes `/chat` API endpoint
* Selects the best tool using semantic similarity
* Executes the chosen tool
* Saves all queries into longterm memory
* Returns structured JSON response

### **Key Features**

| Feature          | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| Semantic Routing | Uses SentenceTransformer to compare user query with tool descriptions |
| 4 Tools          | Positive Prompt, Negative Prompt, Student Marks DB, Suicide Detection |
| Memory           | Saves user interactions into longterm_memory.json                     |
| API              | Fully REST-based FastAPI endpoint `/chat`                             |
| Embeddings       | Uses all-MiniLM-L6-v2 model                                           |

### **Input Format**

```json
{
  "user_input": "your query here"
}
```

### **Output Example**

```json
{
  "routed_tool": "positive-prompt",
  "router_score": 0.88,
  "tool_output": { ... }
}
```

---

## 🔹 **2. agent_frontend.py (Streamlit UI)**

A clean frontend that allows users to interact with the API visually.

<img width="691" height="369" alt="image" src="https://github.com/user-attachments/assets/2a291c26-4d80-49b0-8e0a-e019b736c2cd" />


### **Purpose**

* Send user queries to the FastAPI backend
* Display intelligent agent responses
* Show router score, selected tool, output, and JSON debug view
* Check backend server health

### **How It Works**

1. User enters a query
2. Streamlit sends it to the backend via POST request
3. Displays formatted tool output
4. Shows full JSON response

---

## 🔹 **3. longterm_memory.json**

### **Purpose**

Stores **all user queries**, tool selections, and timestamps.

### **Used For**

* Memory-based query retrieval
* Debugging
* Future personalization

### **Format**

```json
[
  {
    "type": "user_query",
    "text": "give positive prompt",
    "timestamp": 1732502000,
    "tool_used": "positive-prompt"
  }
]
```

---

# 🛠️ **Tools Implemented in the Agent**

| Tool Name           | Purpose                               | Example Input                       |
| ------------------- | ------------------------------------- | ----------------------------------- |
| **positive-prompt** | Returns uplifting suggestions         | "Give me a positive prompt"         |
| **negative-prompt** | Filters negative words from prompts   | "Remove negativity"                 |
| **student-marks**   | Add/get student marks                 | "Add 95 marks for Chris in History" |
| **suicide-related** | Detects risky language, gives support | "I want to end my life"             |

---

# 📡 **API Endpoint**

### **POST** `/chat`

Sends a query to backend and returns structured JSON.

**Example Request**

```bash
curl -X POST http://127.0.0.1:8000/chat \
-H "Content-Type: application/json" \
-d '{"user_input":"Add 90 marks for John in Maths"}'
```

---

# 🖥️ How to Run the Project

## **1️⃣ Start FastAPI Backend**

```bash
uvicorn agent_backend:app --reload
```

Backend runs at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## **2️⃣ Start Streamlit Frontend**

```bash
streamlit run agent_frontend.py
```

Frontend runs at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

# 🖼️ Screenshots



## 🖼️ UI Screenshot
<img width="691" height="388" alt="image" src="https://github.com/user-attachments/assets/533d163f-9bd4-43df-8a4b-c9ac2d60fdba" />


## 📝 Input/Output Example
<img width="691" height="371" alt="image" src="https://github.com/user-attachments/assets/a3136ebd-0168-49eb-b585-9025cee8926f" />


---

# 📦 Technologies Used

* **Python 3.10+**
* **FastAPI**
* **Streamlit**
* **LangChain**
* **Sentence Transformers**
* **NumPy**
* **Uvicorn**
* **Requests**

---


