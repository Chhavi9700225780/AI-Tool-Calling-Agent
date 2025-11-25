
---

# 📘 LangChain AI-Tool-Calling Agent 

## 🧩 **Overview**

This project is a **LangChain Router-Agent** that uses **tool calling, prompt classification, memory**, and **FastAPI + Streamlit** to build an AI system that automatically routes user queries to different tools:

* Positive message generator
* Negative prompt handler
* Suicide-related helper
* Student marks management tool
* Default conversational LLM

The system uses **Google Gemini 2.5 Flash**, **LangChain (0.1.x)**, **FastAPI backend**, and a **Streamlit frontend**.

---

## 🎯 **Purpose of the Project**

This project demonstrates how to build:

### ✔️ A Router Agent

Using LangChain’s `RunnableBranch`, the system detects the intent and selects the correct tool.

### ✔️ Multi-tool AI System

Four tools + default LLM pipeline.

### ✔️ Full-Stack LLM App

* **Backend** → FastAPI
* **Frontend** → Streamlit
* **Memory** → ConversationBufferMemory (per session)

### ✔️ Local Interaction + API

Perfect for learning LLM agent architectures using LangChain.

---

## 🚀 **Key Features**

### 🧠 **1. Intent Classification Router**

Classifies each incoming query into:

* `positive`
* `negative`
* `marks`
* `suicide`
* `default`

### 🔧 **2. Tools Implemented**

| Tool                 | Description                       |
| -------------------- | --------------------------------- |
| Positive Prompt Tool | Motivational message              |
| Negative Prompt Tool | Warns about negativity            |
| Suicide Safety Tool  | Sends supportive + safety message |
| Student Marks Tool   | Add & Retrieve student marks      |
| Default LLM Tool     | Handles general questions         |

---

## 🏗️ **Project Architecture**

<img width="1344" height="768" alt="Gemini_Generated_Image_gdn7ljgdn7ljgdn7" src="https://github.com/user-attachments/assets/56cbd7df-c1bc-432a-8eb0-83180eb5b32c" />


---

## 📁 **Folder Structure**

```
AI-Tool-Calling-Agent/
│
├── LangChainToolCallingAgentCode/
│   ├── main.py
│   ├── requirements.txt
│
├── agent_backend.py        # FastAPI Server
├── agent_frontend.py       # Streamlit Client UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ **Tech Stack**

* **Python 3.10+**
* **FastAPI**
* **Streamlit**
* **LangChain 0.1.x**
* **Google Gemini 2.5 Flash**
* **Requests**
* **Uvicorn**

---

## ⚙️ **Installation & Setup**

### **1. Clone the Repository**

```bash
git clone <your-repo-url>
cd AI-Tool-Calling-Agent
```

---

### **2. Install Requirements**

Use one common `requirements.txt`:

```
pip install -r requirements.txt
```

---

### **3. Set Environment Variables**

Create `.env`:

```
GOOGLE_API_KEY=your_api_key_here
```

---

### **4. Run FastAPI Backend**

```bash
uvicorn agent_backend:app --reload
```

Backend will run at:

```
http://localhost:8000
```

---

### **5. Run Streamlit Frontend**

```bash
streamlit run agent_frontend.py
```

Streamlit UI opens at:

```
http://localhost:8501
```

---

## 🖼️ **Screenshots (Add Here)**



### 📌 **Streamlit UI**

<img width="692" height="284" alt="image" src="https://github.com/user-attachments/assets/8ef8ea4c-a514-4851-baeb-ea4a3f64f745" />
<img width="693" height="352" alt="image" src="https://github.com/user-attachments/assets/7248209d-4320-46ec-92a6-46e5e02553ff" />
<img width="692" height="197" alt="image" src="https://github.com/user-attachments/assets/e942aa80-3706-420c-92f5-68387490fcfa" />




### 📌 **FastAPI Endpoint Test**


<img width="691" height="358" alt="image" src="https://github.com/user-attachments/assets/a335d43a-1091-4368-9edb-696e417a4fc1" />
<img width="692" height="239" alt="image" src="https://github.com/user-attachments/assets/20626fca-0304-4d80-81d0-f627c073dd69" />
<img width="692" height="330" alt="image" src="https://github.com/user-attachments/assets/9ba32c24-3c82-48c8-8ac3-cfa126245aa6" />


---

## 🔌 **API Endpoint (FastAPI)**

### **POST /query**

#### **Request Body**

```json
{
  "query": "add alice marks 92 for math",
  "session_id": "user-123"
}
```

#### **Response**

```json
{
  "response": "Added marks: Alice - math: 92"
}
```

---

## 🧪 **Examples to Try**

```
1. add alice marks 92 for math
2. get marks of alice for math
3. say something positive
4. say something negative
5. I feel hopeless
6. What is the capital of Spain?
```

---

## 🧵 **How Memory Works**

* Each user is assigned a **unique session ID**
* Each session stores conversation using:

```
ConversationBufferMemory
```

* Backend stores memory inside `session_memories` dictionary.

---

## 🐞 **Troubleshooting**

### ❗ “convert_system_message_to_human” Error

Fix applied in code:

```python
convert_system_message_to_human=True
```

### ❗ CORS Error

Backend includes:

```python
allow_origins=["*"]
```

### ❗ Streamlit Not Updating

Clear cache:

```bash
streamlit cache clear
```


---

## 🧑‍💻 **Author**

**Chhavi**

Infosys Virtual Internship 6.0

AI Tool-Calling Agent (LangChain)


