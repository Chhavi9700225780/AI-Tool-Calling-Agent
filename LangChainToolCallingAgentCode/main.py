# ============================================
# IMPORTS
# ============================================
from langchain_google_genai import ChatGoogleGenerativeAI      # For Gemini LLM replies
from langchain_community.embeddings import HuggingFaceEmbeddings  # For local text embeddings
from langchain_core.runnables import RunnableBranch, RunnablePassthrough  # For routing logic
from langchain_community.vectorstores import FAISS           # For vector similarity search
from langchain.schema import Document                        # To store intent text
from langchain.memory import ConversationBufferMemory        # To store chat history
from dotenv import load_dotenv                               # To load API keys
import re                                                    # For command matching

# Load environment variables from .env file
load_dotenv()

# ============================================
# 1. MEMORY
# ============================================
# Stores previous user–AI conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Temporary in-memory storage for student marks
marks_memory = {}

# ============================================
# 2. LLM (FOR DEFAULT RESPONSES ONLY)
# ============================================
# Gemini model is used only for normal questions
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)

# ============================================
# 3. EMBEDDINGS MODEL (FOR SEMANTIC SIMILARITY)
# ============================================
# Local model for converting text into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================
# 4. TOOLS
# ============================================
# Tool for positive responses
def positive_prompt_tool(request: str):
    return f"Stay strong! {request}"

# Tool for negative responses
def negative_prompt_tool(request: str):
    return f"Negative prompt: Avoid negativity like '{request}'"

# Tool for sensitive suicide-related queries
def suicide_related_tool(request: str):
    return (
        "I'm really sorry you feel this way. "
        "Please reach out to someone you trust or a nearby helpline immediately. "
        "You matter, and there are people who care about you."
    )

# Tool to add and retrieve student marks
def student_marks_tool(request: str):
    global marks_memory

    if not isinstance(request, str):
        request = str(request)

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

    # If command format is incorrect
    return "Use 'add alice marks 92 for math' or 'get marks of alice for math'."

# ============================================
# 5. SEMANTIC INTENT DOCUMENTS
# ============================================
# These define how each intent looks in meaning (for similarity search)
documents = [
    Document(
        page_content="say something positive motivation encouragement happy inspire",
        metadata={"label": "positive"}
    ),

    Document(
        page_content="say something negative sad angry complain frustrated upset",
        metadata={"label": "negative"}
    ),

    Document(
        page_content=(
            "add student marks get student marks update marks "
            "add alice marks 90 for math "
            "get marks of alice for math "
            "store exam score retrieve exam score"
        ),
        metadata={"label": "marks"}
    ),

    Document(
        page_content="suicide self harm kill myself die depression hopeless end life",
        metadata={"label": "suicide"}
    ),

    Document(
        page_content="general question normal chat ask anything information",
        metadata={"label": "default"}
    )
]

# ============================================
# 6. FAISS VECTOR DATABASE
# ============================================
# Converts intent documents into vectors and stores them
vector_db = FAISS.from_documents(documents, embeddings)

# ============================================
# 7. SEMANTIC ROUTER FUNCTION
# ============================================
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

# ============================================
# 8. ROUTER CONDITIONS
# ============================================
# These act like if–else checks for routing
def is_positive(x): return x["decision"] == "positive"
def is_negative(x): return x["decision"] == "negative"
def is_marks(x): return x["decision"] == "marks"
def is_suicide(x): return x["decision"] == "suicide"

# ============================================
# 9. TOOL BRANCHES
# ============================================
# Each branch connects an intent to its tool
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

default_branch = RunnablePassthrough.assign(
    output=lambda x: f"I can help! Here's my reply:\n\n{llm.invoke(x.get('request')).content}"
)

# ============================================
# 10. DELEGATION LOGIC
# ============================================
# Decides which tool is executed
delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch
)

# ============================================
# 11. COORDINATOR AGENT (SEMANTIC ROUTING)
# ============================================
# Full pipeline: input → semantic routing → tool → output
coordinator_agent = (
    RunnablePassthrough()
    .assign(decision=lambda x: semantic_router(x["request"]))
    | delegation_chain
    | (lambda x: x["output"])
)

# ============================================
# 12. MAIN LOOP
# ============================================
print("\n=== LangChain Semantic Router with Memory ===")
print("Try:")
print("- add alice marks 92 for math")
print("- get marks of alice for math")
print("- say something positive")
print("- say something negative")
print("- any normal question\n")

while True:
    query = input("\nEnter your query: ")

    if query.lower() in ["exit", "quit"]:
        break

    # Get response from coordinator agent
    result = coordinator_agent.invoke({"request": query})

    # Save conversation in memory
    memory.save_context(
        {"human": query},
        {"ai": result}
    )

    print("\nOUTPUT:", result)
