from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import re

load_dotenv()

# ============================================================
# 1. MEMORY
# ============================================================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

marks_memory = {}   # in-memory marks DB


# ============================================================
# 2. LLM with FIX for SYSTEM MESSAGE ERROR
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)


# ============================================================
# 3. TOOLS
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

    if not isinstance(request, str):
        request = str(request)

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
# 4. ROUTER PROMPT
# ============================================================
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Classify the user's request into exactly ONE label:
        positive, negative, marks, suicide, default"""
     ),
    ("user", "{request}")
])

router_chain = router_prompt | llm | StrOutputParser()


# ============================================================
# 5. ROUTER CONDITIONS
# ============================================================
def is_positive(x): return x["decision"] == "positive"
def is_negative(x): return x["decision"] == "negative"
def is_marks(x): return x["decision"] == "marks"
def is_suicide(x): return x["decision"] == "suicide"


# ============================================================
# 6. TOOL BRANCHES
# ============================================================
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


# ============================================================
# 7. DELEGATION LOGIC
# ============================================================
delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch
)


# ============================================================
# 8. COORDINATOR
# ============================================================
coordinator_agent = (
    RunnablePassthrough()
    .assign(decision=router_chain)
    | delegation_chain
    | (lambda x: x["output"])
)


# ============================================================
# 9. LOOP
# ============================================================
print("\n=== LangChain Router with Memory ===")
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

    # Get LLM/tool result
    result = coordinator_agent.invoke({"request": query})

    # Save to memory
    memory.save_context({"human": query}, {"ai": result})

    print("\nOUTPUT:", result)
