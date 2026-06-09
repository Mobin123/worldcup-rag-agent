import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

# ─────────────────────────────────────────
# STEP 1 — Load API Keys
# ─────────────────────────────────────────
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# STEP 2 — Load FAISS Vector Store
# ─────────────────────────────────────────
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def rewrite_query(question, chat_history):
    # If no chat history, no rewriting needed
    if not chat_history:
        return question
    
    # Build last 4 messages for context (last 2 exchanges)
    recent_history = chat_history[-4:]
    history_text = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    
    # Ask Groq to rewrite the question
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
"content": """You are a query rewriter.
ONLY rewrite a question if it contains these EXACT signals:
- Pronouns: 'he', 'she', 'they', 'it'
- Vague references: 'that tournament', 'that match', 'same year'
- Incomplete questions: 'Who was the runner up?', 'What was the score?'

DO NOT rewrite if:
- Question asks for a comparison across ALL World Cups
- Question contains 'most', 'highest', 'lowest', 'best', 'worst'
- Question is already complete and specific
- Question asks about rankings or records

EXAMPLES - DO NOT REWRITE THESE:
'Which World Cup had most goals?' → return EXACTLY as is
'Which team won the most titles?' → return EXACTLY as is
'Who hosted the 1986 World Cup?' → return EXACTLY as is

EXAMPLES - REWRITE THESE:
'Who was the runner up?' → 'Who was the runner up in the 1986 World Cup?'
'What was the score?' → 'What was the score in the 1930 World Cup Final?'

Return ONLY the question. Nothing else. No explanation."""
            },
            {
                "role": "user",
                "content": f"""Conversation history:
{history_text}

Follow-up question: {question}

Rewrite this as a standalone question:"""
            }
        ],
        temperature=0
    )
    
    rewritten = response.choices[0].message.content.strip()
    print(f"\nOriginal question: {question}")
    print(f"Rewritten question: {rewritten}\n")
    return rewritten

# ─────────────────────────────────────────
# STEP 3 — RAG Pipeline
# ─────────────────────────────────────────
def get_answer(question, chat_history):
    # Load vector store
    vector_store = load_vector_store()

    search_query = rewrite_query(question, chat_history)
    # Search for relevant documents
    relevant_docs = vector_store.similarity_search(search_query, k=10)
    # ADD IT HERE ↓
    print("\n=== FAISS Retrieved These Documents ===")
    for i, doc in enumerate(relevant_docs):
        print(f"\nDoc {i+1}:\n{doc.page_content}")
    print("=======================================\n")
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build conversation history for Groq
    messages = [
        {
            "role": "system",
            "content": f"""You are a FIFA World Cup expert assistant. 
Answer questions using ONLY the context provided below.
If the answer is not in the context, say 'I don't have that information.'
Be precise and factual.

Context:
{context}"""
        }
    ]

    # Add chat history
    for msg in chat_history:
        messages.append(msg)

    # Add current question
    messages.append({"role": "user", "content": question})

    # Get answer from Groq
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.1
    )

    return response.choices[0].message.content

# ─────────────────────────────────────────
# STEP 4 — Streamlit Chat UI
# ─────────────────────────────────────────
st.title("⚽ World Cup AI Assistant")
st.caption("Ask me anything about FIFA World Cup!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if question := st.chat_input("Ask your question here..."):

    # Show user question
    with st.chat_message("user"):
        st.write(question)

    # Save user question to history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching World Cup data..."):
            answer = get_answer(question, st.session_state.messages[:-1])
        st.write(answer)

    # Save answer to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })