import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain primitives
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="📝 RAG Q&A", layout="wide")
st.title("📝 RAG Q&A with Multiple PDFs + Chat History")

# Sidebar
with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs → Ask questions → Get answers")

# Accept key from input OR .env
api_key = api_key_input or os.getenv("GROQ_API_KEY")
if not api_key:
    st.warning("🔑 Please enter your Groq API Key (or set GROQ_API_KEY in .env).")
    st.stop()

# Embeddings & LLM (initialize only after we have a key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")

# ── Upload PDFs ────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader("📚 Upload PDF files", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload one or more PDFs to begin.")
    st.stop()

all_docs = []
tmp_paths = []

for pdf in uploaded_files:
    # write to a temp file so PyPDFLoader can read it
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()
    tmp_paths.append(tmp.name)

    loader = PyPDFLoader(tmp.name)
    docs = loader.load()
    for d in docs:
        d.metadata["source_file"] = pdf.name
    all_docs.extend(docs)

st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")

# Clean up temp files ASAP
for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass


# ── Chunking ──────────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(all_docs)

# ── Vectorstore (fresh per upload; avoid stale persistence) ───────────────────
vectorstore = Chroma.from_documents(splits, embeddings)  # in-memory for reliability
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

st.sidebar.write(f"🔍 Indexed {len(splits)} chunks for retrieval")

# ── Helper: format docs for stuffing ───────────────────────────────────────────
def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# ── Prompts ───────────────────────────────────────────────────────────────────
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that rewrites the user's latest question into a "
     "standalone search query, using the chat history for context. "
     "Return only the rewritten query, no preamble."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT RAG assistant. you MUST nswer using only the provided context.\n"
     "If the context does not contain  the answer, tehn reply exactly:"
     "Out of scope- not found in provided documents.'\n"
     "Do NOT use outside knowledge.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ── Session state for chat history ────────────────────────────────────────────
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

# ── Chat UI ───────────────────────────────────────────────────────────────────
session_id = st.text_input("🆔 Session ID", value="default_session")
user_q = st.chat_input("💬 Ask a question...")

if user_q:
    history = get_history(session_id)

    # 1) Rewrite question with history → standalone search query
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )
    try:
        standalone_q = llm.invoke(rewrite_msgs).content.strip()
    except Exception as e:
        st.error(f"LLM rewrite error: {e}")
        st.stop()

    # 2) Retrieve docs for the rewritten question (Runnable retriever in LC 1.x)

    docs = retriever.invoke(standalone_q)  # <-- FIX: use .invoke(), not get_relevant_documents()

    if not docs:
        st.chat_message("assistant").write("Out of Scope - not found in provided document.")
    

    # 3) Build context string
    context_str = _join_docs(docs)

    # 4) Ask final question with stuffed context
    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        context=context_str
    )
    try:
        answer = llm.invoke(qa_msgs).content
    except Exception as e:
        st.error(f"LLM answer error: {e}")
        st.stop()

    # 5) Render + persist to chat history
    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)
    history.add_user_message(user_q)
    history.add_ai_message(answer)

    # ── Debug / Transparency panels ────────────────────────────────────────────
    with st.expander("🧪 Debug: Rewritten Query & Retrieval"):
        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")
    if docs:
        with st.expander("📑 Retrieved Chunks"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**")
                st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
    else:
        st.warning("No relevant chunks were retrieved. Try asking a simpler or more specific question, or upload a document that contains the answer.")
