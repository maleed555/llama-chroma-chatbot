import streamlit as st
import os
import chromadb
import tempfile
import shutil
import time

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --------------------------------------------------
# 1️⃣ APP CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🦙 Llama-Chroma Assistant",
    page_icon="🦙",
    layout="wide",
)

# --------------------------------------------------
# 2️⃣ CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
<style>
.stApp {
    background-color:#0f172a;
    color:#f1f5f9;
    font-family:'Inter',sans-serif;
}
.app-header{text-align:center;margin-bottom:25px;}
.app-header h1{font-size:3rem;font-weight:800;color:#38bdf8;}
.app-header p{font-size:1.1rem;color:#e2e8f0;}

.card{
    background:#1e293b;
    border-radius:15px;
    padding:20px;
    margin-bottom:20px;
}

.chat-container{
    display:flex;
    flex-direction:column;
    gap:10px;
    max-width:900px;
    margin:auto;
    height:65vh;
    overflow-y:auto;
    padding-bottom:120px;
}

.chat-message{
    border-radius:12px;
    padding:14px;
    max-width:70%;
    word-break:break-word;
}

.chat-message.user{
    background:#2563eb;
    color:white;
    align-self:flex-end;
}

.chat-message.assistant{
    background:#0ea5e9;
    color:#0f172a;
    align-self:flex-start;
}

.stChatInput{
    position:fixed;
    bottom:20px;
    left:50%;
    transform:translateX(-50%);
    width:70%;
    max-width:900px;
    z-index:999;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3️⃣ HEADER
# --------------------------------------------------
st.markdown(
    """
<div class="app-header">
    <h1>🦙 Llama-Chroma Assistant</h1>
    <p>Upload documents and chat with your knowledge base</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 4️⃣ HELPERS
# --------------------------------------------------
DB_PATH = os.path.join(os.getcwd(), "chroma_db")

def safe_rmtree(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception:
        pass

def calculate_doc_stats(docs):
    stats = {}
    for d in docs:
        src = d.metadata.get("file_name", "Unknown")
        stats[src] = len(d.text.split())
    return stats

# --------------------------------------------------
# 5️⃣ UPLOAD + RESET
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

api_key = st.secrets.get("GOOGLE_API_KEY")

uploaded_files = st.file_uploader(
    "Upload knowledge base (PDF, DOCX, TXT)",
    accept_multiple_files=True,
)

if st.button("🗑️ Reset Database"):
    db = chromadb.PersistentClient(path=DB_PATH)
    try:
        db.delete_collection("streamlit_rag")
    except:
        pass

    safe_rmtree(DB_PATH)

    for k in list(st.session_state.keys()):
        del st.session_state[k]

    st.success("Database reset successfully")
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 6️⃣ INIT MODELS (FIXED)
# --------------------------------------------------
@st.cache_resource
def init_models():
    Settings.llm = Gemini(
        model="models/gemini-1.0-pro",   # ✅ STABLE & AVAILABLE
        api_key=api_key,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"  # ✅ FREE
    )

    Settings.chunk_size = 1024

# --------------------------------------------------
# 7️⃣ VECTOR STORE
# --------------------------------------------------
def get_index():
    os.makedirs(DB_PATH, exist_ok=True)

    db = chromadb.PersistentClient(path=DB_PATH)
    collection = db.get_or_create_collection("streamlit_rag")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    new_upload = (
        "last_files" not in st.session_state
        or st.session_state.last_files != [f.name for f in uploaded_files]
    )

    if uploaded_files and new_upload:
        with tempfile.TemporaryDirectory() as tmp:
            for f in uploaded_files:
                with open(os.path.join(tmp, f.name), "wb") as out:
                    out.write(f.getbuffer())

            docs = SimpleDirectoryReader(tmp).load_data()
            st.session_state.doc_stats = calculate_doc_stats(docs)

            index = VectorStoreIndex.from_documents(
                docs, storage_context=storage_context
            )

        st.session_state.last_files = [f.name for f in uploaded_files]
        return index

    return VectorStoreIndex.from_vector_store(vector_store)

# --------------------------------------------------
# 8️⃣ CHAT
# --------------------------------------------------
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY missing in Streamlit Secrets")
    st.stop()

init_models()

index = get_index()
query_engine = index.as_query_engine(similarity_top_k=3)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for m in st.session_state.messages:
    st.markdown(
        f'<div class="chat-message {m["role"]}">{m["content"]}</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask your documents…"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    if (
        "word" in prompt.lower()
        and "document" in prompt.lower()
        and "doc_stats" in st.session_state
    ):
        answer = "\n".join(
            f"{k}: {v} words"
            for k, v in st.session_state.doc_stats.items()
        )
    else:
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            answer = response.response

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()
