import streamlit as st
import os
import chromadb
import tempfile
import shutil
import time
from dotenv import load_dotenv

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
load_dotenv()

st.set_page_config(
    page_title="🦙 Llama-Chroma Assistant",
    page_icon="🦙",
    layout="wide",
)

# --------------------------------------------------
# 2️⃣ CSS (YOUR THEME + FIXED INPUT)
# --------------------------------------------------
st.markdown(
    """
<style>
.stApp {
    background-color: #0f172a;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

.app-header { text-align:center; margin-bottom:25px; }
.app-header h1 { font-size:3rem; font-weight:800; color:#38bdf8; }
.app-header p { font-size:1.1rem; color:#e2e8f0; }

.card {
    background-color:#1e293b;
    border-radius:15px;
    padding:20px;
    margin-bottom:20px;
}

.chat-container {
    display:flex;
    flex-direction:column;
    gap:10px;
    max-width:900px;
    margin:auto;
    height:65vh;
    overflow-y:auto;
    padding-bottom:120px;
}

.chat-message {
    border-radius:12px;
    padding:14px;
    max-width:70%;
    word-break:break-word;
}

.chat-message.user {
    background:#2563eb;
    color:#fff;
    align-self:flex-end;
}

.chat-message.assistant {
    background:#0ea5e9;
    color:#0f172a;
    align-self:flex-start;
}

/* FIX CHAT INPUT */
.stChatInput {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 70%;
    max-width: 900px;
    z-index: 999;
}

.stChatInput textarea {
    background-color:#1e293b !important;
    color:#f1f5f9 !important;
    border-radius:12px !important;
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
def safe_rmtree(path, retries=5, delay=0.5):
    for _ in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            break
        except PermissionError:
            time.sleep(delay)


def calculate_doc_stats(documents):
    stats = {}
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        stats[source] = len(doc.text.split())
    return stats


# --------------------------------------------------
# 5️⃣ UPLOAD SECTION
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

api_key = os.getenv("GOOGLE_API_KEY")
uploaded_files = st.file_uploader(
    "Upload knowledge base (PDF, DOCX, TXT)",
    accept_multiple_files=True,
)

if st.button("🗑️ Reset Database"):
    try:
        db = chromadb.PersistentClient(path="./chroma_db")
        for col in db.list_collections():
            db.delete_collection(col.name)
        safe_rmtree("./chroma_db")
        st.session_state.clear()
        st.success("Database reset successfully")
    except Exception as e:
        st.error(e)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 6️⃣ INIT MODELS
# --------------------------------------------------
@st.cache_resource
def init_models():
    Settings.llm = Gemini(
        model_name="models/gemini-3-flash-preview",
        api_key=api_key,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.chunk_size = 1024


# --------------------------------------------------
# 7️⃣ VECTOR STORE
# --------------------------------------------------
def get_index():
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection("streamlit_rag")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    new_upload = (
        "last_files" not in st.session_state
        or st.session_state.last_files != [f.name for f in uploaded_files]
    )

    if uploaded_files and new_upload:
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())

            documents = SimpleDirectoryReader(tmpdir).load_data()
            st.session_state.doc_stats = calculate_doc_stats(documents)

            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )

        st.session_state.last_files = [f.name for f in uploaded_files]
        return index

    return VectorStoreIndex.from_vector_store(vector_store)


# --------------------------------------------------
# 8️⃣ CHAT
# --------------------------------------------------
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found")
    st.stop()

init_models()

index = get_index()
query_engine = index.as_query_engine(similarity_top_k=3)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    role = msg["role"]
    st.markdown(
        f'<div class="chat-message {role}">{msg["content"]}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div id="chat-end"></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input
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
        answer = query_engine.query(prompt).response

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()

