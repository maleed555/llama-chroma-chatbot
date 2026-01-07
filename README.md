# 🦙 Llama-Chroma AI Document Chatbot

A ChatGPT-style AI web application that allows users to upload documents and chat with them using LlamaIndex, Google Gemini, and ChromaDB, built with Streamlit.

This app is suitable for knowledge bases, reports, policies, educational documents, and internal company data.

---

FEATURES

- Upload and chat with PDF, DOCX, TXT, CSV, XLSX files
- Google Gemini powered responses
- Semantic search using vector embeddings
- Persistent storage using ChromaDB
- Clean chat-style interface
- Fast performance with cached models
- Ready for Streamlit Cloud deployment

---

TECH STACK

Core Framework:
- LlamaIndex

AI Models:
- Google Gemini
- HuggingFace Embeddings

Vector Database:
- ChromaDB (Persistent)

UI:
- Streamlit

---

REQUIREMENTS

llama-index-core==0.14.12
llama-index-readers-file==0.5.6
llama-index-llms-gemini==0.6.1
llama-index-embeddings-huggingface==0.6.1
google-generativeai==0.8.6
chromadb==1.4.0
llama-index-vector-stores-chroma==0.5.5
docx2txt==0.9
pypdf==6.5.0
pandas==2.2.3
openpyxl==3.1.5
streamlit==1.52.2
python-dotenv==1.2.1

---

PROJECT STRUCTURE

llama-chroma-chatbot/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Ignore venv, chroma_db, env
├── chroma_db/              # Persistent vector store (auto-generated)
└── .env                    # Environment variables (local only)

---

IMPORTANT PATHS USED IN CODE

ChromaDB persistent storage path:

./chroma_db

Temporary upload processing:

tempfile.TemporaryDirectory()

Main app entry point:

app.py

---

ENVIRONMENT VARIABLES

Create a `.env` file locally:

GOOGLE_API_KEY=your_google_gemini_api_key

Example usage in code:

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

DO NOT push `.env` to GitHub.

---

STREAMLIT CLOUD SECRETS

Add this in Streamlit Cloud → App Settings → Secrets:

GOOGLE_API_KEY = "your_google_gemini_api_key_here"

Usage in code works automatically via os.getenv()

---

VECTOR STORE INITIALIZATION (CODE SNIPPET)

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("streamlit_rag")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

---

DOCUMENT LOADING (CODE SNIPPET)

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(tmpdir).load_data()

---

INDEX CREATION (CODE SNIPPET)

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

---

QUERYING DOCUMENTS (CODE SNIPPET)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    streaming=True
)

response = query_engine.query("Ask a question here")

---

RUN LOCALLY

Create virtual environment:

python -m venv .venv

Activate (Windows):

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run app:

streamlit run app.py

---

DEPLOY ON STREAMLIT CLOUD

1. Push this project to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New App"
4. Select your repository
5. Branch: main
6. File path: app.py
7. Add Secrets (GOOGLE_API_KEY)
8. Click Deploy

---

USE CASES

- Company knowledge base chatbot
- Policy and compliance assistant
- Educational document Q&A
- Business reports analysis
- Internal AI assistant
- Client-facing RAG chatbot projects

---

NOTES

- Uploaded documents are embedded and stored persistently in ChromaDB
- Re-uploading documents rebuilds the index
- Database reset is supported from UI
- Suitable for freelancing and SaaS demos

---

LICENSE

Free to use for personal, educational, and commercial projects.

---

AUTHOR

Built using LlamaIndex, Google Gemini, ChromaDB, and Streamlit.
Designed for AI freelancers, startups, and production-ready RAG systems.
