import os
import io
from typing import List, Dict, Optional, Tuple

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np
import httpx
import tiktoken
from openai import OpenAI

# ---------------------------
# Config & helpers
# ---------------------------

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="wide")

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PREFER_OLLAMA = os.getenv("PREFER_OLLAMA", "false").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def choose_backend() -> str:
    """
    Decide whether to use OpenAI or Ollama, based on env and preference.
    Returns 'openai' or 'ollama'.
    """
    if PREFER_OLLAMA:
        return "ollama"
    if OPENAI_API_KEY:
        return "openai"
    # Fallback: try Ollama
    return "ollama"

BACKEND = choose_backend()

# Token counter (rough) for display
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

# ---------------------------
# PDF loading & chunking
# ---------------------------

def read_pdf_to_text(file_bytes: bytes) -> str:
    """Extract plain text from a PDF bytes object."""
    with io.BytesIO(file_bytes) as f:
        reader = PdfReader(f)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages)

def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    Simple character-based chunker. In practice consider token-based chunking.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    # strip empties
    return [c.strip() for c in chunks if c and c.strip()]

# ---------------------------
# Embeddings & Vector store
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_embedder():
    # Use sentence-transformers locally for embeddings (no API needed)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def create_vector_store(chunks: List[str]):
    """
    Build an in-memory Chroma collection with sentence-transformer embeddings.
    """
    client = chromadb.Client(chromadb.config.Settings(allow_reset=True))
    # fresh in-memory store per upload
    client.reset()
    collection = client.create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"source": "uploaded_pdf", "index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection

def retrieve(query: str, collection, k: int = 5) -> List[Dict]:
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    # package nicely
    out = []
    for i, d in enumerate(docs):
        out.append({"text": d, "meta": metas[i]})
    return out

# ---------------------------
# LLM backends
# ---------------------------

def call_ollama(system: str, messages: List[Dict], model: str = DEFAULT_OLLAMA_MODEL, host: str = DEFAULT_OLLAMA_HOST) -> str:
    """
    Call Ollama /api/chat if available; fallback to /api/generate with a baked prompt.
    We try /api/chat first.
    """
    try:
        payload = {
            "model": model,
            "messages": [{"role":"system","content":system}] + messages,
            "stream": False,
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{host}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "").strip()
    except Exception:
        # Fallback to /api/generate
        prompt = system + "\n\n" + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
        payload = {"model": model, "prompt": prompt, "stream": False}
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{host}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()

def call_openai(system: str, messages: List[Dict], model: str = DEFAULT_OPENAI_MODEL) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def generate_answer(query: str, contexts: List[Dict]) -> str:
    """
    Build a RAG-style prompt and call the selected backend.
    """
    context_block = "\n\n---\n\n".join([c["text"] for c in contexts])
    system = (
        "You are a precise assistant for question answering over a provided PDF. "
        "Cite only from the given context. If the answer isn't in the context, say you don't know."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Context (quoted from the PDF; use what's relevant, don't hallucinate):\n{context_block}"
        "\n\nAnswer clearly and concisely. If uncertain, say so."
    )
    messages = [{"role": "user", "content": user}]

    if BACKEND == "openai" and OPENAI_API_KEY:
        return call_openai(system, messages)
    else:
        return call_ollama(system, messages)

# ---------------------------
# UI
# ---------------------------

st.title("📄 PDF Q&A Chatbot")
st.caption("Upload a PDF and ask questions about it. No ChatGPT custom bot required.")

with st.sidebar:
    st.subheader("LLM Backend")
    st.write(f"Using: **{('OpenAI' if (BACKEND=='openai' and OPENAI_API_KEY) else 'Ollama')}**")
    if BACKEND == "openai" and not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not set — falling back to Ollama.")
    st.markdown("---")
    st.subheader("Index Settings")
    chunk_size = st.slider("Chunk size (characters)", min_value=500, max_value=3000, value=1200, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=500, value=200, step=50)
    top_k = st.slider("Top-K retrieved", min_value=2, max_value=10, value=5, step=1)

# State: collection & chat
if "collection" not in st.session_state:
    st.session_state.collection = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, content)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded is not None:
    with st.spinner("Reading & indexing PDF..."):
        pdf_bytes = uploaded.read()
        text = read_pdf_to_text(pdf_bytes)
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            st.error("I couldn't extract text from that PDF. Try a different file.")
        else:
            st.session_state.collection = create_vector_store(chunks)
            st.success(f"Indexed {len(chunks)} chunks from your PDF.")

st.divider()

# Chat input
if st.session_state.collection is None:
    st.info("Upload a PDF first.")
else:
    # Show chat history
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    user_msg = st.chat_input("Ask the PDF a question...")
    if user_msg:
        st.session_state.history.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                contexts = retrieve(user_msg, st.session_state.collection, k=top_k)
                answer = generate_answer(user_msg, contexts)

                # Show sources small expander
                with st.expander("View retrieved context"):
                    for i, c in enumerate(contexts, 1):
                        st.markdown(f"**Match {i}** (chunk #{c['meta'].get('index','?')})")
                        st.code(c["text"][:1200] + ("..." if len(c["text"])>1200 else ""), language="markdown")

                st.markdown(answer)
                st.session_state.history.append(("assistant", answer))

# Footer
st.caption("Prototype: in-memory store, single PDF at a time. Swap to a persisted Chroma/FAISS DB for multi-file & re-usable indexes.")
