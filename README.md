# PDF Q&A Chatbot (Local Prototype)

A tiny, **no-ChatGPT-custom-bot** prototype that lets users upload a PDF and ask questions about it.
- UI: Streamlit
- Retrieval: Chroma (in-memory) + sentence-transformers
- LLM: choose **OpenAI** (set `OPENAI_API_KEY`) *or* **Ollama** locally (default if available)

## Quick Start

### 1) Python env
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Choose an LLM backend

**Option A — OpenAI (hosted):**
```bash
export OPENAI_API_KEY=sk-...   # Windows: set OPENAI_API_KEY=...
# Optional (defaults below):
# export OPENAI_MODEL=gpt-4o-mini
```

**Option B — Ollama (local):**
- Install Ollama: https://ollama.com
- Pull a model (e.g. llama3.1): `ollama pull llama3.1`
- Make sure Ollama is running (default: `http://localhost:11434`).
- Optional:
```bash
export OLLAMA_MODEL=llama3.1
export OLLAMA_HOST=http://localhost:11434
```

> If **both** OPENAI_API_KEY and Ollama are available, the app will prefer **OpenAI** unless you set `PREFER_OLLAMA=true`.

### 3) Run it
```bash
streamlit run app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501).

## Notes
- This is a prototype: it keeps embeddings in-memory (not persisted). Uploading a new PDF replaces the store.
- For multi-file or persistent stores, move Chroma to a disk path and index multiple docs.
- See `app.py` for inline comments and places to tweak chunk size, overlap, etc.

