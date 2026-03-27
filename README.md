# RAG Chat Bot

A hybrid RAG (Retrieval-Augmented Generation) application for querying your own documents. It combines **BM25** (keyword matching) and **FAISS** (semantic vector search) to retrieve relevant information from uploaded files, and exposes the index as MCP tools for Claude Code.

## Prerequisites

- Python 3.10+
- (Optional) API key for generative answers in `bot.py`: Google Gemini, OpenAI, or a GitHub Models token.

## Setup

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # Git Bash
    # venv\Scripts\activate       # CMD/PowerShell
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure environment variables** *(only needed for `bot.py`)*:
    Create a `.env` file:
    ```text
    GEMINI_API_KEY=your_gemini_key_here
    OPENAI_API_KEY=your_openai_key_here
    GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
    ```
    The bot tries Gemini first, then GitHub Models, then OpenAI. All are optional — it falls back to retrieval-only mode if none are set.

## Indexing Documents

### Option A — Web UI (recommended)

```bash
python web_app.py
```

Open `http://localhost:8000`, upload 1–10 documents, and click **Build Index**. Supported formats: **PDF**, **PPTX**, **DOC/DOCX**.

The page shows live indexing progress and confirms when the MCP servers are ready to use.

### Option B — Command line

```bash
python build_db.py                         # scans project dir for .pdf / .pptx / .docx
python build_db.py file1.pdf file2.pptx   # explicit file paths
```

Both options produce `faiss_index/` and `bm25_retriever.pkl`, which the MCP servers and `bot.py` load on startup.

## Usage

### Interactive chat bot

```bash
python bot.py
```

- **Retrieval-only mode** (default): returns the most relevant raw passages.
- **Generative mode**: if an API key is present, an LLM synthesizes an answer from the retrieved context.
- Detects domain keywords (e.g. `vincit`, `cost`, `ai`, `llm`, `copilot`, `rag`) and highlights them.
- Source attribution shows the slide number (PPTX), page number (PDF), or document name.

---

## MCP Servers

Both MCP servers are configured in `.mcp.json` and start automatically when Claude Code opens this project.

> **Note:** `.mcp.json` contains absolute paths. Update the Python executable path and script paths to match your local clone before use.

### ragserver — Hybrid Search (FAISS + BM25)

**File:** `rag_server.py`
**Search type:** Ensemble — BM25 (weight 0.5) + FAISS semantic search (weight 0.5), top 3 chunks per retriever.
**Best for:** Conceptual or semantic queries where exact keywords may not appear in the source.

```
use ragserver mcp — how does Vincit approach test automation?
```

Logs: `rag_server_debug.log`

### ragbm25 — BM25 Keyword Search

**File:** `rag_bm25.py`
**Search type:** Pure BM25, top 5 chunks.
**Best for:** Exact keyword or terminology lookups (tool names, company names, section headings).

```
use ragbm25 mcp — what does the Siili handbook say about hallucinations?
```

Logs: `rag_bm25_debug.log`

### ragfaiss — FAISS-only Semantic Search

**File:** `rag_faiss.py`
**Search type:** Pure FAISS semantic vector search, top 5 chunks.
**Best for:** Meaning-based queries where exact keywords may not appear in the source. Uses `all-MiniLM-L6-v2` embeddings.

```
use ragfaiss mcp — what approaches are recommended for AI-powered development?
```

Logs: `rag_faiss_debug.log`

### When to use which

| Situation | Use |
|---|---|
| Broad topic / conceptual question | `ragserver` or `ragfaiss` |
| Exact term, tool name, or phrase | `ragbm25` |
| Want more result chunks | `ragbm25` or `ragfaiss` (both return 5) |
| Query mixes keywords and meaning | `ragserver` |
| Pure semantic similarity search | `ragfaiss` |

All servers require the indexes to exist. If they are missing, run `web_app.py` or `build_db.py` first.
