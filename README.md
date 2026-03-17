# RAG Chat Bot for Sales & AI Materials

This application allows you to chat with the content of `Testing & QA sales materials.pptx` and the `SIILI-Handbook_for_AI-Powered_Developers-pages.pdf`. It uses a Hybrid Search (RAG) approach combining **BM25** (keyword matching) and **FAISS** (semantic vector search) to retrieve relevant information from both slides and document pages.

## Prerequisites

- Python 3.10+
- (Optional) OpenAI API Key, Google Gemini API Key, or GitHub Models token for generative answers.

## Setup

1.  **Create and Activate Virtual Environment** (if not already active):
    ```bash
    python -m venv venv
    source venv/Scripts/activate # For bash-like shells (Git Bash, etc.)
    # venv\Scripts\activate      # For CMD/PowerShell
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables** *(only needed for `bot.py` — skip this step if using Claude with the MCP servers)*:
    Create a `.env` file in the root directory and add your API keys:
    ```text
    GEMINI_API_KEY=your_gemini_key_here
    OPENAI_API_KEY=your_openai_key_here
    GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
    ```
    *(The bot prioritizes Gemini, then GitHub Models, then OpenAI as fallbacks.)*

4.  **Ingest Data**:
    Build the database (Vector Index + BM25 Index) from the PPTX and PDF files:
    ```bash
    python build_db.py
    ```

## Usage

Run the chat bot:
```bash
python bot.py
```

### Features

- **Multi-Document Support**: Automatically indexes and retrieves from both PowerPoint presentations and PDF documents.
- **Keyword Detection**: The bot specifically looks for keywords related to Vincit, costs, automation, and AI (e.g., `vincit`, `cost`, `ai`, `llm`, `copilot`, `rag`).
- **Hybrid Search**: Combines BM25 keyword search and FAISS semantic understanding for high-precision retrieval.
- **Intelligent Metadata Display**: Clearly indicates whether information was found on a specific Slide or Page, including the source filename.
- **Modes**:
    - **Retrieval Only (Default)**: If no API key is set, the bot returns the most relevant raw text passages.
    - **Generative Mode**: If `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `GITHUB_PERSONAL_ACCESS_TOKEN` is set, the bot uses an LLM to generate synthesized answers based on the retrieved context.

### Example Interaction

```text
You: What are the benefits of using a coding assistant like Cursor?
[Bot]: I noticed you're asking about: ai, cursor
[Bot]: According to the Handbook for AI-Powered Developers, Cursor positions itself as an AI-powered IDE...
...
```

---

## MCP Servers

Both documents can also be queried directly from Claude Code (or any MCP-compatible client) via two MCP servers. The servers are configured in [`.mcp.json`](.mcp.json) and start automatically when Claude Code opens this project.

> **Note:** The absolute paths in `.mcp.json` point to the original developer's machine. Update them to match your own environment before use — both the Python executable path and the script paths need to reflect your local clone location.

### ragserver — Hybrid Search (FAISS + BM25)

**File:** `rag_server.py`
**Search type:** Ensemble retriever — BM25 keyword search (weight 0.5) + FAISS semantic vector search (weight 0.5), returns top 3 chunks per retriever.

Best for: **semantic / conceptual queries** where exact keywords may not appear in the source text.

**Tool:** `search_documents(query: str)`

**Example usage in Claude Code:**
```
use ragserver mcp — how does Vincit approach test automation?
```

**Logs:** `rag_server_debug.log` in the project root.

---

### ragbm25 — BM25 Keyword Search Only

**File:** `rag_bm25.py`
**Search type:** Pure BM25 keyword retriever, returns top 5 chunks.

Best for: **exact keyword / terminology lookups** where you know a specific word or phrase appears in the source (e.g. tool names, company names, section headings).

**Tool:** `search_documents(query: str)`

**Example usage in Claude Code:**
```
use ragbm25 mcp — what does the Siili handbook say about hallucinations?
```

**Logs:** `rag_bm25_debug.log` in the project root.

---

### When to use which

| Situation | Recommended server |
|---|---|
| Broad topic / conceptual question | `ragserver` (hybrid) |
| Exact term, tool name, or phrase | `ragbm25` |
| Want more result chunks (up to 5) | `ragbm25` |
| Query mixes keywords and meaning | `ragserver` |

### Prerequisites for MCP servers

The MCP servers require the FAISS index and BM25 index to exist. If you haven't already, run:

```bash
python build_db.py
```

This generates `faiss_index/` and `bm25_retriever.pkl` which both servers load on startup.
