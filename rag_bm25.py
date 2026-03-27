import asyncio
import os
import pickle
import sys
import warnings
import logging

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_bm25_debug.log")
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.debug("rag_bm25.py starting")

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from mcp.server.fastmcp import FastMCP

BM25_PATH = os.path.join(PROJECT_DIR, "bm25_retriever.pkl")

mcp = FastMCP("RagBM25")

logging.debug("loading retriever from pickle...")
with open(BM25_PATH, "rb") as f:
    _retriever = pickle.load(f)
    _retriever.k = 5
logging.debug("retriever loaded")


@mcp.tool()
async def search_documents(query: str) -> str:
    """Search the RAG document database (BM25 keyword search) and return relevant chunks.
    Use this whenever the user asks about topics covered in the indexed documents,
    such as Vincit, Siili, AI-powered development, testing, sales materials, HIFK or related topics.
    """
    logging.debug(f"search_documents called: {query!r}")
    docs = await asyncio.to_thread(_retriever.invoke, query)
    logging.debug(f"invoke() returned {len(docs)} docs")

    if not docs:
        return "No relevant documents found for this query."

    results = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        slide = doc.metadata.get("slide")
        page = doc.metadata.get("page")
        location = f"slide {slide}" if slide else f"page {page}" if page else f"chunk {i}"
        results.append(f"[{i}] {source} ({location}):\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    logging.debug("calling mcp.run()")
    mcp.run()
    logging.debug("mcp.run() returned")
