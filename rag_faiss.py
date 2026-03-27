import asyncio
import os
import sys
import warnings
import logging

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_faiss_debug.log")
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.debug("rag_faiss.py starting")

warnings.filterwarnings("ignore")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("FAISS_OPT_LEVEL", "AVX2")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

logging.debug("importing langchain / torch / sentence-transformers ...")
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
logging.debug("imports done")

from mcp.server.fastmcp import FastMCP

FAISS_PATH = os.path.join(PROJECT_DIR, "faiss_index")

logging.debug("loading FAISS retriever...")
_retriever = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logging.debug("embeddings loaded")
    vectorstore = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    logging.debug("FAISS retriever loaded")
except Exception:
    logging.exception("FAILED to load FAISS retriever")

mcp = FastMCP("RagFAISS")


@mcp.tool()
async def search_documents(query: str) -> str:
    """Search the RAG document database (FAISS semantic search) and return relevant chunks.
    Use this whenever the user asks about topics covered in the indexed documents,
    such as Vincit, Siili, AI-powered development, testing, sales materials, HIFK,  or related topics.
    """
    logging.debug(f"search_documents called: {query!r}")
    if _retriever is None:
        return "Error: FAISS retriever failed to load. Check rag_faiss_debug.log for details."
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
