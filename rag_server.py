import asyncio
import os
import pickle
import sys
import warnings
import logging

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_server_debug.log")
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.debug("rag_server.py starting")

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from mcp.server.fastmcp import FastMCP

BM25_PATH = os.path.join(PROJECT_DIR, "bm25_retriever.pkl")
FAISS_PATH = os.path.join(PROJECT_DIR, "faiss_index")

mcp = FastMCP("RagTest")
logging.debug("FastMCP initialized, retriever will load on first call")

_retriever = None


def _load_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    logging.debug("loading retriever (FAISS + BM25 ensemble)...")
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain.retrievers.ensemble import EnsembleRetriever
        except ImportError:
            from langchain_classic.retrievers.ensemble import EnsembleRetriever

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
        bm25_retriever.k = 3
    _retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    logging.debug("retriever loaded")
    return _retriever


@mcp.tool()
async def search_documents(query: str) -> str:
    """Search the RAG document database (FAISS + BM25 ensemble) and return relevant chunks.
    Use this whenever the user asks about topics covered in the indexed documents,
    such as Vincit, Siili, AI-powered development, testing, sales materials, or related topics.
    """
    logging.debug(f"search_documents called: {query!r}")
    retriever = await asyncio.to_thread(_load_retriever)
    docs = await asyncio.to_thread(retriever.invoke, query)
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
