import os
import pickle
from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader

PPTX_FILE = "Testing & QA sales materials.pptx"
PDF_FILE = "SIILI-Handbook_for_AI-Powered_Developers-pages.pdf"
BM25_PATH = "bm25_retriever.pkl"
FAISS_PATH = "faiss_index"

def load_pptx(file_path):
    print(f"Loading {file_path}...")
    prs = Presentation(file_path)
    text_data = []
    for i, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text.append(shape.text)
        
        full_slide_text = "\n".join(slide_text).strip()
        if full_slide_text:
            text_data.append(Document(page_content=full_slide_text, metadata={"source": file_path, "slide": i + 1}))
    return text_data

def load_pdf(file_path):
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    return loader.load()

def build_database():
    all_docs = []
    
    # 1. Load Data
    if os.path.exists(PPTX_FILE):
        docs = load_pptx(PPTX_FILE)
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} slides from {PPTX_FILE}.")
    else:
        print(f"Warning: {PPTX_FILE} not found.")

    if os.path.exists(PDF_FILE):
        docs = load_pdf(PDF_FILE)
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from {PDF_FILE}.")
    else:
        print(f"Warning: {PDF_FILE} not found.")

    if not all_docs:
        print("Error: No documents found to index.")
        return

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)
    print(f"Created {len(splits)} text chunks total.")

    # 3. BM25 Index
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5  # Retrieve top 5
    
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 index saved to {BM25_PATH}")

    # 4. Vector Index (FAISS)
    print("Building Vector index (this may take a moment to download embeddings)...")
    # Using a standard, small, efficient model suitable for CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"FAISS index saved to {FAISS_PATH}")

    print("Ingestion complete.")

if __name__ == "__main__":
    build_database()
