import os
import sys
import pickle
import json
from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BM25_PATH = os.path.join(PROJECT_DIR, "bm25_retriever.pkl")
FAISS_PATH = os.path.join(PROJECT_DIR, "faiss_index")


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
            text_data.append(Document(
                page_content=full_slide_text,
                metadata={"source": file_path, "slide": i + 1},
            ))
    return text_data


def load_pdf(file_path):
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_docx(file_path):
    print(f"Loading {file_path}...")
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    return [Document(page_content=full_text, metadata={"source": file_path})]


def load_json(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    full_text = json.dumps(data, ensure_ascii=False, indent=2)
    return [Document(page_content=full_text, metadata={"source": file_path})]


def build_database(file_paths):
    """Build BM25 and FAISS indexes from a list of PDF/PPTX file paths.

    Returns the number of text chunks indexed.
    Raises ValueError if no documents could be loaded.
    """
    all_docs = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping.")
            continue
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pptx":
            docs = load_pptx(file_path)
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} slides from {file_path}.")
        elif ext == ".pdf":
            docs = load_pdf(file_path)
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {file_path}.")
        elif ext in (".doc", ".docx"):
            docs = load_docx(file_path)
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} section(s) from {file_path}.")
        elif ext == ".json":
            docs = load_json(file_path)
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} section(s) from {file_path}.")
        else:
            print(f"Warning: Unsupported file type '{ext}' for {file_path}, skipping.")

    if not all_docs:
        raise ValueError("No documents could be loaded from the provided file paths.")

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)
    print(f"Created {len(splits)} text chunks total.")

    # BM25 Index
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 index saved to {BM25_PATH}")

    # FAISS Vector Index
    print("Building vector index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"FAISS index saved to {FAISS_PATH}")

    print("Ingestion complete.")
    return len(splits)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        # Scan project directory for supported documents (excluding uploads/)
        paths = [
            os.path.join(PROJECT_DIR, f)
            for f in os.listdir(PROJECT_DIR)
            if os.path.isfile(os.path.join(PROJECT_DIR, f))
            and os.path.splitext(f)[1].lower() in (".pdf", ".pptx", ".doc", ".docx", ".json")
        ]
        if not paths:
            print("No supported files found (.pdf, .pptx, .doc, .docx, .json). Pass file paths as arguments.")
            sys.exit(1)
        print(f"Found {len(paths)} document(s): {', '.join(os.path.basename(p) for p in paths)}")
    build_database(paths)
