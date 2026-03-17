import os
import pickle
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Fix import for EnsembleRetriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Try importing LLM integrations
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    from langchain_openai import ChatOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_openai import ChatOpenAI as GitHubChatModel

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False

BM25_PATH = "bm25_retriever.pkl"
FAISS_PATH = "faiss_index"
KEYWORDS = [
    "vincit",
    "cost",
    "automate",
    "automation",
    "pricing",
    "quote",
    "estimate",
    "roi",
    "return on investment",
    "savings",
    "efficiency",
    "benefits",
    "advantages",
    "value proposition",
    "competitive edge",
    "sales",
    "proposal",
    "customer",
    "client",
    "industry",
    "case study",
    "success story",
    "implementation",
    "deployment",
    "integration",
    "support",
    "training",
    "onboarding",
    "features",
    "capabilities",
    "functionality",
    "performance",
    "scalability",
    "security",
    "compliance",
    "customization",
    "flexibility",
    "ai",
    "llm",
    "generative ai",
    "rag",
    "retrieval-augmented generation",
    "prompt engineering",
    "coding assistant",
    "copilot",
    "cursor",
    "aider",
    "devin",
    "agent",
    "software engineering agent",
    "framework",
    "tool",
    "plugin",
    "extension",
    "api",
    "sdk",
    "documentation",
    "whitepaper",
    "test" 
      
]


def load_resources():
    print("Loading resources...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load FAISS
    try:
        vectorstore = FAISS.load_local(
            FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        sys.exit(1)

    # Load BM25
    try:
        with open(BM25_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
            bm25_retriever.k = 2
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        sys.exit(1)

    # Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def get_llm():
    # Prioritize GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")

    if api_key:
        api_key = api_key.strip()

    # Try Gemini models
    if api_key and HAS_GOOGLE:
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-pro",
        ]
        for model_name in models_to_try:
            try:
                print(f"Testing Google Gemini ({model_name})...")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    convert_system_message_to_human=True,
                )
                # Test with a simple call to validate the API key and quota
                llm.invoke("test")
                print(f"✓ Using Google Gemini ({model_name})")
                return llm
            except Exception as e:
                print(f"✗ Gemini ({model_name}) failed: {e}")
                continue

    # Try GitHub Models as fallback
    if github_token and HAS_GITHUB:
        models_to_try = ["gpt-4o", "gpt-4o-mini"]
        for model_name in models_to_try:
            try:
                print(f"Testing GitHub Model ({model_name})...")
                llm = GitHubChatModel(
                    model=model_name,
                    api_key=github_token,
                    base_url="https://models.inference.ai.azure.com",
                )
                # Test with a simple call to validate the token
                llm.invoke("test")
                print(f"✓ Using GitHub Model ({model_name})")
                return llm
            except Exception as e:
                print(f"✗ GitHub Model ({model_name}) failed: {e}")
                continue

    # Try OpenAI as fallback
    if openai_key and HAS_OPENAI:
        try:
            print("Testing OpenAI GPT...")
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            # Test with a simple call
            llm.invoke("test")
            print("✓ Using OpenAI GPT")
            return llm
        except Exception as e:
            print(f"✗ OpenAI failed: {e}")

    return None


def get_fallback_llm():
    """Get fallback LLM (GitHub or OpenAI) when primary fails"""
    github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Try GitHub Models first
    if github_token and HAS_GITHUB:
        models_to_try = ["gpt-4o", "gpt-4o-mini"]
        for model_name in models_to_try:
            try:
                print(f"[Fallback] Trying GitHub Model ({model_name})...")
                llm = GitHubChatModel(
                    model=model_name,
                    api_key=github_token,
                    base_url="https://models.inference.ai.azure.com",
                )
                llm.invoke("test")
                print(f"✓ Switched to GitHub Model ({model_name})")
                return llm
            except Exception as e:
                print(f"✗ GitHub Model ({model_name}) failed: {e}")
                continue

    # Try OpenAI as last resort
    if openai_key and HAS_OPENAI:
        try:
            print("[Fallback] Trying OpenAI...")
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            llm.invoke("test")
            print("✓ Switched to OpenAI GPT")
            return llm
        except Exception as e:
            print(f"✗ OpenAI failed: {e}")

    return None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    if not os.path.exists(FAISS_PATH) or not os.path.exists(BM25_PATH):
        print("Indexes not found. Run 'build_db.py' first.")
        return

    retriever = load_resources()
    llm = get_llm()

    print("\n" + "=" * 50)
    print("RAG Chat Bot Initialized")
    print(f"Keywords to trigger: {', '.join(KEYWORDS[:5])}... ({len(KEYWORDS)} total)")
    if llm:
        print("LLM Mode: ENABLED")
    else:
        print("LLM Mode: DISABLED (Retrieval Only)")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break

        if query.lower() in ["exit", "quit"]:
            break

        if not query:
            continue

        # Keyword Check
        found_keywords = [k for k in KEYWORDS if k in query.lower()]
        if found_keywords:
            print(f"[Bot]: I noticed you're asking about: {', '.join(found_keywords)}")

        # Retrieval
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            print(f"[Bot]: Retrieval Error: {e}")
            continue

        context = format_docs(docs)

        if not docs:
            # No docs found - use general LLM knowledge if available
            if llm:
                print("[Bot]: No documents found, using general knowledge...")
                try:
                    general_response = llm.invoke(query)
                    print(f"[Bot]: {general_response.content}")
                except Exception as e:
                    print(f"[Bot]: Error: {e}")
            else:
                print(
                    "[Bot]: I couldn't find any relevant information in the documents."
                )
            continue

        if llm:
            # RAG Generation with fallback
            try:
                template = """Answer the question based on the following context. If the context doesn't contain relevant information, say so and provide a general answer instead.

Context:
{context}

Question: {question}

Answer:"""
                prompt = ChatPromptTemplate.from_template(template)
                chain = (
                    {"context": lambda x: context, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                response = chain.invoke(query)

                # Check if response indicates no relevant context
                no_context_phrases = [
                    "does not contain information",
                    "no information",
                    "not mentioned",
                    "not provided in the context",
                    "context does not",
                ]

                # Only fallback to general knowledge if NO keywords were found
                if (
                    any(phrase in response.lower() for phrase in no_context_phrases)
                    and not found_keywords
                ):
                    # Fallback to general knowledge
                    print("[Bot]: Context not relevant, using general knowledge...")
                    general_response = llm.invoke(query)
                    print(f"[Bot]: {general_response.content}")
                else:
                    print(f"[Bot]: {response}")
            except Exception as e:
                error_str = str(e)
                # Check if quota exhausted
                if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"[Bot]: Quota exhausted. Trying fallback LLM...")
                    fallback_llm = get_fallback_llm()
                    if fallback_llm:
                        llm = fallback_llm  # Switch to fallback
                        try:
                            # Retry with fallback LLM
                            prompt = ChatPromptTemplate.from_template(template)
                            chain = (
                                {
                                    "context": lambda x: context,
                                    "question": RunnablePassthrough(),
                                }
                                | prompt
                                | llm
                                | StrOutputParser()
                            )
                            response = chain.invoke(query)
                            print(f"[Bot]: {response}")
                        except Exception as retry_error:
                            print(f"[Bot]: Fallback LLM also failed: {retry_error}")
                            print("[Bot]: Switching to Retrieval View.")
                            print("-" * 20)
                            for i, doc in enumerate(docs):
                                source_info = doc.metadata.get('slide') or doc.metadata.get('page')
                                source_type = "Slide" if "slide" in doc.metadata else "Page" if "page" in doc.metadata else "Source"
                                print(
                                    f"--- {source_type} {source_info or i+1} (from {os.path.basename(doc.metadata.get('source', 'unknown'))}) ---"
                                )
                                print(doc.page_content.strip())
                                print()
                            print("-" * 20)
                    else:
                        print(
                            "[Bot]: No fallback LLM available. Switching to Retrieval View."
                        )
                        print("-" * 20)
                        for i, doc in enumerate(docs):
                            source_info = doc.metadata.get('slide') or doc.metadata.get('page')
                            source_type = "Slide" if "slide" in doc.metadata else "Page" if "page" in doc.metadata else "Source"
                            print(
                                f"--- {source_type} {source_info or i+1} (from {os.path.basename(doc.metadata.get('source', 'unknown'))}) ---"
                            )
                            print(doc.page_content.strip())
                            print()
                        print("-" * 20)
                else:
                    print(f"[Bot]: Error generating response with LLM: {e}")
                    print("[Bot]: Switching to Retrieval View due to error.")
                    # Fallback to printing docs
                    print("-" * 20)
                    for i, doc in enumerate(docs):
                        source_info = doc.metadata.get('slide') or doc.metadata.get('page')
                        source_type = "Slide" if "slide" in doc.metadata else "Page" if "page" in doc.metadata else "Source"
                        print(
                            f"--- {source_type} {source_info or i+1} (from {os.path.basename(doc.metadata.get('source', 'unknown'))}) ---"
                        )
                        print(doc.page_content.strip())
                        print()
                    print("-" * 20)
        else:
            # Retrieval Only
            print(
                f"[Bot]: Here is the relevant information I found from the RAG database:"
            )
            print("-" * 20)
            for i, doc in enumerate(docs):
                source_info = doc.metadata.get('slide') or doc.metadata.get('page')
                source_type = "Slide" if "slide" in doc.metadata else "Page" if "page" in doc.metadata else "Source"
                print(
                    f"--- {source_type} {source_info or i+1} (from {os.path.basename(doc.metadata.get('source', 'unknown'))}) ---"
                )
                print(doc.page_content.strip())
                print()
            print("-" * 20)


if __name__ == "__main__":
    main()
