import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SOURCE_PATH = Path(os.getenv(
    "RAG_SOURCE_PATH",
    "/home/junaid/Agentic_Rag/dataset/income-tax-act-2023-english (1) (1).md"
)).expanduser()
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", Path(__file__).parent / "chroma_store")).expanduser()
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "legal_text")
