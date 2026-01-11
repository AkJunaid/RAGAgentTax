from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(source_path: Path):
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    loader = PyPDFLoader(str(source_path)) if source_path.suffix.lower() == ".pdf" else TextLoader(str(source_path), encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from: {source_path}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)
