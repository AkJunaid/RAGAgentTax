from pathlib import Path
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

def init_vectorstore(pages_split, embeddings, persist_dir: Path, collection_name: str):
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name
    )
    print("Created ChromaDB vector store!")
    return vectorstore

def init_retriever(vectorstore):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
