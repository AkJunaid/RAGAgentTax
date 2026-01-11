from langchain_community.embeddings import HuggingFaceEmbeddings

def init_embeddings():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded!")
    return embeddings
