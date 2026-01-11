from config import GROQ_API_KEY, SOURCE_PATH, CHROMA_DIR, CHROMA_COLLECTION
from llm_module import init_llm, SYSTEM_PROMPT
from embeddings_module import init_embeddings
from loader_module import load_documents, split_documents
from vectorstore_module import init_vectorstore, init_retriever
from tools_module import create_retriever_tool
from agent_module import build_graph
from langchain_core.messages import HumanMessage

def run_agent(rag_agent):
    print("\n=== RAG AGENT ===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

def main():
    llm_instance = init_llm(GROQ_API_KEY)
    embeddings = init_embeddings()
    docs = load_documents(SOURCE_PATH)
    pages_split = split_documents(docs)
    vectorstore = init_vectorstore(pages_split, embeddings, CHROMA_DIR, CHROMA_COLLECTION)
    retriever = init_retriever(vectorstore)
    retriever_tool = create_retriever_tool(retriever)
    tools_dict = {retriever_tool.name: retriever_tool}
    rag_agent = build_graph(llm_instance, tools_dict, SYSTEM_PROMPT)
    run_agent(rag_agent)

if __name__ == "__main__":
    main()
