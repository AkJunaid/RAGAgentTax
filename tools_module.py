from langchain_core.tools import tool

def create_retriever_tool(retriever):
    @tool
    def retriever_tool(query: str) -> str:
        """Search and return information from legal tax documents."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found."
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    return retriever_tool
