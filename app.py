import streamlit as st
from config import GROQ_API_KEY, SOURCE_PATH, CHROMA_DIR, CHROMA_COLLECTION
from llm_module import init_llm, SYSTEM_PROMPT
from embeddings_module import init_embeddings
from loader_module import load_documents, split_documents
from vectorstore_module import init_vectorstore, init_retriever
from tools_module import create_retriever_tool
from agent_module import build_graph
from langchain_core.messages import HumanMessage, AIMessage
import os

# Page configuration
st.set_page_config(
    page_title="RAG Agent - Income Tax Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    """Initialize the RAG agent with caching"""
    with st.spinner("🔄 Initializing RAG Agent..."):
        try:
            llm_instance = init_llm(GROQ_API_KEY)
            embeddings = init_embeddings()
            docs = load_documents(SOURCE_PATH)
            pages_split = split_documents(docs)
            vectorstore = init_vectorstore(pages_split, embeddings, CHROMA_DIR, CHROMA_COLLECTION)
            retriever = init_retriever(vectorstore)
            retriever_tool = create_retriever_tool(retriever)
            tools_dict = {retriever_tool.name: retriever_tool}
            rag_agent = build_graph(llm_instance, tools_dict, SYSTEM_PROMPT)
            return rag_agent, True, None
        except Exception as e:
            return None, False, str(e)

def main():
    # Header
    st.title("📚 Agentic RAG System")
    st.markdown("### Income Tax Act 2023 - AI Assistant")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This intelligent assistant uses:
        - **LangChain** & **LangGraph** for agent orchestration
        - **Groq LLM** for response generation
        - **ChromaDB** for vector search
        - **Retrieval-Augmented Generation** for accurate answers
        """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # Display configuration info
        if os.path.exists(SOURCE_PATH):
            st.success("✅ Document loaded")
        else:
            st.error("❌ Document not found")
        
        if GROQ_API_KEY:
            st.success("✅ API Key configured")
        else:
            st.error("❌ API Key missing")
        
        st.markdown("---")
        st.header("💡 Tips")
        st.markdown("""
        - Ask specific questions about tax laws
        - Request explanations of sections
        - Compare different tax provisions
        - Get examples and clarifications
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        agent, success, error = initialize_agent()
        if success:
            st.session_state.agent = agent
            st.session_state.agent_ready = True
        else:
            st.session_state.agent_ready = False
            st.error(f"❌ Failed to initialize agent: {error}")
            st.stop()

    # Check if agent is ready
    if not st.session_state.agent_ready:
        st.error("⚠️ Agent not initialized. Please check your configuration.")
        st.stop()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the Income Tax Act..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Invoke the agent
                    messages = [HumanMessage(content=prompt)]
                    result = st.session_state.agent.invoke({"messages": messages})
                    
                    # Extract the response
                    response = result['messages'][-1].content
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with LangChain, LangGraph & Streamlit | "
        "Powered by Groq"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
