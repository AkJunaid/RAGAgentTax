# Agentic RAG System

An intelligent Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph for querying legal documents using AI agents.

## Overview

This project implements an agentic RAG system that uses a conversational AI agent to answer questions about legal documents (specifically the Income Tax Act 2023). The system leverages vector embeddings for efficient document retrieval and an LLM (via Groq) for generating contextual answers.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for LLM, embeddings, vector store, tools, and agent logic
- **LangGraph Agent**: Intelligent agent that can decide when to retrieve documents and when to respond directly
- **Vector Search**: Uses ChromaDB for efficient semantic search over document chunks
- **Conversational Interface**: Interactive CLI for querying the system
- **Document Processing**: Automatic document loading and chunking for optimal retrieval

## Project Structure

```
Agentic_Rag/
├── main.py                 # Entry point and orchestration
├── config.py              # Configuration and environment variables
├── llm_module.py          # LLM initialization (Groq)
├── embeddings_module.py   # Embedding model setup
├── loader_module.py       # Document loading and splitting
├── vectorstore_module.py  # ChromaDB vector store setup
├── tools_module.py        # Retriever tool creation
├── agent_module.py        # LangGraph agent logic
├── dataset/               # Source documents
│   └── income-tax-act-2023-english (1) (1).md
└── chroma_store/          # Vector database persistence
```

## Prerequisites

- Python 3.8+
- Groq API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Agentic_Rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install langchain langchain-groq langchain-community langchain-chroma langgraph python-dotenv
```

4. Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
RAG_SOURCE_PATH=/path/to/your/document.md
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION=legal_text
```

## Usage

Run the agent:
```bash
python main.py
```

The system will start an interactive session where you can ask questions:
```
=== RAG AGENT ===

What is your question: What are the tax rates for individuals?

=== ANSWER ===
[Agent's response based on retrieved documents]
```

To exit, type `exit` or `quit`.

## How It Works

1. **Document Loading**: Documents are loaded from the specified source path and split into manageable chunks
2. **Embedding**: Text chunks are converted to vector embeddings using HuggingFace models
3. **Vector Store**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Agent Loop**: 
   - User asks a question
   - Agent decides if it needs to retrieve documents
   - If needed, retrieves relevant chunks using the retriever tool
   - LLM generates a response based on retrieved context
   - Response is displayed to the user

## Components

### LLM Module
Initializes the Groq LLM with the configured API key and system prompt.

### Embeddings Module
Sets up the embedding model for converting text to vectors.

### Loader Module
Handles document loading and splitting into chunks for processing.

### Vector Store Module
Manages ChromaDB for storing and retrieving document embeddings.

### Tools Module
Creates the retriever tool that the agent can use to search documents.

### Agent Module
Implements the LangGraph state machine for agent decision-making and tool execution.

## Configuration

Key configuration options in `config.py`:
- `GROQ_API_KEY`: Your Groq API key
- `SOURCE_PATH`: Path to source documents
- `CHROMA_DIR`: Directory for ChromaDB persistence
- `CHROMA_COLLECTION`: Collection name in ChromaDB

