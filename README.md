# Local RAG Assistant

## Problem
Built a Retrieval-Augmented Generation assistant that answers questions from custom documents using grounded context.

## Architecture
User Query
→ Embeddings
→ Vector Search
→ Retrieved Context
→ Phi3 LLM
→ Guardrailed Response

## Tech Stack
- Python
- Ollama
- Phi3-mini
- SentenceTransformers
- ChromaDB

## Features
- Document chunking
- Vector retrieval
- Grounded responses
- Hallucination mitigation
- Output guardrails

## Known Limitations
- Basic chunking
- Simple confidence checks

## Future Improvements
- Persistent vector DB
- Multi-document support
- Streamlit UI

## Sample Questions

Examples:

- What is cloud computing?
- What are the service models?
- Is Oracle Cloud listed as provider?
