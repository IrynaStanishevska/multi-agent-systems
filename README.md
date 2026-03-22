# Research Multi-Agent with RAG

A research agent with a custom ReAct loop and a local RAG pipeline.

## Features

- Web search and page reading
- Local knowledge base search
- Hybrid retrieval: semantic search + BM25
- Cross-encoder reranking
- Markdown report generation

## Stack

- LangChain
- FAISS
- OpenAI Embeddings
- Sentence Transformers
- DuckDuckGo Search

## Usage

```bash
python ingest.py
python main.py
