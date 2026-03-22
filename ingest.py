import os
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import Settings

settings = Settings()


def load_documents():
    docs = []
    data_path = Path(settings.data_dir)

    for file in data_path.glob("*"):
        if file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix in [".txt", ".md"]:
            docs.extend(TextLoader(str(file)).load())

    return docs


def ingest():
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} docs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    print("🔨 Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(settings.index_dir, exist_ok=True)
    vectorstore.save_local(settings.index_dir)

    # Save chunks for BM25
    with open(f"{settings.index_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Ingestion complete")


if __name__ == "__main__":
    ingest()
