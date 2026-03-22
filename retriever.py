import pickle

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder

from config import Settings

settings = Settings()


def get_retriever():
    # Load FAISS
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vectorstore = FAISS.load_local(
        settings.index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )

    # Load BM25
    with open(f"{settings.index_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.retrieval_top_k

    # Ensemble
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    # Cross-encoder reranker
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(query, docs):
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: settings.rerank_top_n]]

    def retrieve(query: str):
        docs = ensemble.invoke(query)
        return rerank(query, docs)

    return retrieve
