import pickle
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from config import Settings

settings = Settings()


def get_retriever():
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )

    vectorstore = FAISS.load_local(
        settings.index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )

    with open(f"{settings.index_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.retrieval_top_k

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    def hybrid_retrieve(query: str):
        semantic_docs = semantic_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)

        combined_scores = defaultdict(float)
        doc_map = {}

        for rank, doc in enumerate(semantic_docs, start=1):
            key = (
                doc.page_content,
                tuple(sorted(doc.metadata.items())) if doc.metadata else ()
            )
            combined_scores[key] += 1.0 / rank
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs, start=1):
            key = (
                doc.page_content,
                tuple(sorted(doc.metadata.items())) if doc.metadata else ()
            )
            combined_scores[key] += 1.0 / rank
            doc_map[key] = doc

        ranked_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc_map[key] for key, _ in ranked_docs[: settings.retrieval_top_k]]

    def rerank(query: str, docs):
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: settings.rerank_top_n]]

    def retrieve(query: str):
        docs = hybrid_retrieve(query)
        return rerank(query, docs)

    return retrieve