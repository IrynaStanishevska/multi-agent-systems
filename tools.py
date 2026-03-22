from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS

from config import Settings
from retriever import get_retriever

settings = Settings()
retriever = get_retriever()


def _truncate(text: str, max_length: int) -> str:
    return text[:max_length] + ("\n\n[Truncated]" if len(text) > max_length else "")


def web_search(query: str) -> list[dict]:
    try:
        results = list(DDGS().text(query, max_results=settings.max_search_results))
        return [
            {
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


def read_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Invalid URL"

        html = trafilatura.fetch_url(url)
        text = trafilatura.extract(html)

        return _truncate(text or "", settings.max_url_content_length)
    except Exception as e:
        return str(e)


def write_report(filename: str, content: str) -> str:
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(exist_ok=True)

    path = output_dir / filename
    path.write_text(content, encoding="utf-8")

    return f"Saved to {path}"


def knowledge_search(query: str) -> str:
    docs = retriever(query)

    if not docs:
        return "No results found"

    result = []
    for i, doc in enumerate(docs):
        result.append(f"[Doc {i+1}]\n{doc.page_content[:500]}")

    return "\n\n".join(result)
