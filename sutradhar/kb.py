"""The RAG system, packaged so the agent can use it as a TOOL.

Same ingredients as Days 4-6: chunk -> embed -> store in ChromaDB -> similarity search.
Kept deliberately small; the point of Day 9 is the AGENT around it, not the RAG itself.
"""
import re
from pathlib import Path

from . import config


def chunk_document(text: str, max_chars: int = 700):
    """Split on blank lines (paragraphs), then pack paragraphs up to ~max_chars.
    Section headings like '2.1 Casual Leave' stay attached to the paragraph that follows."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _embed(texts):
    from openai import OpenAI  # lazy: keeps pure-logic tests importable without the package

    client = OpenAI()
    out = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in out.data]


class KnowledgeBase:
    def __init__(self, path=config.CHROMA_DIR, name="enterprise_kb"):
        import chromadb

        self.client = chromadb.PersistentClient(path=path)
        self.name = name

    def build(self, kb_dir=config.KB_DIR):
        """(Re)create the collection from every .txt in knowledge_base/."""
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        col = self.client.create_collection(self.name, metadata={"hnsw:space": "cosine"})
        ids, docs, metas = [], [], []
        for f in sorted(Path(kb_dir).glob("*.txt")):
            for i, ch in enumerate(chunk_document(f.read_text(encoding="utf-8"))):
                ids.append(f"{f.stem}-{i}")
                docs.append(ch)
                metas.append({"source": f.name, "chunk_index": i})
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=_embed(docs))
        return len(docs)

    def search(self, query: str, k: int = config.TOP_K):
        col = self.client.get_collection(self.name)
        res = col.query(query_embeddings=_embed([query]), n_results=k)
        return [
            {"text": d, "source": m["source"], "distance": round(dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


def format_results(results) -> str:
    """Turn raw hits into the text the LLM will read as the tool's return value."""
    if not results:
        return "NO_RESULTS"
    best = results[0]["distance"]
    verdict = ("STRONG" if best <= config.STRONG_MATCH else "WEAK" if best <= config.WEAK_MATCH else "NO_RELEVANT_MATCH")
    lines = [f"MATCH_QUALITY: {verdict} (best distance {best})"]
    for r in results:
        lines.append(f"[source: {r['source']} | distance {r['distance']}]\n{r['text']}")
    return "\n\n".join(lines)
