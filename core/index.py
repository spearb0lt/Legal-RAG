"""
ChromaDB-backed persistent index over Chunks.

Index is local-only — no API needed beyond the embedder. Chunk metadata is stored
alongside vectors so retrieval returns full citation context in one round-trip.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings

from . import config
from .citation import Chunk
from .embeddings import embed_documents, embed_query


_client: chromadb.PersistentClient | None = None
_collection = None


def _client_singleton() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def get_collection():
    """Return (and lazily create) the legal-corpus collection."""
    global _collection
    if _collection is not None:
        return _collection
    client = _client_singleton()
    _collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def reset_index() -> None:
    """Drop and recreate the collection. Destructive."""
    global _collection
    client = _client_singleton()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    _collection = None
    get_collection()


def _flatten_metadata(c: Chunk) -> dict:
    md = {
        "doc_id": c.doc_id,
        "doc_title": c.doc_title,
        "doc_type": c.doc_type,
        "section_marker": c.section_marker,
        "paragraph_index": c.paragraph_index,
        "jurisdiction": c.jurisdiction,
        "source_url": c.source_url,
        "source_note": c.source_note,
    }
    if c.extra:
        md["extra_json"] = json.dumps(c.extra, ensure_ascii=False)
        # Promote to top-level so ChromaDB can filter on them directly
        if "source_task" in c.extra:
            md["source_task"] = c.extra["source_task"]
        if "outcome" in c.extra and c.extra["outcome"] != -1:
            md["outcome"] = int(c.extra["outcome"])
    return md


def _reconstruct_extra(md: dict) -> dict:
    """Rebuild extra dict from stored metadata."""
    extra: dict = {}
    if "extra_json" in md:
        try:
            extra = json.loads(md["extra_json"])
        except (json.JSONDecodeError, TypeError):
            pass
    # Ensure promoted fields are present (handles old chunks missing extra_json)
    if "source_task" in md:
        extra["source_task"] = md["source_task"]
    if "outcome" in md:
        extra["outcome"] = int(md["outcome"])
    return extra


def add_chunks(chunks: list[Chunk], batch: int = 256) -> int:
    """Embed and add chunks. Returns count added."""
    if not chunks:
        return 0
    from tqdm import tqdm
    coll = get_collection()
    total = 0
    n_batches = (len(chunks) + batch - 1) // batch
    for i in tqdm(range(0, len(chunks), batch), total=n_batches, desc="embedding+indexing", unit="batch"):
        sub = chunks[i : i + batch]
        ids = [c.chunk_id for c in sub]
        docs = [c.text for c in sub]
        metas = [_flatten_metadata(c) for c in sub]
        vecs = embed_documents(docs)
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
        total += len(sub)
    return total


def vector_search(
    query: str,
    top_k: int = 10,
    where_filter: dict | None = None,
) -> list[Chunk]:
    """Pure vector retrieval (no BM25). Returns chunks ordered by similarity.

    where_filter: optional ChromaDB `where` clause, e.g. {"source_task": "cjpe"}.
    """
    coll = get_collection()
    qvec = embed_query(query)
    query_kwargs: dict = dict(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        query_kwargs["where"] = where_filter
    res = coll.query(**query_kwargs)
    out: list[Chunk] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for cid, text, md in zip(ids, docs, metas):
        out.append(
            Chunk(
                chunk_id=cid,
                doc_id=md.get("doc_id", ""),
                doc_title=md.get("doc_title", ""),
                doc_type=md.get("doc_type", ""),
                section_marker=md.get("section_marker", "") or "",
                paragraph_index=int(md.get("paragraph_index", 0)),
                text=text,
                jurisdiction=md.get("jurisdiction", "IN"),
                source_url=md.get("source_url", "") or "",
                source_note=md.get("source_note", "") or "",
                extra=_reconstruct_extra(md),
            )
        )
    return out


def all_chunks() -> list[Chunk]:
    """Return every chunk in the collection. Used to (re)build BM25 indexes."""
    coll = get_collection()
    res = coll.get(include=["documents", "metadatas"])
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    out: list[Chunk] = []
    for cid, text, md in zip(ids, docs, metas):
        out.append(
            Chunk(
                chunk_id=cid,
                doc_id=md.get("doc_id", ""),
                doc_title=md.get("doc_title", ""),
                doc_type=md.get("doc_type", ""),
                section_marker=md.get("section_marker", "") or "",
                paragraph_index=int(md.get("paragraph_index", 0)),
                text=text,
                jurisdiction=md.get("jurisdiction", "IN"),
                source_url=md.get("source_url", "") or "",
                source_note=md.get("source_note", "") or "",
                extra=_reconstruct_extra(md),
            )
        )
    return out


def count() -> int:
    return get_collection().count()
