"""
Hybrid retriever: vector (Gemini embeddings + Chroma) + BM25 (rank_bm25).

Scores are combined with Reciprocal Rank Fusion (Cormack et al., 2009).
RRF is parameter-light and robust to score-scale mismatch between two retrievers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from rank_bm25 import BM25Okapi

from . import config, index
from .citation import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RetrievalHit:
    chunk: Chunk
    vector_rank: int | None
    bm25_rank: int | None
    score: float


class HybridRetriever:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def _refresh_bm25(self) -> None:
        chunks = index.all_chunks()
        self._chunks = chunks
        if not chunks:
            self._bm25 = None
            return
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_type_filter: list[str] | None = None,
    ) -> list[RetrievalHit]:
        top_k = top_k or config.RETRIEVAL_TOP_K
        if not query.strip():
            return []

        if self._bm25 is None or not self._chunks:
            self._refresh_bm25()

        vector_hits = index.vector_search(query, top_k=top_k * 2)
        vector_ids = {c.chunk_id: rank for rank, c in enumerate(vector_hits)}

        bm25_ids: dict[str, int] = {}
        bm25_hits: list[Chunk] = []
        if self._bm25 and self._chunks:
            scores = self._bm25.get_scores(_tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[: top_k * 2]
            bm25_hits = [self._chunks[i] for i in order]
            bm25_ids = {c.chunk_id: rank for rank, c in enumerate(bm25_hits)}

        candidates: dict[str, Chunk] = {}
        for c in vector_hits:
            candidates[c.chunk_id] = c
        for c in bm25_hits:
            candidates.setdefault(c.chunk_id, c)

        fused: list[RetrievalHit] = []
        for cid, c in candidates.items():
            if doc_type_filter and c.doc_type not in doc_type_filter:
                continue
            vr = vector_ids.get(cid)
            br = bm25_ids.get(cid)
            score = 0.0
            if vr is not None:
                score += 1.0 / (self.rrf_k + vr)
            if br is not None:
                score += 1.0 / (self.rrf_k + br)
            fused.append(RetrievalHit(chunk=c, vector_rank=vr, bm25_rank=br, score=score))

        fused.sort(key=lambda h: -h.score)
        return fused[:top_k]


@lru_cache(maxsize=1)
def default_retriever() -> HybridRetriever:
    return HybridRetriever()
