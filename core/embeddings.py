"""
Embedding backends for legal-rag.

Set LEGAL_RAG_EMBEDDER in your .env to choose the backend:

  Backend  | Value      | Required env vars          | Recommended model
  ---------|------------|----------------------------|------------------------------------------
  Gemini   | gemini     | GOOGLE_API_KEY             | gemini-embedding-001 (768-dim, default)
  Local    | local      | (none — fully offline)     | all-MiniLM-L6-v2 (384-dim) or BAAI/bge-base-en-v1.5 (768-dim)
  Voyage   | voyage     | VOYAGE_API_KEY             | voyage-law-2 (1024-dim, legal-tuned)
  HF API   | hf_api     | HF_TOKEN (recommended)     | any HF Inference API model

Examples (.env):
  LEGAL_RAG_EMBEDDER=local
  LEGAL_RAG_LOCAL_EMBED_MODEL=BAAI/bge-base-en-v1.5

  LEGAL_RAG_EMBEDDER=voyage
  VOYAGE_API_KEY=pa-...
  LEGAL_RAG_VOYAGE_MODEL=voyage-law-2

  LEGAL_RAG_EMBEDDER=hf_api
  HF_TOKEN=hf_...
  LEGAL_RAG_HF_API_MODEL=BAAI/bge-base-en-v1.5

IMPORTANT — dimension mismatch: if you switch backends you MUST rebuild the index:
  python -m data.ingest.build_all --reset --il-tur
"""
from __future__ import annotations

import os
import time
from typing import Sequence

from . import config

_EMBEDDER_BACKEND = os.environ.get("LEGAL_RAG_EMBEDDER", "gemini").lower()
_LOCAL_MODEL_NAME = os.environ.get(
    "LEGAL_RAG_LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
_VOYAGE_MODEL = config.LEGAL_RAG_VOYAGE_MODEL or "voyage-law-2"
_HF_API_MODEL = config.LEGAL_RAG_HF_API_MODEL or "sentence-transformers/all-MiniLM-L6-v2"

# BGE models need a retrieval-specific prefix on the query side for best accuracy
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_gemini_client = None
_local_model = None
_voyage_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Install google-genai: pip install -U google-genai"
            ) from e
        _gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _gemini_client


def _load_local():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
    return _local_model


def _load_voyage():
    global _voyage_client
    if _voyage_client is None:
        if not config.VOYAGE_API_KEY:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. Add VOYAGE_API_KEY=pa-... to your .env file.\n"
                "Get a key at https://www.voyageai.com/"
            )
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "Install voyageai: pip install -U voyageai"
            ) from e
        _voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _voyage_client


def _gemini_model_name() -> str:
    n = (config.GEMINI_EMBEDDING_MODEL or "gemini-embedding-001").strip()
    return n[len("models/"):] if n.startswith("models/") else n


def _apply_bge_prefix(texts: list[str], task_type: str) -> list[str]:
    if task_type == "RETRIEVAL_QUERY" and "bge" in _LOCAL_MODEL_NAME.lower():
        return [_BGE_QUERY_PREFIX + t for t in texts]
    return texts


def embed_texts(
    texts: Sequence[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """
    Embed a batch of texts.
    task_type: "RETRIEVAL_DOCUMENT" when indexing, "RETRIEVAL_QUERY" at query time.
    Local and HF API backends ignore task_type (except BGE prefix for local queries).
    """
    if not texts:
        return []

    # ── Local (sentence-transformers) ──────────────────────────────────────
    if _EMBEDDER_BACKEND == "local":
        prepared = _apply_bge_prefix(list(texts), task_type)
        model = _load_local()
        vecs = model.encode(prepared, show_progress_bar=False, normalize_embeddings=True)
        return vecs.tolist()

    # ── Voyage AI ──────────────────────────────────────────────────────────
    if _EMBEDDER_BACKEND == "voyage":
        vc = _load_voyage()
        input_type = "query" if task_type == "RETRIEVAL_QUERY" else "document"
        out: list[list[float]] = []
        batch = 64  # Voyage AI recommended batch size
        for i in range(0, len(texts), batch):
            chunk = list(texts[i: i + batch])
            result = vc.embed(chunk, model=_VOYAGE_MODEL, input_type=input_type)
            out.extend([list(e) for e in result.embeddings])
        return out

    # ── HuggingFace Inference API ──────────────────────────────────────────
    if _EMBEDDER_BACKEND == "hf_api":
        import requests
        import numpy as np

        api_url = (
            f"https://api-inference.huggingface.co"
            f"/pipeline/feature-extraction/{_HF_API_MODEL}"
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.HF_TOKEN:
            headers["Authorization"] = f"Bearer {config.HF_TOKEN}"
        out = []
        batch = 32  # conservative for HF free-tier rate limits
        for i in range(0, len(texts), batch):
            chunk = list(texts[i: i + batch])
            payload = {"inputs": chunk, "options": {"wait_for_model": True}}
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        api_url, headers=headers, json=payload, timeout=60
                    )
                    resp.raise_for_status()
                    raw = resp.json()
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(
                    f"HF Inference API embedding failed after retries: {last_err!r}"
                )
            for item in raw:
                # sentence-transformers → List[float] directly
                # other transformers → List[List[float]] (token-level) → mean pool
                if item and isinstance(item[0], list):
                    vec = np.array(item).mean(axis=0).tolist()
                else:
                    vec = item
                out.append(vec)
        return out

    # ── Gemini (default) ───────────────────────────────────────────────────
    from google.genai import types

    model_name = _gemini_model_name()
    cfg = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=config.GEMINI_EMBEDDING_DIM,
    )
    out = []
    batch = 50  # gemini-embedding-001 limit is 100; stay conservative
    for i in range(0, len(texts), batch):
        chunk = list(texts[i: i + batch])
        last_err = None
        for attempt in range(3):
            try:
                result = _gemini().models.embed_content(
                    model=model_name,
                    contents=chunk,
                    config=cfg,
                )
                out.extend([list(e.values) for e in result.embeddings])
                break
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(
                f"Gemini embedding failed after retries: {last_err!r}"
            )
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    return embed_texts(list(texts), task_type="RETRIEVAL_DOCUMENT")
