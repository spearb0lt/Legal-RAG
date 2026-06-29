"""
IL-TUR Prior Case Retrieval (PCR) evaluation.

For each query case, retrieve top-k from the candidate pool and check whether the
gold-relevant cases appear. Reports Recall@1, Recall@5, Recall@10, and MRR@10.

This is a STAND-ALONE eval index: it builds its own ChromaDB collection so it
doesn't pollute the production corpus, and tears it down at the end (unless
--keep-index is passed).

Run:
    python -m eval.il_tur_pcr                       # 100 queries, top-10
    python -m eval.il_tur_pcr --n-queries 500       # bigger run
    python -m eval.il_tur_pcr --keep-index          # persist for inspection

Requires HF_TOKEN in .env if the IL-TUR PCR task is gated.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from core import config
from core.chunker import chunk_document
from core.embeddings import embed_documents, embed_query


_EVAL_COLLECTION = "il_tur_pcr_eval"


def _load_pcr(split: str = "test"):
    """Return the IL-TUR PCR dataset for a given split, or None."""
    try:
        from datasets import load_dataset
        return load_dataset(
            "Exploration-Lab/IL-TUR", "pcr", split=split, token=config.HF_TOKEN
        )
    except Exception as e:
        print(f"[pcr-eval] could not load test split: {e!r}")
        try:
            from datasets import load_dataset
            return load_dataset(
                "Exploration-Lab/IL-TUR", "pcr", split="validation", token=config.HF_TOKEN
            )
        except Exception as e2:
            print(f"[pcr-eval] could not load validation split either: {e2!r}")
            return None


def _extract_query_relevant_pair(row: dict[str, Any]) -> tuple[str, str, list[str]] | None:
    """
    IL-TUR PCR row -> (query_id, query_text, list of gold-relevant doc_ids).

    The exact schema of the PCR split has varied across HF dataset revisions;
    we look for the most common field names and skip rows that don't match.
    """
    query_id = None
    for k in ("query_id", "id", "qid", "case_id"):
        if k in row and row[k]:
            query_id = str(row[k])
            break

    query_text = None
    for k in ("query_text", "query", "text", "case_proceeding", "case_text", "document"):
        v = row.get(k)
        if isinstance(v, str) and len(v) > 200:
            query_text = v
            break

    relevant: list[str] = []
    for k in ("relevant_candidates", "relevant", "gold", "labels", "candidate_relevant"):
        v = row.get(k)
        if v:
            if isinstance(v, list):
                relevant = [str(x) for x in v]
            elif isinstance(v, dict) and "ids" in v:
                relevant = [str(x) for x in v["ids"]]
            elif isinstance(v, str):
                relevant = [v]
            break

    if not query_id or not query_text or not relevant:
        return None
    return query_id, query_text, relevant


def _load_candidates(split: str = "train"):
    try:
        from datasets import load_dataset
        return load_dataset(
            "Exploration-Lab/IL-TUR", "pcr_candidates", split=split, token=config.HF_TOKEN
        )
    except Exception:
        return None


def _candidate_id_text(row: dict[str, Any]) -> tuple[str, str] | None:
    cid = None
    for k in ("doc_id", "id", "candidate_id", "case_id"):
        if k in row and row[k]:
            cid = str(row[k])
            break
    text = None
    for k in ("text", "document", "case_text", "case_proceeding", "body"):
        v = row.get(k)
        if isinstance(v, str) and len(v) > 200:
            text = v
            break
    if not cid or not text:
        return None
    return cid, text


def _build_eval_index(candidates: list[tuple[str, str]]) -> None:
    """Stand up an isolated Chroma collection of PCR candidates."""
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        client.delete_collection(_EVAL_COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(_EVAL_COLLECTION, metadata={"hnsw:space": "cosine"})

    print(f"[pcr-eval] embedding {len(candidates)} candidates ...")
    batch = 64
    for i in range(0, len(candidates), batch):
        sub = candidates[i : i + batch]
        ids = [cid for cid, _ in sub]
        texts = [t[:5000] for _, t in sub]
        vecs = embed_documents(texts)
        coll.upsert(ids=ids, documents=texts, embeddings=vecs)
        if (i // batch) % 5 == 0:
            print(f"  embedded {min(i + batch, len(candidates))}/{len(candidates)}")
    print("[pcr-eval] candidate index ready.")


def _retrieve(query_text: str, top_k: int = 10) -> list[str]:
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_collection(_EVAL_COLLECTION)
    qvec = embed_query(query_text[:5000])
    res = coll.query(query_embeddings=[qvec], n_results=top_k, include=["distances"])
    return res.get("ids", [[]])[0]


def _metrics(ranked_ids: list[str], gold: set[str], cutoffs=(1, 5, 10)) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in cutoffs:
        hits_at_k = [i for i in ranked_ids[:k] if i in gold]
        out[f"recall@{k}"] = float(len(hits_at_k) > 0)
    mrr = 0.0
    for rank, doc_id in enumerate(ranked_ids[:10], start=1):
        if doc_id in gold:
            mrr = 1.0 / rank
            break
    out["mrr@10"] = mrr
    return out


def run_eval(
    *,
    n_queries: int = 100,
    keep_index: bool = False,
    query_split: str = "test",
    candidate_split: str = "train",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    queries_ds = _load_pcr(query_split)
    if queries_ds is None:
        return {"error": "Could not load IL-TUR PCR queries"}

    pairs: list[tuple[str, str, list[str]]] = []
    for row in queries_ds:
        rec = _extract_query_relevant_pair(row)
        if rec is None:
            continue
        pairs.append(rec)
        if len(pairs) >= n_queries:
            break
    if not pairs:
        return {"error": "No usable (query, relevant) pairs extracted from IL-TUR PCR. "
                          "The schema may have changed; inspect a sample row in fetch_il_tur.py."}
    print(f"[pcr-eval] {len(pairs)} usable queries")

    cands_ds = _load_candidates(candidate_split)
    candidates: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    if cands_ds is not None:
        for row in cands_ds:
            ct = _candidate_id_text(row)
            if ct and ct[0] not in seen_ids:
                candidates.append(ct)
                seen_ids.add(ct[0])
    else:
        print("[pcr-eval] candidate split missing — using query texts as their own pool")
        for qid, qtext, _ in pairs:
            if qid not in seen_ids:
                candidates.append((qid, qtext))
                seen_ids.add(qid)

    for qid, qtext, gold in pairs:
        for g in gold:
            if g not in seen_ids:
                candidates.append((g, ""))
                seen_ids.add(g)

    candidates = [(cid, txt) for cid, txt in candidates if txt]
    print(f"[pcr-eval] {len(candidates)} candidates after dedup")

    _build_eval_index(candidates)

    per_query: list[dict[str, Any]] = []
    aggregate = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0}
    for i, (qid, qtext, gold) in enumerate(pairs):
        ranked = _retrieve(qtext, top_k=10)
        m = _metrics(ranked, set(gold))
        per_query.append({"query_id": qid, "gold": list(gold), "retrieved": ranked, **m})
        for k in aggregate:
            aggregate[k] += m[k]
        if (i + 1) % 10 == 0:
            print(
                f"  q {i+1}/{len(pairs)}  R@1={aggregate['recall@1']/(i+1):.3f}  "
                f"R@5={aggregate['recall@5']/(i+1):.3f}  R@10={aggregate['recall@10']/(i+1):.3f}  "
                f"MRR@10={aggregate['mrr@10']/(i+1):.3f}"
            )

    for k in aggregate:
        aggregate[k] /= len(pairs)

    result = {
        "dataset": "Exploration-Lab/IL-TUR :: pcr",
        "n_queries": len(pairs),
        "n_candidates": len(candidates),
        "query_split": query_split,
        "candidate_split": candidate_split,
        "metrics": aggregate,
        "elapsed_s": time.perf_counter() - t0,
        "embedder": __import__("os").environ.get("LEGAL_RAG_EMBEDDER", "gemini"),
        "embedding_model": config.GEMINI_EMBEDDING_MODEL,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    out_path = config.EVAL_RESULTS_DIR / f"il_tur_pcr_{int(time.time())}.json"
    out_path.write_text(
        json.dumps({"summary": result, "per_query": per_query}, indent=2),
        encoding="utf-8",
    )
    print(f"[pcr-eval] wrote {out_path}")

    if not keep_index:
        client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        try:
            client.delete_collection(_EVAL_COLLECTION)
        except Exception:
            pass

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--keep-index", action="store_true")
    ap.add_argument("--query-split", default="test")
    ap.add_argument("--candidate-split", default="train")
    args = ap.parse_args()
    result = run_eval(
        n_queries=args.n_queries,
        keep_index=args.keep_index,
        query_split=args.query_split,
        candidate_split=args.candidate_split,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
