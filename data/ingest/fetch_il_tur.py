"""
Loader for the IL-TUR benchmark (Exploration-Lab/IL-TUR on HuggingFace).
ACL 2024 long paper, arXiv:2407.05399.

Tasks and what we index from each
──────────────────────────────────
PCR  - Prior Case Retrieval
       Indexes: train/dev/test_CANDIDATES splits (full case judgments)
       Skips:   train/dev/test_QUERIES splits (short query texts — evaluation only)
       Available: 7,070 candidate cases total

LSI  - Legal Statute Identification
       Indexes: ONLY the 'statutes' split (100 actual Indian statute texts)
       Skips:   train/dev/test splits — these are case passage QUERIES with
                anonymized <SECTION>/<ACT> placeholders, used only for evaluation

CJPE - Court Judgment Prediction with Explanation
       Indexes: case fact texts (doc_type='case')
       Useful for: questions about judgment reasoning, criminal/civil outcomes

BAIL - Bail Prediction
       Indexes: bail application texts (doc_type='case')
       Useful for: questions about bail conditions under specific offenses

Set HF_TOKEN in .env for authenticated (faster, gated) downloads.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from core import config
from core.chunker import chunk_document
from core.citation import Chunk

_DATASET = "Exploration-Lab/IL-TUR"


def _safe_outcome(label) -> int:
    """Convert a dataset label to 0/1/-1. Guards against list/dict labels in multi splits."""
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return int(label)
    if isinstance(label, bool):
        return 1 if label else 0
    if isinstance(label, str):
        try:
            return int(label)
        except (ValueError, TypeError):
            pass
    return -1  # list, dict, None, or unparseable → unknown outcome


def _ensure_hf_login() -> None:
    token = config.HF_TOKEN
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def _safe_load(task: str, split: str | None = None):
    from datasets import load_dataset
    _ensure_hf_login()
    try:
        if split:
            return load_dataset(_DATASET, task, split=split, token=config.HF_TOKEN)
        return load_dataset(_DATASET, task, token=config.HF_TOKEN)
    except Exception as e:
        print(f"[il-tur] could not load task={task!r} split={split!r}: {e!r}")
        return None


def _first_text_field(row: dict[str, Any]) -> str | None:
    for key in (
        "text",
        "document",
        "case_text",
        "case_proceeding",
        "input",
        "context",
        "facts",
        "judgment",
        "body",
        "doc_text",
        "candidate",
        "precedent",
        "precedent_text",
        "passage",
        "content",
        "decision",
        "judgment_text",
        "proceedings",
        "full_text",
        "case_body",
        "text_en",
    ):
        v = row.get(key)
        if v is None:
            continue
        # IL-TUR stores paragraphs as a Python list — join to a single string
        if isinstance(v, list):
            joined = "\n".join(str(x) for x in v if x)
            if len(joined) > 50:
                return joined
        elif isinstance(v, str) and len(v) > 50:
            return v
    return None


def _first_id_field(row: dict[str, Any], fallback: str) -> str:
    for key in ("id", "case_id", "doc_id", "uid"):
        v = row.get(key)
        if v:
            return str(v)
    return fallback


def _warn_zero(task: str, split: str, ds) -> None:
    sample_row = next(iter(ds[split]), None)
    if sample_row:
        print(f"[il-tur] WARNING: 0 rows loaded from task={task!r} split={split!r}.")
        print(f"[il-tur]   Columns present: {list(sample_row.keys())}")
        for k, v in list(sample_row.items())[:3]:
            try:
                print(f"[il-tur]   {k!r}: {str(v)[:120]!r}")
            except UnicodeEncodeError:
                print(f"[il-tur]   {k!r}: <non-ASCII value, type={type(v).__name__}>")


# ── PCR ───────────────────────────────────────────────────────────────────────

def load_il_tur_pcr_corpus(max_docs: int = 500) -> list[Chunk]:
    """
    Load IL-TUR PCR candidate case documents (full court judgments).
    Total available: 7,070 (train 4320 + dev 1023 + test 1727 candidates).
    Raises max_docs to load more; set to 7070 for the full corpus.
    """
    chunks: list[Chunk] = []
    ds = _safe_load("pcr")
    if ds is None:
        return chunks

    all_splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    # _QUERIES splits are short evaluation texts — skip them
    candidate_splits = [s for s in all_splits if "candidate" in s]
    splits = candidate_splits if candidate_splits else all_splits

    seen = 0
    for split in splits:
        if seen >= max_docs:
            break
        for i, row in enumerate(ds[split]):
            if seen >= max_docs:
                break
            text = _first_text_field(row)
            if not text:
                continue
            case_id = _first_id_field(row, fallback=f"pcr-{split}-{i}")
            chunks.extend(
                chunk_document(
                    title=f"IL-TUR PCR case {case_id}",
                    doc_type="case",
                    text=text,
                    source_url=f"https://huggingface.co/datasets/{_DATASET}",
                    source_note=f"IL-TUR (ACL 2024) — PCR — split={split} — id={case_id}",
                )
            )
            seen += 1

    if seen == 0 and splits:
        _warn_zero("pcr", splits[0], ds)
    print(f"[il-tur] loaded {seen} PCR cases -> {len(chunks)} chunks")
    return chunks


# ── LSI ───────────────────────────────────────────────────────────────────────

def load_il_tur_lsi_statutes(max_docs: int = 100) -> list[Chunk]:
    """
    Load IL-TUR LSI statute texts.
    ONLY the 'statutes' split (100 entries) contains actual Indian statute texts.
    The train/dev/test splits are evaluation QUERIES — case passage excerpts with
    anonymized <SECTION>/<ACT> placeholders — and must NOT be indexed.
    max_docs defaults to 100 (the complete statute set).
    """
    chunks: list[Chunk] = []
    ds = _safe_load("lsi")
    if ds is None:
        return chunks

    all_splits = list(ds.keys()) if hasattr(ds, "keys") else []
    if "statutes" in all_splits:
        splits = ["statutes"]
    else:
        # Fallback: prefer candidate-like names; otherwise warn
        candidate_splits = [s for s in all_splits if "candidate" in s or "statute" in s]
        splits = candidate_splits if candidate_splits else all_splits
        if splits != ["statutes"]:
            print(f"[il-tur] WARNING: 'statutes' split not found in LSI. "
                  f"Available: {all_splits}. Falling back to: {splits}")

    seen = 0
    for split in splits:
        if seen >= max_docs:
            break
        for i, row in enumerate(ds[split]):
            if seen >= max_docs:
                break
            text = _first_text_field(row)
            if not text:
                continue
            sid = _first_id_field(row, fallback=f"lsi-{split}-{i}")
            chunks.extend(
                chunk_document(
                    title=f"IL-TUR LSI statute {sid}",
                    doc_type="statute",
                    text=text,
                    source_url=f"https://huggingface.co/datasets/{_DATASET}",
                    source_note=f"IL-TUR (ACL 2024) — LSI — split={split} — id={sid}",
                )
            )
            seen += 1

    if seen == 0 and splits:
        _warn_zero("lsi", splits[0], ds)
    print(f"[il-tur] loaded {seen} LSI statutes -> {len(chunks)} chunks")
    return chunks


# ── CJPE ──────────────────────────────────────────────────────────────────────

def load_il_tur_cjpe_cases(max_docs: int = 500) -> list[Chunk]:
    """
    Load IL-TUR CJPE (Court Judgment Prediction with Explanation) case texts.
    These are court case facts + judgment reasoning texts.
    Enriches the corpus for queries about judgment outcomes and judicial reasoning.
    Requires downloading the 'cjpe' config from HuggingFace (not cached by default).
    """
    chunks: list[Chunk] = []
    ds = _safe_load("cjpe")
    if ds is None:
        return chunks

    all_splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    # Use all splits for corpus building (CJPE is a classification task;
    # there is no separate 'candidate' vs 'query' distinction for corpus docs)
    splits = all_splits

    seen = 0
    for split in splits:
        if seen >= max_docs:
            break
        for i, row in enumerate(ds[split]):
            if seen >= max_docs:
                break
            text = _first_text_field(row)
            if not text:
                continue
            case_id = _first_id_field(row, fallback=f"cjpe-{split}-{i}")
            label = row.get("label")
            new_chunks = chunk_document(
                title=f"IL-TUR CJPE case {case_id}",
                doc_type="case",
                text=text,
                source_url=f"https://huggingface.co/datasets/{_DATASET}",
                source_note=f"IL-TUR (ACL 2024) — CJPE — split={split} — id={case_id}",
            )
            outcome_extra = {
                "source_task": "cjpe",
                "outcome": _safe_outcome(label),
            }
            for ch in new_chunks:
                ch.extra = outcome_extra
            chunks.extend(new_chunks)
            seen += 1

    if seen == 0 and splits:
        _warn_zero("cjpe", splits[0], ds)
    print(f"[il-tur] loaded {seen} CJPE cases -> {len(chunks)} chunks")
    return chunks


# ── BAIL ──────────────────────────────────────────────────────────────────────

def _bail_text(row: dict[str, Any]) -> str | None:
    """
    BAIL text field is a dict: {'facts-and-arguments': [...], 'judge-opinion': [...]}.
    Both values are lists of Hindi-language sentences. Join all parts.
    """
    t = row.get("text")
    if isinstance(t, dict):
        parts = []
        for key in ("facts-and-arguments", "judge-opinion"):
            v = t.get(key)
            if isinstance(v, list):
                parts.extend(str(x) for x in v if x)
            elif isinstance(v, str) and v:
                parts.append(v)
        joined = "\n".join(parts)
        return joined if len(joined) > 50 else None
    return _first_text_field(row)


def load_il_tur_bail_cases(max_docs: int = 500) -> list[Chunk]:
    """
    Load IL-TUR BAIL (Bail Prediction) application texts.
    These are bail application texts from Indian criminal courts (mostly in Hindi).
    Enriches the corpus for queries about bail conditions and criminal procedure.
    Note: BAIL text is in Hindi (Devanagari script) — English-only embedding models
    will have limited cross-lingual retrieval effectiveness.
    Requires downloading the 'bail' config from HuggingFace (not cached by default).
    """
    chunks: list[Chunk] = []
    ds = _safe_load("bail")
    if ds is None:
        return chunks

    all_splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    splits = all_splits

    seen = 0
    for split in splits:
        if seen >= max_docs:
            break
        for i, row in enumerate(ds[split]):
            if seen >= max_docs:
                break
            text = _bail_text(row)
            if not text:
                continue
            case_id = _first_id_field(row, fallback=f"bail-{split}-{i}")
            label = row.get("label")
            new_chunks = chunk_document(
                title=f"IL-TUR BAIL case {case_id}",
                doc_type="case",
                text=text,
                source_url=f"https://huggingface.co/datasets/{_DATASET}",
                source_note=f"IL-TUR (ACL 2024) — BAIL — split={split} — id={case_id} — district={row.get('district', '')}",
            )
            outcome_extra = {
                "source_task": "bail",
                "outcome": _safe_outcome(label),
            }
            for ch in new_chunks:
                ch.extra = outcome_extra
            chunks.extend(new_chunks)
            seen += 1

    if seen == 0 and splits:
        _warn_zero("bail", splits[0], ds)
    print(f"[il-tur] loaded {seen} BAIL cases -> {len(chunks)} chunks")
    return chunks


# ── CLI preview ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcr-max", type=int, default=300)
    ap.add_argument("--lsi-max", type=int, default=100)
    ap.add_argument("--cjpe-max", type=int, default=300)
    ap.add_argument("--bail-max", type=int, default=300)
    ap.add_argument("--cjpe", action="store_true")
    ap.add_argument("--bail", action="store_true")
    args = ap.parse_args()

    chunks = (
        load_il_tur_pcr_corpus(args.pcr_max)
        + load_il_tur_lsi_statutes(args.lsi_max)
    )
    if args.cjpe:
        chunks += load_il_tur_cjpe_cases(args.cjpe_max)
    if args.bail:
        chunks += load_il_tur_bail_cases(args.bail_max)

    out_path = config.DATA_DIR / "il_tur_chunks_preview.json"
    out_path.write_text(
        json.dumps([c.to_dict() for c in chunks[:5]], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Preview of first 5 chunks written to {out_path}")
