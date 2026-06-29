"""
Build the v2 corpus (seed + PCR + LSI only, NO CJPE/BAIL) into data/chroma_v2/.
This is the independent corpus used by app2.py.

Run from the legal-rag directory:
    python scripts/build_v2.py
"""
import os
import sys
from pathlib import Path

# Set env vars BEFORE any core import
_HERE = Path(__file__).resolve().parent.parent  # legal-rag/
os.environ["LEGAL_RAG_CHROMA_DIR"] = str(_HERE / "data" / "chroma_v2")
os.environ["LEGAL_RAG_CHROMA_COLLECTION"] = "indian_legal_v2"

# Now safe to import core
sys.path.insert(0, str(_HERE))
from core import config, index
from data.ingest.load_local import load_local_corpus
from data.ingest.fetch_il_tur import load_il_tur_pcr_corpus, load_il_tur_lsi_statutes
from time import perf_counter

print(f"[build_v2] ChromaDB dir : {config.CHROMA_DIR}")
print(f"[build_v2] Collection   : {config.CHROMA_COLLECTION}")

print("[build_v2] Resetting collection ...")
index.reset_index()

t0 = perf_counter()

print(f"[build_v2] Loading local seed corpus from {config.RAW_DIR} ...")
chunks = load_local_corpus()
print(f"[build_v2]   seed chunks: {len(chunks)}")

print("[build_v2] Pulling IL-TUR PCR (up to 300 cases) ...")
chunks.extend(load_il_tur_pcr_corpus(300))

print("[build_v2] Pulling IL-TUR LSI statutes (up to 100) ...")
chunks.extend(load_il_tur_lsi_statutes(100))

print(f"[build_v2] Total chunks to embed: {len(chunks)}")
added = index.add_chunks(chunks)
print(f"[build_v2] Upserted {added} chunks into '{config.CHROMA_COLLECTION}'")
print(f"[build_v2] Collection size: {index.count()} chunks")
print(f"[build_v2] Done in {perf_counter() - t0:.1f}s")
