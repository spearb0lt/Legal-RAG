"""
End-to-end corpus build:
  1. load curated seeds from data/raw/
  2. (optional) download IL-TUR HF datasets
  3. embed everything via the configured embedder
  4. upsert into ChromaDB

Run:
    python -m data.ingest.build_all                              # seed only
    python -m data.ingest.build_all --il-tur                    # + PCR (300) + LSI (100 statutes)
    python -m data.ingest.build_all --il-tur --pcr-max 7070     # full PCR corpus
    python -m data.ingest.build_all --il-tur --cjpe --bail      # + CJPE + BAIL cases
    python -m data.ingest.build_all --reset                     # wipe & rebuild
"""
from __future__ import annotations

import argparse
from time import perf_counter

from core import config, index

from .load_local import load_local_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Drop existing Chroma collection first")
    ap.add_argument("--il-tur", action="store_true", help="Pull IL-TUR PCR + LSI statutes")
    ap.add_argument("--pcr-max", type=int, default=300,
                    help="Max PCR candidate cases to index (max 7070)")
    ap.add_argument("--lsi-max", type=int, default=100,
                    help="Max LSI statutes to index (max 100 — the full statute set)")
    ap.add_argument("--cjpe", action="store_true",
                    help="Also pull CJPE case judgment texts (downloads ~Xmb if not cached)")
    ap.add_argument("--cjpe-max", type=int, default=500,
                    help="Max CJPE cases to index")
    ap.add_argument("--bail", action="store_true",
                    help="Also pull BAIL application texts (downloads ~Xmb if not cached)")
    ap.add_argument("--bail-max", type=int, default=500,
                    help="Max BAIL cases to index")
    args = ap.parse_args()

    if args.reset:
        print("[build] resetting Chroma collection ...")
        index.reset_index()

    t0 = perf_counter()
    print(f"[build] loading local seed corpus from {config.RAW_DIR} ...")
    chunks = load_local_corpus()
    print(f"[build]   seed chunks: {len(chunks)}")

    if args.il_tur or args.cjpe or args.bail:
        from .fetch_il_tur import (
            load_il_tur_pcr_corpus,
            load_il_tur_lsi_statutes,
            load_il_tur_cjpe_cases,
            load_il_tur_bail_cases,
        )

    if args.il_tur:
        print("[build] pulling IL-TUR PCR ...")
        chunks.extend(load_il_tur_pcr_corpus(args.pcr_max))
        print("[build] pulling IL-TUR LSI statutes ...")
        chunks.extend(load_il_tur_lsi_statutes(args.lsi_max))

    if args.cjpe:
        print("[build] pulling IL-TUR CJPE ...")
        chunks.extend(load_il_tur_cjpe_cases(args.cjpe_max))

    if args.bail:
        print("[build] pulling IL-TUR BAIL ...")
        chunks.extend(load_il_tur_bail_cases(args.bail_max))

    print(f"[build] total chunks to embed: {len(chunks)}")
    added = index.add_chunks(chunks)
    print(f"[build] upserted {added} chunks into '{config.CHROMA_COLLECTION}'")
    print(f"[build] collection size: {index.count()} chunks total")
    print(f"[build] done in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
