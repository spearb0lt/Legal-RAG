"""
Loader for the curated seed corpus in data/raw/.

Each .txt file uses a small header (TITLE:, DOC_TYPE:, JURISDICTION:, CITATION:,
SOURCE_URL:, SOURCE_NOTE:) followed by the body text. This loader parses headers,
hands the body to the chunker, and returns ready-to-index Chunks.
"""
from __future__ import annotations

import re
from pathlib import Path

from core import config
from core.chunker import chunk_document
from core.citation import Chunk

_HEADER_RE = re.compile(r"^([A-Z_]+):\s*(.+)$")


def _parse_file(path: Path) -> tuple[dict, str]:
    headers: dict[str, str] = {}
    body_lines: list[str] = []
    reading_body = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not reading_body:
            m = _HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2).strip()
                continue
            if line.strip() == "":
                reading_body = True
                continue
            reading_body = True
        body_lines.append(line)
    return headers, "\n".join(body_lines).strip()


def load_local_corpus() -> list[Chunk]:
    """Load every .txt file under data/raw/ recursively and chunk it."""
    base = config.RAW_DIR
    chunks: list[Chunk] = []
    for path in sorted(base.rglob("*.txt")):
        headers, body = _parse_file(path)
        if not body:
            continue
        title = headers.get("TITLE") or path.stem
        doc_type = (headers.get("DOC_TYPE") or path.parent.name or "statute").lower()
        source_url = headers.get("SOURCE_URL", "")
        source_note = headers.get("SOURCE_NOTE", "")
        jurisdiction = headers.get("JURISDICTION", "IN")
        chunks.extend(
            chunk_document(
                title=title,
                doc_type=doc_type,
                text=body,
                source_url=source_url,
                source_note=source_note,
                jurisdiction=jurisdiction,
            )
        )
    return chunks


if __name__ == "__main__":
    chunks = load_local_corpus()
    print(f"Loaded {len(chunks)} chunks from {config.RAW_DIR}")
    if chunks:
        sample = chunks[0]
        print(f"  Example: {sample.short_label()} (chunk_id={sample.chunk_id})")
