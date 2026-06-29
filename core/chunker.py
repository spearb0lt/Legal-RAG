"""
Paragraph-aware chunker for Indian legal text.

Behavior:
  - Splits on blank lines and on common Indian-legal section markers
    ("Section 5.", "Article 14.", "Clause (a)", "Para 12.").
  - Merges very short fragments into neighbors so a one-line heading
    doesn't become a standalone retrievable chunk.
  - Splits any chunk that exceeds `max_chars` on sentence boundaries
    with a small overlap, preserving order so paragraph_index stays monotonic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .citation import Chunk, make_chunk_id, make_doc_id

_SECTION_PATTERNS = [
    r"^\s*Section\s+\d+[A-Z]?\.",
    r"^\s*Article\s+\d+[A-Z]?\.",
    r"^\s*\d+\.\s+",
    r"^\s*\(\d+\)\s+",
    r"^\s*[IVXLC]+\.\s+",
    r"^\s*Chapter\s+[IVXLC0-9]+",
    r"^\s*PART\s+[IVXLC0-9]+",
]
_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), flags=re.MULTILINE)


_SENT_END_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z(])")


@dataclass
class ChunkerConfig:
    min_chars: int = 200
    max_chars: int = 1800
    overlap_chars: int = 200


def _split_by_blank_lines(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_by_section_markers(paragraph: str) -> list[str]:
    parts: list[str] = []
    matches = list(_SECTION_RE.finditer(paragraph))
    if not matches:
        return [paragraph]
    last = 0
    for m in matches:
        if m.start() > last:
            head = paragraph[last : m.start()].strip()
            if head:
                parts.append(head)
        last = m.start()
    tail = paragraph[last:].strip()
    if tail:
        parts.append(tail)
    return parts or [paragraph]


def _split_long(paragraph: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = _SENT_END_RE.split(paragraph)
    parts: list[str] = []
    current = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip() if current else s.strip()
        else:
            if current:
                parts.append(current)
            if overlap_chars and parts and len(parts[-1]) > overlap_chars:
                current = parts[-1][-overlap_chars:] + " " + s.strip()
            else:
                current = s.strip()
    if current:
        parts.append(current)
    return parts


def _merge_short(parts: list[str], min_chars: int) -> list[str]:
    merged: list[str] = []
    for p in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = merged[-1] + "\n" + p
        else:
            merged.append(p)
    return merged


def _extract_section_marker(paragraph: str) -> str:
    head = paragraph.lstrip()[:120]
    m = _SECTION_RE.search(head)
    if not m:
        return ""
    return m.group(0).strip().rstrip(".")


def chunk_document(
    *,
    title: str,
    doc_type: str,
    text: str,
    source_url: str = "",
    source_note: str = "",
    jurisdiction: str = "IN",
    config: ChunkerConfig | None = None,
) -> list[Chunk]:
    cfg = config or ChunkerConfig()
    doc_id = make_doc_id(doc_type, title)

    paragraphs = _split_by_blank_lines(text)
    refined: list[str] = []
    for p in paragraphs:
        refined.extend(_split_by_section_markers(p))
    refined = _merge_short(refined, cfg.min_chars)

    chunks: list[Chunk] = []
    p_index = 0
    for paragraph in refined:
        for piece in _split_long(paragraph, cfg.max_chars, cfg.overlap_chars):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, p_index),
                    doc_id=doc_id,
                    doc_title=title,
                    doc_type=doc_type,
                    section_marker=_extract_section_marker(piece),
                    paragraph_index=p_index,
                    text=piece.strip(),
                    jurisdiction=jurisdiction,
                    source_url=source_url,
                    source_note=source_note,
                )
            )
            p_index += 1
    return chunks
