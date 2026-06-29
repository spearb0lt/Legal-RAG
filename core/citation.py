"""
Citation primitives. Every retrievable unit is a `Chunk` with a stable `chunk_id`
that downstream citations can reference. Citations survive index rebuilds because
chunk_id is derived deterministically from (doc_id, paragraph_index).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    """One paragraph-sized unit of retrievable text with full provenance."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    section_marker: str
    paragraph_index: int
    text: str
    jurisdiction: str = "IN"
    source_url: str = ""
    source_note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def short_label(self) -> str:
        if self.section_marker:
            return f"{self.doc_title} — {self.section_marker}"
        return f"{self.doc_title} ¶{self.paragraph_index}"

    def citation_text(self) -> str:
        parts = [self.doc_title]
        if self.section_marker:
            parts.append(self.section_marker)
        parts.append(f"¶{self.paragraph_index}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_chunk_id(doc_id: str, paragraph_index: int) -> str:
    raw = f"{doc_id}::p{paragraph_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def make_doc_id(doc_type: str, title: str) -> str:
    raw = f"{doc_type}::{title}".lower().strip()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
