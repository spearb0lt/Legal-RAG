"""
End-to-end pipeline:
  question -> router (Groq) -> retriever (Chroma+BM25) -> synthesizer (Gemini, structured JSON)

Returns a `LegalAnswer` with answer text + verified citations. Each citation is checked
against the chunks actually retrieved so the model can't fabricate chunk_ids.

Also provides `predict_outcome()` for case outcome similarity search:
  case description -> filtered vector search (CJPE/BAIL) -> outcome analysis (Gemini)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config, index
from .llm import gemini_json
from .prompts import (
    OUTCOME_PREDICTION_PROMPT_TEMPLATE,
    OUTCOME_PREDICTION_SCHEMA,
    OUTCOME_PREDICTION_SYSTEM,
    SYNTH_PROMPT_TEMPLATE,
    SYNTH_SCHEMA,
    render_outcome_sources,
    render_sources_block,
)
from .retriever import RetrievalHit, default_retriever
from .router import RouteDecision, route


@dataclass
class Citation:
    source_tag: str
    chunk_id: str
    quote: str
    support_reason: str = ""
    verified: bool = False


@dataclass
class LegalAnswer:
    question: str
    answer: str
    citations: list[Citation]
    hits: list[RetrievalHit]
    route_decision: RouteDecision
    confidence: str = "medium"
    caveats: list[str] = field(default_factory=list)
    raw_model_output: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [f"### Answer\n\n{self.answer}\n"]
        if self.citations:
            lines.append("### Citations\n")
            for c in self.citations:
                hit = next(
                    (h for h in self.hits if h.chunk.chunk_id == c.chunk_id), None
                )
                label = hit.chunk.citation_text() if hit else c.source_tag
                marker = "[verified]" if c.verified else "[unverified]"
                lines.append(f"- **{c.source_tag}** {marker} — {label}")
                if c.quote:
                    lines.append(f"  > {c.quote}")
        if self.caveats:
            lines.append("\n### Caveats")
            for cv in self.caveats:
                lines.append(f"- {cv}")
        return "\n".join(lines)


def _verify_citation(citation_data: dict, hits: list[RetrievalHit]) -> Citation:
    """Mark a citation as verified iff its chunk_id matches a retrieved chunk."""
    cid = (citation_data.get("chunk_id") or "").strip()
    hit_ids = {h.chunk.chunk_id for h in hits}
    return Citation(
        source_tag=citation_data.get("source_tag", ""),
        chunk_id=cid,
        quote=citation_data.get("quote", ""),
        support_reason=citation_data.get("support_reason", ""),
        verified=cid in hit_ids,
    )


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    use_pro_model: bool = False,
    skip_router: bool = False,
) -> LegalAnswer:
    if skip_router:
        decision = RouteDecision(
            intent="general_legal_qa",
            doc_type_filter=[],
            rewritten_query=question.strip(),
            keywords=[],
        )
    else:
        try:
            decision = route(question)
        except Exception:
            # Router (Groq) unavailable — fall back to raw query, no filter
            decision = RouteDecision(
                intent="general_legal_qa",
                doc_type_filter=[],
                rewritten_query=question.strip(),
                keywords=[],
            )

    if decision.intent == "out_of_scope":
        return LegalAnswer(
            question=question,
            answer=(
                "This question doesn't appear to be about Indian law. "
                "Please ask about statutes, cases, doctrines, or procedure."
            ),
            citations=[],
            hits=[],
            route_decision=decision,
            confidence="high",
            caveats=["Out-of-scope query — no retrieval performed."],
        )

    retriever = default_retriever()
    try:
        hits = retriever.search(
            decision.rewritten_query or question,
            top_k=top_k or config.RETRIEVAL_TOP_K,
            doc_type_filter=decision.doc_type_filter or None,
        )
    except Exception as retrieval_err:
        return LegalAnswer(
            question=question,
            answer=(
                "Retrieval failed — the vector index may be unavailable or the "
                "embedding model encountered an error. "
                f"_(Error: {retrieval_err})_\n\n"
                "Try rebuilding the corpus: `python -m data.ingest.build_all --reset --il-tur`"
            ),
            citations=[],
            hits=[],
            route_decision=decision,
            confidence="low",
            caveats=["Retrieval error — check embedder config and ChromaDB index."],
        )

    if not hits:
        return LegalAnswer(
            question=question,
            answer=(
                "I couldn't find any source in the indexed corpus that addresses this. "
                "Try rephrasing, or check that the corpus has been built "
                "(run `python -m data.ingest.build_all`)."
            ),
            citations=[],
            hits=[],
            route_decision=decision,
            confidence="low",
            caveats=["Empty retrieval — corpus may be unbuilt or query is out of domain."],
        )

    sources_block = render_sources_block(hits)
    prompt = SYNTH_PROMPT_TEMPLATE.format(question=question, sources=sources_block)

    model = config.GEMINI_HEAVY_MODEL if use_pro_model else config.GEMINI_SYNTHESIS_MODEL
    try:
        data = gemini_json(
            prompt=prompt,
            schema=SYNTH_SCHEMA,
            model=model,
            max_output_tokens=2048,
        )
    except Exception as llm_err:
        return LegalAnswer(
            question=question,
            answer=(
                "The language model is temporarily unavailable. "
                "The sources below were retrieved — you can review them directly.\n\n"
                f"_(API error: {llm_err})_"
            ),
            citations=[],
            hits=hits,
            route_decision=decision,
            confidence="low",
            caveats=["Gemini synthesis unavailable — retrieved sources shown without analysis."],
        )

    verified_cites = [_verify_citation(c, hits) for c in data.get("citations", []) or []]
    return LegalAnswer(
        question=question,
        answer=(data.get("answer") or "").strip(),
        citations=verified_cites,
        hits=hits,
        route_decision=decision,
        confidence=data.get("confidence", "medium"),
        caveats=list(data.get("caveats") or []),
        raw_model_output=data,
    )


# ── Outcome Prediction ────────────────────────────────────────────────────────

_TASK_META = {
    "cjpe": {
        "description": "Court Judgment Prediction — find similar Indian court cases and their final judgments",
        "favorable_label": "Judgment Accepted (petitioner wins)",
        "unfavorable_label": "Judgment Rejected (petitioner loses)",
    },
    "bail": {
        "description": "Bail Prediction — find similar Indian bail applications and whether bail was granted",
        "favorable_label": "Bail Granted",
        "unfavorable_label": "Bail Denied",
        "language_note": "NOTE: BAIL corpus is in Hindi (Devanagari). Retrieval quality from English queries may be limited.",
    },
}


@dataclass
class SimilarCaseOutcome:
    doc_id: str
    doc_title: str
    outcome: int        # 0 = unfavorable, 1 = favorable, -1 = unknown
    outcome_label: str
    score: float
    snippet: str
    source_note: str


@dataclass
class OutcomePrediction:
    case_description: str
    task: str
    similar_cases: list[SimilarCaseOutcome]
    hits: list          # raw RetrievalHit objects
    favorable_count: int
    unfavorable_count: int
    similar_case_summary: str
    key_factors: list[str]
    assessment: str
    confidence: str = "medium"
    caveats: list[str] = field(default_factory=list)
    raw_model_output: dict = field(default_factory=dict)

    def favorable_label(self) -> str:
        return _TASK_META.get(self.task, {}).get("favorable_label", "Favorable")

    def unfavorable_label(self) -> str:
        return _TASK_META.get(self.task, {}).get("unfavorable_label", "Unfavorable")

    def to_markdown(self) -> str:
        total = self.favorable_count + self.unfavorable_count
        pct = int(100 * self.favorable_count / total) if total else 0
        lines = [
            f"### Outcome Analysis\n",
            f"**Similar cases found:** {total} &nbsp;|&nbsp; "
            f"**{self.favorable_label()}:** {self.favorable_count} ({pct}%) &nbsp;|&nbsp; "
            f"**{self.unfavorable_label()}:** {self.unfavorable_count}\n",
            f"#### Summary of Similar Cases\n{self.similar_case_summary}\n",
            f"#### Key Factors\n" + "\n".join(f"- {f}" for f in self.key_factors) + "\n",
            f"#### Assessment\n{self.assessment}\n",
        ]
        if self.caveats:
            lines.append("#### Caveats\n" + "\n".join(f"- {c}" for c in self.caveats))
        return "\n".join(lines)


def predict_outcome(
    case_description: str,
    *,
    task: str = "cjpe",
    top_k: int = 12,
    use_pro_model: bool = False,
) -> OutcomePrediction:
    """
    Find similar CJPE or BAIL cases and predict the likely outcome.

    task: "cjpe" for court judgment prediction, "bail" for bail grant prediction.
    vector_search returns Chunk objects ordered by similarity (best first).
    """
    meta = _TASK_META.get(task, _TASK_META["cjpe"])

    # Retrieve only chunks tagged with this dataset (filtered by source_task metadata)
    where_filter = {"source_task": task}
    chunks = index.vector_search(case_description, top_k=top_k, where_filter=where_filter)

    # Deduplicate by doc_id — keep first occurrence (best similarity rank)
    seen_docs: set[str] = set()
    similar_cases: list[SimilarCaseOutcome] = []
    for chunk in chunks:
        if chunk.doc_id in seen_docs:
            continue
        seen_docs.add(chunk.doc_id)
        dedup_rank = len(similar_cases)  # position in deduplicated list
        outcome = chunk.extra.get("outcome", -1)
        if outcome == 1:
            outcome_label = meta["favorable_label"]
        elif outcome == 0:
            outcome_label = meta["unfavorable_label"]
        else:
            outcome_label = "Unknown"
        similar_cases.append(SimilarCaseOutcome(
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            outcome=outcome,
            outcome_label=outcome_label,
            score=round(1.0 - dedup_rank / top_k, 3),
            snippet=chunk.text[:600],
            source_note=chunk.source_note,
        ))

    # Count outcomes
    favorable = sum(1 for c in similar_cases if c.outcome == 1)
    unfavorable = sum(1 for c in similar_cases if c.outcome == 0)

    if not similar_cases:
        return OutcomePrediction(
            case_description=case_description,
            task=task,
            similar_cases=[],
            hits=chunks,
            favorable_count=0,
            unfavorable_count=0,
            similar_case_summary="No similar cases found in the indexed corpus.",
            key_factors=[],
            assessment=(
                f"No {task.upper()} cases are indexed yet. "
                f"Rebuild the corpus with `python -m data.ingest.build_all --{task}`."
            ),
            confidence="low",
            caveats=["Corpus may be unbuilt or the task filter returned no results."],
        )

    sources_block = render_outcome_sources(similar_cases)
    language_note = meta.get("language_note", "")

    prompt = OUTCOME_PREDICTION_PROMPT_TEMPLATE.format(
        task_description=meta["description"] + (f"\n{language_note}" if language_note else ""),
        case_description=case_description,
        sources=sources_block,
        favorable_label=meta["favorable_label"],
        unfavorable_label=meta["unfavorable_label"],
        n_favorable=favorable,
        n_unfavorable=unfavorable,
        n_total=len(similar_cases),
    )

    model = config.GEMINI_HEAVY_MODEL if use_pro_model else config.GEMINI_SYNTHESIS_MODEL
    try:
        data = gemini_json(
            prompt=prompt,
            schema=OUTCOME_PREDICTION_SCHEMA,
            system_instruction=OUTCOME_PREDICTION_SYSTEM,
            model=model,
            max_output_tokens=2048,
        )
    except Exception as llm_err:
        total = favorable + unfavorable
        pct = int(100 * favorable / total) if total else 0
        return OutcomePrediction(
            case_description=case_description,
            task=task,
            similar_cases=similar_cases,
            hits=chunks,
            favorable_count=favorable,
            unfavorable_count=unfavorable,
            similar_case_summary=(
                f"Found {len(similar_cases)} similar case(s): "
                f"{favorable} {meta['favorable_label']} ({pct}%), "
                f"{unfavorable} {meta['unfavorable_label']}. "
                "AI analysis unavailable — review the similar cases above for details."
            ),
            key_factors=[],
            assessment=(
                f"Analysis unavailable (API error: {llm_err}). "
                "You can still review the similar cases and their outcomes above."
            ),
            confidence="low",
            caveats=["Gemini analysis temporarily unavailable. Similar cases retrieved successfully."],
            raw_model_output={},
        )

    return OutcomePrediction(
        case_description=case_description,
        task=task,
        similar_cases=similar_cases,
        hits=chunks,
        favorable_count=favorable,
        unfavorable_count=unfavorable,
        similar_case_summary=(data.get("similar_case_summary") or "").strip(),
        key_factors=list(data.get("key_factors") or []),
        assessment=(data.get("assessment") or "").strip(),
        confidence=data.get("confidence", "medium"),
        caveats=list(data.get("caveats") or []),
        raw_model_output=data,
    )
