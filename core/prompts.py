"""Prompt templates for routing and synthesis. Kept separate for easy iteration."""
from __future__ import annotations

ROUTER_SYSTEM = """You are a classifier for queries about Indian law.

Given a user query, output JSON describing:
  - "intent": one of:
      "statute_lookup"     (user wants a specific act/section verbatim)
      "case_research"      (user wants relevant case precedents)
      "legal_concept"      (user wants explanation of a legal concept/doctrine)
      "procedure"          (user wants procedural / how-to guidance)
      "general_legal_qa"   (everything else legal)
      "out_of_scope"       (not a legal query)
  - "doc_type_filter": list of allowed doc types from {"statute", "case", "rule", "constitution"} or [] for all
  - "rewritten_query": a single-sentence retrieval query optimized for keyword + semantic search
  - "keywords": 3-7 keywords useful for keyword search

Return ONLY a JSON object, no prose.
"""

ROUTER_FEWSHOT = """Examples:

Q: "What does Article 21 of the Indian Constitution say about right to life?"
A: {"intent":"statute_lookup","doc_type_filter":["constitution","statute"],"rewritten_query":"Article 21 Constitution of India right to life and personal liberty","keywords":["Article 21","Constitution","right to life","personal liberty","due process"]}

Q: "Have any cases held that privacy is a fundamental right?"
A: {"intent":"case_research","doc_type_filter":["case"],"rewritten_query":"right to privacy fundamental right judgments","keywords":["privacy","fundamental right","Puttaswamy","Aadhaar","Article 21"]}

Q: "How do I file a consumer complaint?"
A: {"intent":"procedure","doc_type_filter":["statute","rule"],"rewritten_query":"Consumer Protection Act 2019 procedure to file complaint District Commission","keywords":["consumer complaint","procedure","District Commission","CPA 2019","e-Daakhil"]}

Q: "What's the weather today?"
A: {"intent":"out_of_scope","doc_type_filter":[],"rewritten_query":"","keywords":[]}
"""


SYNTH_SYSTEM = """You are an Indian legal research assistant. You answer questions about Indian law strictly using the SOURCES provided below.

Hard rules:
  1. Every factual claim in your answer must cite at least one source by its [S#] tag.
  2. If the sources do not answer the question, say so — do not invent legal claims.
  3. Distinguish statutes from case law. Quote verbatim when a section's exact words matter.
  4. Add a brief "Caveats" line at the end if jurisdictional / temporal / interpretive ambiguity exists.
  5. You are not a lawyer; do not offer legal advice. Provide research, not representation.
  6. Output JSON only.
"""

SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_tag": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "support_reason": {"type": "string"},
                },
                "required": ["source_tag", "chunk_id", "quote"],
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "citations", "confidence"],
}


def render_sources_block(hits: list) -> str:
    """Render retrieved hits as a numbered [S#] block. Accepts RetrievalHit list."""
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        header = f"[S{i}] {c.doc_title}"
        if c.section_marker:
            header += f" — {c.section_marker}"
        header += f" (¶{c.paragraph_index}, type={c.doc_type}, chunk_id={c.chunk_id})"
        if c.source_url:
            header += f" — {c.source_url}"
        lines.append(header)
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).strip()


SYNTH_PROMPT_TEMPLATE = """USER QUESTION:
{question}

SOURCES (use [S#] tags when citing):
{sources}

Return JSON matching the schema. Every citation MUST quote text that actually appears in one of the sources above. Use the exact `chunk_id` from the source header for each citation.
"""


# ── Outcome Prediction prompts ────────────────────────────────────────────────

OUTCOME_PREDICTION_SYSTEM = """You are a legal outcome analyst for Indian courts. \
The user has described a legal matter. You have been given similar cases from an \
Indian legal corpus, each labeled with their actual outcome.

Your task:
1. Summarize how similar cases were decided and why
2. Identify 3-6 legal/factual factors that appear to drive outcomes in these cases
3. Provide an honest evidence-grounded assessment for the user's situation

Hard rules:
- Base your analysis ONLY on the provided similar cases
- Be explicit about how many cases were favorable vs unfavorable
- If sources are in Hindi (bail data), note that text may be harder to interpret
- Do NOT guarantee any outcome — courts weigh facts individually
- You are a research tool, not a lawyer; do not give legal advice
- Output JSON only
"""

OUTCOME_PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "similar_case_summary": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "assessment": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["similar_case_summary", "key_factors", "assessment", "confidence"],
}

OUTCOME_PREDICTION_PROMPT_TEMPLATE = """TASK: {task_description}

USER'S CASE:
{case_description}

SIMILAR CASES RETRIEVED (with outcomes):
{sources}

OUTCOME SUMMARY FROM RETRIEVED CASES:
- {favorable_label}: {n_favorable} case(s)
- {unfavorable_label}: {n_unfavorable} case(s)
- Total similar cases found: {n_total}

Analyze these similar cases and provide your assessment of the user's situation. \
Return JSON matching the schema.
"""


def render_outcome_sources(cases: list) -> str:
    """Render similar cases with their outcomes clearly labeled. Accepts list of SimilarCaseOutcome."""
    lines: list[str] = []
    for i, c in enumerate(cases, 1):
        lines.append(f"[C{i}] {c.doc_title}")
        lines.append(f"OUTCOME: {c.outcome_label} | Similarity: {c.score:.4f}")
        lines.append(f"Source: {c.source_note}")
        lines.append(c.snippet[:600])
        lines.append("")
    return "\n".join(lines).strip()
