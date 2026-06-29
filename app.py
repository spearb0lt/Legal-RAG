"""
Streamlit chat UI for the citation-grounded Indian legal RAG pipeline.

Run:
    streamlit run app.py

The app surfaces:
  - chat with the corpus, each answer carrying verified [S#] citations
  - a Sources panel that previews the exact paragraphs retrieved
  - a sidebar that shows corpus stats, lets you pick statute-only / case-only
    retrieval, and toggles Gemini Flash vs Pro
  - a one-click "Inspect retrieval" mode that runs retrieval without LLM
    synthesis — useful for debugging
  - a "Case Outcome Prediction" mode: describe your case → find similar
    CJPE (court judgment) or BAIL cases → get outcome statistics + assessment
"""
from __future__ import annotations

import time

import streamlit as st

from core import config, index, synthesis
from core.retriever import default_retriever


st.set_page_config(
    page_title="Indian Legal RAG — Citation-grounded",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Session state ----------------

def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("doc_type_filter", [])
    st.session_state.setdefault("use_pro", False)
    st.session_state.setdefault("inspect_only", False)
    st.session_state.setdefault("skip_router", False)
    st.session_state.setdefault("outcome_mode", False)
    st.session_state.setdefault("outcome_task", "cjpe")
    st.session_state.setdefault("outcome_history", [])


_init_state()


# ---------------- Sidebar ----------------

with st.sidebar:
    st.title("⚖️ Indian Legal RAG")
    st.caption("Citation-grounded. Gemini synthesis + Groq routing.")

    try:
        n_chunks = index.count()
    except Exception as e:
        n_chunks = -1
        st.error(f"ChromaDB unavailable: {e}")
    if n_chunks == 0:
        st.warning(
            "Corpus is empty. Build it first:\n\n"
            "```bash\npython -m data.ingest.build_all --il-tur\n```"
        )
    elif n_chunks > 0:
        st.success(f"{n_chunks:,} chunks indexed.")

    st.divider()

    # ── Mode selector ──────────────────────────────────────────────────────────
    st.subheader("Mode")
    st.session_state["outcome_mode"] = st.toggle(
        "Case Outcome Prediction",
        value=st.session_state["outcome_mode"],
        help=(
            "Describe your case → find similar CJPE (court judgment) or BAIL cases "
            "in the corpus → get outcome statistics and an assessment. "
            "Distinct from regular Q&A."
        ),
    )

    if st.session_state["outcome_mode"]:
        st.session_state["outcome_task"] = st.radio(
            "Dataset",
            options=["cjpe", "bail"],
            format_func=lambda x: (
                "CJPE — Court Judgment (English)" if x == "cjpe"
                else "BAIL — Bail Application (Hindi ⚠️)"
            ),
            index=0 if st.session_state["outcome_task"] == "cjpe" else 1,
            help=(
                "CJPE: 500 English court judgment texts with accepted/rejected labels.\n"
                "BAIL: 500 Hindi bail application texts with granted/denied labels. "
                "English queries will have limited retrieval effectiveness on Hindi text."
            ),
        )
        if st.session_state["outcome_task"] == "bail":
            st.warning(
                "⚠️ BAIL data is in Hindi. English-language case descriptions will "
                "retrieve Hindi documents with limited accuracy. Results may be unreliable."
            )
    else:
        st.subheader("Retrieval filters")
        st.session_state["doc_type_filter"] = st.multiselect(
            "Restrict to doc types",
            options=["constitution", "statute", "rule", "case"],
            default=[],
            help="Empty = no filter. Useful for statute-only or case-only lookups.",
        )

    st.divider()
    st.subheader("Synthesis model")
    st.session_state["use_pro"] = st.toggle(
        f"Use {config.GEMINI_HEAVY_MODEL} (slower, higher quality)",
        value=st.session_state["use_pro"],
        help=f"Default is {config.GEMINI_SYNTHESIS_MODEL}.",
    )

    if not st.session_state["outcome_mode"]:
        st.session_state["inspect_only"] = st.toggle(
            "Inspect retrieval only (skip LLM)",
            value=st.session_state["inspect_only"],
            help="Returns just the retrieved chunks. Free, fast, useful for debugging.",
        )

        st.session_state["skip_router"] = st.toggle(
            "Skip router (use raw question)",
            value=st.session_state["skip_router"],
            help=(
                "Bypasses the Groq routing/query-rewrite step. "
                "Faster and saves one API call, but loses query rewriting, "
                "doc-type auto-filtering, and out-of-scope detection."
            ),
        )

    st.divider()
    if st.button("🧹 Clear chat history"):
        st.session_state["history"] = []
        st.session_state["outcome_history"] = []
        st.rerun()

    st.divider()
    st.caption(
        "**Models in use**\n\n"
        f"- Router: `{config.GROQ_ROUTER_MODEL}`\n"
        f"- Synth (default): `{config.GEMINI_SYNTHESIS_MODEL}`\n"
        f"- Synth (heavy): `{config.GEMINI_HEAVY_MODEL}`\n"
        f"- Embeddings: `{config.GEMINI_EMBEDDING_MODEL}`"
    )


# ---------------- Main pane ----------------

if st.session_state["outcome_mode"]:
    # ── Outcome Prediction mode ────────────────────────────────────────────────
    task = st.session_state["outcome_task"]
    task_name = "Court Judgment" if task == "cjpe" else "Bail Application"

    st.title(f"Case Outcome Prediction — {task_name}")
    st.caption(
        "Describe your case facts and charges. The system will find the most similar "
        f"{'court judgment' if task == 'cjpe' else 'bail application'} cases in the corpus, "
        "show their actual outcomes, and provide an assessment of what you might expect. "
        "**This is research, not legal advice.**"
    )

    # Outcome history render
    for turn in st.session_state["outcome_history"]:
        with st.chat_message("user"):
            st.write(turn["case_description"])
        with st.chat_message("assistant"):
            st.markdown(turn["markdown"])
            if turn.get("similar_cases"):
                with st.expander(
                    f"📂 Similar cases ({turn['favorable_count']} favorable / "
                    f"{turn['unfavorable_count']} unfavorable)",
                    expanded=False,
                ):
                    for i, c in enumerate(turn["similar_cases"], 1):
                        icon = "✅" if c["outcome"] == 1 else ("❌" if c["outcome"] == 0 else "❓")
                        st.markdown(
                            f"**{icon} [{i}] {c['doc_title']}** — `{c['outcome_label']}`"
                        )
                        st.caption(c["source_note"])
                        st.code(c["snippet"], language="markdown")
            if turn.get("confidence"):
                st.caption(f"Confidence: **{turn['confidence']}** • took {turn.get('elapsed_s', 0):.1f}s")

    # Outcome input
    outcome_prompt = st.chat_input(
        f"Describe the facts and charges of your {'case' if task == 'cjpe' else 'bail matter'} …"
    )

    if outcome_prompt:
        if n_chunks == 0:
            st.error("Corpus is empty. Build it first.")
        else:
            with st.chat_message("user"):
                st.write(outcome_prompt)
            with st.chat_message("assistant"):
                t0 = time.perf_counter()
                with st.spinner(f"Finding similar {task_name.lower()} cases …"):
                    try:
                        pred = synthesis.predict_outcome(
                            outcome_prompt,
                            task=task,
                            use_pro_model=st.session_state["use_pro"],
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        st.stop()
                elapsed = time.perf_counter() - t0

                markdown_out = pred.to_markdown()
                st.markdown(markdown_out)

                similar_dicts = [
                    {
                        "doc_title": c.doc_title,
                        "outcome": c.outcome,
                        "outcome_label": c.outcome_label,
                        "score": c.score,
                        "snippet": c.snippet,
                        "source_note": c.source_note,
                    }
                    for c in pred.similar_cases
                ]

                if pred.similar_cases:
                    with st.expander(
                        f"📂 Similar cases ({pred.favorable_count} favorable / "
                        f"{pred.unfavorable_count} unfavorable)",
                        expanded=True,
                    ):
                        for i, c in enumerate(pred.similar_cases, 1):
                            icon = "✅" if c.outcome == 1 else ("❌" if c.outcome == 0 else "❓")
                            st.markdown(
                                f"**{icon} [{i}] {c.doc_title}** — `{c.outcome_label}` "
                                f"— similarity rank {i}"
                            )
                            st.caption(c.source_note)
                            st.code(c.snippet, language="markdown")

                st.caption(
                    f"Confidence: **{pred.confidence}** • "
                    f"{len(pred.similar_cases)} unique cases found • took {elapsed:.1f}s"
                )

                st.session_state["outcome_history"].append({
                    "case_description": outcome_prompt,
                    "markdown": markdown_out,
                    "similar_cases": similar_dicts,
                    "favorable_count": pred.favorable_count,
                    "unfavorable_count": pred.unfavorable_count,
                    "confidence": pred.confidence,
                    "elapsed_s": elapsed,
                    "task": task,
                })

else:
    # ── Regular Q&A mode ──────────────────────────────────────────────────────
    st.title("Ask an Indian-law question")
    st.caption(
        "Every claim in the answer is tagged with [S#] citations linked to verified "
        "paragraphs of the indexed corpus. If your question isn't grounded in the "
        "corpus, the system will say so rather than guess."
    )

    example_questions = [
        "Is the right to privacy a fundamental right under the Indian Constitution?",
        "What is the pecuniary jurisdiction of a District Consumer Commission?",
        "What did the Supreme Court hold about Section 66A of the IT Act?",
        "What are the Vishaka guidelines?",
        "Explain the basic structure doctrine.",
    ]
    cols = st.columns(len(example_questions))
    for col, q in zip(cols, example_questions):
        with col:
            if st.button(q, use_container_width=True, key=f"ex_{hash(q)}"):
                st.session_state["pending_q"] = q
                st.rerun()

    # Render Q&A history
    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["markdown"])
            if turn.get("hits"):
                with st.expander(f"📚 Sources ({len(turn['hits'])} retrieved)", expanded=False):
                    for i, hit in enumerate(turn["hits"], start=1):
                        c = hit["chunk"]
                        verified_marker = "✅" if hit.get("verified_cited") else "•"
                        header = f"{verified_marker} **[S{i}]** {c['doc_title']}"
                        if c["section_marker"]:
                            header += f" — {c['section_marker']}"
                        header += f" — ¶{c['paragraph_index']} — score {hit['score']:.4f}"
                        if c["source_url"]:
                            header += f" — [source]({c['source_url']})"
                        st.markdown(header)
                        st.code(c["text"], language="markdown")
            if turn.get("route_intent"):
                st.caption(
                    f"Routed as **{turn['route_intent']}** "
                    f"(rewritten: _{turn['rewritten_query']}_) "
                    f"• confidence: {turn.get('confidence', 'n/a')} "
                    f"• took {turn.get('elapsed_s', 0):.1f}s"
                )

    # Input / processing
    def _serialize_hit(hit, cited_ids: set[str]):
        return {
            "chunk": hit.chunk.to_dict(),
            "score": hit.score,
            "vector_rank": hit.vector_rank,
            "bm25_rank": hit.bm25_rank,
            "verified_cited": hit.chunk.chunk_id in cited_ids,
        }

    pending = st.session_state.pop("pending_q", None)
    prompt = pending or st.chat_input("Ask about Indian statutes, cases, or doctrines …")

    if prompt:
        n_chunks_now = 0
        try:
            n_chunks_now = index.count()
        except Exception:
            pass
        if n_chunks_now == 0:
            st.error(
                "The corpus is empty. Build it first by running:\n\n"
                "```bash\npython -m data.ingest.build_all --il-tur\n```"
            )
        else:
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                t0 = time.perf_counter()
                with st.spinner("Routing → retrieving → synthesising …"):

                    if st.session_state["inspect_only"]:
                        hits: list = []
                        try:
                            hits = default_retriever().search(
                                prompt,
                                top_k=config.RETRIEVAL_TOP_K,
                                doc_type_filter=st.session_state["doc_type_filter"] or None,
                            )
                        except Exception as exc:
                            st.error(f"Retrieval error: {exc}")
                        elapsed = time.perf_counter() - t0
                        cited_ids: set[str] = set()
                        md_out = f"_Retrieval-only — {len(hits)} hits in {elapsed:.1f}s._"
                        if not hits:
                            md_out += "\n\nNo relevant chunks found."
                        record = {
                            "question": prompt,
                            "markdown": md_out,
                            "hits": [_serialize_hit(h, cited_ids) for h in hits],
                            "route_intent": "(skipped)",
                            "rewritten_query": prompt,
                            "confidence": "n/a",
                            "elapsed_s": elapsed,
                        }
                        st.markdown(md_out)

                    else:
                        try:
                            ans = synthesis.answer_question(
                                prompt,
                                top_k=config.RETRIEVAL_TOP_K,
                                use_pro_model=st.session_state["use_pro"],
                                skip_router=st.session_state["skip_router"],
                            )
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")
                            st.stop()
                        elapsed = time.perf_counter() - t0
                        if (
                            st.session_state["doc_type_filter"]
                            and ans.hits
                            and not any(
                                h.chunk.doc_type in st.session_state["doc_type_filter"]
                                for h in ans.hits
                            )
                        ):
                            st.info(
                                "Note: your doc-type filter excluded all retrievable chunks; "
                                "consider clearing the filter."
                            )
                        cited_ids = {c.chunk_id for c in ans.citations if c.verified}
                        record = {
                            "question": prompt,
                            "markdown": ans.to_markdown(),
                            "hits": [_serialize_hit(h, cited_ids) for h in ans.hits],
                            "route_intent": ans.route_decision.intent + (" (router skipped)" if st.session_state["skip_router"] else ""),
                            "rewritten_query": ans.route_decision.rewritten_query,
                            "confidence": ans.confidence,
                            "elapsed_s": elapsed,
                        }
                        st.markdown(ans.to_markdown())

                st.session_state["history"].append(record)

                if record.get("hits"):
                    with st.expander(f"📚 Sources ({len(record['hits'])} retrieved)", expanded=True):
                        for i, hit in enumerate(record["hits"], start=1):
                            c = hit["chunk"]
                            verified_marker = "✅" if hit.get("verified_cited") else "•"
                            header = f"{verified_marker} **[S{i}]** {c['doc_title']}"
                            if c["section_marker"]:
                                header += f" — {c['section_marker']}"
                            header += f" — ¶{c['paragraph_index']} — score {hit['score']:.4f}"
                            if c["source_url"]:
                                header += f" — [source]({c['source_url']})"
                            st.markdown(header)
                            st.code(c["text"], language="markdown")
                st.caption(
                    f"Routed as **{record['route_intent']}** "
                    f"(rewritten: _{record['rewritten_query']}_) "
                    f"• confidence: {record['confidence']} "
                    f"• took {record['elapsed_s']:.1f}s"
                )

st.divider()
st.caption(
    "⚠ This tool provides research, not legal advice. Verify every citation against "
    "the official source before any reliance. The corpus includes curated seeds + "
    "(optionally) IL-TUR (ACL 2024) cases & statutes."
)
