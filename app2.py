"""
Indian Legal RAG — Q&A only (seed + PCR + LSI corpus).
No CJPE / BAIL outcome prediction. Uses an independent ChromaDB at data/chroma_v2/.

Run:
    streamlit run app2.py
"""
from __future__ import annotations

import os
from pathlib import Path

# Point to the v2 corpus BEFORE importing any core module.
# This ensures config.py reads these values when it first loads.
_HERE = Path(__file__).resolve().parent
os.environ.setdefault("LEGAL_RAG_CHROMA_DIR", str(_HERE / "data" / "chroma_v2"))
os.environ.setdefault("LEGAL_RAG_CHROMA_COLLECTION", "indian_legal_v2")

import time

import streamlit as st

from core import config, index, synthesis
from core.retriever import default_retriever


st.set_page_config(
    page_title="Indian Legal RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("doc_type_filter", [])
    st.session_state.setdefault("use_pro", False)
    st.session_state.setdefault("inspect_only", False)
    st.session_state.setdefault("skip_router", False)


_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ Indian Legal RAG")
    st.caption("Citation-grounded answers · Gemini synthesis · Groq routing")

    try:
        n_chunks = index.count()
    except Exception as e:
        n_chunks = -1
        st.error(f"ChromaDB unavailable: {e}")

    if n_chunks == 0:
        st.warning(
            "Corpus is empty. Build it:\n\n"
            "```bash\npython scripts/build_v2.py\n```"
        )
    elif n_chunks > 0:
        st.success(f"{n_chunks:,} chunks indexed.")

    st.divider()
    st.subheader("Retrieval filters")
    st.session_state["doc_type_filter"] = st.multiselect(
        "Restrict to doc types",
        options=["constitution", "statute", "rule", "case"],
        default=[],
        help="Empty = no filter.",
    )

    st.divider()
    st.subheader("Synthesis model")
    st.session_state["use_pro"] = st.toggle(
        f"Use {config.GEMINI_HEAVY_MODEL} (slower, higher quality)",
        value=st.session_state["use_pro"],
        help=f"Default is {config.GEMINI_SYNTHESIS_MODEL}.",
    )

    st.session_state["inspect_only"] = st.toggle(
        "Inspect retrieval only (skip LLM)",
        value=st.session_state["inspect_only"],
        help="Returns retrieved chunks without calling any LLM. Free and instant.",
    )

    st.session_state["skip_router"] = st.toggle(
        "Skip router",
        value=st.session_state["skip_router"],
        help="Bypasses Groq query-rewriting. Saves one API call; uses raw question.",
    )

    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state["history"] = []
        st.rerun()

    st.divider()
    st.caption(
        "**Models**\n\n"
        f"- Router: `{config.GROQ_ROUTER_MODEL}`\n"
        f"- Synth (default): `{config.GEMINI_SYNTHESIS_MODEL}`\n"
        f"- Synth (heavy): `{config.GEMINI_HEAVY_MODEL}`\n"
        f"- Embeddings: local MiniLM-L6-v2"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Ask an Indian-law question")
st.caption(
    "Answers are grounded in the indexed corpus with verified [S#] citations. "
    "If the corpus doesn't cover a topic, the system says so rather than guessing."
)

# Example question buttons
example_questions = [
    "Is the right to privacy a fundamental right?",
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

# Render history
for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["markdown"])
        if turn.get("hits"):
            with st.expander(f"📚 Sources ({len(turn['hits'])} retrieved)", expanded=False):
                for i, hit in enumerate(turn["hits"], start=1):
                    c = hit["chunk"]
                    marker = "✅" if hit.get("verified_cited") else "•"
                    hdr = f"{marker} **[S{i}]** {c['doc_title']}"
                    if c["section_marker"]:
                        hdr += f" — {c['section_marker']}"
                    hdr += f" — ¶{c['paragraph_index']} — score {hit['score']:.4f}"
                    if c["source_url"]:
                        hdr += f" — [source]({c['source_url']})"
                    st.markdown(hdr)
                    st.code(c["text"], language="markdown")
        if turn.get("route_intent"):
            st.caption(
                f"Routed as **{turn['route_intent']}** "
                f"(rewritten: _{turn['rewritten_query']}_) "
                f"• confidence: {turn.get('confidence', 'n/a')} "
                f"• {turn.get('elapsed_s', 0):.1f}s"
            )


# ── Input & processing ────────────────────────────────────────────────────────

def _serialize_hit(hit, cited_ids: set) -> dict:
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
    if n_chunks <= 0:
        st.error(
            "Corpus is empty. Build it first:\n\n"
            "```bash\npython scripts/build_v2.py\n```"
        )
    else:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            t0 = time.perf_counter()

            if st.session_state["inspect_only"]:
                with st.spinner("Retrieving …"):
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
                md_out = f"_Retrieval-only — {len(hits)} hits in {elapsed:.1f}s._"
                if not hits:
                    md_out += "  \n_No relevant chunks found._"
                st.markdown(md_out)
                cited_ids: set = set()
                record = {
                    "question": prompt,
                    "markdown": md_out,
                    "hits": [_serialize_hit(h, cited_ids) for h in hits],
                    "route_intent": "(inspect only)",
                    "rewritten_query": prompt,
                    "confidence": "n/a",
                    "elapsed_s": elapsed,
                }

            else:
                with st.spinner("Routing → retrieving → synthesising …"):
                    ans = synthesis.answer_question(
                        prompt,
                        top_k=config.RETRIEVAL_TOP_K,
                        use_pro_model=st.session_state["use_pro"],
                        skip_router=st.session_state["skip_router"],
                    )
                elapsed = time.perf_counter() - t0

                if (
                    st.session_state["doc_type_filter"]
                    and ans.hits
                    and not any(
                        h.chunk.doc_type in st.session_state["doc_type_filter"]
                        for h in ans.hits
                    )
                ):
                    st.info("Your doc-type filter excluded all retrievable chunks — consider clearing it.")

                st.markdown(ans.to_markdown())
                cited_ids = {c.chunk_id for c in ans.citations if c.verified}
                record = {
                    "question": prompt,
                    "markdown": ans.to_markdown(),
                    "hits": [_serialize_hit(h, cited_ids) for h in ans.hits],
                    "route_intent": ans.route_decision.intent
                        + (" (router skipped)" if st.session_state["skip_router"] else ""),
                    "rewritten_query": ans.route_decision.rewritten_query,
                    "confidence": ans.confidence,
                    "elapsed_s": elapsed,
                }

            st.session_state["history"].append(record)

            if record.get("hits"):
                with st.expander(
                    f"📚 Sources ({len(record['hits'])} retrieved)", expanded=True
                ):
                    for i, hit in enumerate(record["hits"], start=1):
                        c = hit["chunk"]
                        marker = "✅" if hit.get("verified_cited") else "•"
                        hdr = f"{marker} **[S{i}]** {c['doc_title']}"
                        if c["section_marker"]:
                            hdr += f" — {c['section_marker']}"
                        hdr += f" — ¶{c['paragraph_index']} — score {hit['score']:.4f}"
                        if c["source_url"]:
                            hdr += f" — [source]({c['source_url']})"
                        st.markdown(hdr)
                        st.code(c["text"], language="markdown")

            st.caption(
                f"Routed as **{record['route_intent']}** "
                f"(rewritten: _{record['rewritten_query']}_) "
                f"• confidence: {record['confidence']} "
                f"• {record['elapsed_s']:.1f}s"
            )

st.divider()
st.caption(
    "⚠ Research tool only — not legal advice. "
    "Verify every citation against the official source before relying on it."
)
