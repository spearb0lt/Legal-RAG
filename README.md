# Indian Legal RAG — Citation-grounded chatbot

A Streamlit-hosted Retrieval-Augmented Generation app over Indian statutes + cases that **forces every answer to carry verifiable paragraph-level citations** to the indexed corpus.

Built to address a gap surfaced by IL-TUR (ACL 2024): frontier LLMs underperform Indian-domain SOTA models on retrieval-heavy Indian legal tasks. This project pairs an Indian-legal corpus with hybrid retrieval (vector embeddings + BM25, RRF-fused) and a synthesis step that refuses to make claims its retrieval didn't ground.

The system has two distinct modes:
- **Q&A mode** — ask questions about Indian statutes, cases, and doctrines; get cited answers
- **Case Outcome Prediction mode** — describe your case; find similar past cases; see their verdicts + an assessment of what you might expect

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Stack](#stack)
3. [Setup](#setup)
4. [Corpus & Datasets](#corpus--datasets)
   - [Seed corpus (local)](#1-seed-corpus-local)
   - [PCR — Prior Case Retrieval](#2-il-tur-pcr--prior-case-retrieval)
   - [LSI — Legal Statute Identification](#3-il-tur-lsi--legal-statute-identification)
   - [CJPE — Court Judgment Prediction with Explanation](#4-il-tur-cjpe--court-judgment-prediction-with-explanation)
   - [BAIL — Bail Prediction](#5-il-tur-bail--bail-prediction)
5. [Case Outcome Prediction feature](#case-outcome-prediction-feature)
6. [Build commands](#build-commands)
7. [Embedding model options](#embedding-model-options)
8. [Query router — what it does & when to skip it](#query-router)
9. [Streamlit UI](#streamlit-ui)
10. [Evaluation (PCR benchmark)](#evaluation)
11. [Adding your own corpus](#adding-your-own-corpus)
12. [Repository layout](#repository-layout)
13. [Disclaimers](#disclaimers)

---

## How it works

Given a question about Indian law, the pipeline runs four stages:

```
User question
      │
      ▼
┌─────────────┐    Groq Llama 3.3 70B    ┌──────────────────────┐
│   ROUTER    │ ─────────────────────▶   │  RouteDecision       │
└─────────────┘                          │  intent, doc filter, │
                                         │  rewritten query     │
                                         └──────────┬───────────┘
                                                    │
                                                    ▼
                                      ┌─────────────────────────┐
                                      │   HYBRID RETRIEVER      │
                                      │  vector (Chroma) + BM25 │
                                      │  fused by RRF (k=60)    │
                                      └──────────┬──────────────┘
                                                 │ top-8 chunks
                                                 ▼
                                      ┌─────────────────────────┐
                                      │   SYNTHESIZER           │
                                      │  Gemini 2.5 Flash/Pro   │
                                      │  forced JSON + citation │
                                      │  verification           │
                                      └──────────┬──────────────┘
                                                 │
                                                 ▼
                                      Answer with [S#] citations
                                      each marked verified / unverified
```

1. **Route** — Groq classifies intent (`statute_lookup` / `case_research` / `legal_concept` / `procedure` / `general_legal_qa`) and rewrites the query for better retrieval. Out-of-scope queries are rejected early.
2. **Retrieve** — Hybrid: top-k vector search (Chroma) + top-k BM25 (`rank_bm25`), scores combined via Reciprocal Rank Fusion. Optional `doc_type_filter` (statute / case / rule / constitution).
3. **Synthesize** — Gemini sees the question + numbered `[S#]` source paragraphs and must output a JSON object: answer text, citations (each with `chunk_id` + supporting quote), confidence, and caveats. Every `chunk_id` is cross-checked against the retrieved set — citations pointing to chunks that weren't retrieved are flagged **[unverified]** in the UI.
4. **Display** — Streamlit shows the grounded answer, a collapsible Sources panel, and per-turn routing/timing telemetry.

The pipeline never falls back to ungrounded model knowledge. If retrieval returns nothing, it says so.

---

## Stack

| Layer | Choice |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io) — `streamlit run app.py` |
| **Synthesis LLM** | Google **Gemini 2.5 Flash** (default) / **Gemini 2.5 Pro** (heavy) — `google-genai` SDK |
| **Router LLM** | Groq **Llama 3.3 70B Versatile** — fast, cheap, structured JSON |
| **Embeddings** | 4 backends — see [Embedding model options](#embedding-model-options) |
| **Vector DB** | [ChromaDB](https://www.trychroma.com) — local persistent, cosine similarity |
| **Lexical retrieval** | `rank_bm25` (BM25Okapi) over in-memory chunk corpus |
| **Fusion** | Reciprocal Rank Fusion (Cormack et al., 2009), k=60 |
| **Datasets** | Curated seeds in `data/raw/` + IL-TUR (ACL 2024, `Exploration-Lab/IL-TUR`) |
| **Eval** | IL-TUR Prior Case Retrieval (Recall@1/5/10, MRR@10) |

---

## Setup

```bash
cd legal-rag

# 1. Create virtual env
python -m venv .venv
. .venv/Scripts/activate          # Windows
# . .venv/bin/activate            # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure .env (copy and fill in)
cp .env.example .env              # or edit .env directly
# Required keys:
#   GOOGLE_API_KEY     — for Gemini synthesis (and embeddings if LEGAL_RAG_EMBEDDER=gemini)
#   GROQ_API_KEY       — for Llama 3.3 70B routing
# Optional but recommended:
#   HF_TOKEN           — for authenticated IL-TUR downloads (faster, avoids rate limits)

# 4. Build the corpus (see Build commands section)
python -m data.ingest.build_all --il-tur        # recommended starting point

# 5. Run the app
streamlit run app.py
# On Windows, you can also double-click run.bat (auto-kills port conflicts)
```

Open the URL Streamlit prints (default: http://localhost:8501).

> **Windows tip:** If the app closes immediately or you see a port conflict error, run `run.bat` from the `legal-rag/` directory — it automatically kills any lingering process on port 8501 before starting. Always keep the terminal window open while using the app; closing it stops the server.

---

## Corpus & Datasets

The index is built from multiple sources, each contributing different types of legal knowledge. You control which sources are included at build time.

### 1. Seed corpus (local)

**Location:** `data/raw/statutes/` and `data/raw/cases/`
**Size:** ~35 chunks (small, high-quality summaries)
**Language:** English
**Always included:** Yes (loaded by every `build_all` invocation)

Curated summaries of landmark Indian legal documents — key constitutional provisions, landmark Supreme Court cases, and fundamental statutes. These are deliberately short and high-precision: they exist to ensure the corpus isn't empty and to anchor retrieval for the most commonly asked questions.

> **Note:** Several seed files are SUMMARIES, not full bare-act text. The `SOURCE_NOTE` field in each file marks the quality. For production-grade retrieval, replace these with full statute text from `indiacode.nic.in`.

**Good for:**
- The Indian Constitution (fundamental rights, DPSP, basic structure doctrine)
- Landmark Supreme Court judgments (Kesavananda Bharati, Maneka Gandhi, Vishaka, etc.)
- Core statutes (IPC, CrPC, Consumer Protection Act)

**Example questions:**
- *"Is the right to privacy a fundamental right under the Indian Constitution?"*
- *"What is the basic structure doctrine and which case established it?"*
- *"What are the Vishaka guidelines for sexual harassment at the workplace?"*
- *"Explain the triple test for Article 21 from Maneka Gandhi v. Union of India."*
- *"What does Article 32 say about the right to constitutional remedies?"*

---

### 2. IL-TUR PCR — Prior Case Retrieval

**Source:** `Exploration-Lab/IL-TUR` on HuggingFace, task=`pcr`
**Available:** 7,070 candidate cases (train 4,320 + dev 1,023 + test 1,727)
**Default indexed:** 300 (configurable with `--pcr-max`)
**Language:** English
**Chunk size:** ~19 chunks per case (full judgment text, paragraph-split)
**Build flag:** `--il-tur` (includes PCR automatically)

Full Supreme Court of India judgment texts from the PCR task of IL-TUR (ACL 2024). These are the "candidate" cases that appear as retrieval targets in the Prior Case Retrieval benchmark — actual court judgments covering a wide range of civil, criminal, and constitutional matters from across Indian legal history.

**What is indexed vs skipped:**
- ✅ Indexed: `*_CANDIDATES` splits — full judgment texts (corpus docs)
- ❌ Skipped: `*_QUERIES` splits — short query case excerpts used for evaluation only

**Good for:**
- Finding precedent cases on specific legal issues
- Research into Supreme Court reasoning patterns
- Questions about how courts have interpreted specific statutes
- Civil, criminal, and constitutional case outcomes

**Example questions:**
- *"What did the Supreme Court hold about the scope of Article 142?"*
- *"Find cases where the court interpreted 'public purpose' under land acquisition law."*
- *"What is the precedent on bail in cases involving non-bailable offences?"*
- *"How has the Supreme Court interpreted Section 302 IPC in cases of circumstantial evidence?"*
- *"What cases deal with contempt of court and the limits of free speech?"*
- *"Find precedents on the doctrine of res judicata in civil suits."*

**To index more PCR cases:**
```bash
python -m data.ingest.build_all --il-tur --pcr-max 1000   # 1000 cases
python -m data.ingest.build_all --il-tur --pcr-max 7070   # full corpus (~7K cases)
```

---

### 3. IL-TUR LSI — Legal Statute Identification

**Source:** `Exploration-Lab/IL-TUR` on HuggingFace, task=`lsi`
**Available:** 100 Indian statute texts (`statutes` split)
**Default indexed:** 100 (complete — the full statute set)
**Language:** English
**Build flag:** `--il-tur` (includes LSI automatically)

The 100 actual Indian statute texts from the LSI task. The LSI benchmark tests a model's ability to identify which statute section a case passage invokes — but the actual statute texts (the "documents") are what we index, not the query passages.

**What is indexed vs skipped:**
- ✅ Indexed: `statutes` split — 100 actual Indian statute texts
- ❌ Skipped: `train`, `dev`, `test` splits — these are case passage QUERIES with anonymized `<SECTION>/<ACT>` placeholders, meaningless for retrieval

**Good for:**
- Statute interpretation questions
- Finding specific provisions within Indian acts
- Questions like "what does Section X of Act Y say?"
- Comparing provisions across related statutes

**Example questions:**
- *"What does the Indian Evidence Act say about admissibility of confessions?"*
- *"What are the offences under the Prevention of Corruption Act?"*
- *"Explain the provisions of the Arbitration and Conciliation Act on interim relief."*
- *"What does the Motor Vehicles Act say about compensation for accident victims?"*
- *"What are the punishments under the NDPS Act for drug trafficking?"*
- *"What is the limitation period for filing a civil suit under the Limitation Act?"*

---

### 4. IL-TUR CJPE — Court Judgment Prediction with Explanation

**Source:** `Exploration-Lab/IL-TUR` on HuggingFace, task=`cjpe`
**Available:** ~42,465 cases across all splits
  - expert: 56 | single_train: 5,082 | single_dev: 2,511
  - multi_train: 32,305 | multi_dev: 994 | test: 1,517
**Default indexed:** 500 (configurable with `--cjpe-max`)
**Language:** English
**Build flag:** `--cjpe`

Full court judgment texts with binary outcome labels (accepted/rejected) and expert-annotated explanation sentences. CJPE is the richest source for understanding judicial reasoning — each document contains the facts, arguments, and the court's reasoning. Expert annotations (`expert_1` through `expert_5`) mark the most legally significant sentences in each judgment.

**What we index:** The full `text` field (complete judgment text). The label and expert annotations are metadata, not indexed content.

**Good for:**
- Questions about judicial reasoning and outcome prediction
- Understanding the factors that led to a specific type of ruling
- Research into how courts decide criminal, civil, and constitutional matters
- Finding cases where specific arguments were accepted or rejected
- Legal research requiring deep understanding of judgment logic

**Example questions:**
- *"What factors do courts consider when assessing whether a civil appeal has merit?"*
- *"How do courts decide landlord-tenant disputes under the Rent Control Act?"*
- *"What reasoning do courts use in murder cases with only circumstantial evidence?"*
- *"In what circumstances will a court reverse a lower court's factual findings?"*
- *"What are the grounds on which courts typically dismiss criminal appeals?"*
- *"How has the judiciary interpreted 'grave and sudden provocation' as a defence?"*

**To index more CJPE cases (CJPE has ~42K available):**
```bash
python -m data.ingest.build_all --cjpe --cjpe-max 2000
```

---

### 5. IL-TUR BAIL — Bail Prediction

**Source:** `Exploration-Lab/IL-TUR` on HuggingFace, task=`bail`
**Available:** ~336,849 bail applications across all splits
  - train_all: 123,742 | dev_all: 17,707 | test_all: 35,400
  - train_specific: 124,341 | dev_specific: 15,929 | test_specific: 36,579
**Default indexed:** 500 (configurable with `--bail-max`)
**Language:** Primarily Hindi (Devanagari script)
**Build flag:** `--bail`

Bail application texts from Indian district courts, primarily from Uttar Pradesh (districts like Agra, Lucknow, etc.). Each record contains the facts and arguments presented in the bail application, the judge's opinion, and a binary label (bail granted/denied).

**Important language note:** BAIL data is in Hindi. English-only embedding models (like MiniLM-L6-v2) will produce suboptimal cross-lingual retrieval results — an English question will not retrieve Hindi documents as effectively as it would English ones. For best results with BAIL, use a multilingual embedding model.

**Text structure:** Each bail record's `text` is a dict:
```json
{
  "facts-and-arguments": ["sentence 1", "sentence 2", ...],
  "judge-opinion": ["sentence 1", ...]
}
```

**Good for:**
- Research into bail conditions and how courts decide bail in Indian criminal courts
- Understanding bail decisions under specific offences (theft, assault, murder, etc.)
- Hindi-language legal queries about bail procedure
- Comparative research on bail outcomes across districts

**Example questions (best in Hindi or with multilingual embedder):**
- *"Under what circumstances is bail typically denied in assault cases in UP courts?"*
- *"What factors does the judge consider for bail in non-bailable offences?"*
- *"What is the typical bail condition for property-related offences?"*
- *"How do courts assess flight risk when deciding bail applications?"*

---

### Dataset summary

| Dataset | Source | Size available | Default indexed | Language | Added by |
|---|---|---|---|---|---|
| Seed corpus | `data/raw/` | ~35 chunks | All | English | Always |
| PCR cases | IL-TUR `pcr` | 7,070 cases | 300 | English | `--il-tur` |
| LSI statutes | IL-TUR `lsi` | 100 statutes | 100 | English | `--il-tur` |
| CJPE cases | IL-TUR `cjpe` | ~42,465 cases | 500 | English | `--cjpe` |
| BAIL applications | IL-TUR `bail` | ~336,849 apps | 500 | Hindi | `--bail` |

---

## Case Outcome Prediction feature

This is the second primary mode of the application. Instead of asking a legal question, you describe your case — and the system finds the most similar historical cases in the corpus, shows their actual verdicts, and provides an evidence-grounded assessment of what you might expect.

### How it works

```
Your case description
        │
        ▼
┌──────────────────────────────┐
│  Filtered vector search       │
│  source_task = "cjpe"/"bail"  │  ← only searches this dataset's chunks
└──────────────┬───────────────┘
               │ top-12 similar chunks
               ▼
┌──────────────────────────────┐
│  Deduplication by case        │  ← one entry per unique case (best match)
│  Outcome extraction from      │  ← reads stored 0/1 label: granted/denied
│  ChromaDB metadata            │
└──────────────┬───────────────┘
               │ similar cases list + outcome counts
               ▼
┌──────────────────────────────┐
│  Gemini analysis              │
│  • summary of similar cases  │
│  • key deciding factors       │
│  • your likely outcome        │
│  • confidence + caveats       │
└──────────────────────────────┘
```

Each CJPE and BAIL chunk has its `outcome` label (0 or 1) stored directly in ChromaDB metadata, so retrieval and outcome lookup happen in one step with no extra API calls.

### CJPE — Court Judgment Prediction

**What it does:** Find similar Supreme Court of India judgment cases and see whether the court accepted or rejected the petition.

**Label meaning:**
- `1` = Judgment Accepted (court ruled in favor of petitioner / appeal succeeded)
- `0` = Judgment Rejected (court ruled against petitioner / appeal dismissed)

**When to use:** You have a civil or criminal appeal or petition and want to know how courts have ruled on similar matters.

**How to describe your case:**
> Be specific. Include: type of matter (civil appeal, criminal appeal, writ petition), the core legal issue, the lower court's decision, and the primary argument. The more specific you are, the more relevant the retrieved cases will be.

**Example inputs:**
```
I filed a civil appeal against a High Court ruling that dismissed my tenancy 
rights under the Rent Control Act. The lower court held that the landlord was 
entitled to evict me for personal use. I have been a tenant for 22 years with 
no default in rent payment.
```
```
This is a criminal appeal against conviction under IPC Section 302. The 
prosecution relied entirely on circumstantial evidence. There was no 
eyewitness. The accused was allegedly identified by a single witness who 
saw him near the scene two hours before the incident.
```
```
Writ petition challenging the termination of a government employee on 
grounds of misconduct. The inquiry was conducted without giving adequate 
opportunity to cross-examine witnesses. No formal chargesheet was served.
```

**Example questions for Q&A mode using CJPE knowledge:**
- *"What factors do courts consider when assessing whether a civil appeal has merit?"*
- *"How do courts decide landlord-tenant disputes under rent control law?"*
- *"What reasoning do courts use in murder cases with only circumstantial evidence?"*
- *"In what circumstances will a court reverse a lower court's factual findings?"*

---

### BAIL — Bail Prediction

**What it does:** Find similar bail applications from Indian district courts and see whether bail was granted or denied.

**Label meaning:**
- `1` = Bail Granted
- `0` = Bail Denied

**Language note:** ⚠️ The BAIL corpus is in **Hindi (Devanagari script)**. The bail applications were filed in UP district courts. If you describe your case in English, the vector search will retrieve Hindi documents — the semantic match will be weaker than for CJPE. Retrieval accuracy improves if you use a multilingual embedding model.

**When to use:** You or someone you know has been arrested and is considering a bail application. You want to know how courts have decided bail in similar criminal matters.

**How to describe your case:**
> Include: the offence(s) charged (IPC/special act section numbers), custody duration, prior criminal record, seriousness of the alleged offence, personal circumstances (family, employment), and the court district if known.

**Example inputs:**
```
Arrested under IPC Section 307 (attempt to murder) during a property dispute. 
In custody for 45 days. No prior criminal record. Sole breadwinner for family 
of 4. Co-accused already granted bail. Matter is from Agra district.
```
```
Accused under IPC Section 420 (cheating) and 406 (criminal breach of trust) 
in a business dispute. Charge is for alleged fraud of Rs. 5 lakhs. In custody 
for 20 days. No flight risk. Has permanent residence and business in the city.
```

**Example questions for Q&A mode using BAIL knowledge:**
- *"What factors do courts consider when deciding bail in non-bailable offences?"*
- *"Under what circumstances is bail typically denied in assault cases?"*
- *"What conditions are commonly imposed when granting bail for property offences?"*

---

### Limitations and honest caveats

| Limitation | Detail |
|---|---|
| Corpus size | Only 500 CJPE + 500 BAIL cases indexed by default (out of 42K+ and 336K+ available). More cases → better predictions. Scale with `--cjpe-max` / `--bail-max`. |
| Not a predictor | The system finds SIMILAR cases — it does not run a trained classifier. Use it for research, not for certainty. |
| BAIL is in Hindi | English queries find Hindi documents imperfectly. A multilingual embedder improves this. |
| Labels are binary | Courts weigh many factors. A 60/40 split in similar cases means the outcome is genuinely uncertain. |
| Not legal advice | Always consult a lawyer for actual legal matters. |

---

## Build commands

```bash
# Seed corpus only (35 chunks, always fast)
python -m data.ingest.build_all

# Recommended starting point: seed + PCR 300 + LSI 100
python -m data.ingest.build_all --il-tur

# Add court judgment reasoning (CJPE) — good for judgment analysis questions
python -m data.ingest.build_all --il-tur --cjpe

# Add bail applications (BAIL, Hindi) — for bail-specific research
python -m data.ingest.build_all --il-tur --bail

# Full recommended corpus (PCR+LSI+CJPE+BAIL)
python -m data.ingest.build_all --il-tur --cjpe --bail

# Scale up PCR (more precedent cases)
python -m data.ingest.build_all --il-tur --pcr-max 1000 --cjpe --bail

# Full PCR corpus (~7K cases — takes 10-20 min depending on embedder)
python -m data.ingest.build_all --il-tur --pcr-max 7070 --cjpe --bail

# Wipe and rebuild from scratch (required when switching embedding models)
python -m data.ingest.build_all --reset --il-tur --cjpe --bail

# Custom limits
python -m data.ingest.build_all --il-tur --pcr-max 500 --lsi-max 100 --cjpe --cjpe-max 2000 --bail --bail-max 1000
```

> **After switching embedding models, always use `--reset`.** ChromaDB stores embedding vectors alongside documents. If you switch from `gemini` (768-dim) to `local/MiniLM` (384-dim), inserting into an existing collection causes a dimension mismatch error.

---

## Embedding model options

Set `LEGAL_RAG_EMBEDDER` in your `.env` file to choose the backend:

| Backend | `.env` setting | Dim | Speed | Notes |
|---|---|---|---|---|
| **Gemini** (default) | `LEGAL_RAG_EMBEDDER=gemini` | 768 | Fast (API) | Requires `GOOGLE_API_KEY`. Best overall quality for English legal text. |
| **Local MiniLM** | `LEGAL_RAG_EMBEDDER=local` + `LEGAL_RAG_LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2` | 384 | ~5s/batch (CPU) | No API needed. 22M params. Good quality, fast on CPU. |
| **Local BGE** | `LEGAL_RAG_EMBEDDER=local` + `LEGAL_RAG_LOCAL_EMBED_MODEL=BAAI/bge-base-en-v1.5` | 768 | ~45s/batch (CPU) | 110M params. Higher quality than MiniLM but much slower on CPU. |
| **Voyage AI** | `LEGAL_RAG_EMBEDDER=voyage` | 1024 | Fast (API) | `voyage-law-2` is specifically fine-tuned for legal text. Requires `VOYAGE_API_KEY`. Best for legal domain. |
| **HF Inference API** | `LEGAL_RAG_EMBEDDER=hf_api` | varies | Free/slow | HuggingFace free tier API. Rate-limited. |

**Example `.env` for local embedding (no API):**
```
LEGAL_RAG_EMBEDDER=local
LEGAL_RAG_LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Example `.env` for Voyage AI (best legal quality):**
```
LEGAL_RAG_EMBEDDER=voyage
VOYAGE_API_KEY=pa-...
LEGAL_RAG_VOYAGE_MODEL=voyage-law-2
```

> **Where are embeddings stored?** Vectors are stored inside ChromaDB at `data/chroma/`. If you switch embedding models, run `build_all --reset` to wipe and rebuild — you cannot mix vectors from different models in the same collection. The HuggingFace datasets themselves are cached separately at `~/.cache/huggingface/hub/`.

---

## Query router

The router (Groq Llama 3.3 70B) runs before every query. It does three things:

1. **Intent classification** — decides what kind of query it is:
   - `statute_lookup` → looking for specific statute text
   - `case_research` → searching for precedent cases
   - `legal_concept` → asking about a doctrine or principle
   - `procedure` → asking about a legal procedure
   - `general_legal_qa` → general Indian law question
   - `out_of_scope` → not about Indian law — rejected with no retrieval

2. **Query rewriting** — rewrites the user's question into a retrieval-optimised form (e.g., removes filler words, adds key legal terms).

3. **Doc-type filtering** — sets a `doc_type_filter` (e.g., `statute` or `case`) based on intent, so retrieval is restricted to the relevant document type.

**Why the router helps:** A user asking *"What does Section 66A say?"* is clearly looking for statute text. Without the router, the retrieval would scan all doc types. With the router, it restricts to `doc_type=statute`, dramatically improving precision.

**When to skip the router:**
- When you're in a hurry and want to skip the Groq API call (~0.5s saved)
- When your question is already precise and doesn't need rewriting
- When you're debugging retrieval directly and don't want Groq to interfere
- When your question spans multiple doc types and you don't want auto-filtering

**Trade-offs of skipping the router:**

| With router | Without router |
|---|---|
| Query is rewritten for retrieval | Raw question is used as-is |
| Doc-type auto-filtering applied | No doc-type filter |
| Out-of-scope queries rejected early | All queries go to retrieval |
| +0.5–1s latency (Groq API call) | Faster |
| Groq API key consumed | No Groq usage |

**How to skip:** Toggle **"Skip router (use raw question)"** in the Streamlit sidebar, or call `answer_question(q, skip_router=True)` in code.

---

## Streamlit UI

```bash
streamlit run app.py
# Windows: double-click run.bat (handles port conflicts automatically)
```

**Sidebar controls:**

| Control | What it does |
|---|---|
| **Case Outcome Prediction** toggle | Switch between Q&A mode and Outcome Prediction mode |
| Dataset (in Outcome mode) | CJPE (English court judgments) or BAIL (Hindi bail applications) |
| Restrict to doc types | Filter retrieval to `statute`, `case`, `rule`, or `constitution` only (Q&A mode) |
| Use Gemini 2.5 Pro | Switch synthesis to the heavier Pro model (slower, higher quality) |
| Inspect retrieval only | Skip LLM synthesis entirely — just shows which chunks were retrieved (Q&A mode) |
| Skip router | Bypass Groq routing/rewriting — use raw question for retrieval (Q&A mode) |
| Clear chat history | Wipe the session |

**Reading the Q&A UI:**
- Each answer carries `[S1]`, `[S2]`, ... citation markers
- `✅` = citation verified (chunk_id matched a retrieved chunk)
- `•` = citation unverified (model cited a chunk_id it didn't actually have)
- The footer shows: intent, rewritten query, confidence, elapsed time

**Reading the Outcome Prediction UI:**
- `✅` = favorable outcome (judgment accepted / bail granted)
- `❌` = unfavorable outcome (judgment rejected / bail denied)
- Outcome statistics shown prominently: "X favorable / Y unfavorable"
- Assessment paragraph from Gemini below the statistics
- Similar cases expandable panel with actual case snippets

---

## Evaluation

The PCR evaluation benchmarks the retrieval system against IL-TUR's held-out queries:

```bash
# Run on 100 queries (fast, ~1-2 min)
python -m eval.il_tur_pcr

# Run on more queries for tighter confidence intervals
python -m eval.il_tur_pcr --n-queries 500

# Keep the eval Chroma collection after run (for inspection)
python -m eval.il_tur_pcr --keep-index
```

The script:
1. Builds an **isolated** Chroma collection from PCR candidate cases
2. Runs each held-out query case through the retriever
3. Computes **Recall@1**, **Recall@5**, **Recall@10**, and **MRR@10**
4. Writes a JSON summary + per-query breakdown to `eval/results/`

> **Fill in your results here after running:**
>
> ```
> Dataset:   Exploration-Lab/IL-TUR :: pcr
> Queries:   <N>
> Candidates: <M>
> Embedder:  <LEGAL_RAG_EMBEDDER value>
> Recall@1:  <fill>
> Recall@5:  <fill>
> Recall@10: <fill>
> MRR@10:    <fill>
> ```

---

## Adding your own corpus

Drop any `.txt` file into `data/raw/statutes/` or `data/raw/cases/` with this header:

```
TITLE: <human-readable title>
DOC_TYPE: <constitution | statute | rule | case>
JURISDICTION: IN
CITATION: <e.g. (2017) 10 SCC 1>
SOURCE_URL: <link to authoritative source>
SOURCE_NOTE: <optional note about verification>

<blank line>
<body text>
```

Then rebuild (no `--reset` needed; upsert is safe):
```bash
python -m data.ingest.build_all --il-tur --cjpe --bail
```

The chunker recognises Indian legal section markers (`Section`, `Article`, `(1)`, `Chapter`) and splits on paragraph boundaries with ~350-token target chunk size and 60-token overlap.

---

## Repository layout

```
legal-rag/
├── app.py                          # Streamlit chat UI
├── core/
│   ├── config.py                   # .env loading, paths, model names
│   ├── llm.py                      # Gemini + Groq clients with retries
│   ├── embeddings.py               # 4 embedding backends (gemini/local/voyage/hf_api)
│   ├── citation.py                 # Chunk dataclass + stable SHA1 chunk_id
│   ├── chunker.py                  # Paragraph-aware chunker with section markers
│   ├── index.py                    # ChromaDB persistent index (upsert, query)
│   ├── retriever.py                # Hybrid vector + BM25, RRF fusion
│   ├── prompts.py                  # Router few-shot + synthesis system prompt
│   ├── router.py                   # Groq classifier → RouteDecision
│   └── synthesis.py                # answer_question() — the end-to-end pipeline
├── data/
│   ├── raw/
│   │   ├── statutes/               # Curated statute summaries (add full text here)
│   │   └── cases/                  # Curated case summaries
│   ├── ingest/
│   │   ├── load_local.py           # Reads every .txt under data/raw/
│   │   ├── fetch_il_tur.py         # IL-TUR loader (PCR/LSI/CJPE/BAIL)
│   │   ├── scrape_indiankanoon.py  # ToS-safe stub (raises by default)
│   │   └── build_all.py            # CLI orchestrator
│   └── chroma/                     # Persistent vector store (auto-created)
├── eval/
│   ├── il_tur_pcr.py               # PCR benchmark: R@k, MRR
│   └── results/                    # JSON eval outputs
├── requirements.txt
├── .env                            # API keys (do not commit)
└── .gitignore
```

---

## Disclaimers

- **Not legal advice.** This is a research tool. Verify every citation against the official source before any reliance.
- **Curated statute seeds are SUMMARIES.** Several `data/raw/statutes/*.txt` files are short curated summaries (clearly marked with `SOURCE_NOTE`). For production use, replace with full bare-act text from `indiacode.nic.in`.
- **Indian Kanoon is intentionally not scraped.** See `data/ingest/scrape_indiankanoon.py` for the rationale and lawful alternatives.
- **BAIL dataset is in Hindi.** English-only embedding models will have limited effectiveness retrieving Hindi documents from English queries.

---

## License

MIT for the code. Dataset usage governed by upstream sources — `Exploration-Lab/IL-TUR` is CC-BY-NC-SA per the IL-TUR paper (ACL 2024, arXiv:2407.05399); check before redistribution.
