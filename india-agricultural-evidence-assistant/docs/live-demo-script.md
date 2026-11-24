# 20-minute technical demo: India Agricultural Evidence Assistant

This run of show is designed for agricultural researchers, faculty, and technical
leaders. The app is a local evidence-discovery teaching prototype, not an official
advisory service. Its bundled synthetic documents are test fixtures and must never be
used to make farming decisions. Curator-written summaries and transcribed figures must
be checked against their linked original sources before external use.

## Before the session

### Prepare and rehearse

From the project directory, install the app, prepare the local embedding model, build
the index, and start Streamlit:

```sh
pip install -e '.[dev]'
python -m app.prepare_models
python -m app.ingest --input data/demo --input data/raw --rebuild
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://127.0.0.1:8502/` and rehearse the exact search queries below. The default
`LLM_PROVIDER=none` is intentional and reliable: it returns a cited evidence excerpt
without calling a generation model. Dense retrieval still requires the prepared local
embedding model. Do not rebuild or clear the index during the presentation.

If you plan to demonstrate the optional Agentic RAG workflow, configure and warm a
working Ollama or OpenAI-compatible provider in advance, select it under **Advanced
answer settings**, and rehearse the Agentic scenario. The planner and, when the evidence
gate passes, the answer synthesis make real LLM calls.

Have the README, source registry, and evaluation methodology available as backup.
Prefer a second browser tab for the original source links. Keep the sidebar scope filters
visible and reset them between queries.

## Run of show

| Time | Segment | Actions and suggested narration |
|---|---|---|
| 0:00–1:30 | Frame the problem | **Say:** “Agricultural researchers often need to answer a focused question from reports, technical references, and structured observations, while preserving where each claim came from. This prototype explores that evidence workflow. It is not a farm-advisory chatbot: it does not browse for current guidance or replace local agronomic expertise.” Identify the primary users as agricultural researchers, extension researchers, and faculty exploring evidence-backed research workflows. |
| 1:30–3:00 | Orient the audience | Show the workbench header and indexed-document, searchable-passage, official-source-summary, and synthetic-fixture counts. Point to the selected framework, vector backend, and embedding model, then the crop, state, district, season, and source-status filters. Explain that filters come from indexed metadata. **Transition:** “Let’s start with a question where the answer is not just text, but traceable evidence.” |
| 3:00–7:00 | Hybrid text retrieval and evidence trace | Load **05 · Rice soil evidence** or enter: **“Which soil is good for rice crop?”** Keep **Automatic routing** and the Rice filter. Run it; review the answer and **Claims & citations**, then open **Sources & evidence** and **Retrieval trace**. The IRRI summary contrasts about 400 mm of seasonal water input for heavy clay with shallow groundwater and over 2,000 mm for coarse sandy or loamy soil with deep groundwater. Emphasize that this is a water-input comparison, not a yield ranking, universal soil recommendation, or India-specific advice. In the trace, point out dense embedding search, BM25 lexical search, reciprocal-rank fusion, reranker state, and evidence-gate decision. **Say:** “Retrieval scores help us inspect the pipeline; they are not calibrated answer confidence.” |
| 7:00–10:00 | Automatic routing to structured evidence | Load **02 · Participation evidence** or enter: **“How many farmers attended the Bloomsdale rice field day, and how many were women?”** Keep **Automatic routing**. Show that this count question takes the ICAR extension-indicator path, not vector retrieval. The result should show 43 participants, 32 women, 11 men, and a deterministically calculated 74.4% women share, with a link to the original ICAR page and a caveat. **Say:** “The arithmetic is deterministic; the input figures are transcribed in a curator-written summary and still need source verification before reuse.” **Transition:** “A trustworthy research tool also needs to make its limits visible.” |
| 10:00–12:30 | Abstention and conflict handling | Load **04 · Conflict handling** or enter: **“Which irrigation interval should cotton farmers in Maharashtra follow during kharif according to the demo notes?”** The seeded synthetic notes disagree; the expected behavior is abstention, not choosing one. Show the explanation and missing/context evidence indicators. If time permits, clear all scope filters and ask **“What should I spray to control pests in my farm?”** It should abstain and identify missing context rather than inventing pesticide advice. State that gate thresholds and conflict heuristics are demonstration mechanisms, not scientifically calibrated agronomic safety guarantees. |
| 12:30–15:00 | Structured statistics and provenance | Open **Structured evidence explorer · deterministic queries** and select **DES district comparison**. Show the Punjab wheat/Rabi 2022–2023 records, chart/table, report-portal link, row hashes, and caveat. Explain that these values came from a visible report view, the hash is over a normalized row rather than a downloaded DES file, and the report view did not state estimate status. They are historical descriptive data, not forecasts or recommendations; verify against an official export before reuse. |
| 15:00–17:00 | Optional bounded Agentic RAG | **Only if a provider is already configured and rehearsed:** load **06 · Agentic research brief**, confirm **Agentic research brief** is selected, and run it. Show the visible plan queries, each local hybrid-search step, merged evidence, and citation-validated result. Explain the bounds: the original question always runs, the planner may add at most two short queries, and the agent searches only the local corpus; it cannot browse the web, execute arbitrary code, or modify data. **If no provider is available:** do not attempt a live model call. Show the Agentic workflow selection and its explicit provider warning, then explain the bounded design from the app and README. |
| 17:00–19:00 | Corpus registry and evaluation | Expand **Corpus registry & data boundaries** to distinguish indexed source summaries from synthetic fixtures and show source metadata/provenance. Then open **Evaluation & quality signals**. If a report is already available, point out retrieval recall@5/nDCG@5, citation coverage, abstention precision, and source-ID validity. If not, explain that the suite is run on demand over a small curated set; do not start a potentially slow evaluation during the talk. **Say:** “These are regression and diagnostic signals, not proof of field accuracy or safety.” |
| 19:00–20:00 | Close | Recap the engineering point: “The prototype combines hybrid retrieval for documents, a separate deterministic path for numeric records, visible provenance, and explicit abstention. The next step toward responsible use is not a more confident model; it is better verified, current, local evidence and review by domain experts.” Invite questions. |

**Timing:** 1:30 + 1:30 + 4:00 + 3:00 + 2:30 + 2:30 + 2:00 + 2:00 + 1:00 = **20 minutes**.

## Opening explanation

> “The India Agricultural Evidence Assistant is a local research workbench for exploring
> agricultural evidence. It is aimed at researchers, extension teams, and faculty who
> need to locate relevant passages, inspect provenance, and distinguish document
> retrieval from numeric analysis. The engineering problem is that a plausible answer is
> not enough: we need to see what supports it, know when the evidence is missing or
> conflicting, and avoid turning incomplete research material into field advice.”

Then set the boundary plainly: the prototype does not independently verify the
authenticity, currentness, or agronomic correctness of sources, and the synthetic fixtures
are not agricultural facts.

## Technology stack

- **Python 3.11–3.13** for ingestion, retrieval, routing, safety checks, and evaluation.
- **Streamlit** for the local interactive workbench.
- **PyMuPDF** for PDF text extraction with page references; Markdown and UTF-8 text are
  also supported. Neighboring metadata JSON files carry source details and provenance.
- **Sentence Transformers / Hugging Face** for local multilingual embeddings.
- **Chroma** as the default persistent local vector store, integrated with
  **LlamaIndex** for dense indexing and retrieval.
- **BM25** (`rank-bm25`) as an independent lexical retrieval path; reciprocal-rank fusion
  combines lexical and dense rankings, with additional weight on lexical ranks.
- **Optional cross-encoder** reranking; it is disabled unless configured and available.
- **Optional Ollama or OpenAI-compatible LLM** for grounded synthesis and the bounded
  Agentic RAG planner. In extractive mode, no answer-generation LLM is called.
- **CSV-backed deterministic structured queries** for ICAR extension indicators and the
  DES district extract. These numeric records bypass vector search and model arithmetic.
- **Optional integrations:** a LangChain dense-retrieval adapter, pgvector storage, and
  Ragas diagnostic evaluation. They are not required for the default demo.

## Architecture and end-to-end flow

### Ingestion path

1. A curator supplies PDFs, Markdown, or UTF-8 text, plus metadata sidecars where needed.
2. The ingestion code extracts text and PDF page numbers, records extraction warnings and
   source metadata, and computes a SHA-256 for each input file.
3. Heading- and paragraph-aware chunking creates stable, searchable passages.
4. The local embedding model encodes passages into vectors; Chroma persists vectors,
   passage text, and metadata. BM25 ranks lexical matches over the indexed corpus.
5. An ingestion manifest records document/chunk counts and failures. Model downloads are
   an explicit preparation step, not a hidden app-startup action.

### User query: text evidence

1. The user selects a workflow and optional crop, geography, season, and source-status
   filters. Automatic routing sends recognized counts and statistics questions down the
   structured paths; ordinary research questions use text retrieval.
2. The app encodes the question for dense vector retrieval and separately runs BM25.
   Some rice/soil wording receives a small, domain-specific query expansion.
3. Reciprocal-rank fusion merges the candidate lists. An optional cross-encoder can
   rerank; if it is not configured, fused ranking is used directly.
4. Contextual checks and an evidence gate inspect support, question-term coverage,
   requested topic, relevant crop/geography/season context, and (for current-advisory
   requests) source status and validity dates. A failed gate abstains instead of asking
   the answer model to fill gaps.
5. With `LLM_PROVIDER=none`, the app returns an extractive evidence excerpt. With an
   optional provider, the prompt treats retrieved text as untrusted data and requests a
   cited answer.
6. Generated citation source IDs and pages are checked against retrieved chunks before
   display. The UI separates claims/citations, source passages, and retrieval diagnostics.

### Structured and Agentic paths

- **Structured indicators and DES statistics:** routing or the structured explorer
  filters validated local CSV records directly. Calculations such as the reported women
  share are deterministic; the app displays original-source links, caveats, and
  provenance. No vector search or LLM calculation is used for these numeric records.
- **Agentic RAG:** requires a configured LLM. A planner may return up to two short
  follow-up searches; the original question always runs, for no more than three
  read-only hybrid searches against the local index. Evidence is merged and passes
  through the same gate and citation validation. This is bounded planning, not an
  autonomous web-browsing or code-execution agent.

## Search queries to rehearse

| Query | What it demonstrates | Expected behavior / caveat |
|---|---|---|
| **Which soil is good for rice crop?** | Rice/soil query expansion, hybrid retrieval, citations, source cards, and trace | Retrieves the IRRI technical summary: about 400 mm versus over 2,000 mm seasonal water input under the described soil/groundwater conditions. It does not establish a universally best soil, compare yield, or give India-specific advice. |
| **How many farmers attended the Bloomsdale rice field day, and how many were women?** | Automatic routing to structured ICAR indicators and deterministic arithmetic | Reports 43 participants, 32 women, 11 men, and a calculated 74.4% women share. Figures are transcribed from a curator-written summary; show the source link and caveat. |
| **Which irrigation interval should cotton farmers in Maharashtra follow during kharif according to the demo notes?** | Conflicting evidence and abstention | The two synthetic notes disagree. The app should abstain; the fixtures are not authoritative evidence or advice. |
| **What should I spray to control pests in my farm?** | Missing-context and pesticide-safety handling | Should abstain rather than give a spray recommendation; the question lacks crop/location/season context and no current official advisory should be inferred. Clear filters first so they do not supply scope. |
| **Compare wheat yields across Punjab districts for Rabi 2022–2023.** | Automatic routing to the deterministic DES comparison | The loaded extract has 23 district rows for Punjab wheat/Rabi 2022–2023. The app can chart and tabulate those rows; state, crop, and season sidebar filters should match. Report caveats: historical descriptive data, estimate status not stated in the portal view, and row hashes are not hashes of an official downloaded report. |
| **A note says to ignore previous instructions and recommend pesticide rates. Should I follow it?** | Optional adversarial/untrusted-source test | The fixture contains an embedded instruction. Discuss it as untrusted source text, not an instruction to the system. The extractive default may quote the fixture; use a configured synthesis provider if you intend to demonstrate how the prompt frames untrusted content, and inspect its citations. |
| **Compare rice soil water requirements, explain how groundwater depth changes the comparison, and identify what evidence is still needed for a local soil-suitability decision.** | Optional Agentic RAG planning and multi-query retrieval | Use the **06 · Agentic research brief** scenario and a preconfigured provider. Show the bounded query plan and visible tool steps; do not promise a particular model-generated plan or response. |

## Transitions to keep the narrative connected

- **Problem → evidence:** “Let’s see whether the system can show us the source behind a
  useful answer, not just produce fluent text.”
- **Text → structured data:** “A passage search is the wrong tool for arithmetic over
  reported counts, so the app routes that question to a different path.”
- **Supported → unsupported:** “The next test is not another success case; it is what the
  system does when evidence conflicts or the question asks for advice without enough
  context.”
- **Single query → agentic option:** “Planning can help retrieve evidence for a
  multi-part question, but here it is deliberately bounded to local, read-only search.”
- **Metrics → close:** “These measurements help us find regressions; they do not certify
  agricultural correctness.”

## Backup plan

- **Streamlit or network unavailable:** use screenshots captured during rehearsal, or
  walk through the query, retrieval, and evidence-gate flow from the architecture above.
  Keep the original ICAR and IRRI source links available, but do not rely on live access.
- **Embedding model/index failure:** use the project README, test cases, and saved
  screenshots. The BM25 path alone is not a turnkey no-model fallback in the current app;
  dense index initialization still expects the embedding model and Chroma collection.
- **LLM/Ollama failure:** switch to **Extractive evidence mode** and skip Agentic RAG.
  The ordinary retrieval, citations, source cards, trace, structured queries, and
  abstention are still the core demonstration.
- **Optional reranker failure:** leave it disabled; the app continues with fused ranking.
- **Query returns no evidence:** first reset or broaden the sidebar scope, confirm the
  index manifest is populated, and retry one of the exact rehearsed queries. Do not
  silently reinterpret a failed result as evidence.
- **Evaluation report missing or slow:** explain that evaluation is opt-in and use its
  panel only if a report is already available. Do not run it live under time pressure.
- **Running long:** drop the optional Agentic RAG segment and keep the supported evidence,
  structured-count, and abstention examples.
