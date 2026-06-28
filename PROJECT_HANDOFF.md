# PROJECT HANDOFF — Customer-Feedback Topic Modeling

> Read this first to understand the whole project and continue it in a new
> chat. It documents the goal, every decision, what we built, what we
> learned, the current state, and how to run it. (Mixed EN/AR notes.)

---

## 1. Goal

Take ~10,000 messy, multilingual customer-feedback comments (Arabic MSA +
Saudi/Gulf/Egyptian dialect + English + code-switched, with typos) and
**automatically discover the main topics**, assign every comment to a
topic, and produce **business-ready** output: topic name, keywords,
representative comments, sentiment, severity, recommended actions, an
executive summary, CSV/Excel reports and charts.

Currently tested on **250 rows** (`sfda_customer_feedback.csv`) as a smoke
test. Real target ≈ 10k.

## 2. Hard constraints (from the user)

- **CPU-only must work** (a GPU may be used if present, but is optional).
- **8 GB RAM** friendly.
- **Do NOT use an LLM per row** — the LLM runs **once per TOPIC only**
  (topics are bounded, the user expects ≤ ~100 topics no matter the data
  size, so LLM calls stay tiny).
- Use **embeddings + clustering** for all comments (the cheap part).
- **Fully offline / no server / no cloud / no API.** Everything local.
- Minimal, automatic cleaning; keep Arabic + English; keep negation words.
- Beginner-friendly, modular, single entry point.

## 3. Final architecture (pipeline)

```
CSV/Excel
  -> load (data_loader)              auto-detect csv/xlsx, pick comment column
  -> clean (cleaning)                URLs/emoji/repeated-punct + light Arabic
                                     normalization. KEEPS duplicates (adds a
                                     `frequency` column) and originals.
  -> embeddings (embeddings)         sentence-transformers on CPU, cached to disk
  -> UMAP (engine_sklearn)           reduce to ~5 dims (cosine)
  -> HDBSCAN (engine_sklearn)        density clustering on the UMAP space
                                     (euclidean — correct after UMAP)
                                     + optional KMeans fallback (big+noisy only)
  -> topic quality check (quality)   per-topic size/%/keywords/flags BEFORE LLM
  -> c-TF-IDF keywords (keywords)    Arabic-aware, negation-safe
  -> representative comments (sampling)  centroid-based, deduped, AR/EN mix
  -> local LLM (naming)              Qwen per TOPIC: name/desc/root_cause/
                                     severity/sentiment/3 actions + exec summary
                                     validate -> stricter retry -> keyword fallback
  -> reports (reporting) + charts    CSVs + Excel (8 sheets) + PNG charts
```

## 4. Key decisions + rationale

**Embedding model — currently `intfloat/multilingual-e5-large`** (set in
`config.EMBEDDING = "e5large"`, full precision, NO quantization — user
insisted: do not quantize the embedding).
- We compared (same params, 250 rows, no LLM):
  | model | dim | topics | noise | silhouette |
  |---|---|---|---|---|
  | MiniLM (paraphrase-multilingual-MiniLM-L12-v2) | 384 | 20 | 16.8% | 0.168 |
  | e5-base | 768 | 14 | 6.8% | 0.060 |
  | **e5-large** | 1024 | 17 | **4.8%** | 0.080 |
- e5-large gave the **lowest noise / best coverage**, so the user chose it.
  (Earlier in the project MiniLM looked best; the cleaner re-test favored
  e5-large. MiniLM is still kept as a fallback option in EMBEDDING_MODELS.)
- e5 models REQUIRE a `"query: "` prefix on every text — handled
  automatically in `embeddings.py` (detects "e5" in the model name).

**Clustering — hand-rolled sklearn (NOT BERTopic).**
- We A/B-tested BERTopic vs a hand-rolled `UMAP -> sklearn.HDBSCAN ->
  c-TF-IDF`. They are the same algorithm; the hand-rolled one is simpler,
  fewer fragile deps, and gives full control. BERTopic + standalone
  `hdbscan` were removed.
- **Dynamic params by dataset size** (`params.py`): tiny (<500) = smoke
  test (min_cluster_size 5); medium (500–3000); large (>=3000,
  min_cluster_size ~50, capped 100). Prevents 250-tuned params from
  fragmenting 10k.
- **KMeans fallback** only when `n >= 1000 AND noise > 30%` (small samples
  never auto-fallback).

**Local LLM — Qwen3-4B GGUF via `llama-cpp-python`.**
- `config.LLM_REPO="Qwen/Qwen3-4B-GGUF"`, `LLM_FILE="Qwen3-4B-Q4_K_M.gguf"`.
- Runs **once per topic**, schema-constrained JSON (so output is always
  valid JSON; also suppresses Qwen3 `<think>`), then strict validation
  (6 keys, enum severity/sentiment, exactly 3 actions, 2–4-word non-generic
  name) -> one stricter retry -> deterministic keyword fallback.
- We compared 3B vs 4B: **4B reads sentiment better** (catches positive/
  neutral, less "everything is negative"); difference is real but modest;
  with time not a constraint and topics bounded, 4B is the pick.
- Prompt asks for Arabic business_name, description, root_cause, severity,
  sentiment, recommended_actions (3). Important rule baked in: *"topic size/
  percentage indicate frequency only, they do NOT raise severity by
  themselves."*

**Offline — `config.OFFLINE = True`** sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` so nothing ever hits the network. Models load
from `models/` (local) or the local HF cache. All pip wheels are bundled
in `offline/wheels/`.

## 5. CRITICAL gotchas we solved (don't re-break these)

1. **Arabic diacritics regex bug** — an early cleaning regex range spanned
   the Arabic LETTER block and DELETED all Arabic (left only the comma).
   Fixed: the diacritics class must NOT include U+0621–U+064A. (See
   `cleaning.py` `_AR_DIACRITICS`.)
2. **PCA→50 dims fed to HDBSCAN = 0 topics (all noise).** Density
   clustering needs LOW dims (~5), like UMAP. Use UMAP (default) or PCA=5.
3. **numpy/pandas pinning** — global machine had numpy 2.4 / pandas 3.0
   which break numba/umap. The venv pins numpy 1.26.4, pandas 2.2.3,
   sklearn 1.6.1, etc. A `pip install --force-reinstall` once bumped numpy
   to 2.x and broke numba — had to restore 1.26.4.
4. **GitHub "Download ZIP" breaks Git LFS files** — wheels (LFS) become
   tiny placeholder text files => `pip` says "Wheel is invalid". Get real
   files via the **Release asset / committed zip / git clone + git lfs
   pull**, NOT the green "Download ZIP" button. The offline machine can't
   reach GitHub anyway, so transfer by **USB**.
5. **llama-cpp prebuilt CPU wheel required AVX512** -> Windows error
   `0xc000001d` (illegal instruction) on CPUs without AVX512 (i7-10750H
   and the target i7-14700 both lack AVX512). FIX: we rebuilt
   `llama-cpp-python 0.3.31` for **CPU + AVX2** (no AVX512, no CUDA):
   `CMAKE_ARGS=-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON
   -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_FMA=ON -DGGML_F16C=ON`.
   Verified Qwen loads + runs on pure CPU. This AVX2 wheel is the one
   bundled in `offline/wheels/`.
6. **llama-cpp `.tar.gz` (sdist) won't install offline** (needs a compiler
   + hits Windows long-path errors). Always ship the prebuilt `.whl`.

## 6. Machines & environment

- **Dev machine** (this one): user "HP", Windows 10, Python 3.11.9,
  Intel i7-10750H (AVX2, no AVX512), **NVIDIA GTX 1650 Ti 4GB + CUDA 12.6**.
  Here `llama-cpp-python 0.3.31` is installed as a **CUDA/GPU build** in
  `.venv`. GPU offload works (`offloaded 37/37 layers`).
- **Offline target machine**: user "oa.ghamdi", **locked-down, CMD only,
  NO internet, NO venv allowed**, Intel **i7-14700** (AVX2, no AVX512),
  CPU only. Install with `pip install --user --no-index`. Uses the bundled
  **AVX2 CPU** wheel.

Pinned versions: python 3.11, numpy 1.26.4, scipy 1.13.1, pandas 2.2.3,
scikit-learn 1.6.1, sentence-transformers 3.3.1, transformers 4.46.3,
sentencepiece 0.2.0, umap-learn 0.5.7, numba 0.60.0, torch 2.4.1+cpu,
openpyxl 3.1.5, matplotlib 3.9.2, llama-cpp-python 0.3.31.

## 7. Repo & current state

- **GitHub (private):** https://github.com/Osamhkk-Ai/feedback-topic-modeling
  (account Osamhkk-Ai). Latest commit at handoff: `5520e04`.
- Tracked via **Git LFS**: `offline/wheels/*.whl` and `*.zip`.
- In the repo: code, `config.py`, `requirements.txt`, `README.md`,
  `run_pipeline.py`, `run_llm.py`, `src/`, `offline/wheels/` (all wheels
  incl. the AVX2 llama-cpp), `feedback-topic-modeling-offline.zip` (full
  bundle), `models/README.txt`.
- **NOT in git** (too big / private): the actual models, `outputs/`,
  `.venv/`, the data CSV.
- Models on disk (HF cache) kept: `multilingual-e5-large` (~2.2GB),
  `Qwen3-4B-GGUF` (~2.4GB), MiniLM (fallback). We deleted experiment
  caches (e5-small/base, qwen2.5-3b, jina-v3, gte-reranker, unsloth) to
  free disk (was down to 2.1GB free, now ~12GB).

## 8. File / module map

```
run_pipeline.py    MAIN entry: full pipeline (load->...->LLM->reports)
run_llm.py         SEPARATE entry: LLM stage only, AFTER embeddings/topics
                   (reads cached embeddings + checkpoint; resumable)
config.py          ALL settings (OFFLINE, EMBEDDING, LLM_*, params, paths)
src/
  data_loader.py   load csv/xlsx, pick column
  cleaning.py      clean + Arabic normalize, KEEP duplicates (+frequency)
  embeddings.py    sentence embeddings, cached, e5 "query:" prefix, offline
  params.py        dynamic clustering params by n + KMeans-fallback rule
  engine_sklearn.py UMAP -> HDBSCAN (+ KMeans fallback) -> TopicResult
  keywords.py      Arabic-aware c-TF-IDF + auto names; NEGATION_WORDS kept
  sampling.py      representative comments (centroid, deduped, AR/EN)
  quality.py       per-topic quality check + flags (before LLM)
  evaluation.py    metrics: n_topics, noise, silhouette, NPMI coherence
  llm_prompts.py   prompt builders + TOPIC_SCHEMA + validation + fallback
  naming.py        llama-cpp backend + apply_naming (validate/retry/fallback,
                   resumable via out_path) + executive_report
  reporting.py     build frames, CSVs, Excel (8 sheets), charts, validation
  charts.py        matplotlib (no seaborn): top topics, distribution, noise
  topic_result.py  shared TopicResult dataclass
  checkpoint.py    save/load pre-LLM result (used by run_llm.py; NOT wired
                   into run_pipeline.py)
outputs/           results (flat): feedback_with_topics.csv, topic_summary.csv,
                   topic_samples.csv, noise_comments.csv, quality_check.csv,
                   executive_summary.txt, topic_modeling_report.xlsx,
                   llm_input.json, llm_output.json, charts/, cache/
```

Note: `run_pipeline.py` does NOT currently call `checkpoint.py` (only
`run_llm.py` does). If you want resume across the whole pipeline, wire
checkpoint save into `run_pipeline.py` Phase 1.

## 9. How to run

**Dev machine (this one, GPU, has venv):**
```cmd
cd C:\Users\HP\Desktop\project\tt
.venv\Scripts\activate.bat
python run_pipeline.py
```

**Offline machine (i7-14700, CMD only, no venv, no internet):**
```cmd
:: install packages from the bundled wheels (no internet, no venv)
python -m pip install --user --no-index --find-links=offline\wheels "torch==2.4.1+cpu"
python -m pip install --user --no-index --find-links=offline\wheels -r requirements.txt
python -m pip install --user --no-index --find-links=offline\wheels llama-cpp-python
:: put models:  models\e5large\  (from HF)  and  models\Qwen3-4B-Q4_K_M.gguf
:: in config.py set EMBEDDING_LOCAL_DIR and LLM_LOCAL_FILE + INPUT_PATH/COMMENT_COLUMN
python run_pipeline.py
```

**Separate LLM run (run model after embeddings are done):**
```cmd
:: 1) embeddings + topics only:  set LLM_ENABLED = False  ->  python run_pipeline.py
:: 2) model only:                set LLM_ENABLED = True   ->  python run_llm.py
```

## 10. Key config knobs (config.py)

`OFFLINE` (True), `INPUT_PATH`, `COMMENT_COLUMN`, `MIN_WORDS`,
`EMBEDDING` ("e5large"), `EMBEDDING_LOCAL_DIR`, `EMBED_BATCH_SIZE` (16),
`LLM_ENABLED`, `LLM_REPO`/`LLM_FILE`, `LLM_LOCAL_FILE`,
`LLM_N_GPU_LAYERS` (-1 = GPU if present, 0 = CPU), `LLM_MAX_TOKENS`,
`LLM_SAMPLES_PER_TOPIC` (8), `LLM_MAX_RETRIES` (1), `AUTO_PARAMS` (True),
`DIM_REDUCER` ("umap"), `MIN_DF`, `TOP_KEYWORDS`, `REP_COMMENTS_MIN/MAX`.

## 11. Outputs (in `outputs/`)

`feedback_with_topics.csv` (every comment + topic, name, sentiment,
severity), `topic_summary.csv` (count, %, keywords, description,
root_cause, sentiment, severity, recommended_actions), `topic_samples.csv`,
`noise_comments.csv`, `quality_check.csv`, `executive_summary.txt`,
`topic_modeling_report.xlsx` (Overview, Feedback, Topic Summary,
Representative Comments, Noise, Quality Check, Settings, Charts),
`llm_input.json` / `llm_output.json`, `charts/`.

## 12. Open items / possible next steps

- Run on the real ~10k dataset (params auto-scale; expect 30–80 topics).
- Optionally wire `checkpoint.py` into `run_pipeline.py` for full resume.
- Sentiment/severity from a 3B/4B model are decent but not perfect on a
  small sample; validate on real data; a bigger model improves naming.
- The data file `sfda_customer_feedback.csv` is LABELED (main_topic,
  sub_topic, sentiment, dialect_style...) — can be used to measure
  discovered-vs-true topic agreement (not yet done).
- Consider a `.whl`-only mini-bundle for quick fixes (we already ship the
  AVX2 llama-cpp wheel separately).
```
