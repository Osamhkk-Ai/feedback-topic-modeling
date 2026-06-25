# Customer-Feedback Topic Modeling (Arabic / English / mixed)

Runs **fully offline on CPU**, **no virtual environment**, **CMD only**.
No server, no cloud, no API. The local LLM runs **once per topic** (never
per comment).

```
CSV/Excel -> clean -> embeddings -> UMAP -> HDBSCAN -> quality check
-> keywords -> representative comments -> local LLM -> reports + charts
```

> Target machine: **Windows 64-bit + Python 3.11** installed system-wide.
> All commands are **CMD** (Command Prompt). No `venv` is created.

---

## STEP 0 — Get the project (IMPORTANT — read first)

Download the offline bundle from the **Releases** page (it contains the
REAL pip wheels):

**https://github.com/Osamhkk-Ai/feedback-topic-modeling/releases**

→ download the asset **`feedback-topic-modeling-offline.zip`** and unzip it,
then copy the folder by USB to the offline machine.

⚠️ Do **NOT** use the green **"Code → Download ZIP"** button (or `git clone`
without Git LFS). That turns the wheels into tiny broken placeholder files
and `pip` will report **"Wheel is invalid"**. Always use the **Release
asset** above.

---

## STEP 1 — Install the packages (OFFLINE, no venv)

Open **CMD** in the project folder and install straight from the bundled
wheels (no internet, no virtual environment):

```cmd
cd "C:\path\to\feedback-topic-modeling"
python -m pip install --user --no-index --find-links=offline\wheels "torch==2.4.1+cpu"
python -m pip install --user --no-index --find-links=offline\wheels -r requirements.txt
python -m pip install --user --no-index --find-links=offline\wheels llama-cpp-python
```

- `--user` = install for your user (works even when the machine is locked
  down / global Python is read-only).
- `--no-index` = never go online; use only `offline\wheels`.
- If `python` is not found, use `py -3.11` instead of `python`.

## STEP 2 — Put the models in `models\`

The offline machine can't download, so copy these by USB:

1. **Embedding — e5-large** → put the folder at `models\e5large\`
   (must contain `model.safetensors`, `config.json`, `tokenizer.json`,
   `sentencepiece.bpe.model`, `modules.json`, ...).
   Download source (on an online PC): https://huggingface.co/intfloat/multilingual-e5-large
2. **LLM — Qwen GGUF** → put the file at `models\Qwen3-4B-Q4_K_M.gguf`
   Download source: https://huggingface.co/Qwen/Qwen3-4B-GGUF

Then open `config.py` and set the local paths:
```python
EMBEDDING_LOCAL_DIR = "models/e5large"
LLM_LOCAL_FILE = "models/Qwen3-4B-Q4_K_M.gguf"
```

## STEP 3 — Point to your data + column

In `config.py`:
```python
INPUT_PATH = "my_feedback.csv"        # .csv / .xlsx / .xls (put the file in the project folder)
COMMENT_COLUMN = "customer_feedback"  # the column that holds the text
```
(Wrong column name? The program prints the available columns.)

## STEP 4 — Run

```cmd
cd "C:\path\to\feedback-topic-modeling"
python run_pipeline.py
```
(or `py -3.11 run_pipeline.py`)

You'll see staged progress (STAGE 1/6 ... 6/6) with percentages.

---

## Results — `outputs\`

- `feedback_with_topics.csv` — every comment + topic, name, sentiment, severity
- `topic_summary.csv` — topic, count, %, keywords, description, root cause, actions
- `topic_samples.csv` — representative comments per topic
- `noise_comments.csv` — unclassified comments
- `quality_check.csv` — per-topic quality + flags
- `executive_summary.txt` — overall findings + recommendations
- `topic_modeling_report.xlsx` — full report (8 sheets + charts)
- `llm_input.json` / `llm_output.json` — what the LLM received / returned

---

## Settings (config.py)

| Setting | Meaning |
|---|---|
| `INPUT_PATH` | data file (CSV/Excel) |
| `COMMENT_COLUMN` | text column |
| `EMBEDDING_LOCAL_DIR` | local embedding model folder (`models/e5large`) |
| `LLM_LOCAL_FILE` | local GGUF model file |
| `LLM_N_GPU_LAYERS` | `-1` = use GPU if present, `0` = CPU only |
| `LLM_ENABLED` | `False` to skip the LLM (keep automatic keyword names) |
| `EMBED_BATCH_SIZE` | lower (8) if RAM is tight |
| `MIN_WORDS` | drop comments shorter than this |

`OFFLINE = True` is already set — nothing ever contacts a server.

---

## Troubleshooting

- **`Wheel '...' is invalid`** → the wheels came from a GitHub ZIP (LFS files
  become placeholders). Use the wheels from the manually-copied folder
  (`offline\wheels`), not a GitHub "Download ZIP".
- **`python` not found** → use `py -3.11` instead of `python`.
- **pip writes to "user" location** → that's expected here (no venv); it's fine.
- **`llama-cpp` build error / long paths** → use the bundled prebuilt
  `llama_cpp_python-*-win_amd64.whl` (already in `offline\wheels`); do not
  install the source `.tar.gz`.
- **Out of memory** → set `EMBED_BATCH_SIZE = 8` and `LLM_N_GPU_LAYERS = 0`.
- **Arabic shows as boxes in charts** → expected; Arabic is correct in the
  CSV/Excel data sheets (`utf-8-sig`). Charts use topic IDs.

---

## Project layout

```
run_pipeline.py   single entry point
config.py         all settings
offline\wheels\   bundled pip packages (offline install)
models\           your local models (e5large folder + Qwen gguf)
requirements.txt
src\              pipeline modules
```
