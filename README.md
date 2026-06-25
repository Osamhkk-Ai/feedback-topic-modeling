# Customer-Feedback Topic Modeling (Arabic / English / mixed)

Runs **fully offline on CPU** (uses GPU automatically if present). No
server, no cloud, no API. The local LLM runs **once per topic** (never
per comment).

```
CSV/Excel -> clean -> embeddings -> UMAP -> HDBSCAN -> quality check
-> keywords -> representative comments -> local LLM -> reports + charts
```

All commands below are **Windows CMD** (Command Prompt).

---

## STEP 1 — Create the environment

```cmd
cd C:\Users\HP\Desktop\project\tt
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
```

## STEP 2 — Install the packages

**Option A — normal install (needs internet once):**
```cmd
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install llama-cpp-python
```

**Option B — OFFLINE install (wheels are ALREADY bundled in the repo):**

All Windows/Python-3.11 wheels are committed under `offline\wheels\` (via
Git LFS). After cloning, install with NO internet:
```cmd
git lfs install
git lfs pull
pip install --no-index --find-links=offline\wheels "torch==2.4.1+cpu"
pip install --no-index --find-links=offline\wheels -r requirements.txt
pip install --no-index --find-links=offline\wheels llama-cpp-python
```
(These wheels target Windows x64 + Python 3.11. The Qwen model is NOT
bundled — see STEP 5.)

## STEP 3 — Put your data in the project folder

Copy your file into `C:\Users\HP\Desktop\project\tt` (CSV or Excel), e.g. `my_feedback.csv`.
Then open `config.py` and set the path:
```python
INPUT_PATH = "my_feedback.csv"
```

## STEP 4 — Choose the comment column

In `config.py`, set the column that holds the feedback text:
```python
COMMENT_COLUMN = "customer_feedback"
```
(If you type a wrong name, the program prints the list of available columns.)

## STEP 5 — Put the model you downloaded yourself

**Download the models manually from Hugging Face:**
- Embedding (e5-large): **https://huggingface.co/intfloat/multilingual-e5-large**
  (download the whole repo: `model.safetensors`, `config.json`, `tokenizer.json`,
  `sentencepiece.bpe.model`, `modules.json`, etc.)
- LLM (Qwen3-4B GGUF): **https://huggingface.co/Qwen/Qwen3-4B-GGUF**
  (download just the file `Qwen3-4B-Q4_K_M.gguf`)

Then put them in the `models` folder:

1. **LLM (GGUF file)** — e.g. `Qwen3-4B-Q4_K_M.gguf`
   - Put it here:  `models\Qwen3-4B-Q4_K_M.gguf`
   - In `config.py` set:
     ```python
     LLM_LOCAL_FILE = "models/Qwen3-4B-Q4_K_M.gguf"
     ```

2. **Embedding model (folder)** — e.g. the e5-large folder
   - Put it here:  `models\e5large\`  (the folder with `model.safetensors`,
     `config.json`, `tokenizer.json`, `sentencepiece.bpe.model`, etc.)
   - In `config.py` set:
     ```python
     EMBEDDING_LOCAL_DIR = "models/e5large"
     ```

To swap a model later, just drop a different file/folder in `models\` and update that line.
(`OFFLINE = True` is already set in `config.py`, so nothing ever goes online.)

## Moving to a FULLY-OFFLINE machine (no internet at all)

A machine with no internet can't `git clone` or download models. So copy
everything by USB:

1. On the online PC, the project folder already contains the wheels
   (`offline\wheels\`). Also put the embedding model in `models\e5large\`.
2. Copy the WHOLE project folder (code + `offline\wheels\` + `models\`) to USB
   → paste it on the offline PC. (Qwen GGUF: place it in `models\` there too.)
3. On the offline PC:
   ```cmd
   py -3.11 -m venv .venv
   .venv\Scripts\activate.bat
   pip install --no-index --find-links=offline\wheels "torch==2.4.1+cpu"
   pip install --no-index --find-links=offline\wheels -r requirements.txt
   pip install --no-index --find-links=offline\wheels llama-cpp-python
   ```
4. In `config.py` point to the local models:
   ```python
   EMBEDDING_LOCAL_DIR = "models/e5large"
   LLM_LOCAL_FILE = "models/Qwen3-4B-Q4_K_M.gguf"
   ```
5. `python run_pipeline.py`  — runs with zero internet (`OFFLINE = True`).

## STEP 6 — Run

```cmd
.venv\Scripts\activate.bat
python run_pipeline.py
```

You will see staged progress (STAGE 1/6 ... 6/6) with percentages.

---

## Where the results go

Folder: `outputs\run\`
- `feedback_with_topics.csv` — every comment + its topic, name, sentiment, severity
- `topic_summary.csv` — topic, count, %, keywords, description, root cause, actions
- `topic_samples.csv` — representative comments per topic
- `noise_comments.csv` — unclassified comments
- `quality_check.csv` — per-topic quality + flags
- `executive_summary.txt` — overall findings + recommendations
- `topic_modeling_report.xlsx` — full Excel report (8 sheets + charts)
- `llm_input.json` / `llm_output.json` — exactly what the LLM received/returned

---

## Useful settings (config.py)

| Setting | Meaning |
|---|---|
| `INPUT_PATH` | your data file (CSV/Excel) |
| `COMMENT_COLUMN` | the text column |
| `LLM_LOCAL_FILE` | path to your GGUF model |
| `EMBEDDING_LOCAL_DIR` | path to your embedding model folder |
| `LLM_N_GPU_LAYERS` | `-1` = use GPU if present, `0` = CPU only |
| `LLM_ENABLED` | `False` to skip the LLM (keeps automatic keyword names) |
| `MIN_WORDS` | drop comments shorter than this |
| `AUTO_PARAMS` | auto-scale clustering to dataset size |

---

## Project layout

```
run_pipeline.py   single entry point
config.py         all settings
models\           your downloaded models (offline)
requirements.txt
src\              pipeline modules (load, clean, embed, cluster, keywords,
                  quality, naming, reporting, charts)
```
