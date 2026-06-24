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

**Option B — OFFLINE install (download once, install on the offline PC):**

On a PC WITH internet, download every package into a folder:
```cmd
mkdir offline_packages
pip download "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu -d offline_packages
pip download -r requirements.txt -d offline_packages
pip download llama-cpp-python -d offline_packages
```
Copy the `offline_packages` folder to the offline PC, then:
```cmd
pip install --no-index --find-links=offline_packages "torch==2.4.1"
pip install --no-index --find-links=offline_packages -r requirements.txt
pip install --no-index --find-links=offline_packages llama-cpp-python
```

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

Download these two yourself and put them in the `models` folder:

1. **LLM (GGUF file)** — e.g. `Qwen3-4B-Q4_K_M.gguf`
   - Put it here:  `models\Qwen3-4B-Q4_K_M.gguf`
   - In `config.py` set:
     ```python
     LLM_LOCAL_FILE = "models/Qwen3-4B-Q4_K_M.gguf"
     ```

2. **Embedding model (folder)** — e.g. the MiniLM folder
   - Put it here:  `models\minilm\`  (the folder with config.json, etc.)
   - In `config.py` set:
     ```python
     EMBEDDING_LOCAL_DIR = "models/minilm"
     ```

To swap a model later, just drop a different file/folder in `models\` and update that line.
(`OFFLINE = True` is already set in `config.py`, so nothing ever goes online.)

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
