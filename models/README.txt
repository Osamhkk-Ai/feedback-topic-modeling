LOCAL MODELS FOLDER
===================

Put models here to run the pipeline FULLY OFFLINE (no Hugging Face download).

1) Local LLM (GGUF for llama-cpp)
   - Place a .gguf file here, e.g.:  models/Qwen3-4B-Q4_K_M.gguf
   - In config.py set:
        LLM_LOCAL_FILE = "models/Qwen3-4B-Q4_K_M.gguf"
   - To swap models, just drop a different .gguf and update that line.

2) Local embedding model (sentence-transformers folder)
   - Place the model folder here, e.g.:  models/minilm/
     (the folder that contains config.json, modules.json, etc.)
   - In config.py set:
        EMBEDDING_LOCAL_DIR = "models/minilm"

If these are left empty in config.py, the models are downloaded once from
Hugging Face and cached. After that, you can set the environment variable
HF_HUB_OFFLINE=1 to force offline use of the cached copies.
