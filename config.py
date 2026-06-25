# =====================================================================
# config.py  —  ALL settings in one place. Edit this file, not the code.
# =====================================================================
import os

# ---------------------------------------------------------------------
# 0) OFFLINE  — block ALL network access. Models load only from ./models
#    or the local Hugging Face cache. Nothing ever contacts a server.
# ---------------------------------------------------------------------
OFFLINE = True
if OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------------
# 1) INPUT  (CSV or Excel; choose the text column without touching code)
# ---------------------------------------------------------------------
INPUT_PATH = "sfda_customer_feedback.csv"   # .csv / .xlsx / .xls
COMMENT_COLUMN = "customer_feedback"        # the column that holds the comments
MIN_WORDS = 3                               # drop comments shorter than this

# ---------------------------------------------------------------------
# 2) MODELS  (put local models in ./models for fully offline runs)
# ---------------------------------------------------------------------
MODELS_DIR = "models"

# Embedding model: a Hugging Face name (downloaded once) OR a local folder.
EMBEDDING = "e5large"           # full precision, no quantization
EMBEDDING_MODELS = {
    "e5large": "intfloat/multilingual-e5-large",     # best coverage in tests (1024-dim)
    "e5base": "intfloat/multilingual-e5-base",       # lighter alternative
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
EMBEDDING_LOCAL_DIR = ""        # e.g. "models/e5large" to load the embedder fully offline
EMBED_BATCH_SIZE = 16           # e5-large is heavier; 16 is RAM-safe on CPU (raise if you have headroom)

# Local LLM (llama-cpp). Used ONLY per-topic (after clustering), never per row.
LLM_ENABLED = True
LLM_REPO = "Qwen/Qwen3-4B-GGUF"             # HF GGUF repo (first run downloads ~2.6GB)
LLM_FILE = "Qwen3-4B-Q4_K_M.gguf"           # quantized Q4 file
LLM_LOCAL_FILE = ""             # e.g. "models/Qwen3-4B-Q4_K_M.gguf" to run fully offline
LLM_CTX = 2048
LLM_THREADS = 4
LLM_N_GPU_LAYERS = -1           # -1 = use GPU if the CUDA build is present; 0 = CPU only
LLM_MAX_TOKENS = 768
LLM_SAMPLES_PER_TOPIC = 8       # representative comments sent to the LLM per topic (6-12)
LLM_MAX_RETRIES = 1             # retry once with a stricter prompt if JSON invalid

# ---------------------------------------------------------------------
# 3) CLUSTERING  (parameters scale automatically with dataset size)
# ---------------------------------------------------------------------
AUTO_PARAMS = True              # compute UMAP/HDBSCAN params from n (see src/params.py)

DIM_REDUCER = "umap"           # "umap" (best) or "pca" (no compile)
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
PCA_COMPONENTS = 5
LOW_MEMORY = True

# KMeans fallback only for BIG datasets with high noise (never on small samples):
KMEANS_FALLBACK_MIN_N = 1000
KMEANS_FALLBACK_NOISE = 0.30

# Manual fallbacks (used only when AUTO_PARAMS = False):
MIN_TOPIC_SIZE = 50
UMAP_N_NEIGHBORS = 30
HDBSCAN_MIN_SAMPLES = None

# ---------------------------------------------------------------------
# 4) KEYWORDS + REPRESENTATIVE COMMENTS
# ---------------------------------------------------------------------
MIN_DF = 2
NGRAM_RANGE = (1, 2)
TOP_KEYWORDS = 10
REP_COMMENTS_MIN = 6
REP_COMMENTS_MAX = 12

# ---------------------------------------------------------------------
# 5) OUTPUT
# ---------------------------------------------------------------------
OUTPUT_DIR = "outputs"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
VALIDATION_TOP_TOPICS = 15
VALIDATION_SAMPLES = 5
