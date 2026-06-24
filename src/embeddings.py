# =====================================================================
# embeddings.py  —  Sentence embeddings (CPU-friendly, offline-capable).
#
# - Loads the embedder from a LOCAL folder (config.EMBEDDING_LOCAL_DIR)
#   when provided, otherwise by HF name (downloaded once, then cached).
# - Adds the e5 "query:" prefix automatically when the model is an e5 model.
# - Batches encoding and caches the result to disk (keyed by content hash).
# =====================================================================

import os
import gc
import hashlib
import numpy as np


def _docs_hash(texts):
    h = hashlib.md5()
    h.update(str(len(texts)).encode())
    h.update("\n".join(texts).encode("utf-8"))
    return h.hexdigest()[:10]


def _resolve_model(config):
    """Return (model_path_or_name, is_e5)."""
    local = getattr(config, "EMBEDDING_LOCAL_DIR", "")
    if local and os.path.isdir(local):
        name = local
    else:
        name = config.EMBEDDING_MODELS[config.EMBEDDING]
    is_e5 = "e5" in config.EMBEDDING.lower() or "e5" in str(name).lower()
    return name, is_e5


def embed(texts, config, use_cache=True):
    """Return a float32 (n_docs x dim) embedding matrix for `texts`."""
    texts = list(texts)
    name, is_e5 = _resolve_model(config)
    cache_path = os.path.join(config.CACHE_DIR,
                              f"emb_{config.EMBEDDING}_{_docs_hash(texts)}.npy")

    if use_cache and os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            print(f"  using cached embeddings -> {emb.shape}")
            return emb

    from sentence_transformers import SentenceTransformer
    print(f"  loading embedder: {name}")
    model = SentenceTransformer(name, device="cpu")

    inputs = [f"query: {t}" for t in texts] if is_e5 else texts   # e5 needs the prefix

    print(f"  encoding {len(inputs)} comments (batch={config.EMBED_BATCH_SIZE})...")
    emb = model.encode(
        inputs,
        batch_size=config.EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, emb)
    del model
    gc.collect()
    print(f"  embeddings ready -> {emb.shape} (cached)")
    return emb
