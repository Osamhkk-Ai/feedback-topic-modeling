# =====================================================================
# checkpoint.py  —  Persist the PRE-LLM result to disk.
#
# Why: clustering (embed -> UMAP -> HDBSCAN -> keywords) is the expensive,
# deterministic part. We save it BEFORE the slow LLM stage so that:
#   - if RAM fills / the LLM crashes, the pre-LLM results are kept,
#   - re-running resumes from the checkpoint (no re-clustering),
#   - the LLM stage itself saves each topic as it finishes (resumable).
#
# The checkpoint is keyed by a signature of the cleaned comments, so it is
# reused only when the input is unchanged.
# =====================================================================

import os
import json
import hashlib
import numpy as np

from src.topic_result import TopicResult


def docs_signature(docs):
    h = hashlib.md5()
    h.update(str(len(docs)).encode())
    h.update("\n".join(docs).encode("utf-8"))
    return h.hexdigest()[:12]


def _dir(config):
    d = os.path.join(config.CACHE_DIR, "checkpoint")
    os.makedirs(d, exist_ok=True)
    return d


def save_result(result, metrics, stats, sig, config):
    d = _dir(config)
    np.save(os.path.join(d, "doc_topics.npy"), np.asarray(result.doc_topics))
    payload = {
        "sig": sig,
        "engine": result.engine,
        "model": result.model,
        "runtime_sec": result.runtime_sec,
        "params": result.params,
        "topic_keywords": {str(k): v for k, v in result.topic_keywords.items()},
        "topic_sizes": {str(k): v for k, v in result.topic_sizes.items()},
        "topic_auto_names": {str(k): v for k, v in result.topic_auto_names.items()},
        "metrics": metrics,
        "stats": stats,
    }
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[checkpoint] pre-LLM result saved -> {d}")


def load_result(sig, config):
    """Return (result, metrics, stats) if a checkpoint matches `sig`, else None."""
    d = _dir(config)
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return None
    try:
        data = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    if data.get("sig") != sig:
        return None
    result = TopicResult(
        engine=data["engine"],
        model=data["model"],
        doc_topics=np.load(os.path.join(d, "doc_topics.npy")),
        topic_keywords={int(k): v for k, v in data["topic_keywords"].items()},
        topic_sizes={int(k): v for k, v in data["topic_sizes"].items()},
        topic_auto_names={int(k): v for k, v in data["topic_auto_names"].items()},
        runtime_sec=data["runtime_sec"],
        params=data["params"],
    )
    return result, data["metrics"], data["stats"]
