# =====================================================================
# engine_sklearn.py  —  Hand-rolled topic modeling (no BERTopic needed).
#
# Pipeline:  embeddings -> UMAP (or PCA) -> HDBSCAN -> c-TF-IDF keywords.
# Parameters scale with dataset size (src/params.py). KMeans only runs as
# a fallback for BIG datasets with high noise (never on small samples).
# =====================================================================

import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans

from src.keywords import ctfidf_keywords, auto_topic_name
from src.topic_result import TopicResult
from src.params import compute_params, should_run_kmeans_fallback


def _reduce_dims(embeddings, config, n_neighbors):
    """Reduce embeddings to a low-dim space (UMAP preferred, PCA fallback)."""
    if config.DIM_REDUCER == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=config.UMAP_N_COMPONENTS,
                min_dist=config.UMAP_MIN_DIST,
                metric=config.UMAP_METRIC,          # cosine on the raw embeddings
                low_memory=config.LOW_MEMORY,
                random_state=42,
            )
            print(f"[sklearn] UMAP (n_neighbors={n_neighbors}, metric=cosine)")
            return reducer.fit_transform(embeddings)
        except Exception as exc:
            print(f"[sklearn] UMAP unavailable/failed ({exc}); using PCA")

    n_comp = min(config.PCA_COMPONENTS, embeddings.shape[1], max(2, embeddings.shape[0] - 1))
    print(f"[sklearn] PCA (n_components={n_comp})")
    return PCA(n_components=n_comp, random_state=42).fit_transform(embeddings)


def run(docs, embeddings, config, model_key):
    """Run the hand-rolled engine and return a TopicResult."""
    t0 = time.time()
    n = len(docs)

    p = compute_params(n, config)
    print(f"[sklearn] dataset tier: {p['tier']}  "
          f"(min_cluster_size={p['min_cluster_size']}, min_samples={p['min_samples']})")
    if p["warning"]:
        print(f"[WARNING] {p['warning']}")

    reduced = _reduce_dims(embeddings, config, p["umap_n_neighbors"])

    # NOTE: HDBSCAN uses euclidean here ON PURPOSE. After UMAP we are in a
    # reduced numeric space where euclidean distance is appropriate; the
    # cosine geometry was already captured inside UMAP. This is not a bug.
    labels = HDBSCAN(
        min_cluster_size=max(2, p["min_cluster_size"]),
        min_samples=p["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(reduced).astype(int)

    noise_pct = float((labels == -1).mean())

    # KMeans fallback ONLY for big datasets with high noise (see params.py).
    kmeans_used = should_run_kmeans_fallback(n, noise_pct, config)
    if kmeans_used:
        k = min(50, max(10, n // 250))
        print(f"[sklearn] noise {noise_pct:.0%} high on big dataset -> KMeans fallback (k={k})")
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(reduced).astype(int)
        noise_pct = 0.0

    keywords = ctfidf_keywords(docs, labels, config, topn=config.TOP_KEYWORDS)
    sizes = {int(t): int((labels == t).sum()) for t in sorted(set(labels.tolist()))}
    auto_names = {t: auto_topic_name(kws) for t, kws in keywords.items()}

    runtime = time.time() - t0
    n_topics = len([t for t in sizes if t != -1])
    print(f"[sklearn] {n_topics} topics, {sizes.get(-1, 0)} noise, {runtime:.1f}s")

    return TopicResult(
        engine="sklearn",
        model=model_key,
        doc_topics=labels,
        topic_keywords=keywords,
        topic_sizes=sizes,
        topic_auto_names=auto_names,
        runtime_sec=runtime,
        params={
            "reducer": config.DIM_REDUCER,
            "umap_n_neighbors": p["umap_n_neighbors"],
            "min_cluster_size": p["min_cluster_size"],
            "min_samples": p["min_samples"],
            "clusterer": "kmeans" if kmeans_used else "hdbscan",
            "kmeans_fallback": kmeans_used,
            "is_smoke_test": p["is_smoke_test"],
            "warning": p["warning"],
            "tier": p["tier"],
            "noise_pct": round(noise_pct, 4),
        },
    )
