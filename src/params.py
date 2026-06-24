# =====================================================================
# params.py  —  Dynamic clustering parameters based on dataset size.
#
# Settings that work for 250 comments are WRONG for 10,000. Tiny
# min_cluster_size on big data fragments into too many topics; large
# values on small data merge everything. So we scale parameters by
# n = number of clean comments.
# =====================================================================


def compute_params(n, config):
    """Return a dict of clustering params + smoke-test flags for n comments."""
    if not getattr(config, "AUTO_PARAMS", True):
        # Manual override: use the fixed values from config.py
        return {
            "n": n,
            "umap_n_neighbors": config.UMAP_N_NEIGHBORS,
            "min_cluster_size": config.MIN_TOPIC_SIZE,
            "min_samples": config.HDBSCAN_MIN_SAMPLES,
            "is_smoke_test": n < 500,
            "warning": "",
            "tier": "manual",
        }

    warning = ""
    if n < 500:
        tier = "smoke_test (<500)"
        umap_n_neighbors = 15 if n >= 150 else 10
        min_cluster_size = 5
        min_samples = 2
        is_smoke_test = True
        warning = ("Small dataset smoke test only. "
                   "Do not judge final topic quality from this sample.")
    elif n < 3000:
        tier = "medium (500-3000)"
        umap_n_neighbors = 20
        min_cluster_size = max(10, int(n * 0.01))
        min_samples = 4
        is_smoke_test = False
    else:
        tier = "large (>=3000)"
        umap_n_neighbors = 40
        min_cluster_size = max(30, int(n * 0.005))
        # for ~10k this lands around 50; cap so it doesn't get huge
        min_cluster_size = min(min_cluster_size, 100)
        min_samples = 8
        is_smoke_test = False

    return {
        "n": n,
        "umap_n_neighbors": umap_n_neighbors,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "is_smoke_test": is_smoke_test,
        "warning": warning,
        "tier": tier,
    }


def should_run_kmeans_fallback(n, noise_percentage, config):
    """KMeans fallback ONLY for big datasets with high noise.

    Small samples naturally produce unstable noise, so never auto-fallback
    for n < KMEANS_FALLBACK_MIN_N even if noise is high.
    """
    if n < config.KMEANS_FALLBACK_MIN_N:
        return False
    return noise_percentage > config.KMEANS_FALLBACK_NOISE
