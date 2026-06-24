# =====================================================================
# quality.py  —  Topic quality check BEFORE the LLM step.
#
# A great LLM summary is useless if the cluster itself is bad. This
# produces a per-topic table (size, %, keywords, sample comments) plus
# warning flags for tiny/unclear/near-duplicate topics, so you can judge
# cluster quality before naming.
# =====================================================================

import numpy as np
import pandas as pd

from src import sampling

SIMILAR_THRESHOLD = 0.90   # centroid cosine above this => possibly the same topic


def topic_quality_check(result, embeddings, clean_df, config):
    labels = np.asarray(result.doc_topics)
    n = labels.size
    n_noise = int((labels == -1).sum())
    noise_pct = round(100 * n_noise / max(n, 1), 2)

    tids = result.topic_ids()
    centroids = {t: embeddings[labels == t].mean(axis=0) for t in tids}
    tiny_threshold = max(5, int(0.01 * n))

    rows = []
    for t in tids:
        size = result.topic_sizes.get(t, 0)
        pct = round(100 * size / max(n, 1), 2)
        kws = result.topic_keywords.get(t, [])
        reps = sampling.representative_comments(result, embeddings, clean_df, t, 3)

        # nearest other topic by centroid cosine
        best_other, best_sim = None, -1.0
        for t2 in tids:
            if t2 == t:
                continue
            c1, c2 = centroids[t], centroids[t2]
            cs = float(c1 @ c2 / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-9))
            if cs > best_sim:
                best_sim, best_other = cs, t2

        flags = []
        if size < tiny_threshold:
            flags.append("توبيك صغير")
        if not kws:
            flags.append("بدون كلمات واضحة")
        if best_other is not None and best_sim > SIMILAR_THRESHOLD:
            flags.append(f"يشبه التوبيك {best_other}")

        rows.append({
            "topic_id": t,
            "topic_size": size,
            "topic_percentage": pct,
            "noise_percentage": noise_pct,
            "top_keywords": ", ".join(kws[:config.TOP_KEYWORDS]),
            "representative_comments": " || ".join(reps),
            "most_similar_topic": best_other,
            "similarity": round(best_sim, 3) if best_other is not None else None,
            "flags": "; ".join(flags),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("topic_size", ascending=False).reset_index(drop=True)
    return df
