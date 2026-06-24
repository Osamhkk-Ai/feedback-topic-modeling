# =====================================================================
# sampling.py  —  Pick representative comments for a topic.
#
# Rules:
#   - use ORIGINAL comment text (not cleaned)
#   - prefer comments closest to the topic centroid (most typical)
#   - drop exact duplicates (so the LLM doesn't see the same line twice)
#   - include both Arabic and English examples when the topic has both
# =====================================================================

import numpy as np


def _has_arabic(text):
    return any("؀" <= c <= "ۿ" for c in str(text))


def representative_comments(result, embeddings, clean_df, tid, k):
    """Up to k representative ORIGINAL comments for topic `tid`."""
    labels = np.asarray(result.doc_topics)
    idx = np.where(labels == tid)[0]
    if idx.size == 0:
        return []

    # closeness to centroid (embeddings are L2-normalized -> cosine ~ dot)
    centroid = embeddings[idx].mean(axis=0)
    sub = embeddings[idx]
    sims = (sub @ centroid) / (np.linalg.norm(sub, axis=1) * np.linalg.norm(centroid) + 1e-9)
    order = idx[np.argsort(-sims)]

    originals = clean_df["original_comment"].tolist()
    cleaned = clean_df["cleaned_text"].tolist()

    # unique candidates, closest first
    uniq, seen = [], set()
    for i in order:
        key = cleaned[i].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((originals[i], _has_arabic(cleaned[i])))

    picked = [c for c, _ in uniq[:k]]

    # ensure language diversity if the topic genuinely has both
    has_ar = any(flag for _, flag in uniq)
    has_en = any(not flag for _, flag in uniq)
    if has_ar and has_en and len(uniq) > k:
        flags = [f for _, f in uniq[:k]]
        if all(flags):                      # all Arabic -> bring one English
            for c, f in uniq[k:]:
                if not f:
                    picked[-1] = c
                    break
        elif not any(flags):                # all English -> bring one Arabic
            for c, f in uniq[k:]:
                if f:
                    picked[-1] = c
                    break
    return picked
