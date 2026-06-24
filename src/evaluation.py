# =====================================================================
# evaluation.py  —  Objective metrics + the recommendation rubric.
#
# Metrics per run:
#   - n_topics, n_noise, noise_rate, coverage
#   - silhouette       (cluster separation, cosine; higher is better)
#   - npmi_coherence   (do a topic's keywords co-occur? higher is better)
#   - topic_balance    (1 - Gini of topic sizes; higher = more balanced)
#   - runtime_sec
#
# These let us COMPARE runs objectively instead of guessing.
# =====================================================================

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

from src.keywords import AR_TOKEN_PATTERN


def _gini(values):
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)


def npmi_coherence(docs, keywords_dict, config, topn=10):
    """Average NPMI coherence over topics, using document co-occurrence.

    NPMI(wi, wj) in [-1, 1]; we average over all keyword pairs in a topic,
    then over topics. Uses a binary doc-term matrix (no gensim dependency).
    """
    candidates = sorted({w for kws in keywords_dict.values() for w in kws[:topn]})
    if len(candidates) < 2:
        return 0.0

    cv = CountVectorizer(
        token_pattern=AR_TOKEN_PATTERN,
        ngram_range=config.NGRAM_RANGE,
        min_df=1,
        binary=True,
        vocabulary=candidates,
    )
    X = cv.fit_transform(docs).tocsc()         # (n_docs, n_candidates) 0/1
    vocab = cv.vocabulary_
    n_docs = X.shape[0]
    if n_docs == 0:
        return 0.0

    doc_freq = {w: int(X[:, vocab[w]].count_nonzero()) for w in candidates}

    topic_scores = []
    for kws in keywords_dict.values():
        words = [w for w in kws[:topn] if w in vocab and doc_freq[w] > 0]
        if len(words) < 2:
            continue
        pair_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wi, wj = words[i], words[j]
                co = int(X[:, vocab[wi]].multiply(X[:, vocab[wj]]).count_nonzero())
                if co == 0:
                    pair_scores.append(-1.0)
                    continue
                p_i = doc_freq[wi] / n_docs
                p_j = doc_freq[wj] / n_docs
                p_ij = co / n_docs
                pair_scores.append(np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij)))
        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(topic_scores)) if topic_scores else 0.0


def evaluate(result, embeddings, docs, config):
    """Compute the metric dict for one run."""
    labels = np.asarray(result.doc_topics)
    n = labels.size
    n_noise = int((labels == -1).sum())
    topic_ids = [t for t in set(labels.tolist()) if t != -1]
    n_topics = len(topic_ids)
    noise_rate = n_noise / n if n else 0.0

    # silhouette on non-noise points (needs >=2 clusters)
    sil = None
    mask = labels != -1
    if mask.sum() > 2 and len(set(labels[mask].tolist())) >= 2:
        try:
            sil = float(silhouette_score(embeddings[mask], labels[mask], metric="cosine"))
        except Exception:
            sil = None

    coherence = npmi_coherence(docs, result.topic_keywords, config, topn=config.TOP_KEYWORDS)
    sizes_no_noise = [s for t, s in result.topic_sizes.items() if t != -1]
    balance = (1.0 - _gini(sizes_no_noise)) if n_topics > 0 else 0.0

    return {
        "engine": result.engine,
        "model": result.model,
        "n_topics": n_topics,
        "n_noise": n_noise,
        "noise_rate": round(noise_rate, 4),
        "coverage": round(1 - noise_rate, 4),
        "silhouette": round(sil, 4) if sil is not None else None,
        "npmi_coherence": round(coherence, 4),
        "topic_balance": round(balance, 4),
        "runtime_sec": round(result.runtime_sec, 2),
    }


