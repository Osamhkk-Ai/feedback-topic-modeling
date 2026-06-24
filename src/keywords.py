# =====================================================================
# keywords.py  —  Arabic-aware keyword extraction (c-TF-IDF) + auto names.
#
# c-TF-IDF = "class-based TF-IDF": treat each topic (cluster) as one big
# document, then find the words that are characteristic of that topic.
# This is the same idea BERTopic uses; we implement it here so the
# hand-rolled engine has no extra dependencies.
# =====================================================================

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Keep tokens that are >=2 chars and contain word chars OR Arabic letters.
# (?u) = unicode; ؀-ۿ is the Arabic block.
AR_TOKEN_PATTERN = r"(?u)\b[\w؀-ۿ]{2,}\b"

# A small, practical Arabic stopword list (function words / fillers).
ARABIC_STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "كان", "كانت", "ما", "لا", "لم", "لن", "ان", "أن", "إن", "او", "أو",
    "ثم", "كل", "بعد", "قبل", "عند", "هو", "هي", "هم", "انا", "أنا", "نحن", "انت",
    "أنت", "كما", "قد", "كذلك", "حتى", "اذا", "إذا", "لكن", "بل", "يا", "و", "ف",
    "ب", "ل", "ك", "هناك", "بين", "دون", "غير", "فيه", "فيها", "لها", "له", "به",
    "بها", "علي", "اليه", "إليه", "منها", "منه", "عليه", "عليها", "وهو", "وهي",
    "كنت", "يكون", "تكون", "شي", "شيء", "جدا", "جداً", "اي", "أي", "هل", "نعم",
    # --- common Saudi/Gulf/Egyptian dialect fillers (not topical) ---
    "مو", "اللي", "شنو", "وش", "ويش", "بس", "وانا", "ليش", "كذا", "شو", "ايش",
    "وين", "ولا", "مش", "فين", "ايه", "ازاي", "دي", "ده", "عشان", "علشان",
    "يعني", "طيب", "لي", "لسا", "لسه", "للحين", "حقي", "حق", "عاد", "احنا",
    "انتو", "انتم", "هالشي", "كيف", "وشو", "خلاص", "زي", "كده", "برضه", "برضو",
    "عم", "قاعد", "قاعده", "صار", "صارلي", "بقالي", "بقاله", "مره", "مرة",
}


# Negation words must NEVER be treated as stopwords — they flip sentiment.
# We subtract them from BOTH the Arabic list and sklearn's English list
# (which otherwise contains no/not/never/none...).
NEGATION_WORDS = {
    "لا", "ما", "مو", "مب", "ليس", "ليست", "لست", "بدون", "بلا", "لم", "لن",
    "no", "not", "never", "none", "nor", "without", "cannot",
}


def get_stopwords():
    """Combined Arabic + English stopwords, with negation words preserved."""
    english = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
    arabic = set(ARABIC_STOPWORDS) - NEGATION_WORDS
    return list(english | arabic)


def build_vectorizer(config, min_df=None):
    """A CountVectorizer tuned for mixed Arabic/English short text.

    min_df defaults to config.MIN_DF. BERTopic applies the vectorizer to
    the few per-topic grouped documents, so it passes min_df=1 to avoid
    pruning every term.
    """
    return CountVectorizer(
        token_pattern=AR_TOKEN_PATTERN,
        stop_words=get_stopwords(),
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF if min_df is None else min_df,
    )


def ctfidf_keywords(docs, labels, config, topn=10):
    """Return {topic_id: [keyword, ...]} using class-based TF-IDF.

    docs   : list[str]
    labels : array-like of topic ids per doc (-1 means noise; skipped)
    """
    labels = np.asarray(labels)
    unique = sorted({int(t) for t in labels.tolist()})

    vec = build_vectorizer(config)
    try:
        X = vec.fit_transform(docs)                  # (n_docs, n_terms) counts
    except ValueError:
        # happens if vocabulary is empty (e.g. everything filtered out)
        return {t: [] for t in unique if t != -1}

    terms = np.array(vec.get_feature_names_out())
    term_freq_total = np.asarray(X.sum(axis=0)).ravel()   # term freq across all docs
    n_classes = max(len([t for t in unique if t != -1]), 1)
    A = X.sum() / n_classes                                # avg tokens per class

    keywords = {}
    for tid in unique:
        if tid == -1:
            continue
        mask = labels == tid
        class_counts = np.asarray(X[mask].sum(axis=0)).ravel()
        total_in_c = class_counts.sum()
        if total_in_c == 0:
            keywords[tid] = []
            continue
        tf = class_counts / total_in_c
        idf = np.log(1.0 + A / np.maximum(term_freq_total, 1e-9))
        ctfidf = tf * idf
        order = np.argsort(ctfidf)[::-1][:topn]
        keywords[tid] = [terms[i] for i in order if ctfidf[i] > 0]
    return keywords


def auto_topic_name(keywords, n=3):
    """Build a readable machine name from the top keywords.

    e.g. ["ticket", "closed", "solved"] -> "ticket_closed_solved"
    """
    if not keywords:
        return "topic_unknown"
    parts = []
    for kw in keywords[:n]:
        token = kw.replace(" ", "_")
        if token not in parts:
            parts.append(token)
    return "_".join(parts) if parts else "topic_unknown"
