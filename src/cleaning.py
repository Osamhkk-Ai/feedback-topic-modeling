# =====================================================================
# cleaning.py  —  Careful cleaning that KEEPS the original feedback.
#
# Key principles (per requirements):
#   - never overwrite the original comment (kept as `original_comment`)
#   - produce a separate `cleaned_text` column
#   - DO NOT drop duplicates: repeated comments are a business demand
#     signal. We keep them all and add a `frequency` count instead.
#   - remove URLs, emojis, repeated punctuation, excess whitespace
#   - light Arabic normalization (no meaning change)
#   - NEVER strip negation words (لا، ما، مو، ليس، not, no, never) — those
#     are removed nowhere; negation matters for sentiment.
# =====================================================================

import re
import pandas as pd

# Arabic diacritics (tashkeel), superscript alef, Quranic marks.
# Written as \u escapes ON PURPOSE so the ranges are unambiguous and do
# NOT include Arabic letters (U+0621–U+064A) or digits (U+0660–U+0669).
_AR_DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]"
)
_TATWEEL = re.compile("ـ")                     # kashida / elongation char
_URL = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")
_ELONGATION = re.compile(r"(.)\1{2,}")              # 3+ repeats -> 2 ("حلوووو")
_REPEAT_PUNCT = re.compile("([!?.,:\\-_،؛]){2,}")   # "!!!" -> "!"
# Emoji / pictographs / symbols to drop (keep letters, digits, punctuation):
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols, pictographs, emoji
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # flags
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002190-\U000021FF"   # arrows
    "]+",
    flags=re.UNICODE,
)


def normalize_spaces(text):
    return _SPACES.sub(" ", text).strip()


def normalize_arabic(text):
    """Light Arabic normalization (alef/ya/ta unification, diacritics, tatweel)."""
    text = _AR_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = (text.replace("أ", "ا")    # أ -> ا
                .replace("إ", "ا")    # إ -> ا
                .replace("آ", "ا")    # آ -> ا
                .replace("ى", "ي")    # ى -> ي
                .replace("ة", "ه"))   # ة -> ه
    text = _ELONGATION.sub(r"\1\1", text)
    return text


def clean_text(text):
    """Produce cleaned_text WITHOUT changing meaning or dropping negation."""
    text = str(text)
    text = _URL.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = normalize_arabic(text)
    text = normalize_spaces(text)
    return text


def _word_count(text):
    return len(text.split())


def clean_comments(df, col, min_words=3):
    """Return (clean_df, stats).

    clean_df columns: original_row_index, original_comment, cleaned_text, frequency
    Duplicates are KEPT (frequency = how many times the cleaned_text appears).
    stats keys: n_original, removed_null, removed_short, duplicate_count, n_after
    """
    n_original = len(df)

    work = pd.DataFrame({
        "original_row_index": df.index,
        "original_comment": df[col].astype("string"),
    })

    # 1) null / empty
    is_null = work["original_comment"].isna() | (work["original_comment"].str.strip() == "")
    removed_null = int(is_null.sum())
    work = work[~is_null].copy()

    # 2) clean
    work["cleaned_text"] = work["original_comment"].map(clean_text)
    empty_after = work["cleaned_text"].str.strip() == ""
    removed_null += int(empty_after.sum())
    work = work[~empty_after].copy()

    # 3) very short comments
    is_short = work["cleaned_text"].map(_word_count) < min_words
    removed_short = int(is_short.sum())
    work = work[~is_short].copy()

    # 4) KEEP duplicates — just count them as a demand signal
    work = work.reset_index(drop=True)
    freq = work["cleaned_text"].map(work["cleaned_text"].value_counts())
    work["frequency"] = freq.astype(int)
    duplicate_count = int(len(work) - work["cleaned_text"].nunique())

    work["original_comment"] = work["original_comment"].astype(str)
    clean_df = work[["original_row_index", "original_comment", "cleaned_text", "frequency"]].copy()

    stats = {
        "n_original": n_original,
        "removed_null": removed_null,
        "removed_short": removed_short,
        "duplicate_count": duplicate_count,
        "n_after": len(clean_df),
    }
    return clean_df, stats


def print_stats(stats):
    print("\n[cleaning] ----------------------------------------")
    print(f"  original rows           : {stats['n_original']}")
    print(f"  removed null/empty      : {stats['removed_null']}")
    print(f"  removed very short      : {stats['removed_short']}")
    print(f"  duplicates kept (signal): {stats['duplicate_count']}")
    print(f"  rows after cleaning     : {stats['n_after']}")
    print("[cleaning] ----------------------------------------\n")
