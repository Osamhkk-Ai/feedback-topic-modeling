# =====================================================================
# topic_result.py  —  One shared shape for the output of ANY engine.
#
# Both engines (BERTopic and the hand-rolled sklearn one) return a
# TopicResult, so evaluation / reporting / naming code is identical
# regardless of which engine produced it.
# =====================================================================

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TopicResult:
    engine: str                      # "sklearn" or "bertopic"
    model: str                       # "minilm" or "e5"
    doc_topics: np.ndarray           # topic id per doc, aligned to clean_df rows
    topic_keywords: dict             # {topic_id: [keyword, ...]}
    topic_sizes: dict                # {topic_id: count}  (-1 = noise)
    topic_auto_names: dict           # {topic_id: "kw1_kw2_kw3"}
    runtime_sec: float
    params: dict = field(default_factory=dict)

    # Filled later by the optional local-LLM step (otherwise empty):
    topic_names: dict = field(default_factory=dict)   # {topic_id: "Business Name"}
    topic_meta: dict = field(default_factory=dict)    # {topic_id: {summary, sentiment, action}}

    def topic_ids(self, include_noise=False):
        ids = sorted(self.topic_sizes.keys())
        if not include_noise:
            ids = [t for t in ids if t != -1]
        return ids

    def name_of(self, tid):
        """Best available display name: LLM business name > auto name > fallback."""
        if self.topic_names.get(tid):
            return self.topic_names[tid]
        if tid == -1:
            return "noise"
        return self.topic_auto_names.get(tid, f"topic_{tid}")
