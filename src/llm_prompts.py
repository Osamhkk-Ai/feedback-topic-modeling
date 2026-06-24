# =====================================================================
# llm_prompts.py  —  Builds the EXACT text sent to the local LLM, plus
# strict validation / retry / fallback helpers.
# =====================================================================

import json
import re

import numpy as np

from src import sampling

# Schema CONSTRAINS the LLM to valid JSON with all fields + allowed enums.
TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "business_name": {"type": "string"},
        "description": {"type": "string"},
        "root_cause": {"type": "string"},
        "severity": {"type": "string", "enum": ["عالية", "متوسطة", "منخفضة"]},
        "sentiment": {"type": "string", "enum": ["إيجابي", "سلبي", "محايد", "مختلط"]},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["business_name", "description", "root_cause",
                 "severity", "sentiment", "recommended_actions"],
}

SEVERITY_VALUES = {"عالية", "متوسطة", "منخفضة"}
SENTIMENT_VALUES = {"إيجابي", "سلبي", "محايد", "مختلط"}
GENERIC_NAMES = {"مشكلة عامة", "خدمة العملاء", "مشكلة", "خدمة", "عام", "تعليقات",
                 "شكوى", "موضوع", "ملاحظات", "غير محدد"}

_EXAMPLE = (
    'التعليقات: "التطبيق بطيء ويعلق"، "شاشة بيضاء وأسكره"، "ثقل بعد التحديث"\n'
    '{"business_name":"بطء التطبيق وتعليقه","description":"العملاء يعانون من بطء التطبيق '
    'وتعليقه المتكرر، وبعضهم يضطر لإغلاقه. تظهر التعليقات أن المشكلة مرتبطة بتجربة استخدام '
    'مزعجة بعد التحديث.","root_cause":"تراجع أداء التطبيق بعد التحديث الأخير حسب ما يظهر من '
    'التعليقات.","severity":"متوسطة","sentiment":"سلبي","recommended_actions":'
    '["مراجعة أداء التطبيق بعد التحديث الأخير","اختبار التطبيق على الأجهزة الأكثر استخدامًا",'
    '"إصدار تحسين عاجل لتقليل التعليق والبطء"]}'
)


def build_topic_prompt(keywords, samples, config, topic_size=None, topic_pct=None, duplicate_note=""):
    """The per-topic analysis prompt (one topic only)."""
    kw = "، ".join(keywords[:10])
    sample_block = "\n".join(f"- {s}" for s in samples[: config.LLM_SAMPLES_PER_TOPIC])

    stats_block = ""
    if topic_size is not None:
        stats_block += f"حجم الموضوع: {topic_size} تعليق ({topic_pct}% من الإجمالي).\n"
    if duplicate_note:
        stats_block += duplicate_note + "\n"

    return (
        "أنت محلل تجربة عملاء. حلّل التعليقات التالية حول موضوع واحد فقط، وأخرج JSON فقط.\n\n"
        "التعليقات قد تكون عربية أو إنجليزية أو مختلطة، وقد تحتوي على لهجات أو أخطاء كتابية.\n\n"
        "قاعدة مهمة: حجم الموضوع ونسبته يدلان على كثرة التكرار فقط، ولا يرفعان severity "
        "وحدهما. حدّد الخطورة بناءً على نوع المشكلة وأثرها على العميل.\n\n"
        "المشاعر:\n"
        '- مدح/شكر/رضا ← "إيجابي"\n'
        '- شكوى/معاناة ← "سلبي"\n'
        '- سؤال/استفسار فقط ← "محايد"\n'
        '- إيجابي وسلبي معًا ← "مختلط"\n\n'
        "الخطورة:\n"
        '- "عالية" فقط للمشكلات الحرجة: مشاكل مالية أو دفع، منع إكمال الخدمة، حجب الوصول '
        "للحساب، مخاطر قانونية أو امتثال، أو فشل متكرر يمنع الإنجاز.\n"
        '- "متوسطة" لمشكلات مزعجة غير حرجة: تأخير، ضعف متابعة، غموض الإجراءات، إزعاج.\n'
        '- "منخفضة" للمدح والشكر والاستفسارات البسيطة والاقتراحات الطفيفة.\n\n'
        "التعليمات:\n"
        "- اعتمد على التعليقات الممثلة كدليل رئيسي، والكلمات المفتاحية كدليل مساعد فقط.\n"
        "- لا تفترض أن كل موضوع شكوى.\n"
        "- لا تضف معلومات غير مذكورة في التعليقات.\n"
        "- business_name: اسم مميز من ٢ إلى ٤ كلمات عربية، بدون أسماء عامة.\n"
        "- description: جملتان إلى ثلاث جمل تصف ما يشعر به العميل فعليًا.\n"
        '- root_cause: السبب الجذري الظاهر، وإذا لم يكن واضحًا اكتب "غير واضح من التعليقات المتاحة".\n'
        "- recommended_actions: ثلاثة إجراءات عملية محددة بالضبط (٣ عناصر).\n"
        "- كل القيم بالعربية الفصحى.\n"
        "- أخرج كائن JSON واحد فقط يبدأ بـ { وينتهي بـ }، بدون أي نص أو Markdown.\n\n"
        "مثال للتنسيق فقط، لا تنسخه:\n" + _EXAMPLE + "\n\n"
        "المطلوب:\n" + stats_block +
        f"الكلمات المفتاحية: {kw}\n"
        f"التعليقات الممثلة:\n{sample_block}\n\n"
        "أخرج JSON فقط:"
    )


def build_retry_prompt(base_prompt, errors):
    """Stricter correction prompt after an invalid output."""
    err = "؛ ".join(errors)
    return (
        base_prompt + "\n\n"
        f"تنبيه: المخرج السابق كان غير صالح بسبب: {err}.\n"
        "صحّح والتزم تمامًا: ٦ مفاتيح بالضبط، severity من (عالية|متوسطة|منخفضة)، "
        "sentiment من (إيجابي|سلبي|محايد|مختلط)، recommended_actions ٣ عناصر بالضبط، "
        "business_name من ٢-٤ كلمات عربية ومميز وغير عام. أرجع JSON فقط."
    )


def build_topic_payload(result, clean_df, embeddings, tid, config):
    """Everything the LLM gets for one topic (also used for the preview file)."""
    labels = np.asarray(result.doc_topics)
    n = labels.size
    size = int(result.topic_sizes.get(tid, 0))
    pct = round(100 * size / max(n, 1), 1)
    samples = sampling.representative_comments(result, embeddings, clean_df, tid,
                                               config.LLM_SAMPLES_PER_TOPIC)
    idx = np.where(labels == tid)[0]
    freq = clean_df["frequency"].to_numpy()
    dup = int((freq[idx] > 1).sum()) if idx.size else 0
    dup_note = ""
    if dup > 0:
        dup_note = (f"ملاحظة: يحتوي الموضوع على {dup} تعليقًا متكررًا "
                    "(التكرار إشارة طلب، ولا يرفع الخطورة).")
    kws = result.topic_keywords.get(tid, [])
    prompt = build_topic_prompt(kws, samples, config, topic_size=size,
                                topic_pct=pct, duplicate_note=dup_note)
    return {
        "topic_id": int(tid),
        "auto_name": result.topic_auto_names.get(tid, ""),
        "topic_size": size,
        "topic_percentage": pct,
        "keywords": kws,
        "samples": samples,
        "duplicate_note": dup_note,
        "prompt": prompt,
    }


def build_llm_inputs(result, clean_df, embeddings, config):
    return [build_topic_payload(result, clean_df, embeddings, tid, config)
            for tid in result.topic_ids()]


def build_exec_prompt(result, config):
    """Executive-summary prompt: all topics + sizes + descriptions/keywords."""
    lines = []
    for tid in result.topic_ids():
        name = result.name_of(tid)
        size = result.topic_sizes.get(tid, 0)
        meta = result.topic_meta.get(tid, {})
        detail = meta.get("description") or "، ".join(result.topic_keywords.get(tid, [])[:6])
        lines.append(f"- {name} (عدد={size}): {detail}")
    topic_block = "\n".join(lines)
    return (
        "أنت محلل تجربة عملاء. هذه المواضيع المكتشفة من ملاحظات العملاء:\n\n"
        f"{topic_block}\n\n"
        "اكتب ملخصًا تنفيذيًا موجزًا بالعربية يحتوي على:\n"
        "1) أبرز ٣-٥ نتائج،\n2) ٣-٥ توصيات عملية.\n"
        "استخدم نقاطًا قصيرة."
    )


# ---------------------------------------------------------------------
# Validation + fallback
# ---------------------------------------------------------------------
def normalize_actions(actions):
    if isinstance(actions, list):
        return [str(a).strip() for a in actions if str(a).strip()]
    s = str(actions).strip()
    return [s] if s else []


def validate_topic_output(data):
    """Return (is_valid, errors) per the strict output contract."""
    errors = []
    for k in ["business_name", "description", "root_cause",
              "severity", "sentiment", "recommended_actions"]:
        if k not in data:
            errors.append(f"المفتاح ناقص: {k}")

    if str(data.get("severity", "")).strip() not in SEVERITY_VALUES:
        errors.append("severity غير صحيح")
    if str(data.get("sentiment", "")).strip() not in SENTIMENT_VALUES:
        errors.append("sentiment غير صحيح")

    acts = normalize_actions(data.get("recommended_actions", []))
    if len(acts) != 3:
        errors.append("recommended_actions يجب أن تكون ٣ عناصر")

    name = str(data.get("business_name", "")).strip()
    wc = len(name.split())
    if not name or name in GENERIC_NAMES or wc < 2 or wc > 4:
        errors.append("business_name يجب أن يكون ٢-٤ كلمات ومميزًا")

    return (len(errors) == 0, errors)


def fallback_from_keywords(keywords, samples):
    """Deterministic fallback when the LLM keeps failing validation."""
    words = []
    for kw in keywords[:4]:
        for w in kw.split():
            if w not in words:
                words.append(w)
    name = " ".join(words[:3]) if len(words) >= 2 else (words[0] if words else "موضوع غير محدد")
    if len(name.split()) < 2:
        name = name + " للعملاء"
    return {
        "business_name": name,
        "description": "تجميع تلقائي من الكلمات المفتاحية والتعليقات الممثلة (تعذّر تحليل النموذج).",
        "root_cause": "غير واضح من التعليقات المتاحة",
        "severity": "متوسطة",
        "sentiment": "محايد",
        "recommended_actions": [
            "مراجعة تعليقات هذا الموضوع يدويًا",
            "تحديد الإجراء المناسب حسب المحتوى",
            "متابعة معدّل تكرار الموضوع",
        ],
    }


def parse_json(text):
    """Pull the first {...} JSON object out of an LLM response (robust).

    Strips Qwen3 <think>...</think> reasoning blocks first so they don't
    interfere with JSON extraction.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None
