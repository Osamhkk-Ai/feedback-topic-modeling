# =====================================================================
# reporting.py  —  Build deliverables: CSVs, Excel report, charts,
# console validation, and the cross-run comparison report.
# =====================================================================

import os
import json
import shutil
import numpy as np
import pandas as pd
from openpyxl.drawing.image import Image as XLImage

from src import charts, sampling

# feedback_with_topics columns (per requirements)
FEEDBACK_COLS = ["original_row_index", "original_comment", "cleaned_text",
                 "topic_id", "topic_name", "keywords", "sentiment", "severity"]
SAMPLE_COLS = ["topic_id", "topic_name", "sample_comment"]


def _rep_count(size, config):
    return min(config.REP_COMMENTS_MAX, max(config.REP_COMMENTS_MIN, min(size, config.REP_COMMENTS_MAX)))


# ---------------------------------------------------------------------
# Build the per-run tables
# ---------------------------------------------------------------------
def build_frames(clean_df, result, config, embeddings=None):
    df = clean_df.reset_index(drop=True).copy()
    labels = np.asarray(result.doc_topics)
    df["topic_id"] = labels

    def kws_of(t):
        return ", ".join(result.topic_keywords.get(t, []))

    def sev_of(t):
        return result.topic_meta.get(t, {}).get("severity", "")

    def sent_of(t):
        return result.topic_meta.get(t, {}).get("sentiment", "")

    df["topic_name"] = df["topic_id"].map(result.name_of)
    df["keywords"] = df["topic_id"].map(kws_of)
    df["sentiment"] = df["topic_id"].map(sent_of)
    df["severity"] = df["topic_id"].map(sev_of)

    feedback = df[FEEDBACK_COLS].copy()

    n_assigned = int((labels != -1).sum())
    summary_rows, sample_rows = [], []
    for t in result.topic_ids(include_noise=False):
        sub = df[df["topic_id"] == t]
        count = len(sub)
        pct = round(100.0 * count / max(n_assigned, 1), 2)
        meta = result.topic_meta.get(t, {})

        # representative comments (centroid-based + deduped) when we have embeddings
        if embeddings is not None:
            reps = sampling.representative_comments(result, embeddings, clean_df, t,
                                                    _rep_count(count, config))
        else:
            reps = sub["original_comment"].head(config.REP_COMMENTS_MIN).tolist()

        row = {
            "topic_id": t,
            "topic_name": result.name_of(t),
            "count": count,
            "percentage": pct,
            "keywords": kws_of(t),
            "business_name": meta.get("business_name", result.name_of(t)),
            "description": meta.get("description", ""),
            "root_cause": meta.get("root_cause", ""),
            "sentiment": meta.get("sentiment", ""),
            "severity": meta.get("severity", ""),
            "recommended_actions": meta.get("recommended_actions", ""),
            "top_sample_1": reps[0] if len(reps) > 0 else "",
            "top_sample_2": reps[1] if len(reps) > 1 else "",
            "top_sample_3": reps[2] if len(reps) > 2 else "",
        }
        summary_rows.append(row)

        for s in reps:
            sample_rows.append({"topic_id": t, "topic_name": result.name_of(t),
                                "sample_comment": s})

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("count", ascending=False).reset_index(drop=True)
    samples = pd.DataFrame(sample_rows, columns=SAMPLE_COLS)
    noise = df[df["topic_id"] == -1][
        ["original_row_index", "original_comment", "cleaned_text", "frequency"]
    ].copy()

    return feedback, summary, samples, noise, df


# ---------------------------------------------------------------------
# Overview + Settings sheets
# ---------------------------------------------------------------------
def build_overview(result, metrics, stats, config):
    p = result.params
    sizes = {t: s for t, s in result.topic_sizes.items() if t != -1}
    largest = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]
    largest_str = "، ".join(f"{result.name_of(t)} ({s})" for t, s in largest)
    rows = [
        ("total_rows", stats["n_original"]),
        ("clean_rows", stats["n_after"]),
        ("duplicate_count", stats["duplicate_count"]),
        ("number_of_topics", metrics["n_topics"]),
        ("noise_count", metrics["n_noise"]),
        ("noise_percentage", f"{round(metrics['noise_rate']*100, 2)}%"),
        ("largest_topics", largest_str),
        ("kmeans_fallback_triggered", "YES" if p.get("kmeans_fallback") else "NO"),
        ("dataset_tier", p.get("tier", "")),
        ("smoke_test", "YES" if p.get("is_smoke_test") else "NO"),
        ("warning", p.get("warning", "")),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_settings(result, metrics, stats, config):
    p = result.params
    rows = [
        ("embedding_model", config.EMBEDDING_MODELS.get(result.model, result.model)),
        ("engine", result.engine),
        ("llm_model", config.LLM_FILE if config.LLM_ENABLED else "disabled"),
        ("dim_reducer", p.get("reducer", "")),
        ("umap_n_neighbors", p.get("umap_n_neighbors", "")),
        ("hdbscan_min_cluster_size", p.get("min_cluster_size", "")),
        ("hdbscan_min_samples", p.get("min_samples", "")),
        ("clusterer_used", p.get("clusterer", "")),
        ("kmeans_fallback", p.get("kmeans_fallback", False)),
        ("dataset_tier", p.get("tier", "")),
        ("n_original_rows", stats["n_original"]),
        ("n_rows_after_cleaning", stats["n_after"]),
        ("removed_null", stats["removed_null"]),
        ("removed_short", stats["removed_short"]),
        ("duplicate_count", stats["duplicate_count"]),
        ("total_topics", metrics["n_topics"]),
        ("noise_comments", metrics["n_noise"]),
        ("noise_rate", metrics["noise_rate"]),
        ("silhouette", metrics["silhouette"]),
        ("npmi_coherence", metrics["npmi_coherence"]),
        ("runtime_sec", metrics["runtime_sec"]),
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


# ---------------------------------------------------------------------
# Write one run's deliverables
# ---------------------------------------------------------------------
def write_run_outputs(run_dir, clean_df, result, metrics, stats, config,
                      embeddings=None, quality_df=None, llm_outputs=None):
    os.makedirs(run_dir, exist_ok=True)
    feedback, summary, samples, noise, _ = build_frames(clean_df, result, config, embeddings)

    # --- CSVs (utf-8-sig so Arabic shows correctly in Excel) ---
    feedback.to_csv(os.path.join(run_dir, "feedback_with_topics.csv"),
                    index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(run_dir, "topic_summary.csv"),
                   index=False, encoding="utf-8-sig")
    samples.to_csv(os.path.join(run_dir, "topic_samples.csv"),
                   index=False, encoding="utf-8-sig")
    noise.to_csv(os.path.join(run_dir, "noise_comments.csv"),
                 index=False, encoding="utf-8-sig")
    if quality_df is not None:
        quality_df.to_csv(os.path.join(run_dir, "quality_check.csv"),
                          index=False, encoding="utf-8-sig")
    if llm_outputs is not None:
        with open(os.path.join(run_dir, "llm_output.json"), "w", encoding="utf-8") as f:
            json.dump(llm_outputs, f, ensure_ascii=False, indent=2)

    # --- charts ---
    chart_dir = os.path.join(run_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    top_png = os.path.join(chart_dir, "top_topics.png")
    dist_png = os.path.join(chart_dir, "topic_distribution.png")
    noise_png = os.path.join(chart_dir, "noise_percentage.png")
    charts.top10_bar(result.topic_sizes, top_png)
    charts.pct_distribution(result.topic_sizes, dist_png)
    charts.noise_bar(result.topic_sizes, noise_png)

    overview = build_overview(result, metrics, stats, config)
    settings = build_settings(result, metrics, stats, config)

    xlsx = os.path.join(run_dir, "topic_modeling_report.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:    # engine pinned on purpose
        overview.to_excel(writer, sheet_name="Overview", index=False)
        feedback.to_excel(writer, sheet_name="Feedback With Topics", index=False)
        summary.to_excel(writer, sheet_name="Topic Summary", index=False)
        samples.to_excel(writer, sheet_name="Representative Comments", index=False)
        noise.to_excel(writer, sheet_name="Noise Comments", index=False)
        if quality_df is not None:
            quality_df.to_excel(writer, sheet_name="Quality Check", index=False)
        settings.to_excel(writer, sheet_name="Settings", index=False)
        ws = writer.book.create_sheet("Charts")
        try:
            ws.add_image(XLImage(top_png), "A1")
            ws.add_image(XLImage(dist_png), "A30")
            ws.add_image(XLImage(noise_png), "A60")
        except Exception as exc:
            ws["A1"] = f"charts unavailable: {exc}"

    print(f"[reporting] wrote outputs -> {run_dir}")
    return xlsx


# ---------------------------------------------------------------------
# LLM input preview (exact payload per topic)
# ---------------------------------------------------------------------
def write_llm_inputs(run_dir, result, clean_df, embeddings, config):
    from src import llm_prompts
    items = llm_prompts.build_llm_inputs(result, clean_df, embeddings, config)

    with open(os.path.join(run_dir, "llm_input.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    lines = ["هذا هو المُدخَل الذي يُرسَل للنموذج اللغوي لكل موضوع.\n"]
    for it in items:
        lines.append("=" * 70)
        lines.append(f"topic_id={it['topic_id']} | size={it['topic_size']} "
                     f"({it['topic_percentage']}%) | auto_name={it['auto_name']}")
        lines.append("=" * 70)
        lines.append(it["prompt"])
        lines.append("")
    with open(os.path.join(run_dir, "llm_input.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[reporting] wrote LLM input preview -> {run_dir}\\llm_input.json (+ .txt)")


# ---------------------------------------------------------------------
# Console validation block
# ---------------------------------------------------------------------
def print_validation(clean_df, result, config, embeddings=None):
    _, summary, _, _, df = build_frames(clean_df, result, config, embeddings)
    labels = np.asarray(result.doc_topics)
    n_topics = len(result.topic_ids())
    n_noise = int((labels == -1).sum())

    print("\n" + "=" * 72)
    print(f" VALIDATION  |  engine={result.engine}  model={result.model}")
    print(f" topics={n_topics}   noise={n_noise}   runtime={result.runtime_sec:.1f}s")
    print("=" * 72)
    if summary.empty:
        print(" No topics formed. Lower min_cluster_size / check data.")
        return
    for _, r in summary.head(config.VALIDATION_TOP_TOPICS).iterrows():
        print(f"\n[Topic {r['topic_id']}] {r['topic_name']}  "
              f"(count={r['count']}, {r['percentage']}%)  "
              f"[{r.get('sentiment','')}/{r.get('severity','')}]")
        print(f"   keywords: {r['keywords']}")
        sub = df[df["topic_id"] == r["topic_id"]]["original_comment"].head(
            config.VALIDATION_SAMPLES).tolist()
        for i, s in enumerate(sub, 1):
            s = (s[:120] + "…") if len(s) > 120 else s
            print(f"   {i}. {s}")
    print()


# ---------------------------------------------------------------------
# Copy recommended run's deliverables to outputs/ root
# ---------------------------------------------------------------------
def copy_to_root(run_dir, config):
    for fname in ["feedback_with_topics.csv", "topic_summary.csv", "topic_samples.csv",
                  "noise_comments.csv", "topic_modeling_report.xlsx",
                  "llm_input.json", "llm_output.json"]:
        src_path = os.path.join(run_dir, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(config.OUTPUT_DIR, fname))
    print(f"[reporting] copied deliverables to {config.OUTPUT_DIR}/")
