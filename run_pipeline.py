# =====================================================================
# run_pipeline.py  —  SINGLE entry point. Run:  python run_pipeline.py
#
#   load -> clean (keep originals + duplicates) -> embeddings -> UMAP
#   -> HDBSCAN (auto params, KMeans fallback) -> topic quality check
#   -> c-TF-IDF keywords -> representative comments -> local Qwen analysis
#   (validate/retry/fallback) -> reports + charts + console summary
#
# Works on CPU end-to-end (GPU used automatically if the CUDA build is
# present). Edit settings in config.py — no need to touch the code.
# =====================================================================

import os
import time

import config
from src import data_loader, cleaning, embeddings, evaluation, quality, reporting
from src.engine_sklearn import run as run_sklearn


def stage(n, total, title):
    print("\n" + "=" * 64)
    print(f"  STAGE {n}/{total}  —  {title}")
    print("=" * 64)


def final_summary(stats, result, metrics, run_dir, seconds):
    p = result.params
    sizes = {t: s for t, s in result.topic_sizes.items() if t != -1}
    largest = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n" + "#" * 64)
    print("  FINAL SUMMARY")
    print("#" * 64)
    print(f"  total rows          : {stats['n_original']}")
    print(f"  clean rows          : {stats['n_after']}")
    print(f"  duplicates (kept)   : {stats['duplicate_count']}")
    print(f"  topics discovered   : {metrics['n_topics']}")
    print(f"  noise               : {metrics['n_noise']} ({round(metrics['noise_rate']*100,1)}%)")
    print(f"  largest topics      : " + "، ".join(f"{result.name_of(t)}({s})" for t, s in largest))
    print(f"  KMeans fallback     : {'YES' if p.get('kmeans_fallback') else 'NO'}")
    print(f"  dataset tier        : {p.get('tier','')}")
    if p.get("warning"):
        print(f"  WARNING             : {p['warning']}")
    print(f"  total runtime       : {round(seconds,1)}s")
    print(f"  output folder       : {os.path.abspath(run_dir)}")
    print("#" * 64 + "\n")


def run():
    t0 = time.time()
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    TOTAL = 6

    # 1) LOAD ----------------------------------------------------------
    stage(1, TOTAL, "Load data")
    df = data_loader.load_dataframe(config.INPUT_PATH)
    col = data_loader.pick_comment_column(df, config.COMMENT_COLUMN)

    # 2) CLEAN ---------------------------------------------------------
    stage(2, TOTAL, "Clean text (keep originals + duplicates)")
    clean_df, stats = cleaning.clean_comments(df, col, config.MIN_WORDS)
    cleaning.print_stats(stats)
    if stats["n_after"] == 0:
        raise ValueError("Nothing left after cleaning. Check COMMENT_COLUMN / MIN_WORDS.")
    docs = clean_df["cleaned_text"].tolist()

    # 3) EMBED ---------------------------------------------------------
    stage(3, TOTAL, f"Embeddings ({config.EMBEDDING}, CPU)")
    emb = embeddings.embed(docs, config)

    # 4) CLUSTER -------------------------------------------------------
    stage(4, TOTAL, "Cluster (UMAP -> HDBSCAN)")
    result = run_sklearn(docs, emb, config, config.EMBEDDING)
    metrics = evaluation.evaluate(result, emb, docs, config)

    # 5) QUALITY + LLM -------------------------------------------------
    stage(5, TOTAL, "Topic quality check + local LLM analysis")
    quality_df = quality.topic_quality_check(result, emb, clean_df, config)
    llm_outputs, exec_text = None, None
    if config.LLM_ENABLED:
        from src import naming
        backend = naming.load_backend(config)
        if backend:
            llm_outputs = naming.apply_naming(result, clean_df, emb, config, backend)
            exec_text = naming.executive_report(result, config, backend)

    # 6) EXPORT  (single flat folder, no duplication) -----------------
    stage(6, TOTAL, "Reports + charts")
    run_dir = config.OUTPUT_DIR
    reporting.write_run_outputs(run_dir, clean_df, result, metrics, stats, config,
                                embeddings=emb, quality_df=quality_df, llm_outputs=llm_outputs)
    reporting.write_llm_inputs(run_dir, result, clean_df, emb, config)
    if exec_text:
        with open(os.path.join(run_dir, "executive_summary.txt"), "w", encoding="utf-8") as f:
            f.write(exec_text)
    reporting.print_validation(clean_df, result, config, embeddings=emb)

    final_summary(stats, result, metrics, run_dir, time.time() - t0)


def main():
    try:
        run()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] File not found: {exc}\n-> Fix INPUT_PATH in config.py")
    except KeyError as exc:
        print(f"\n[ERROR] Column problem: {exc}")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
