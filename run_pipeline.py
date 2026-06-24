# =====================================================================
# run_pipeline.py  —  SINGLE entry point. Run:  python run_pipeline.py
#
#   load -> clean -> embeddings -> UMAP -> HDBSCAN (auto params)
#   -> topic quality check -> c-TF-IDF keywords -> [SAVE pre-LLM results]
#   -> local Qwen analysis (incremental + resumable) -> final report
#
# Design: everything BEFORE the LLM is computed and SAVED to disk first.
# The slow LLM stage then runs on the saved result and writes each topic
# as it finishes. So if RAM fills or the LLM stops, the pre-LLM results
# are kept and a re-run resumes without re-clustering or re-naming.
#
# Runs on CPU end-to-end (GPU used automatically if the CUDA build exists).
# Edit settings in config.py — no need to touch the code.
# =====================================================================

import os
import time

import config
from src import (data_loader, cleaning, embeddings, evaluation, quality,
                 reporting, checkpoint)
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
    run_dir = config.OUTPUT_DIR
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
    sig = checkpoint.docs_signature(docs)

    # 3) EMBED  (cached on disk) ---------------------------------------
    stage(3, TOTAL, f"Embeddings ({config.EMBEDDING}, CPU)")
    emb = embeddings.embed(docs, config)

    # 4) CLUSTER  (reuse checkpoint if the data is unchanged) ----------
    stage(4, TOTAL, "Cluster (UMAP -> HDBSCAN)")
    ck = checkpoint.load_result(sig, config)
    if ck:
        result, metrics, _ = ck
        print("[checkpoint] reused saved pre-LLM result (skipped clustering)")
    else:
        result = run_sklearn(docs, emb, config, config.EMBEDDING)
        metrics = evaluation.evaluate(result, emb, docs, config)
        checkpoint.save_result(result, metrics, stats, sig, config)

    # 5) QUALITY + SAVE PRE-LLM RESULTS --------------------------------
    stage(5, TOTAL, "Topic quality + SAVE pre-LLM results to disk")
    quality_df = quality.topic_quality_check(result, emb, clean_df, config)
    reporting.write_run_outputs(run_dir, clean_df, result, metrics, stats, config,
                                embeddings=emb, quality_df=quality_df, llm_outputs=None)
    reporting.write_llm_inputs(run_dir, result, clean_df, emb, config)
    print("[saved] pre-LLM results are on disk (kept even if the LLM stops)")

    # 6) LLM ANALYSIS (incremental + resumable) + FINAL REPORT ---------
    stage(6, TOTAL, "Local LLM analysis + final report")
    llm_outputs, exec_text = None, None
    if config.LLM_ENABLED:
        from src import naming
        backend = naming.load_backend(config)
        if backend:
            llm_path = os.path.join(run_dir, "llm_output.json")
            llm_outputs = naming.apply_naming(result, clean_df, emb, config, backend,
                                              out_path=llm_path)
            exec_text = naming.executive_report(result, config, backend)
            # re-write reports now WITH the business names/sentiment/severity
            reporting.write_run_outputs(run_dir, clean_df, result, metrics, stats, config,
                                        embeddings=emb, quality_df=quality_df,
                                        llm_outputs=llm_outputs)
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
