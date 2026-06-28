# =====================================================================
# run_llm.py  —  SEPARATE LLM stage (run AFTER the embedding results).
#
# Why a separate script:
#   The embedding + clustering part is the heavy/deterministic step. Do it
#   first, review the topics, THEN run the (slow) local LLM on its own.
#   This file does NOT change run_pipeline.py — it just reuses the same
#   modules. The embeddings are read from the on-disk cache, so the slow
#   step is not repeated, and the LLM stage is resumable (each topic is
#   saved as it finishes; re-running skips topics already done).
#
# How to use (CMD):
#   1) First produce the embeddings + topics (LLM off):
#        - set  LLM_ENABLED = False  in config.py
#        - python run_pipeline.py
#        (this clusters the comments and writes the base reports)
#   2) Then run the model separately:
#        - set  LLM_ENABLED = True   in config.py
#        - python run_llm.py
#        (this loads the cached embeddings + topics and adds the LLM
#         names/summaries, then rewrites the reports with them)
#
# Run it again any time to resume if it was interrupted.
# =====================================================================

import os

import config
from src import (data_loader, cleaning, embeddings, evaluation,
                 quality, reporting, checkpoint)
from src.engine_sklearn import run as run_sklearn


def main():
    if not config.LLM_ENABLED:
        print("[run_llm] LLM_ENABLED is False in config.py — set it to True first.")
        return

    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 1) load + clean (fast, deterministic) -----------------------------
    print("=" * 60 + "\n  run_llm: load + clean\n" + "=" * 60)
    df = data_loader.load_dataframe(config.INPUT_PATH)
    col = data_loader.pick_comment_column(df, config.COMMENT_COLUMN)
    clean_df, stats = cleaning.clean_comments(df, col, config.MIN_WORDS)
    cleaning.print_stats(stats)
    if stats["n_after"] == 0:
        print("[run_llm] nothing to do after cleaning.")
        return
    docs = clean_df["cleaned_text"].tolist()

    # 2) embeddings from CACHE (the slow part is already done) -----------
    print("=" * 60 + f"\n  run_llm: embeddings ({config.EMBEDDING}, from cache)\n" + "=" * 60)
    emb = embeddings.embed(docs, config)

    # 3) topics: from checkpoint if available, else cluster (and save) ---
    sig = checkpoint.docs_signature(docs)
    loaded = checkpoint.load_result(sig, config)
    if loaded:
        result, metrics, _ = loaded
        print("[run_llm] loaded clustering from checkpoint (no re-clustering)")
    else:
        print("[run_llm] no checkpoint -> clustering once")
        result = run_sklearn(docs, emb, config, config.EMBEDDING)
        metrics = evaluation.evaluate(result, emb, docs, config)
        checkpoint.save_result(result, metrics, stats, sig, config)

    # 4) quality check --------------------------------------------------
    quality_df = quality.topic_quality_check(result, emb, clean_df, config)

    # 5) LLM analysis (resumable, saved per topic) ----------------------
    print("=" * 60 + "\n  run_llm: local LLM analysis\n" + "=" * 60)
    from src import naming
    backend = naming.load_backend(config)
    if backend is None:
        print("[run_llm] LLM backend could not load. Check models / config.")
        return
    out_path = os.path.join(config.OUTPUT_DIR, "llm_output.json")
    llm_outputs = naming.apply_naming(result, clean_df, emb, config, backend, out_path=out_path)
    exec_text = naming.executive_report(result, config, backend)

    # 6) rewrite reports WITH the LLM names -----------------------------
    print("=" * 60 + "\n  run_llm: writing reports\n" + "=" * 60)
    reporting.write_run_outputs(config.OUTPUT_DIR, clean_df, result, metrics, stats, config,
                                embeddings=emb, quality_df=quality_df, llm_outputs=llm_outputs)
    reporting.write_llm_inputs(config.OUTPUT_DIR, result, clean_df, emb, config)
    if exec_text:
        with open(os.path.join(config.OUTPUT_DIR, "executive_summary.txt"), "w", encoding="utf-8") as f:
            f.write(exec_text)
    reporting.print_validation(clean_df, result, config, embeddings=emb)
    print(f"\n[run_llm] DONE. LLM results in {os.path.abspath(config.OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
