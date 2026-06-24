# =====================================================================
# naming.py  —  OPTIONAL local-LLM topic analysis (after clustering).
#
# Uses llama-cpp to run a quantized Qwen model fully OFFLINE (no server,
# no cloud). The model runs once PER TOPIC (never per comment), with
# validation -> stricter retry -> deterministic fallback. Degrades
# gracefully: if the model can't load, automatic keyword names are kept.
# =====================================================================

import os
import json
from src import llm_prompts


class _LlamaCppBackend:
    def __init__(self, config):
        from llama_cpp import Llama
        local = getattr(config, "LLM_LOCAL_FILE", "")
        ngl = getattr(config, "LLM_N_GPU_LAYERS", 0)
        if local and os.path.exists(local):
            print(f"[llm] loading local model: {local}  (n_gpu_layers={ngl})")
            self.llm = Llama(model_path=local, n_ctx=config.LLM_CTX,
                             n_threads=config.LLM_THREADS, n_gpu_layers=ngl, verbose=False)
        else:
            print(f"[llm] loading {config.LLM_REPO}/{config.LLM_FILE}  (n_gpu_layers={ngl})")
            self.llm = Llama.from_pretrained(
                repo_id=config.LLM_REPO, filename=config.LLM_FILE,
                n_ctx=config.LLM_CTX, n_threads=config.LLM_THREADS,
                n_gpu_layers=ngl, verbose=False)
        self.max_tokens = config.LLM_MAX_TOKENS

    def chat(self, prompt):
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=self.max_tokens)
        return out["choices"][0]["message"]["content"]

    def chat_json(self, prompt, schema):
        """Schema-constrained generation -> always valid JSON shape."""
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=self.max_tokens,
            response_format={"type": "json_object", "schema": schema})
        return out["choices"][0]["message"]["content"]


def load_backend(config):
    """Load the local LLM, or return None (then automatic names are used)."""
    if not config.LLM_ENABLED:
        return None
    try:
        return _LlamaCppBackend(config)
    except Exception as exc:
        print(f"[llm] unavailable ({exc}). Keeping automatic keyword names.")
        return None


def _generate_valid(backend, payload, config):
    """Return (data, source): LLM -> stricter retry -> keyword fallback."""
    prompt = payload["prompt"]

    def _try(p):
        try:
            return llm_prompts.parse_json(backend.chat_json(p, llm_prompts.TOPIC_SCHEMA)) or {}
        except Exception as exc:
            print(f"      (llm error: {exc})")
            return {}

    data = _try(prompt)
    ok, errors = llm_prompts.validate_topic_output(data)
    if ok:
        return data, "llm"
    for _ in range(max(0, config.LLM_MAX_RETRIES)):
        data = _try(llm_prompts.build_retry_prompt(prompt, errors))
        ok, errors = llm_prompts.validate_topic_output(data)
        if ok:
            return data, "llm_retry"
    return llm_prompts.fallback_from_keywords(payload["keywords"], payload["samples"]), "fallback"


def _build_output(tid, data, source):
    return {
        "topic_id": tid, "source": source,
        "business_name": str(data.get("business_name", "")).strip(),
        "description": str(data.get("description", "")).strip(),
        "root_cause": str(data.get("root_cause", "")).strip(),
        "severity": str(data.get("severity", "")).strip(),
        "sentiment": str(data.get("sentiment", "")).strip(),
        "recommended_actions": llm_prompts.normalize_actions(data.get("recommended_actions", [])),
    }


def _apply_output(result, out):
    tid = out["topic_id"]
    if out["business_name"]:
        result.topic_names[tid] = out["business_name"]
    acts = out["recommended_actions"]
    result.topic_meta[tid] = {
        "business_name": out["business_name"],
        "description": out["description"],
        "root_cause": out["root_cause"],
        "severity": out["severity"],
        "sentiment": out["sentiment"],
        "recommended_actions": " | ".join(acts) if isinstance(acts, list) else acts,
    }


def apply_naming(result, clean_df, embeddings, config, backend, out_path=None):
    """Analyze each topic with the LLM. Saves each result as it finishes
    (out_path) and resumes by skipping topics already saved there."""
    payloads = llm_prompts.build_llm_inputs(result, clean_df, embeddings, config)
    total = len(payloads)

    # resume: load anything already saved and apply it first
    done = {}
    if out_path and os.path.exists(out_path):
        try:
            for o in json.load(open(out_path, encoding="utf-8")):
                done[o["topic_id"]] = o
        except Exception:
            done = {}

    outputs = []
    for o in done.values():
        _apply_output(result, o)
        outputs.append(o)

    remaining = [p for p in payloads if p["topic_id"] not in done]
    if done:
        print(f"[llm] resuming: {len(done)} done, {len(remaining)} remaining of {total}")
    else:
        print(f"[llm] analyzing {total} topics (validate + retry + fallback)...")

    for pl in remaining:
        tid = pl["topic_id"]
        data, source = _generate_valid(backend, pl, config)
        out = _build_output(tid, data, source)
        _apply_output(result, out)
        outputs.append(out)
        if out_path:                      # incremental save -> crash-safe
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
        pct = int(100 * len(outputs) / total)
        print(f"  [{len(outputs)}/{total} {pct:3d}%] topic {tid}: "
              f"{out['business_name'] or '(unnamed)'}  [{source}]")
    return outputs


def executive_report(result, config, backend):
    try:
        return backend.chat(llm_prompts.build_exec_prompt(result, config))
    except Exception as exc:
        return f"(executive report unavailable: {exc})"
