# =====================================================================
# naming.py  —  OPTIONAL local-LLM topic analysis (after clustering).
#
# Uses llama-cpp to run a quantized Qwen model fully OFFLINE (no server,
# no cloud). The model runs once PER TOPIC (never per comment), with
# validation -> stricter retry -> deterministic fallback. Degrades
# gracefully: if the model can't load, automatic keyword names are kept.
# =====================================================================

import os
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


def apply_naming(result, clean_df, embeddings, config, backend):
    """Fill result.topic_names + result.topic_meta. Returns validated outputs."""
    payloads = llm_prompts.build_llm_inputs(result, clean_df, embeddings, config)
    total = len(payloads)
    print(f"[llm] analyzing {total} topics (validate + retry + fallback)...")

    outputs = []
    for i, pl in enumerate(payloads, 1):
        tid = pl["topic_id"]
        data, source = _generate_valid(backend, pl, config)

        name = str(data.get("business_name", "")).strip()
        if name:
            result.topic_names[tid] = name
        actions = llm_prompts.normalize_actions(data.get("recommended_actions", []))
        result.topic_meta[tid] = {
            "business_name": name,
            "description": str(data.get("description", "")).strip(),
            "root_cause": str(data.get("root_cause", "")).strip(),
            "severity": str(data.get("severity", "")).strip(),
            "sentiment": str(data.get("sentiment", "")).strip(),
            "recommended_actions": " | ".join(actions),
        }
        outputs.append({
            "topic_id": tid, "source": source, "business_name": name,
            "description": result.topic_meta[tid]["description"],
            "root_cause": result.topic_meta[tid]["root_cause"],
            "severity": result.topic_meta[tid]["severity"],
            "sentiment": result.topic_meta[tid]["sentiment"],
            "recommended_actions": actions,
        })
        pct = int(100 * i / total)
        print(f"  [{i}/{total} {pct:3d}%] topic {tid}: {name or '(unnamed)'}  [{source}]")
    return outputs


def executive_report(result, config, backend):
    try:
        return backend.chat(llm_prompts.build_exec_prompt(result, config))
    except Exception as exc:
        return f"(executive report unavailable: {exc})"
