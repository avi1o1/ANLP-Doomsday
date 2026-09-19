"""Language-model loading and inference helpers shared by transfer settings."""

from __future__ import annotations

import time


def synchronize():
    import torch

    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def load_language_model(spec, routing=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, revision = spec["model"], spec.get("revision", "main")
    options = {"revision": revision, "torch_dtype": getattr(torch, spec.get("precision", "float16")),
               "attn_implementation": "sdpa", "local_files_only": spec.get("offline", False)}
    if routing:
        devices = torch.cuda.device_count()
        if devices < 2:
            raise ValueError("Routing jobs require two visible GPUs; submit with sbatch train.sbatch routing")
        options.update(device_map=spec.get("device_map", "balanced"),
                       max_memory={i: f"{spec.get('max_memory_gib', 10)}GiB" for i in range(2)})
        if spec.get("quantization"):
            from transformers import BitsAndBytesConfig

            if spec["quantization"] not in {"int8", "nf4"}:
                raise ValueError("Quantization must be null, int8, or nf4")
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=spec["quantization"] == "int8", load_in_4bit=spec["quantization"] == "nf4",
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=options["local_files_only"])
        model = AutoModelForCausalLM.from_pretrained(name, **options)
    except OSError as error:
        if routing or not spec.get("fallback_model") or not any(t in str(error).lower() for t in ("gated", "401", "403")):
            raise
        name, revision = spec["fallback_model"], spec.get("fallback_revision", "main")
        options["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(name, **options)
    if not routing:
        if not torch.cuda.is_available():
            raise ValueError("Language-model transfer workloads must run on a GPU cluster")
        model.to("cuda")
    elif any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("Model placement spilled to CPU; reduce memory use or declare identical quantization")
    model.eval().requires_grad_(False)
    metadata = {"model": name, "revision": getattr(model.config, "_commit_hash", None) or revision,
                "precision": spec.get("precision", "float16"), "quantization": spec.get("quantization"),
                "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}}
    return model, tokenizer, metadata


def model_inputs(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    return ids.to(model.get_input_embeddings().weight.device)


def cache_bytes(cache):
    if cache is None:
        return 0
    if hasattr(cache, "key_cache"):
        return sum(t.numel() * t.element_size() for t in cache.key_cache + cache.value_cache)
    return sum(t.numel() * t.element_size() for layer in cache for t in layer)


def greedy_generate(model, tokenizer, ids, max_new_tokens, stop):
    import torch

    synchronize()
    tick = time.perf_counter()
    with torch.inference_mode():
        output = model(ids, use_cache=True)
        synchronize()
        prefill_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        generated = []
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        peak_cache = cache_bytes(output.past_key_values)
        for step in range(max_new_tokens):
            stop.check()
            token = output.logits[:, -1].argmax(-1)
            generated.append(int(token.item()))
            if generated[-1] in eos or step == max_new_tokens - 1:
                break
            output = model(token[:, None], past_key_values=output.past_key_values, use_cache=True)
            peak_cache = max(peak_cache, cache_bytes(output.past_key_values))
    synchronize()
    return tokenizer.decode(generated, skip_special_tokens=True), {
        "prefill_seconds": prefill_seconds, "decode_seconds": time.perf_counter() - tick,
        "generated_tokens": len(generated), "resident_cache_bytes": peak_cache,
        "peak_cuda_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]}
