"""PyTorch reference Quest-style sparse decode for Llama and Qwen2.

The full KV cache remains resident. Selected KV traffic, resident memory, and
selection overhead are reported separately; this is not a fused speed kernel.
Batch size one, unpadded causal inputs, and dynamic caches are deliberate gates.
"""

from __future__ import annotations

import copy
import math
import time
from contextlib import AbstractContextManager

import numpy as np

from src.scoring import Switches
from src.selection import aggregate_diagnostics, corrected_positive


def page_summary(keys, page_size):
    import torch

    # [tokens, dimensions] -> page extrema. Final partial page contains no padding.
    pages = list(keys.float().split(page_size))
    low = torch.stack([p.amin(0) for p in pages])
    high = torch.stack([p.amax(0) for p in pages])
    length = torch.linalg.vector_norm(torch.cat([low, high], dim=-1), dim=-1)
    return low, high, length


def criticality(query, low, high):
    import torch

    return (torch.where(query >= 0, high, low) * query).sum(-1)


def select_pages(scores, token_count, page_size, budget, sink=False):
    """Whole pages with a hard token cap; mandatory partial current page is charged."""
    import torch

    if page_size <= 0 or budget < page_size or token_count < 1:
        raise ValueError("Require a nonempty sequence and budget at least one page")
    pages = math.ceil(token_count / page_size)
    if len(scores) != pages:
        raise ValueError("Page scores do not match cache length")
    mandatory = [pages - 1]
    if sink and pages > 1:
        mandatory.insert(0, 0)
    def size(p):
        return min(page_size, token_count - p * page_size)
    used = sum(size(p) for p in mandatory)
    if used > budget:
        raise ValueError("Mandatory pages exceed token budget")
    selected = list(mandatory)
    for p in torch.argsort(scores, descending=True, stable=True).tolist():
        if p not in selected and used + size(p) <= budget:
            selected.append(p)
            used += size(p)
    positions = [i for p in sorted(selected) for i in range(p * page_size, min((p + 1) * page_size, token_count))]
    return torch.tensor(positions, device=scores.device, dtype=torch.long), selected


class AttentionController(AbstractContextManager):
    def __init__(self, model, mode="quest", switches=None, budget=512, page_size=16,
                 calibration_samples=64, sink=False, reference=None):
        self.model, self.mode = model, mode
        self.switches = switches or Switches()
        self.budget, self.page_size = budget, page_size
        self.calibration_samples, self.sink = calibration_samples, sink
        self.calibrating = reference is None
        self.reference = copy.deepcopy(reference) if reference else {}
        self.heavy = {}
        self.measurements = {"diagnostic_seconds": 0.0, "selection_seconds": 0.0,
                             "h2o_prefill_statistics_seconds": 0.0,
                             "decode_head_calls": 0, "selected_kv_tokens": 0,
                             "cumulative_resident_kv_observed_bytes": 0, "selected_kv_bytes": 0}
        if mode not in {"full", "quest", "h2o", "corrected"}:
            raise ValueError("Unknown attention mode")

    def __enter__(self):
        from transformers.models.llama import modeling_llama
        from transformers.models.qwen2 import modeling_qwen2

        if self.model.config.model_type not in {"llama", "qwen2"}:
            raise ValueError("Attention adapter supports Llama and Qwen2 only")
        self.originals = []
        self.old_backend = self.model.config._attn_implementation
        self.model.config._attn_implementation = "eager"
        for module in (modeling_llama, modeling_qwen2):
            original = module.eager_attention_forward
            self.originals.append((module, original))
            module.eager_attention_forward = self.forward
        return self

    def __exit__(self, *_):
        for module, original in self.originals:
            module.eager_attention_forward = original
        self.model.config._attn_implementation = self.old_backend

    def freeze(self):
        self.calibrating = False
        if not self.reference:
            raise ValueError("Prefill must precede freezing attention statistics")
        return copy.deepcopy(self.reference)

    def diagnostics(self):
        return aggregate_diagnostics(self.reference.values())

    def _calibrate(self, layer, head, queries, keys, scaling):
        import torch

        tick = time.perf_counter()
        n = len(keys)
        pages = math.ceil(n / self.page_size)
        counts, opportunities, mass = np.zeros(pages), np.zeros(pages), np.zeros(pages)
        norms, positives = [], []
        positions = np.unique(np.linspace(0, n - 1, min(n, self.calibration_samples), dtype=int))
        for position in positions:
            low, high, lengths = page_summary(keys[:position + 1], self.page_size)
            raw = criticality(queries[position].float(), low, high) * scaling
            _, selected = select_pages(raw, position + 1, self.page_size, self.budget, self.sink)
            values = torch.nn.functional.softplus(raw)
            opportunities[:len(raw)] += 1
            counts[selected] += 1
            mass[selected] += values[selected].cpu().numpy()
            norms.extend(lengths.cpu().tolist())
            positives.extend(values[selected].cpu().tolist())
        self.reference[f"{layer}:{head}"] = {
            "counts": counts.tolist(), "opportunities": opportunities.tolist(), "mass": mass.tolist(),
            "average_length": float(np.mean(norms)), "scale": max(float(np.median(positives)), 1e-12),
            "prefill_tokens": n, "causal_sample_positions": positions.tolist()}
        self.measurements["diagnostic_seconds"] += time.perf_counter() - tick

    def forward(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        import torch
        import torch.nn.functional as F

        if query.shape[0] != 1 or dropout:
            raise ValueError("Reference attention requires eval mode, batch one, and no padding")
        batch, heads, qlen, width = query.shape
        n = key.shape[-2]
        if qlen not in (1, n):
            raise ValueError("Use a single dense prefill followed by one-token decode")
        groups = heads // key.shape[1]
        expanded_key = key.repeat_interleave(groups, dim=1)
        expanded_value = value.repeat_interleave(groups, dim=1)
        if qlen > 1:
            if self.calibrating:
                for head in range(heads):
                    self._calibrate(module.layer_idx, head, query[0, head], expanded_key[0, head], scaling)
            if self.mode == "h2o":
                tick = time.perf_counter()
                mass = torch.zeros((heads, n), device=query.device, dtype=torch.float32)
                key_positions = torch.arange(n, device=query.device)
                # Exact causal prefill attention mass, with bounded query chunks.
                for start in range(0, qlen, 64):
                    end = min(start+64, qlen)
                    scores = (query[0, :, start:end].float() @ expanded_key[0].float().transpose(-1, -2)) * scaling
                    causal = key_positions[None, :] > torch.arange(start, end, device=query.device)[:, None]
                    scores.masked_fill_(causal[None], -torch.inf)
                    mass += scores.softmax(-1).sum(-2)
                for head in range(heads):
                    self.heavy[f"{module.layer_idx}:{head}"] = mass[head]
                self.measurements["h2o_prefill_statistics_seconds"] += time.perf_counter() - tick
            # Memory-efficient PyTorch SDPA is available on Turing; no FA2 requirement.
            mask = attention_mask[..., :n] if attention_mask is not None else None
            output = F.scaled_dot_product_attention(query, expanded_key, expanded_value,
                attn_mask=mask, is_causal=mask is None, dropout_p=0.0, scale=scaling)
            return output.transpose(1, 2).contiguous(), None
        if self.calibrating and n == 1:
            for head in range(heads):
                self._calibrate(module.layer_idx, head, query[0, head], expanded_key[0, head], scaling)
        resident = (key.numel() + value.numel()) * key.element_size()
        self.measurements["cumulative_resident_kv_observed_bytes"] += resident
        outputs = []
        for head in range(heads):
            tick = time.perf_counter()
            q, k, v = query[0, head, 0], expanded_key[0, head], expanded_value[0, head]
            name = f"{module.layer_idx}:{head}"
            if self.mode == "h2o":
                previous = self.heavy.get(name, torch.zeros(n, device=k.device))
                self.heavy[name] = F.pad(previous, (0, n - len(previous)))
            if self.mode == "full" or n <= self.budget:
                selected = torch.arange(n, device=k.device)
            elif self.mode == "h2o":
                heavy = self.heavy[name]
                recent = min(self.budget // 2, n)
                old = torch.argsort(heavy[:n-recent], descending=True, stable=True)[:self.budget-recent]
                selected = torch.cat([old, torch.arange(n-recent, n, device=k.device)]).sort().values
                self.heavy[name] = heavy
            else:
                low, high, lengths = page_summary(k, self.page_size)
                score = criticality(q.float(), low, high) * scaling
                if self.mode == "corrected" and self.switches.key != "000":
                    ref = self.reference[name]
                    extra = len(score) - len(ref["counts"])
                    counts = ref["counts"] + [0.0] * extra
                    opportunities = ref["opportunities"] + [0.0] * extra
                    score = corrected_positive(F.softplus(score), counts, opportunities, lengths,
                                               ref["average_length"], self.switches, ref["scale"])
                selected, _ = select_pages(score, n, self.page_size, self.budget, self.sink)
            self.measurements["selection_seconds"] += time.perf_counter() - tick
            weights = torch.softmax((k[selected].float() @ q.float()) * scaling, dim=-1)
            outputs.append((weights[:, None] * v[selected].float()).sum(0).to(query.dtype))
            if self.mode == "h2o" and name in self.heavy:
                self.heavy[name][selected] += weights.detach()
            self.measurements["decode_head_calls"] += 1
            self.measurements["selected_kv_tokens"] += len(selected)
            self.measurements["selected_kv_bytes"] += len(selected) * width * 2 * key.element_size()
        # resident bytes above are cumulative layer-step observations; report peak elsewhere.
        return torch.stack(outputs).reshape(batch, 1, heads, width), None
