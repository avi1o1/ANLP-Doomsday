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
from src.selection import aggregate_diagnostics, corrected_positive, corrected_positive_batched


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


def page_summary_batched(keys, page_size):
    """`page_summary` for every head: keys [heads, tokens, dimensions] -> [heads, pages, ...]."""
    import torch
    import torch.nn.functional as F

    heads, n, width = keys.shape
    pages = math.ceil(n / page_size)
    pad = pages * page_size - n
    # A final partial page is reduced over its real tokens only: pad with the identity of each extremum.
    low_source = F.pad(keys, (0, 0, 0, pad), value=float("inf")) if pad else keys
    high_source = F.pad(keys, (0, 0, 0, pad), value=float("-inf")) if pad else keys
    low = low_source.reshape(heads, pages, page_size, width).amin(2)
    high = high_source.reshape(heads, pages, page_size, width).amax(2)
    length = torch.linalg.vector_norm(torch.cat([low, high], dim=-1), dim=-1)
    return low, high, length


def criticality_batched(query, low, high):
    """query [heads, dimensions]; low and high [heads, pages, dimensions] -> [heads, pages]."""
    import torch

    q = query[:, None, :]
    return (torch.where(q >= 0, high, low) * q).sum(-1)


def select_page_mask(scores, token_count, page_size, budget, sink=False):
    """`select_pages` for every head: a boolean [heads, pages] mask and the selected token count.

    Every page except the mandatory current one holds page_size tokens, so the greedy
    pass over score order keeps exactly the best (budget - mandatory) // page_size of them.
    Mandatory pages are ranked last, and the stable sort keeps the others in the order
    `select_pages` visits them.
    """
    import torch

    if page_size <= 0 or budget < page_size or token_count < 1:
        raise ValueError("Require a nonempty sequence and budget at least one page")
    heads, pages = scores.shape
    if pages != math.ceil(token_count / page_size):
        raise ValueError("Page scores do not match cache length")
    mandatory = [pages - 1]
    if sink and pages > 1:
        mandatory.insert(0, 0)
    sizes = [min(page_size, token_count - p * page_size) for p in mandatory]
    used = sum(sizes)
    if used > budget:
        raise ValueError("Mandatory pages exceed token budget")
    take = min((budget - used) // page_size, pages - len(mandatory))
    ranked = scores.clone()
    ranked[:, mandatory] = -torch.inf
    mask = torch.zeros((heads, pages), dtype=torch.bool, device=scores.device)
    if take > 0:
        best = torch.argsort(ranked, dim=1, descending=True, stable=True)[:, :take]
        mask.scatter_(1, best, True)
    mask[:, mandatory] = True
    return mask, used + take * page_size


def mask_to_positions(mask, token_count, page_size, selected_tokens):
    """Ascending selected token positions per head, [heads, selected_tokens], without a host sync."""
    import torch

    tokens = mask.repeat_interleave(page_size, dim=1)[:, :token_count]
    return torch.argsort((~tokens).to(torch.uint8), dim=1, stable=True)[:, :selected_tokens]


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
        self._tensors = {}
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

    def _calibrate(self, layer, queries, keys, scaling):
        """Causal page-selection statistics for every head of a layer.

        queries and keys are [heads, tokens, dimensions]. For each sampled position the
        page summaries cover only the tokens up to it, as the per-head reference did.
        """
        import torch

        tick = time.perf_counter()
        heads, n, _ = keys.shape
        pages = math.ceil(n / self.page_size)
        device = keys.device
        counts = torch.zeros((heads, pages), dtype=torch.float64, device=device)
        mass = torch.zeros((heads, pages), dtype=torch.float64, device=device)
        opportunities = torch.zeros(pages, dtype=torch.float64, device=device)
        norm_sum, norm_count = torch.zeros(heads, dtype=torch.float64, device=device), 0
        positives = []
        keys = keys.float()
        positions = np.unique(np.linspace(0, n - 1, min(n, self.calibration_samples), dtype=int))
        for position in positions:
            low, high, lengths = page_summary_batched(keys[:, :position + 1], self.page_size)
            raw = criticality_batched(queries[:, position].float(), low, high) * scaling
            mask, _ = select_page_mask(raw, position + 1, self.page_size, self.budget, self.sink)
            values = torch.nn.functional.softplus(raw)
            width = raw.shape[1]
            opportunities[:width] += 1
            counts[:, :width] += mask
            mass[:, :width] += values.double() * mask
            norm_sum += lengths.double().sum(1)
            norm_count += lengths.shape[1]
            positives.append(values[mask].reshape(heads, -1).double())
        positives = torch.cat(positives, dim=1).cpu().numpy()
        counts, mass, opportunities = counts.cpu().numpy(), mass.cpu().numpy(), opportunities.cpu().tolist()
        averages = (norm_sum / norm_count).cpu().tolist()
        for head in range(heads):
            self.reference[f"{layer}:{head}"] = {
                "counts": counts[head].tolist(), "opportunities": opportunities, "mass": mass[head].tolist(),
                "average_length": averages[head], "scale": max(float(np.median(positives[head])), 1e-12),
                "prefill_tokens": n, "causal_sample_positions": positions.tolist()}
        self.measurements["diagnostic_seconds"] += time.perf_counter() - tick

    def _reference_tensors(self, layer, heads, device):
        """The frozen per-head statistics of one layer as tensors, built once."""
        import torch

        cached = self._tensors.get(layer)
        if cached is None:
            refs = [self.reference[f"{layer}:{head}"] for head in range(heads)]
            counts = torch.tensor([r["counts"] for r in refs], dtype=torch.float32, device=device)
            opportunities = torch.tensor([r["opportunities"] for r in refs], dtype=torch.float32, device=device)
            if (counts < 0).any() or (counts > opportunities).any():
                raise ValueError("Selection counts exceed causal opportunities")
            cached = (counts, opportunities,
                      torch.tensor([r["average_length"] for r in refs], dtype=torch.float32, device=device),
                      torch.tensor([r["scale"] for r in refs], dtype=torch.float32, device=device))
            self._tensors[layer] = cached
        return cached

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
                self._calibrate(module.layer_idx, query[0], expanded_key[0], scaling)
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
                self.heavy[module.layer_idx] = mass
                self.measurements["h2o_prefill_statistics_seconds"] += time.perf_counter() - tick
            # Memory-efficient PyTorch SDPA is available on Turing; no FA2 requirement.
            mask = attention_mask[..., :n] if attention_mask is not None else None
            output = F.scaled_dot_product_attention(query, expanded_key, expanded_value,
                attn_mask=mask, is_causal=mask is None, dropout_p=0.0, scale=scaling)
            return output.transpose(1, 2).contiguous(), None
        if self.calibrating and n == 1:
            self._calibrate(module.layer_idx, query[0], expanded_key[0], scaling)
        resident = (key.numel() + value.numel()) * key.element_size()
        self.measurements["cumulative_resident_kv_observed_bytes"] += resident
        q, k, v = query[0, :, 0], expanded_key[0], expanded_value[0]
        layer = module.layer_idx
        torch.cuda.synchronize() if q.is_cuda else None
        tick = time.perf_counter()
        if self.mode == "h2o":
            previous = self.heavy.get(layer, torch.zeros((heads, n), device=k.device))
            self.heavy[layer] = F.pad(previous, (0, n - previous.shape[1]))
        if self.mode == "full" or n <= self.budget:
            selected = None
            count = n
        elif self.mode == "h2o":
            heavy = self.heavy[layer]
            recent = min(self.budget // 2, n)
            old = torch.argsort(heavy[:, :n - recent], dim=1, descending=True, stable=True)[:, :self.budget - recent]
            tail = torch.arange(n - recent, n, device=k.device).expand(heads, -1)
            selected = torch.cat([old, tail], dim=1).sort(dim=1).values
            count = selected.shape[1]
        else:
            low, high, lengths = page_summary_batched(k.float(), self.page_size)
            score = criticality_batched(q.float(), low, high) * scaling
            if self.mode == "corrected" and self.switches.key != "000":
                counts, opportunities, average, scale = self._reference_tensors(layer, heads, k.device)
                score = corrected_positive_batched(F.softplus(score), counts, opportunities, lengths,
                                                   average, self.switches, scale)
            mask, count = select_page_mask(score, n, self.page_size, self.budget, self.sink)
            selected = mask_to_positions(mask, n, self.page_size, count)
        torch.cuda.synchronize() if q.is_cuda else None
        self.measurements["selection_seconds"] += time.perf_counter() - tick
        if selected is None:
            keys_chosen, values_chosen = k, v
        else:
            index = selected[:, :, None].expand(-1, -1, width)
            keys_chosen, values_chosen = k.gather(1, index), v.gather(1, index)
        weights = torch.softmax((keys_chosen.float() @ q.float()[:, :, None]).squeeze(-1) * scaling, dim=-1)
        output = (weights[:, :, None] * values_chosen.float()).sum(1).to(query.dtype)
        if self.mode == "h2o":
            if selected is None:
                self.heavy[layer] += weights.detach()
            else:
                self.heavy[layer].scatter_add_(1, selected, weights.detach())
        self.measurements["decode_head_calls"] += heads
        self.measurements["selected_kv_tokens"] += count * heads
        self.measurements["selected_kv_bytes"] += count * heads * width * 2 * key.element_size()
        # resident bytes above are cumulative layer-step observations; report peak elsewhere.
        return output.reshape(batch, 1, heads, width), None
