"""Inference-only OLMoE routing with native mixture weights and frozen traces."""

from __future__ import annotations

import copy
import types
from contextlib import AbstractContextManager

import numpy as np

from src.scoring import Switches
from src.selection import aggregate_diagnostics, corrected_positive


class RoutingController(AbstractContextManager):
    def __init__(self, model, switches=None, reference=None, mode="corrected", bias_strength=0.1,
                 preserve_experts=0, expected_top_k=8):
        self.model, self.switches = model, switches or Switches()
        self.reference = copy.deepcopy(reference) if reference else {}
        self.calibrating = reference is None
        self.mode, self.bias_strength = mode, bias_strength
        self.preserve_experts, self.expected_top_k = preserve_experts, expected_top_k
        self.last_selections = {}
        if mode not in {"native", "corrected", "frequency_bias"}:
            raise ValueError("Unknown routing mode")

    def __enter__(self):
        self.originals, self.hooks = [], []
        for name, module in self.model.named_modules():
            if module.__class__.__name__ != "OlmoeSparseMoeBlock":
                continue
            if module.top_k != self.expected_top_k:
                raise ValueError(f"Expected {self.expected_top_k} active experts, found {module.top_k}")
            if not 0 <= self.preserve_experts <= module.top_k:
                raise ValueError("Preserved experts must fit within the native top-k budget")
            original = module.forward
            self.originals.append((module, original))
            controller = self

            def forward(block, states, _name=name, _original=original):
                return controller._forward(_name, block, states, _original)

            module.forward = types.MethodType(forward, module)
        if not self.originals:
            raise ValueError("No supported OLMoE blocks found")
        return self

    def __exit__(self, *_):
        for module, original in self.originals:
            module.forward = original

    def freeze(self):
        self.calibrating = False
        for record in self.reference.values():
            record["average_length"] = record["length_sum"] / record["tokens"]
        if not self.reference:
            raise ValueError("Collect label-free reference contexts before freezing")
        return copy.deepcopy(self.reference)

    def diagnostics(self):
        return aggregate_diagnostics(self.reference.values())

    def _observe(self, name, block, flat, logits):
        import torch

        probability = torch.softmax(logits.float(), dim=-1)
        selected = torch.topk(probability, block.top_k, dim=-1).indices
        mask = torch.zeros_like(probability).scatter_(1, selected, 1)
        record = self.reference.setdefault(name, {
            "counts": [0.0] * block.num_experts, "mass": [0.0] * block.num_experts,
            "opportunities": 0, "tokens": 0, "length_sum": 0.0,
            "expert_magnitude_sum": [0.0] * block.num_experts,
            "expert_magnitude_count": [0] * block.num_experts})
        record["counts"] = (np.asarray(record["counts"]) + mask.sum(0).cpu().numpy()).tolist()
        record["mass"] = (np.asarray(record["mass"]) + (probability * mask).sum(0).cpu().numpy()).tolist()
        record["opportunities"] += len(flat)
        record["tokens"] += len(flat)
        record["length_sum"] += float(torch.linalg.vector_norm(logits.float(), dim=-1).sum())
        # Optional ablation: measure only natively selected expert activations, label free.
        if self.preserve_experts:
            for expert, layer in enumerate(block.experts):
                ids = torch.where(mask[:, expert] > 0)[0]
                if len(ids):
                    magnitude = torch.linalg.vector_norm(layer(flat[ids]).float(), dim=-1)
                    record["expert_magnitude_sum"][expert] += float(magnitude.sum())
                    record["expert_magnitude_count"][expert] += len(ids)

    def _forward(self, name, block, states, original):
        import torch

        flat = states.reshape(-1, states.shape[-1])
        # Native forwarding is exact for calibration and for all-off, including tie semantics.
        if self.calibrating:
            output, logits = original(states)
            self._observe(name, block, flat, logits.detach())
            return output, logits
        if self.mode == "native" or (self.mode == "corrected" and self.switches.key == "000" and not self.preserve_experts):
            return original(states)
        ref = self.reference[name]
        logits = block.gate(flat)
        probability = torch.softmax(logits.float(), dim=-1)
        if self.mode == "frequency_bias":
            frequency = torch.tensor(ref["counts"], device=flat.device) / ref["opportunities"]
            score = probability + self.bias_strength * (block.top_k / block.num_experts - frequency)
        else:
            lengths = torch.linalg.vector_norm(logits.float(), dim=-1, keepdim=True)
            score = corrected_positive(probability, ref["counts"], ref["opportunities"], lengths,
                                       ref["average_length"], self.switches)
        if self.preserve_experts:
            magnitude = np.asarray(ref["expert_magnitude_sum"]) / np.maximum(ref["expert_magnitude_count"], 1)
            preserve = np.lexsort((np.arange(len(magnitude)), -magnitude))[:self.preserve_experts]
            score[:, preserve.tolist()] = torch.inf
        selected = torch.argsort(score, descending=True, stable=True)[:, :block.top_k]
        # Corrections change selection only. Combine using ORIGINAL probabilities.
        weights = probability.gather(1, selected)
        if block.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(states.dtype)
        result = torch.zeros_like(flat)
        for expert, layer in enumerate(block.experts):
            token, slot = torch.where(selected == expert)
            if len(token):
                contribution = layer(flat[token]) * weights[token, slot, None]
                result.index_add_(0, token, contribution.to(states.dtype))
        self.last_selections[name] = selected.detach()
        return result.reshape_as(states), logits
