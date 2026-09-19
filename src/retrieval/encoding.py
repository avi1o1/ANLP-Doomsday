"""Frozen E5, dense reference, SPLADE, and deterministic offline fixture encoders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EncodedBatch:
    pooled: np.ndarray
    tokens: list[np.ndarray]
    dense: np.ndarray
    truncated: list[bool]
    token_counts: list[int]
    content_masks: list[np.ndarray] = field(default_factory=list)
    attention_masks: list[np.ndarray] = field(default_factory=list)


class FixtureEncoder:
    """Small deterministic encoder exclusively for engineering fixtures."""

    def __init__(self, dimension=8, **_):
        self.dimension = dimension
        self.revision = "fixture-v1"

    def encode(self, texts, kind="document"):
        tokens = []
        for text in texts:
            words = text.split()
            rows = []
            for word in words:
                seed = int.from_bytes(hashlib.sha256(word.casefold().encode()).digest()[:8], "little")
                rows.append(np.random.default_rng(seed).normal(size=self.dimension))
            tokens.append(np.asarray(rows, dtype=np.float32).reshape(-1, self.dimension))
        pooled = np.stack([t.mean(axis=0) if len(t) else np.zeros(self.dimension) for t in tokens]).astype(np.float32)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return EncodedBatch(pooled, tokens, pooled / np.maximum(norms, 1e-12),
                            [False] * len(texts), [len(t) for t in tokens],
                            [np.ones(len(t), dtype=bool) for t in tokens],
                            [np.ones(len(t), dtype=bool) for t in tokens])


class E5Encoder:
    def __init__(self, model="intfloat/multilingual-e5-base", revision="main", device="cpu",
                 precision="float16", max_length=512, offline=False, **_):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        dtype = torch.float32 if device == "cpu" else getattr(torch, precision)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True,
                                                       local_files_only=offline)
        if not self.tokenizer.is_fast:
            raise ValueError("Content masking requires a fast tokenizer with offsets")
        self.model = AutoModel.from_pretrained(model, revision=revision, torch_dtype=dtype,
                                               local_files_only=offline, attn_implementation="sdpa")
        self.model.to(device).eval().requires_grad_(False)
        self.dimension = self.model.config.hidden_size
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def encode(self, texts, kind="document"):
        torch = self.torch
        prefix = "query: " if kind == "query" else "passage: "
        inputs = [prefix + text for text in texts]
        # Obtain lengths without truncation so truncation statistics are exact.
        lengths = [len(ids) for ids in self.tokenizer(inputs, add_special_tokens=True,
                                                     truncation=False)["input_ids"]]
        encoded = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length,
                                 return_offsets_mapping=True, return_special_tokens_mask=True,
                                 return_tensors="pt")
        offsets = encoded.pop("offset_mapping")
        special = encoded.pop("special_tokens_mask").bool()
        # A token belongs to content when its span reaches beyond the prefix.
        content = (offsets[:, :, 1] > len(prefix)) & ~special & encoded["attention_mask"].bool()
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].bool()
            dense = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            dense = torch.nn.functional.normalize(dense, dim=1)
            tokens = [h[m.to(h.device)].cpu().numpy() for h, m in zip(hidden, content)]
        pooled = np.stack([t.mean(0) if len(t) else np.zeros(self.dimension, dtype=np.float32) for t in tokens])
        return EncodedBatch(pooled, tokens, dense.cpu().numpy(),
                            [n > self.max_length for n in lengths], [len(t) for t in tokens],
                            [m.numpy() for m in content], [m.cpu().numpy() for m in mask])


class SpladeEncoder:
    def __init__(self, model="naver/splade-v3", revision="main", device="cpu", max_length=512,
                 precision="float16", offline=False):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=offline)
        dtype = torch.float32 if device == "cpu" else getattr(torch, precision)
        self.model = AutoModelForMaskedLM.from_pretrained(model, revision=revision, torch_dtype=dtype,
                                                         local_files_only=offline).to(device)
        self.model.eval().requires_grad_(False)
        self.dimension = self.model.config.vocab_size
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def encode(self, texts, kind="document"):
        from scipy import sparse

        batch = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**batch).logits.float()
            values = self.torch.log1p(self.torch.relu(logits)) * batch.attention_mask.unsqueeze(-1)
            values = values.max(dim=1).values.cpu().numpy()
        return sparse.csr_matrix(values)


def make_encoder(config):
    config = dict(config)
    kind = config.pop("kind", "e5")
    if kind == "fixture":
        return FixtureEncoder(**config)
    if kind == "e5":
        return E5Encoder(**config)
    raise ValueError(f"Unsupported encoder: {kind}")
