"""Bounded FP32 basis inference with deterministic feature-ID tie resolution."""

import numpy as np
import torch
from scipy import sparse


def deterministic_topk(values, k):
    """Select largest values; exact boundary ties prefer the lowest feature ID.

    Two partial selections and a linear tie scan avoid sorting the vocabulary.
    Returned indices are in feature-ID order, as in canonical CSR matrices.
    """
    k = min(k, values.shape[1])
    cutoff = values.topk(k, dim=1, sorted=False).values.min(dim=1, keepdim=True).values
    greater = values > cutoff
    equal = values == cutoff
    remaining = k - greater.sum(dim=1, keepdim=True)
    keep = greater | (equal & (equal.cumsum(dim=1) <= remaining))
    ids = torch.arange(values.shape[1], device=values.device).expand_as(values)
    selected = torch.where(keep, ids, values.shape[1]).topk(k, largest=False, dim=1).values
    return selected, values.gather(1, selected)


class TensorBasis:
    def __init__(self, basis, device="cuda", chunk_size=1024):
        self.basis = basis
        self.device = device
        self.chunk_size = chunk_size
        self.arrays = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                       for k, v in basis.arrays.items() if k not in {"reconstruction", "decoder"}}
        if not np.isfinite(basis.scale) or basis.scale <= 0:
            raise ValueError("Basis activation scale must be positive")

    def nearest(self, x, centers, k=1):
        distance = (x.square().sum(1, keepdim=True) + centers.square().sum(1)[None]
                    - 2 * (x @ centers.T)).clamp_min(0)
        ids, negative = deterministic_topk(-distance, k)
        return ids, -negative

    def transform(self, x):
        b, a = self.basis, self.arrays
        kind = b.kind
        if kind == "identity":
            return torch.cat((x.clamp_min(0), (-x).clamp_min(0)), dim=1) / b.scale
        if kind in {"random", "pca"}:
            v = (x - a.get("mean", 0)) @ a["projection"]
            return torch.cat((v.clamp_min(0), (-v).clamp_min(0)), dim=1) / b.scale
        if kind == "kmeans":
            ids, distances = self.nearest(x, a["centers"], b.spec.get("assignments", 4))
            weights = torch.softmax(-distances / a["temperature"], dim=1)
        elif kind in {"pq", "rq"}:
            groups = b.spec["subspaces"] if kind == "pq" else b.spec["stages"]
            residual, columns = x, []
            width = b.dimension // groups
            for i in range(groups):
                values = x[:, i * width:(i+1) * width] if kind == "pq" else residual
                centers = a[f"centers_{i}"]
                chosen, _ = self.nearest(values, centers)
                columns.append(chosen[:, 0] + i * b.spec["codes"])
                if kind == "rq":
                    residual = residual - centers[chosen[:, 0]]
            ids = torch.stack(columns, dim=1)
            weights = torch.ones_like(ids, dtype=torch.float32)
        elif kind == "sae":
            activation = ((x - a["center"]) @ a["encoder"].T + a["bias"]).clamp_min(0)
            ids, weights = deterministic_topk(activation, b.spec["active"])
        else:
            raise ValueError(kind)
        codes = x.new_zeros((len(x), b.vocabulary_size))
        codes.scatter_(1, ids, weights / b.scale)
        return codes

    @torch.inference_mode()
    def aggregate(self, pooled, tokens, granularity):
        if granularity == "pooled":
            result = self.transform(torch.as_tensor(pooled, device=self.device, dtype=torch.float32))
        elif granularity == "token":
            lengths = [len(t) for t in tokens]
            result = torch.zeros((len(tokens), self.basis.vocabulary_size), device=self.device)
            if sum(lengths):
                states = torch.cat([torch.as_tensor(t, device=self.device, dtype=torch.float32) for t in tokens])
                owners = torch.repeat_interleave(torch.arange(len(tokens), device=self.device),
                                                torch.tensor(lengths, device=self.device))
                for start in range(0, len(states), self.chunk_size):
                    codes = self.transform(states[start:start+self.chunk_size])
                    counts = torch.bincount(owners[start:start+self.chunk_size], minlength=len(tokens))
                    # Contiguous segments avoid unordered floating-point scatter-adds.
                    result += torch.segment_reduce(codes, "sum", lengths=counts)
        else:
            raise ValueError(granularity)
        return sparse.csr_matrix(result.cpu().numpy())
