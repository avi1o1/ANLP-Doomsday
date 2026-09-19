"""Collection-aware selection primitives shared by model interventions."""

from __future__ import annotations

import numpy as np

from src.diagnostics import ALL_FEATURES, from_frequencies


def corrected_positive(values, counts, opportunities, lengths, average_length, switches, scale=1.0):
    import torch

    f = values.float() / max(float(scale), 1e-12)
    if not torch.isfinite(f).all() or (f < 0).any():
        raise ValueError("Selection requires finite nonnegative activations")
    counts = torch.as_tensor(counts, device=f.device, dtype=torch.float32)
    opportunities = torch.as_tensor(opportunities, device=f.device, dtype=torch.float32)
    if (counts < 0).any() or (counts > opportunities).any():
        raise ValueError("Selection counts exceed causal opportunities")
    ell = torch.ones_like(torch.as_tensor(lengths, device=f.device, dtype=torch.float32))
    if switches.length and average_length > 0:
        ell = 1 - switches.b + switches.b * torch.as_tensor(lengths, device=f.device).float() / average_length
    if switches.saturation:
        result = (switches.k1 + 1) * f / (f + switches.k1 * ell).clamp_min(1e-12)
    else:
        result = f / ell.clamp_min(1e-12)
    if switches.idf:
        idf = torch.log1p((opportunities - counts + 0.5) / (counts + 0.5))
        # New decode pages have never been observed: neutral weighting, not fabricated counts.
        idf = torch.where(opportunities > 0, idf, torch.ones_like(idf))
        result = result * idf
    return result


def aggregate_diagnostics(traces):
    """One downstream row, averaging diagnostics across reference heads/layers."""
    rows = []
    for trace in traces:
        counts = np.asarray(trace["counts"], float)
        opportunities = np.broadcast_to(np.asarray(trace["opportunities"], float), counts.shape)
        eligible = opportunities > 0
        if eligible.any():
            rows.append(from_frequencies(counts[eligible], opportunities[eligible],
                np.asarray(trace["mass"], float)[eligible],
                float(counts[eligible].sum() / opportunities[eligible].sum())))
    if not rows:
        raise ValueError("No uncorrected calibration observations")
    return {**{key: float(np.mean([row[key] for row in rows])) for key in ALL_FEATURES},
            "dead_feature_fraction": float(np.mean([r["dead_feature_fraction"] for r in rows])),
            "reference_units": len(rows), "aggregation": "equal_reference_unit_mean"}
