"""Unsupervised diagnostics, shared across retrieval and model selection traces."""

from __future__ import annotations

import numpy as np

from src.scoring import CollectionStats, smoothed_idf

PRIMARY_FEATURES = ["frequency_slope", "idf_variance", "frequent_activation_share"]
ALL_FEATURES = ["active_fraction", "frequency_slope", "mass_gini", "head_frequency_mass",
                "idf_variance", "frequent_activation_share"]


def gini(values):
    x = np.sort(np.asarray(values, dtype=np.float64))
    if (x < 0).any() or not np.isfinite(x).all():
        raise ValueError("Gini inputs must be finite and nonnegative")
    total = x.sum()
    if len(x) == 0 or total == 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * total) - (len(x) + 1) / len(x))


def from_frequencies(counts, opportunities, mass, active_fraction):
    counts = np.asarray(counts, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    opportunities = np.broadcast_to(np.asarray(opportunities, dtype=np.float64), counts.shape)
    idf = smoothed_idf(counts, opportunities)
    if counts.ndim != 1 or mass.shape != counts.shape or not len(counts):
        raise ValueError("Expected one nonempty feature vocabulary")
    if (mass < 0).any() or not np.isfinite(mass).all() or not 0 <= active_fraction <= 1:
        raise ValueError("Invalid activation mass or active fraction")
    rate = np.divide(counts, opportunities, out=np.zeros_like(counts), where=opportunities > 0)
    live = counts > 0
    order = np.lexsort((np.arange(len(counts)), -rate))
    head = order[:max(1, int(np.ceil(0.01 * len(counts))))]
    frequencies = np.sort(rate[live])[::-1]
    slope, r2 = 0.0, None
    if len(frequencies) > 1:
        x = np.log(np.arange(1, len(frequencies) + 1))
        y = np.log(frequencies)
        coefficient = np.polyfit(x, y, 1)
        slope = float(-coefficient[0])
        total = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - np.sum((y - np.polyval(coefficient, x)) ** 2) / total) if total > 1e-15 else None
    return {
        "active_fraction": float(active_fraction), "frequency_slope": slope,
        "frequency_fit_r2": r2, "mass_gini": gini(mass),
        "head_frequency_mass": float(rate[head].sum() / rate.sum()) if rate.sum() else 0.0,
        "idf_variance": float(np.var(idf[live])) if live.any() else 0.0,
        "frequent_activation_share": float(mass[head].sum() / mass.sum()) if mass.sum() else 0.0,
        "dead_feature_fraction": float(1 - live.mean()), "vocabulary_size": len(counts),
        "live_features": int(live.sum()), "frequency_fit_defined": len(frequencies) > 1,
    }


def collection_diagnostics(stats: CollectionStats):
    denominator = stats.n_items * len(stats.df)
    return from_frequencies(stats.df, stats.n_items, stats.mass,
                            stats.nnz_sum / denominator if denominator else 0.0)
