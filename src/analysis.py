"""Paired effects, grouped prediction, frozen transfer, and uncertainty estimates."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.artifacts import atomic_json, read_json
from src.config import digest
from src.diagnostics import ALL_FEATURES, PRIMARY_FEATURES

INDIC_LANGUAGES = {"hi", "bn", "te"}
SWITCH_KEYS = [f"{i:03b}" for i in range(8)]
DIAGNOSTIC_PROTOCOL = "occurrence-rate-live-idf-head-ceil-one-percent-v1"


def paired_effect(outcomes, samples=2000, seed=0, weights=None):
    """outcomes is [examples, eight ordered switch configurations]."""
    values = np.asarray(outcomes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Effects require finite, aligned outcomes for all eight configurations")
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(values),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Example weights must be finite, positive, and aligned")
    quality = np.average(values, weights=weights, axis=0)
    delta = float(quality[-1] - quality[0])
    spread = float(quality.std(ddof=0))
    margin = delta / spread if spread > 1e-12 else None
    rng = np.random.default_rng(seed)
    raw, standardized = [], []
    for _ in range(samples):
        indices = rng.integers(0, len(values), len(values))
        resampled = np.average(values[indices], weights=weights[indices], axis=0)
        difference = float(resampled[-1] - resampled[0])
        raw.append(difference)
        denominator = float(resampled.std(ddof=0))
        if denominator > 1e-12:
            standardized.append(difference / denominator)
    raw = np.asarray(raw)
    p = float((1 + np.count_nonzero(np.abs(raw - delta) >= abs(delta))) / (samples + 1)) if samples else None
    return {"quality": dict(zip(SWITCH_KEYS, quality.tolist())), "raw_delta": delta,
            "spread": spread, "margin": margin, "n_examples": len(values),
            "raw_ci95": np.quantile(raw, [0.025, 0.975]).tolist() if samples else None,
            "margin_ci95": np.quantile(standardized, [0.025, 0.975]).tolist() if standardized else None,
            "margin_variance": float(np.var(standardized, ddof=1)) if len(standardized) > 1 else None,
            "bootstrap_valid_margins": len(standardized), "p_value": p,
            "undefined_reason": "zero_configuration_spread" if margin is None else None}


def holm(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1)
    return result.tolist()


def metrics(y, prediction, baseline=None):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if len(y) == 0:
        return {"n": 0}
    residual = y - prediction
    total = np.sum((y - y.mean()) ** 2)
    nonzero = y != 0
    result = {"n": len(y), "mae": float(np.abs(residual).mean()),
              "rmse": float(np.sqrt(np.mean(residual ** 2))),
              "r2": float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
              "sign_accuracy": float(np.mean(np.sign(y[nonzero]) == np.sign(prediction[nonzero]))) if nonzero.any() else None,
              "zero_effect_rows": int((~nonzero).sum()), "auc": None}
    if len(np.unique(y[nonzero] > 0)) == 2:
        result["auc"] = float(roc_auc_score(y[nonzero] > 0, prediction[nonzero]))
    if baseline is not None:
        baseline = np.asarray(baseline)
        base = metrics(y, baseline)
        result["baseline"] = base
        result["incremental_r2"] = result["r2"] - base["r2"] if total > 0 else None
    return result


def feature_matrix(rows, features):
    result = np.asarray([[row["diagnostics"][name] for name in features] for row in rows], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Predictor diagnostics must be finite; do not impute using evaluation rows")
    return result


def precision_weights(rows, cap_ratio=20):
    variance = np.asarray([r.get("margin_variance") or 0 for r in rows], dtype=np.float64)
    positive = variance[variance > 0]
    if not len(positive):
        return np.ones(len(rows))
    floor = np.median(positive) / cap_ratio
    weights = 1 / np.maximum(variance, floor)
    weights = np.minimum(weights, np.median(weights) * cap_ratio)
    return weights / weights.mean()


@dataclass
class FrozenPredictor:
    features: list[str]
    mean: list[float]
    scale: list[float]
    coefficients: list[float]
    intercept: float
    alpha: float
    metadata: dict

    def predict(self, rows):
        x = feature_matrix(rows, self.features)
        return ((x - np.asarray(self.mean)) / np.asarray(self.scale)) @ np.asarray(self.coefficients) + self.intercept

    def save(self, path):
        atomic_json(path, self.__dict__)

    @classmethod
    def load(cls, path):
        model = cls(**read_json(path))
        if model.metadata.get("diagnostic_protocol") != DIAGNOSTIC_PROTOCOL:
            raise ValueError("Predictor diagnostic protocol mismatch")
        return model


def train_model(rows, features, alpha, weighted=False):
    x = feature_matrix(rows, features)
    y = np.asarray([r["margin"] for r in rows], float)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y,
                                  sample_weight=precision_weights(rows) if weighted else None)
    return FrozenPredictor(features, scaler.mean_.tolist(), scaler.scale_.tolist(),
                           model.coef_.tolist(), float(model.intercept_), alpha,
                           {"diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
                            "training_row_ids": [r["row_id"] for r in rows], "weighted": weighted})


def choose_alpha(rows, features, group_axis, weighted=False):
    groups = sorted({r[group_axis] for r in rows})
    if len(groups) < 2:
        return 1.0
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        errors = []
        for group in groups:
            train = [r for r in rows if r[group_axis] != group]
            test = [r for r in rows if r[group_axis] == group]
            if len(train) < 2:
                continue
            pred = train_model(train, features, alpha, weighted).predict(test)
            errors.append(float(np.mean(np.abs(pred - np.asarray([r["margin"] for r in test])))))
        candidates.append((np.mean(errors) if errors else np.inf, alpha))
    return min(candidates)[1]


def grouped_validation(rows, axis, features=PRIMARY_FEATURES, weighted=False):
    rows = [r for r in rows if r.get("margin") is not None]
    if axis not in {"family", "corpus_id", "indic"}:
        raise ValueError("Unknown validation axis")
    groups = ["indic"] if axis == "indic" else sorted({r[axis] for r in rows})
    records, folds = [], []
    for group in groups:
        def held(r):
            return r["language"] in INDIC_LANGUAGES if axis == "indic" else r[axis] == group
        train, test = [r for r in rows if not held(r)], [r for r in rows if held(r)]
        if axis == "indic":
            train = [r for r in train if r["language"] == "en"]
        if not test or len(train) < 3:
            folds.append({"held_out": group, "status": "insufficient_data"})
            continue
        inner_axis = "family" if axis == "family" else "corpus_id"
        alpha = choose_alpha(train, features, inner_axis, weighted)
        model = train_model(train, features, alpha, weighted)
        prediction = model.predict(test)
        y_train = np.asarray([r["margin"] for r in train])
        global_mean = float(y_train.mean())
        majority = float(np.mean(y_train > 0) >= 0.5)
        fold_records = []
        for row, pred in zip(test, prediction):
            corpus_values = [r["margin"] for r in train if r["corpus_id"] == row["corpus_id"]]
            baseline = float(np.mean(corpus_values)) if axis == "family" and corpus_values else global_mean
            record = {"row_id": row["row_id"], "held_out": group, "axis": axis,
                      "corpus_id": row["corpus_id"], "representation_id": row["representation_id"],
                      "collection": row["collection"], "budget": row["budget"],
                      "margin": row["margin"], "prediction": float(pred), "baseline": baseline,
                      "majority_prediction": majority, "alpha": alpha,
                      "training_row_ids": model.metadata["training_row_ids"]}
            records.append(record)
            fold_records.append(record)
        folds.append({"held_out": group, "status": "complete", "alpha": alpha,
                      **metrics([r["margin"] for r in fold_records], [r["prediction"] for r in fold_records],
                                [r["baseline"] for r in fold_records])})
    summary = metrics([r["margin"] for r in records], [r["prediction"] for r in records],
                      [r["baseline"] for r in records])
    nonzero = [r for r in records if r["margin"] != 0]
    summary["majority_sign_accuracy"] = float(np.mean([
        (r["margin"] > 0) == bool(r["majority_prediction"]) for r in nonzero])) if nonzero else None
    tau_groups = {}
    for record in records:
        key = (record["collection"], record["budget"])
        tau_groups.setdefault(key, []).append(record)
    taus = []
    for (collection, budget), group in tau_groups.items():
        if len(group) > 1:
            value = kendalltau([r["margin"] for r in group], [r["prediction"] for r in group]).statistic
            taus.append({"collection": collection, "budget": budget,
                         "tau": float(value) if np.isfinite(value) else None})
    return {"axis": axis, "features": features, "weighted": weighted, "folds": folds,
            "summary": summary, "predictions": records, "ordering": taus}


def cluster_intervals(records, samples=2000, seed=0):
    """Two-way pigeonhole bootstrap of fixed OOF errors (conditional on fitted folds)."""
    if not records:
        return {"status": "insufficient_data"}
    corpora = sorted({r["corpus_id"] for r in records})
    reps = sorted({r["representation_id"] for r in records})
    ci = np.array([corpora.index(r["corpus_id"]) for r in records])
    ri = np.array([reps.index(r["representation_id"]) for r in records])
    y = np.array([r["margin"] for r in records])
    p = np.array([r["prediction"] for r in records])
    rng = np.random.default_rng(seed)
    maes, signs = [], []
    for _ in range(samples):
        cw = np.bincount(rng.integers(0, len(corpora), len(corpora)), minlength=len(corpora))
        rw = np.bincount(rng.integers(0, len(reps), len(reps)), minlength=len(reps))
        weight = cw[ci] * rw[ri]
        if weight.sum():
            maes.append(float(np.average(np.abs(y - p), weights=weight)))
        active = (y != 0) & (weight > 0)
        if active.any():
            signs.append(float(np.average(np.sign(y[active]) == np.sign(p[active]), weights=weight[active])))
    return {"method": "two_way_pigeonhole_fixed_oof", "corpus_groups": len(corpora),
            "representation_groups": len(reps), "bootstrap_samples": samples,
            "mae_ci95": np.quantile(maes, [0.025, 0.975]).tolist() if maes else None,
            "sign_accuracy_ci95": np.quantile(signs, [0.025, 0.975]).tolist() if signs else None}


def _single_thread_worker():
    from threadpoolctl import threadpool_limits
    # Keep the controller alive for the lifetime of this worker.
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)


def parallel_map(function, jobs):
    """Deterministic ordered results, no nested BLAS oversubscription."""
    jobs = list(jobs)
    requested = int(os.environ.get("CSX_ANALYSIS_WORKERS", "1"))
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
    if requested < 1:
        raise ValueError("CSX_ANALYSIS_WORKERS must be positive")
    workers = min(requested, allocated, len(jobs))
    if workers <= 1:
        yield from map(function, jobs)
    else:
        names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in names}
        try:
            # Spawned interpreters must import NumPy with one thread, not merely
            # reduce an already-created large thread pool in the initializer.
            os.environ.update({name: "1" for name in names})
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_single_thread_worker) as pool:
                yield from pool.map(function, jobs)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def ablation_task(job):
    filename, row_id, samples = job
    outcomes = np.load(filename)
    ablations = []
    for bit, component in enumerate(("idf", "saturation", "length")):
        off = [i for i in range(8) if f"{i:03b}"[bit] == "0"]
        on = [i for i in range(8) if f"{i:03b}"[bit] == "1"]
        baseline = outcomes[:, off].mean(axis=1)
        comparison = np.repeat(baseline[:, None], 8, axis=1)
        comparison[:, -1] = outcomes[:, on].mean(axis=1)
        effect = paired_effect(comparison, samples)
        ablations.append({"row_id": row_id, "component": component,
            "comparison_family": f"factorial_main_effect_{component}",
            "definition": "mean on-minus-off over the four settings of the other two switches",
            **{k: effect[k] for k in ("raw_delta", "raw_ci95", "p_value")}})
    return ablations


def validation_task(job):
    rows, axis, features, weighted, name, samples, seed = job
    result = grouped_validation(rows, axis, features, weighted)
    result["model"] = name
    result["uncertainty"] = cluster_intervals(result["predictions"], samples, seed)
    return result


def fit_predictors(rows, directory, samples=2000, seed=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Repeated seeds are robustness experiments, not extra primary observations.
    primary = [r for r in rows if r.get("seed", 0) == 0 and r.get("primary_budget", True) and r.get("margin") is not None
               and not r.get("external", False) and not r.get("control", False)]
    if len(primary) < 4:
        raise ValueError("Need at least four defined primary rows to fit a predictor")
    jobs = [(primary, axis, features, weighted, name, samples, seed)
            for features, name in ((PRIMARY_FEATURES, "primary"), (ALL_FEATURES, "six_features"))
            for weighted in (False, True) for axis in ("family", "corpus_id", "indic")]
    repeated = [r for r in rows if r.get("margin") is not None and not r.get("external") and not r.get("control")]
    if len(repeated) > len(primary):
        jobs.extend((repeated, axis, PRIMARY_FEATURES, False, "repeated_measurements_sensitivity", samples, seed)
                    for axis in ("family", "corpus_id", "indic"))
    results = list(parallel_map(validation_task, jobs))
    reference = next(r for r in results if r["model"] == "primary" and not r["weighted"] and r["axis"] == "corpus_id")
    residual = reference["summary"].get("rmse")
    if residual is None:
        raise ValueError("Corpus-held-out residual spread is required before freezing a transfer predictor")
    alpha = choose_alpha(primary, PRIMARY_FEATURES, "corpus_id")
    frozen = train_model(primary, PRIMARY_FEATURES, alpha)
    frozen.metadata.update(retrieval_oof_rmse=residual, training_manifest_hash=digest(primary),
                           transfer_sign_threshold=0.70, transfer_mae_multiplier=1.5,
                           acceptance_protocol="per-setting-no-refit-v1")
    frozen.save(directory / "predictor.json")
    english = [r for r in primary if r["language"] not in INDIC_LANGUAGES]
    if len(english) >= 3:
        model = train_model(english, PRIMARY_FEATURES, choose_alpha(english, PRIMARY_FEATURES, "corpus_id"))
        model.save(directory / "english_predictor.json")
    atomic_json(directory / "validation.json", results)
    return frozen, results


def evaluate_transfer(predictor, rows):
    defined = [r for r in rows if r.get("margin") is not None]
    if not defined:
        return {"status": "no_defined_margins", "undefined_rows": len(rows)}
    prediction = predictor.predict(defined)
    summary = metrics([r["margin"] for r in defined], prediction)
    reference = predictor.metadata["retrieval_oof_rmse"]
    summary["mae_threshold"] = 1.5 * reference
    summary["passes_point_thresholds"] = (summary["sign_accuracy"] is not None
        and summary["sign_accuracy"] >= 0.70 and summary["mae"] <= 1.5 * reference)
    summary["undefined_rows"] = len(rows) - len(defined)
    summary["predictions"] = [{"row_id": r["row_id"], "margin": r["margin"], "prediction": float(p)}
                              for r, p in zip(defined, prediction)]
    # Keep repeated contexts and budgets of each task together. This is conditional
    # on the frozen model; per-example paired intervals are also retained in each row.
    groups = sorted({r.get("task", r.get("subject") or r.get("benchmark", r["row_id"])) for r in defined})
    group_ids = [groups.index(r.get("task", r.get("subject") or r.get("benchmark", r["row_id"]))) for r in defined]
    rng, maes, signs = np.random.default_rng(0), [], []
    y = np.asarray([r["margin"] for r in defined])
    for _ in range(2000):
        sampled = rng.integers(0, len(groups), len(groups))
        indices = [i for group in sampled for i, g in enumerate(group_ids) if group == g]
        maes.append(float(np.abs(y[indices] - prediction[indices]).mean()))
        nonzero = [i for i in indices if y[i] != 0]
        if nonzero:
            signs.append(float(np.mean(np.sign(y[nonzero]) == np.sign(prediction[nonzero]))))
    summary["uncertainty"] = {"method": "task_group_bootstrap_fixed_predictor", "groups": len(groups),
        "mae_ci95": np.quantile(maes, [0.025, 0.975]).tolist() if len(groups) > 1 else None,
        "sign_accuracy_ci95": np.quantile(signs, [0.025, 0.975]).tolist() if signs and len(groups) > 1 else None,
        "limitation": "single-group intervals undefined" if len(groups) == 1 else "conditional on frozen predictor"}
    return summary
