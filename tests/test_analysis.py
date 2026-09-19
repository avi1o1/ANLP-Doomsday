import numpy as np

from src.analysis import (
    FrozenPredictor,
    evaluate_transfer,
    fit_predictors,
    grouped_validation,
    holm,
    paired_effect,
)
from src.diagnostics import ALL_FEATURES, PRIMARY_FEATURES


def test_zero_spread_and_paired_resampling():
    assert paired_effect(np.ones((10, 8)), 20)["margin"] is None
    values = np.tile(np.arange(8), (10, 1))
    effect = paired_effect(values, 20)
    assert effect["margin"] == 7 / np.std(np.arange(8))
    assert effect["raw_ci95"] == [7, 7]
    assert effect["margin_variance"] < 1e-20
    assert holm([0.04, 0.001, 0.03]) == [0.06, 0.003, 0.06]


def test_grouped_validation_keeps_repeats_and_indic_together():
    rows = []
    for corpus, language in (("english_a", "en"), ("shared_marco", "en"), ("hi", "hi"), ("bn", "bn"), ("te", "te")):
        for family in ("linear", "learned", "quantized"):
            for seed in (0, 1):
                value = len(rows) / 10 - 2
                rows.append(dict(row_id=str(len(rows)), family=family, corpus_id=corpus, language=language,
                                 representation_id=family, collection=corpus, budget="q32d128", seed=seed,
                                 diagnostics=dict(zip(PRIMARY_FEATURES, [value, value**2, 0.4])), margin=value))
    lookup = {r["row_id"]: r for r in rows}
    for axis in ("family", "corpus_id", "indic"):
        result = grouped_validation(rows, axis)
        assert result["predictions"]
        for pred in result["predictions"]:
            test = lookup[pred["row_id"]]
            train = [lookup[i] for i in pred["training_row_ids"]]
            if axis == "indic":
                assert all(r["language"] == "en" for r in train)
                assert test["language"] in {"hi", "bn", "te"}
            else:
                assert all(r[axis] != test[axis] for r in train)


def test_frozen_predictor_roundtrip_and_weighted_effects(tmp_path):
    rows = []
    rng = np.random.default_rng(20)
    for corpus, language in (("a", "en"), ("b", "en"), ("c", "en"), ("h", "hi"), ("n", "bn"), ("t", "te")):
        for family in ("linear", "clustered", "quantized", "learned"):
            x = rng.uniform(size=6)
            rows.append(dict(row_id=str(len(rows)), family=family, corpus_id=corpus, language=language,
                representation_id=family, collection=corpus, budget="primary", seed=0, primary_budget=True,
                diagnostics=dict(zip(ALL_FEATURES, x)), margin=float(2*x[1]-x[4]+x[5]-0.7), margin_variance=0.1))
    predictor, validation = fit_predictors(rows, tmp_path, samples=10)
    restored = FrozenPredictor.load(tmp_path / "predictor.json")
    np.testing.assert_array_equal(predictor.predict(rows), restored.predict(rows))
    assert len(validation) == 12
    before = (tmp_path / "predictor.json").read_bytes()
    report = evaluate_transfer(restored, [{**r, "task": r["corpus_id"]} for r in rows])
    assert report["mae_threshold"] == 1.5*restored.metadata["retrieval_oof_rmse"]
    assert report["uncertainty"]["groups"] == 6
    assert before == (tmp_path / "predictor.json").read_bytes()
    values = np.array([np.arange(8), 2*np.arange(8)])
    effect = paired_effect(values, samples=10, weights=[1, 3])
    assert effect["raw_delta"] == 12.25
