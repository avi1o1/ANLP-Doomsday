"""The batched attention adapter must agree with the per-head reference it replaced.

`attention_adapter_oracle` is the adapter as it was before the rewrite. Selection must be
identical, and reference statistics, logits and greedy tokens must agree to float tolerance.
"""

import copy

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

import attention_adapter_oracle as oracle
from src.attention import adapter
from src.scoring import Switches
from src.selection import corrected_positive, corrected_positive_batched


def tiny(architecture):
    torch.manual_seed(1)
    config_class, model_class = (LlamaConfig, LlamaForCausalLM) if architecture == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    model = model_class(config_class(vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256))
    model.config._attn_implementation = "eager"
    return model.eval()


@pytest.mark.parametrize("tokens", [1, 3, 16, 17, 50, 64, 65, 130])
def test_page_summary_matches_per_head(tokens):
    torch.manual_seed(tokens)
    keys = torch.randn(3, tokens, 8)
    low, high, length = adapter.page_summary_batched(keys, 16)
    for head in range(3):
        reference = oracle.page_summary(keys[head], 16)
        for batched, single in zip((low[head], high[head], length[head]), reference):
            torch.testing.assert_close(batched, single, atol=0, rtol=1e-6)


@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.parametrize("tokens,budget", [(70, 32), (65, 16), (130, 48), (33, 16), (20, 16), (200, 64)])
def test_selection_matches_per_head(tokens, budget, sink):
    torch.manual_seed(tokens + budget)
    pages = -(-tokens // 16)
    # Duplicate scores exercise the stable tie order.
    scores = torch.randint(0, 6, (5, pages)).float()
    try:
        expected = [oracle.select_pages(scores[head], tokens, 16, budget, sink)[0] for head in range(5)]
    except ValueError:
        # Mandatory pages exceed the budget: the batched version must refuse too.
        with pytest.raises(ValueError):
            adapter.select_page_mask(scores, tokens, 16, budget, sink)
        return
    mask, count = adapter.select_page_mask(scores, tokens, 16, budget, sink)
    positions = adapter.mask_to_positions(mask, tokens, 16, count)
    for head in range(5):
        assert expected[head].tolist() == positions[head].tolist()
    assert count == len(expected[0]) and count <= budget


def test_selection_errors_match():
    scores = torch.ones(2, 4)
    with pytest.raises(ValueError):
        adapter.select_page_mask(scores, 64, 16, 8)
    with pytest.raises(ValueError):
        adapter.select_page_mask(scores, 64, 16, 16, sink=True)


@pytest.mark.parametrize("key", ["001", "010", "011", "100", "101", "110", "111"])
def test_corrected_scores_match_per_head(key):
    torch.manual_seed(int(key, 2))
    heads, pages, observed = 4, 12, 9
    values = torch.rand(heads, pages) * 3
    opportunities = torch.randint(1, 30, (heads, observed)).float()
    counts = torch.floor(opportunities * torch.rand(heads, observed))
    lengths = torch.rand(heads, pages) * 5 + 0.1
    average = torch.tensor([2.0, 0.0, 1.5, 3.0])
    scale = torch.tensor([1.0, 0.5, 2.0, 1e-15])
    switches = Switches.parse(key)
    batched = corrected_positive_batched(values, counts, opportunities, lengths, average, switches, scale)
    for head in range(heads):
        extra = pages - observed
        expected = corrected_positive(values[head], counts[head].tolist() + [0.0] * extra,
                                      opportunities[head].tolist() + [0.0] * extra, lengths[head],
                                      float(average[head]), switches, float(scale[head]))
        torch.testing.assert_close(batched[head], expected, atol=1e-6, rtol=1e-5)


def decode(model, controller_class, ids, reference, steps, **options):
    """Greedy decode under one controller; returns the logits at every step and the controller."""
    logits = []
    with torch.inference_mode(), controller_class(model, reference=copy.deepcopy(reference), **options) as ctrl:
        output = model(ids, use_cache=True)
        for _ in range(steps):
            logits.append(output.logits[:, -1].clone())
            token = output.logits[:, -1].argmax(-1)
            output = model(token[:, None], past_key_values=output.past_key_values, use_cache=True)
        return torch.stack(logits), ctrl


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("sink", [False, True])
def test_calibration_matches_per_head(architecture, sink):
    model = tiny(architecture)
    ids = (torch.arange(90)[None] * 7) % 48
    references = []
    for controller_class in (oracle.AttentionController, adapter.AttentionController):
        with torch.inference_mode(), controller_class(model, mode="full", page_size=8, budget=24,
                                                      calibration_samples=20, sink=sink) as ctrl:
            model(ids, use_cache=False)
            references.append(ctrl.freeze())
    old, new = references
    assert old.keys() == new.keys()
    for name in old:
        assert old[name]["counts"] == new[name]["counts"]
        assert old[name]["opportunities"] == new[name]["opportunities"]
        assert old[name]["causal_sample_positions"] == new[name]["causal_sample_positions"]
        np.testing.assert_allclose(old[name]["mass"], new[name]["mass"], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(old[name]["average_length"], new[name]["average_length"], rtol=1e-5)
        np.testing.assert_allclose(old[name]["scale"], new[name]["scale"], rtol=1e-5)


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.parametrize("mode,key", [("full", None), ("quest", None), ("h2o", None), ("corrected", "000"),
                                      ("corrected", "001"), ("corrected", "010"), ("corrected", "100"),
                                      ("corrected", "111")])
def test_decode_matches_per_head(architecture, sink, mode, key):
    model = tiny(architecture)
    ids = (torch.arange(90)[None] * 5 + 3) % 48
    with torch.inference_mode(), oracle.AttentionController(model, mode="full", page_size=8, budget=24,
                                                            calibration_samples=20, sink=sink) as ctrl:
        model(ids, use_cache=False)
        reference = ctrl.freeze()
    options = dict(mode=mode, page_size=8, budget=24, sink=sink)
    if key:
        options["switches"] = Switches.parse(key)
    expected, old = decode(model, oracle.AttentionController, ids, reference, 12, **options)
    actual, new = decode(model, adapter.AttentionController, ids, reference, 12, **options)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    assert actual.argmax(-1).tolist() == expected.argmax(-1).tolist()
    for field in ("decode_head_calls", "selected_kv_tokens", "selected_kv_bytes",
                  "cumulative_resident_kv_observed_bytes"):
        assert new.measurements[field] == old.measurements[field], field
    assert new.reference == reference


def test_quest_and_all_off_generate_identically():
    model = tiny("llama")
    ids = (torch.arange(90)[None] * 3 + 1) % 48
    with torch.inference_mode(), adapter.AttentionController(model, mode="full", page_size=8, budget=24,
                                                             calibration_samples=20) as ctrl:
        model(ids, use_cache=False)
        reference = ctrl.freeze()
    quest, _ = decode(model, adapter.AttentionController, ids, reference, 12, mode="quest", page_size=8, budget=24)
    off, _ = decode(model, adapter.AttentionController, ids, reference, 12, mode="corrected",
                    switches=Switches.parse("000"), page_size=8, budget=24)
    assert torch.equal(quest, off)
