import copy

import numpy as np
import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    OlmoeConfig,
    OlmoeForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from src.attention.adapter import AttentionController, select_pages
from src.routing.adapter import RoutingController
from src.scoring import Switches


def llama(architecture="llama"):
    torch.manual_seed(0)
    config_class, model_class = (LlamaConfig, LlamaForCausalLM) if architecture == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    model = model_class(config_class(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128))
    model.config._attn_implementation = "eager"
    return model.eval()


def test_current_and_sink_pages_charged_to_budget():
    for n in (1, 16, 17, 63, 64, 65):
        scores = torch.arange((n+15)//16, dtype=torch.float32)
        ids, pages = select_pages(scores, n, 16, 32, sink=True)
        assert len(ids) <= 32 and n-1 in ids and 0 in ids
        assert len(set(ids.tolist())) == len(ids)


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
def test_attention_full_noop_causal_calibration_and_quest_parity(architecture):
    model = llama(architecture)
    ids = torch.arange(20)[None] % 32
    with torch.inference_mode():
        native = model(ids, use_cache=True)
        with AttentionController(model, mode="full", page_size=4, budget=8, calibration_samples=10) as ctrl:
            actual = model(ids, use_cache=True)
            reference = ctrl.freeze()
            logits = model(torch.tensor([[3]]), past_key_values=actual.past_key_values, use_cache=True).logits
        torch.testing.assert_close(native.logits, actual.logits, atol=1e-6, rtol=1e-5)
        expected = model(torch.tensor([[3]]), past_key_values=native.past_key_values, use_cache=True).logits
        torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
        for trace in reference.values():
            assert np.all(np.array(trace["counts"]) <= trace["opportunities"])
            assert trace["opportunities"][0] > trace["opportunities"][-1]
        results = []
        for mode in ("quest", "corrected"):
            with AttentionController(model, mode=mode, reference=reference, page_size=4, budget=8) as ctrl:
                output = model(ids, use_cache=True)
                before = copy.deepcopy(ctrl.reference)
                output = model(torch.tensor([[3]]), past_key_values=output.past_key_values, use_cache=True)
                results.append(output.logits)
                assert ctrl.reference == before
                assert ctrl.measurements["selected_kv_tokens"] <= 8 * ctrl.measurements["decode_head_calls"]
        torch.testing.assert_close(*results, rtol=0, atol=0)


def test_h2o_growing_cache_crosses_budget_without_overflow():
    model = llama()
    with torch.inference_mode(), AttentionController(model, mode="h2o", page_size=4, budget=8) as ctrl:
        output = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        reference = ctrl.freeze()
        for _ in range(8):
            output = model(torch.tensor([[5]]), past_key_values=output.past_key_values, use_cache=True)
        assert ctrl.reference == reference
        assert ctrl.measurements["selected_kv_tokens"] <= 8 * ctrl.measurements["decode_head_calls"]
        assert output.logits.isfinite().all()


@pytest.mark.parametrize("normalize", [False, True])
def test_olmoe_native_noop_and_original_gate_mixture(normalize):
    torch.manual_seed(0)
    model = OlmoeForCausalLM(OlmoeConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        num_experts=12, num_experts_per_tok=8, norm_topk_prob=normalize)).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        native = model(ids).logits
        with RoutingController(model) as calibration:
            calibrated = model(ids).logits
            reference = calibration.freeze()
        torch.testing.assert_close(native, calibrated, atol=0, rtol=0)
        with RoutingController(model, reference=reference) as ctrl:
            noop = model(ids).logits
        torch.testing.assert_close(native, noop, atol=0, rtol=0)
        block = next(m for m in model.modules() if m.__class__.__name__ == "OlmoeSparseMoeBlock")
        states = torch.randn(1, 3, 16)
        with RoutingController(model, reference=reference, switches=Switches(idf=True)) as ctrl:
            actual, logits = block(states)
            selected = next(iter(ctrl.last_selections.values()))
        probability = logits.softmax(-1)
        expected = torch.zeros_like(states).view(-1, 16)
        for token in range(3):
            weights = probability[token, selected[token]]
            if normalize:
                weights = weights / weights.sum()
            assert len(set(selected[token].tolist())) == 8
            for expert, weight in zip(selected[token], weights):
                expected[token] += block.experts[expert](states[0, token]) * weight
        torch.testing.assert_close(actual.view(-1, 16), expected, atol=1e-6, rtol=1e-5)


def test_sae_interrupted_training_resumes_identically(tmp_path):
    from src.artifacts import StopRequested
    from src.retrieval.sae import fit_sae

    x = np.random.default_rng(8).normal(size=(32, 4)).astype(np.float32)
    spec = dict(features=8, active=2, epochs=2, batch_size=8, micro_batch_size=4, checkpoint_steps=1)

    class Interrupt:
        requested = True

        def check(self):
            raise StopRequested()

    with pytest.raises(StopRequested):
        fit_sae(x[:24], x[24:], spec, 0, tmp_path / "resume", Interrupt())
    resumed, _ = fit_sae(x[:24], x[24:], spec, 0, tmp_path / "resume", None)
    whole, _ = fit_sae(x[:24], x[24:], spec, 0, tmp_path / "whole", None)
    for key in whole:
        np.testing.assert_array_equal(whole[key], resumed[key])
