"""The mass probe must select exactly the pages the corrected adapter attends to."""

import copy

import pytest
import torch

from src.analysis import SWITCH_KEYS
from src.attention import adapter
from src.attention.mass import MassProbe
from src.scoring import Switches
from test_attention_batched import decode, tiny


def calibrate(model, ids, budget, page_size, sink):
    with torch.inference_mode(), adapter.AttentionController(model, mode="full", page_size=page_size, budget=budget,
                                                             calibration_samples=20, sink=sink) as ctrl:
        model(ids, use_cache=False)
        return ctrl.freeze()


class Recording(MassProbe):
    """Keeps the decode inputs of every probed call for the comparison below."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls_seen = []

    def forward(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.shape[2] == 1 and key.shape[-2] > self.budget:
            self.calls_seen.append((module, query.clone(), key.clone(), value.clone(), scaling))
        return super().forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs)


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("sink", [False, True])
def test_probe_matches_corrected_adapter(architecture, sink):
    model = tiny(architecture)
    ids = (torch.arange(90)[None] * 5 + 3) % 48
    reference = calibrate(model, ids, 24, 8, sink)
    _, probe = decode(model, Recording, ids, reference, 6, budget=24, page_size=8, sink=sink)
    assert probe.calls_seen
    with torch.inference_mode():
        compare(model, reference, probe, sink)


def compare(model, reference, probe, sink):
    for module, query, key, value, scaling in probe.calls_seen:
        masks, q, k = probe.masks(module.layer_idx, query, key, scaling)
        n = key.shape[-2]
        v = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)[0].float()
        for name in SWITCH_KEYS:
            controller = adapter.AttentionController(model, mode="corrected", switches=Switches.parse(name),
                                                     budget=24, page_size=8, sink=sink,
                                                     reference=copy.deepcopy(reference))
            actual, _ = controller.forward(module, query, key, value, None, scaling)
            tokens = masks[name].repeat_interleave(8, dim=1)[:, :n]
            logits = (k @ q[:, :, None]).squeeze(-1) * scaling
            weights = torch.softmax(logits.masked_fill(~tokens, -torch.inf), dim=-1)
            expected = (weights[:, :, None] * v).sum(1).to(query.dtype)
            torch.testing.assert_close(actual.reshape(expected.shape), expected, atol=1e-5, rtol=1e-4)


def test_probe_leaves_full_generation_unchanged():
    model = tiny("llama")
    ids = (torch.arange(90)[None] * 3 + 1) % 48
    reference = calibrate(model, ids, 24, 8, False)
    full, _ = decode(model, adapter.AttentionController, ids, reference, 10, mode="full", budget=24, page_size=8)
    probed, probe = decode(model, MassProbe, ids, reference, 10, budget=24, page_size=8)
    assert torch.equal(full, probed)
    shares = probe.shares()
    assert set(shares) == set(SWITCH_KEYS)
    assert all(0 < value <= 1 + 1e-6 for value in shares.values())


def test_probe_requires_reference():
    with pytest.raises(ValueError):
        MassProbe(tiny("llama"), None, 24, 8)
