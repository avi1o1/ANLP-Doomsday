"""Exact attention mass captured by each switch configuration, the label-free reference of Setting B.

The model decodes with full attention. At every decode step, in every layer and head, the
probe scores the pages as each configuration would, selects them under the same budget and
frozen calibration reference, and records the share of the exact softmax mass that falls on
the selected tokens. The generation itself is the uncorrected one, so no corrected outcome
is read or produced.
"""

from __future__ import annotations

from src.analysis import SWITCH_KEYS
from src.attention.adapter import (AttentionController, criticality_batched, page_summary_batched,
                                   select_page_mask)
from src.scoring import Switches
from src.selection import corrected_positive_batched


class MassProbe(AttentionController):
    def __init__(self, model, reference, budget, page_size, sink=False, keys=SWITCH_KEYS):
        if not reference:
            raise ValueError("The mass probe needs the example's frozen calibration reference")
        super().__init__(model, mode="full", budget=budget, page_size=page_size, sink=sink, reference=reference)
        self.switches = {key: Switches.parse(key) for key in keys}
        self.totals = None
        self.calls = 0

    def masks(self, layer, query, key, scaling):
        """Selected-page masks [heads, pages] per configuration, as the corrected adapter computes them."""
        import torch.nn.functional as F

        heads = query.shape[1]
        n = key.shape[-2]
        q = query[0, :, 0].float()
        k = key.repeat_interleave(heads // key.shape[1], dim=1)[0].float()
        low, high, lengths = page_summary_batched(k, self.page_size)
        raw = criticality_batched(q, low, high) * scaling
        result = {}
        for name, switches in self.switches.items():
            score = raw
            if name != "000":
                counts, opportunities, average, scale = self._reference_tensors(layer, heads, k.device)
                score = corrected_positive_batched(F.softplus(raw), counts, opportunities, lengths,
                                                   average, switches, scale)
            result[name] = select_page_mask(score, n, self.page_size, self.budget, self.sink)[0]
        return result, q, k

    def forward(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        import torch

        n = key.shape[-2]
        if query.shape[2] == 1 and n > self.budget:
            masks, q, k = self.masks(module.layer_idx, query, key, scaling)
            weights = torch.softmax((k @ q[:, :, None]).squeeze(-1) * scaling, dim=-1)
            shares = torch.stack([(weights * mask.repeat_interleave(self.page_size, dim=1)[:, :n]).sum(1).mean()
                                  for mask in masks.values()])
            self.totals = shares if self.totals is None else self.totals + shares
            self.calls += 1
        return super().forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    def shares(self):
        """Mean captured mass per configuration over every probed layer and decode step."""
        if not self.calls:
            return None
        return dict(zip(self.switches, (self.totals / self.calls).tolist()))
