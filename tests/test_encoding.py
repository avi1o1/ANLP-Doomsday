from types import SimpleNamespace

import numpy as np
import torch

from src.retrieval.encoding import E5Encoder


def test_e5_content_masks_exclude_prefix_special_padding_and_dense_keeps_prefix():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            if "return_tensors" not in kwargs:
                return {"input_ids": [[0, 1, 2, 3, 4, 5, 6] for _ in texts]}
            return {"input_ids": torch.tensor([[0, 1, 2, 3, 4, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
                    "offset_mapping": torch.tensor([[[0, 0], [0, 6], [7, 10], [11, 15], [0, 0], [0, 0]]]),
                    "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 1, 1]])}

    class Model:
        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[9., 0.], [7., 0.], [1., 2.], [3., 4.], [5., 0.], [100., 100.]]]))

    encoder = E5Encoder.__new__(E5Encoder)
    encoder.torch, encoder.device, encoder.dimension, encoder.max_length = torch, "cpu", 2, 6
    encoder.tokenizer, encoder.model = Tokenizer(), Model()
    batch = encoder.encode(["one two"], "query")
    np.testing.assert_array_equal(batch.tokens[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(batch.pooled, [[2, 3]])
    np.testing.assert_array_equal(batch.content_masks[0], [False, False, True, True, False, False])
    reference = np.array([5., 1.2])
    np.testing.assert_allclose(batch.dense[0], reference / np.linalg.norm(reference), rtol=1e-6)
    assert batch.truncated == [True]
