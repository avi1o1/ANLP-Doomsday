"""Check actual Torch kernels and exact FAISS retrieval on each allocated GPU."""

import json

import faiss
import numpy as np
import torch

from src.retrieval.baselines import _gpu_dense_ranking


def main():
    if not torch.cuda.is_available() or not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("Both CUDA Torch and GPU FAISS are required")
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(4096, 64)).astype("float32")
    queries = rng.normal(size=(7, 64)).astype("float32")
    ids = [str(i) for i in range(len(vectors))]
    reference = faiss.IndexFlatIP(64)
    reference.add(vectors)
    expected_scores, expected_ids = reference.search(queries, 1000)
    devices = []
    for device in range(torch.cuda.device_count()):
        tensor = torch.from_numpy(vectors).to(f"cuda:{device}")
        torch.testing.assert_close((tensor @ tensor.T).diagonal().cpu(),
                                   torch.from_numpy((vectors * vectors).sum(axis=1)))
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = device
        index = faiss.GpuIndexFlatIP(resources, 64, config)
        index.add(vectors)
        scores, neighbors = index.search(queries, 1000)
        np.testing.assert_allclose(scores, expected_scores, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(neighbors, expected_ids)
        # Exercise the project's wrapper and the deterministic cutoff fallback.
        ranking = _gpu_dense_ranking(index, vectors, queries, ids, [None] * 7,
                                     {"candidate_buffer": 128})
        assert [i for i, _ in ranking[0]] == [str(i) for i in expected_ids[0]]
        index.reset()
        zeros = np.zeros_like(vectors)
        index.add(zeros)
        tied = _gpu_dense_ranking(index, zeros, queries[:1], ids, ["0"],
                                  {"candidate_buffer": 128})[0]
        assert [i for i, _ in tied] == sorted(ids[1:])[:1000]
        devices.append({"device": device, "name": torch.cuda.get_device_name(device), "parity": True})
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "faiss": faiss.__version__, "devices": devices}, indent=2))


if __name__ == "__main__":
    main()
