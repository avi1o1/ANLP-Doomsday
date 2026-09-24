"""One frozen-encoder pass per input batch, shared by all requested dictionaries."""

import time
from contextlib import ExitStack
from pathlib import Path

from scipy import sparse

from src.artifacts import Artifact, atomic_json, jsonl, write_jsonl
from src.config import digest
from src.retrieval.bases import Basis
from src.retrieval.data import Document
from src.retrieval.encoding import make_encoder
from src.retrieval.pipeline import basis_path, chunks, encoded_path, prepared_paths, storage_cap
from src.scoring import sparsify


def encode_shared(config, dataset, side, representations, stop, shard_id=0, num_shards=1):
    if not 0 <= shard_id < num_shards:
        raise ValueError("Invalid shard assignment")
    collection, corpus = prepared_paths(config, dataset)
    parent = corpus if side == "documents" else collection
    source = parent / ("documents.jsonl" if side == "documents" else "queries.jsonl")
    options = config.get("shared_encoding", {})
    backend = options.get("backend", "cuda")
    batch_size = options.get("batch_size", config["encoder"].get("batch_size", 8))
    encoder, adapters = None, {}
    cap = storage_cap(config, dataset, side)
    if backend == "cuda":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
    for number, records in enumerate(chunks(jsonl(source), config.get("shard_size", 4096))):
        if number % num_shards != shard_id:
            continue
        stop.check()
        tick = time.monotonic()
        with ExitStack() as stack:
            pending = []
            for name, granularity, seed in sorted(representations):
                fitted = basis_path(config, name, granularity, seed)
                path = encoded_path(config, dataset, name, granularity, seed, side) / "shards" / f"{number:06d}"
                work = stack.enter_context(Artifact(path, {
                    "encoder": config["encoder"], "side": side, "shard": number,
                    "maximum_cached_support": cap,
                    "ids": digest([r.get("doc_id", r.get("query_id")) for r in records]),
                }, [parent, fitted]))
                if work.reused:
                    continue
                key = name, granularity, seed
                if key not in adapters:
                    basis = Basis.load(fitted)
                    if backend == "cuda":
                        from src.retrieval.gpu_bases import TensorBasis
                        adapter = TensorBasis(basis, chunk_size=options.get("token_chunk_size", 1024))
                    else:
                        adapter = basis
                    adapters[key] = basis, adapter
                pending.append((key, work, path, [], []))
            if not pending:
                continue
            if encoder is None:
                encoder = make_encoder(config["encoder"])
            for key, *_ in pending:
                if encoder.revision != adapters[key][0].fit_metadata["model_revision"]:
                    raise ValueError("Encoder revision changed since fitting")
            ids = [r.get("doc_id", r.get("query_id")) for r in records]
            for batch in chunks(records, batch_size):
                stop.check()
                texts = [Document(**r).content if side == "documents" else r["text"] for r in batch]
                encoded = encoder.encode(texts, "document" if side == "documents" else "query",
                                         device_output=backend == "cuda")
                for key, work, path, matrices, measures in pending:
                    stop.check()
                    matrix = adapters[key][1].aggregate(encoded.pooled, encoded.tokens, key[1])
                    matrices.append(sparsify(matrix, cap))
                    measures.extend({"id": r.get("doc_id", r.get("query_id")),
                                     "truncated": encoded.truncated[i], "content_tokens": encoded.token_counts[i],
                                     "native_nonzeros": int(matrix[i].nnz)} for i, r in enumerate(batch))
            for key, work, path, matrices, measures in pending:
                sparse.save_npz(path / "vectors.npz", sparse.vstack(matrices, format="csr"))
                atomic_json(path / "ids.json", ids)
                write_jsonl(path / "encoding.jsonl", measures)
                work.complete(["vectors.npz", "ids.json", "encoding.jsonl"], items=len(ids),
                              maximum_cached_support=cap, model_revision=encoder.revision,
                              truncated=sum(r["truncated"] for r in measures),
                              content_tokens=sum(r["content_tokens"] for r in measures),
                              native_nonzeros=sum(r["native_nonzeros"] for r in measures),
                              execution={"backend": backend, "shared_encoder": True, "batch_size": batch_size,
                                         "token_chunk_size": options.get("token_chunk_size", 1024)})
        print(f"{dataset} {side} shard {number}: {len(records)} items, {len(pending)} representations, "
              f"{time.monotonic()-tick:.1f}s", flush=True)


def join_shared(config, dataset, side, representations, stop):
    collection, corpus = prepared_paths(config, dataset)
    parent = corpus if side == "documents" else collection
    source = parent / ("documents.jsonl" if side == "documents" else "queries.jsonl")
    counts = []
    for records in chunks(jsonl(source), config.get("shard_size", 4096)):
        stop.check()
        counts.append(len(records))
    for name, granularity, seed in representations:
        directory = encoded_path(config, dataset, name, granularity, seed, side)
        shards = [{"path": str(directory / "shards" / f"{n:06d}"), "count": count}
                  for n, count in enumerate(counts)]
        with Artifact(directory, {"side": side, "shard_size": config.get("shard_size", 4096)},
                      [Path(s["path"]) for s in shards]) as work:
            if not work.reused:
                atomic_json(directory / "shards.json", shards)
                work.complete(["shards.json"], items=sum(counts))
