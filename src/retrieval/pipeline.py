"""Resumable, bounded-memory retrieval stages; scorer changes reuse encodings."""

from __future__ import annotations

import hashlib
import os
import resource
import time
from itertools import islice
from pathlib import Path

import numpy as np
from scipy import sparse

from src.analysis import ablation_task, fit_predictors, holm, paired_effect, parallel_map
from src.artifacts import Artifact, atomic_json, file_hash, jsonl, provenance, read_json, write_jsonl
from src.config import digest, named, root
from src.diagnostics import collection_diagnostics
from src.retrieval.bases import Basis
from src.retrieval.data import Document, corpus_key, deterministic_sample, prepare_dataset, split_documents
from src.retrieval.encoding import make_encoder
from src.retrieval.index import (
    InvertedIndex,
    merge_rankings,
    qrels_mapping,
    reference_metrics,
    standard_metrics,
)
from src.retrieval.postings_cache import PostingsCache
from src.scoring import CollectionStats, configurations, sparsify


def chunks(iterator, size):
    iterator = iter(iterator)
    while batch := list(islice(iterator, size)):
        yield batch


def spec_for(config, dataset):
    return {**config["datasets"][dataset], "name": dataset}


def prepared_paths(config, dataset):
    return (root(config) / "datasets" / "collections" / dataset,
            root(config) / "datasets" / "corpora" / corpus_key(spec_for(config, dataset)))


def fit_dataset(config):
    choices = [k for k, v in config["datasets"].items() if v["role"] == "fit"]
    if len(choices) != 1:
        raise ValueError("Exactly one fitting collection is required")
    if not config.get("fixture", False) and config["datasets"][choices[0]]["corpus_id"] != "msmarco-passage":
        raise ValueError("Research bases must be fit on MS MARCO only")
    return choices[0]


def basis_path(config, basis, granularity, seed):
    return root(config) / "bases" / f"{basis}__{granularity}__s{seed}"


def encoded_path(config, dataset, basis, granularity, seed, side):
    key = corpus_key(spec_for(config, dataset)) if side == "documents" else dataset
    return root(config) / "encoded" / f"{basis}__{granularity}__s{seed}" / side / key


def storage_cap(config, dataset, side):
    """Keep enough aggregated features for EVERY declared support experiment.

    Truncating to max(k) commutes with any smaller deterministic top-k. Native
    experiments retain the full vector. Shared document corpora use a common cap.
    """
    if side == "documents":
        corpus = config["datasets"][dataset]["corpus_id"]
        collections = {n for n, s in config["datasets"].items() if s["corpus_id"] == corpus}
    else:
        collections = {dataset}
    budgets = [b["document" if side == "documents" else "query"] for b in config["budgets"]
               if not b.get("collections") or collections.intersection(b["collections"])]
    return None if None in budgets else max(budgets)


def index_path(config, dataset, basis, granularity, seed, budget):
    return root(config) / "indexes" / f"{basis}__{granularity}__s{seed}" / corpus_key(spec_for(config, dataset)) / budget


def prepare(config, dataset, stop):
    return prepare_dataset(spec_for(config, dataset), root(config), stop)


def encode_sample(config, stop):
    fitting = config["fitting"]
    _, corpus = prepared_paths(config, fit_dataset(config))
    directory = root(config) / "sample"
    with Artifact(directory, {"encoder": config["encoder"], "fitting": fitting}, [corpus]) as work:
        if work.reused:
            return directory
        selected = deterministic_sample((Document(**r) for r in jsonl(corpus / "documents.jsonl")),
                                        fitting["documents"], fitting["seed"])
        if len(selected) < fitting["documents"] and not config.get("fixture", False):
            raise ValueError("Fitting corpus is smaller than the declared sample")
        train, validation = split_documents(selected, fitting["validation_fraction"], fitting["seed"])
        encoder = make_encoder(config["encoder"])
        resolved_path = directory / "resolved_encoder.json"
        if resolved_path.exists() and read_json(resolved_path)["revision"] != encoder.revision:
            raise ValueError("Encoder revision changed during sample recovery; pin it and use a new output root")
        atomic_json(resolved_path, {"revision": encoder.revision})
        metadata = {"model_revision": encoder.revision, "dimension": encoder.dimension}
        outputs = ["resolved_encoder.json"]
        for split, documents in (("train", train), ("validation", validation)):
            capacity = max(1, round(fitting["max_tokens"] * len(documents) / len(selected)))
            quota = max(1, capacity // len(documents))
            pooled_path, token_path = directory / f"{split}_pooled.npy", directory / f"{split}_tokens.npy"
            progress_path = directory / f"{split}_progress.json"
            resume = progress_path.exists()
            progress = read_json(progress_path) if resume else {"documents": 0, "tokens": 0}
            if resume:
                pooled = np.load(pooled_path, mmap_mode="r+")
                tokens = np.load(token_path, mmap_mode="r+")
            else:
                pooled = np.lib.format.open_memmap(pooled_path, mode="w+", dtype="float32",
                                                   shape=(len(documents), encoder.dimension))
                tokens = np.lib.format.open_memmap(token_path, mode="w+", dtype="float32",
                                                   shape=(capacity, encoder.dimension))
            cursor, token_cursor = progress["documents"], progress["tokens"]
            for batch in chunks(documents[cursor:], config["encoder"].get("batch_size", 16)):
                stop.check()
                encoded = encoder.encode([d.content for d in batch])
                pooled[cursor:cursor + len(batch)] = encoded.pooled
                for doc, states in zip(batch, encoded.tokens):
                    n = min(len(states), quota, capacity - token_cursor)
                    seed = int(hashlib.sha256(f"{fitting['seed']}:{doc.doc_id}".encode()).hexdigest()[:16], 16)
                    ids = np.sort(np.random.default_rng(seed).choice(len(states), n, replace=False))
                    tokens[token_cursor:token_cursor + n] = states[ids]
                    token_cursor += n
                cursor += len(batch)
                pooled.flush()
                tokens.flush()
                atomic_json(progress_path, {"documents": cursor, "tokens": token_cursor})
            write_jsonl(directory / f"{split}_ids.jsonl", ({"doc_id": d.doc_id} for d in documents))
            metadata[split] = {"documents": len(documents), "tokens": token_cursor}
            outputs.extend([f"{split}_pooled.npy", f"{split}_tokens.npy", f"{split}_ids.jsonl"])
        # Bounded full-document activations calibrate scales AFTER aggregation.
        calibration = train[:fitting.get("scale_documents", 1024)]
        for number, batch in enumerate(chunks(calibration, config["encoder"].get("batch_size", 16))):
            path = directory / "scale" / f"{number:06d}.npz"
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                stop.check()
                encoded = encoder.encode([d.content for d in batch])
                offsets = np.cumsum([0] + [len(x) for x in encoded.tokens])
                with path.with_suffix(".tmp").open("wb") as handle:
                    np.savez(handle, pooled=encoded.pooled, tokens=np.concatenate(encoded.tokens), offsets=offsets)
                path.with_suffix(".tmp").replace(path)
            outputs.append(str(path.relative_to(directory)))
        metadata["sample_ids_hash"] = digest([d.doc_id for d in selected])
        atomic_json(directory / "sample.json", metadata)
        work.complete(outputs + ["sample.json"], **metadata)
    return directory


def fit_basis(config, name, granularity, seed, stop):
    sample = root(config) / "sample"
    info = read_json(sample / "sample.json")
    spec = named(config["bases"], name)
    directory = basis_path(config, name, granularity, seed)
    shared = basis_path(config, spec["dictionary_basis"], spec["dictionary_granularity"], seed) if spec.get("dictionary_basis") else None
    with Artifact(directory, {"basis": spec, "granularity": granularity, "seed": seed},
                  [sample] + ([shared] if shared else [])) as work:
        if work.reused:
            return directory
        suffix = "pooled" if granularity == "pooled" else "tokens"
        data = []
        for split in ("train", "validation"):
            x = np.load(sample / f"{split}_{suffix}.npy", mmap_mode="r")
            data.append(x if suffix == "pooled" else x[:info[split]["tokens"]])
        if shared:
            basis = Basis.load(shared)
            basis.spec = spec
            basis.fit_metadata["shared_dictionary_source"] = str(shared)
        else:
            basis = Basis(spec, info["dimension"], seed).fit(
                data[0], data[1], directory / "checkpoints", stop, config["encoder"].get("device", "cpu"))
        activations = []
        for path in sorted((sample / "scale").glob("*.npz")):
            stop.check()
            with np.load(path) as batch:
                states = [batch["tokens"][a:b] for a, b in zip(batch["offsets"][:-1], batch["offsets"][1:])]
                values = basis.aggregate(batch["pooled"], states, granularity, raw=True)
                activations.append(values.data)
        positive = np.concatenate(activations)
        basis.scale = float(np.median(positive)) if len(positive) else 1.0
        basis.fit_metadata.update(scale_policy="median_positive_train_document_aggregate",
                                  scale_documents=config["fitting"].get("scale_documents", 1024),
                                  model_revision=info["model_revision"])
        basis.save(directory)
        work.complete(["basis.json", "weights.npz"], **basis.fit_metadata)
    return directory


def _encode_side(config, dataset, name, granularity, seed, side, stop, shard_id=None, num_shards=1):
    collection, corpus = prepared_paths(config, dataset)
    parent = corpus if side == "documents" else collection
    directory = encoded_path(config, dataset, name, granularity, seed, side)
    fitted = basis_path(config, name, granularity, seed)
    basis = Basis.load(fitted)
    encoder = None
    cache_cap = storage_cap(config, dataset, side)
    shards = []
    source_file = "documents.jsonl" if side == "documents" else "queries.jsonl"
    for number, records in enumerate(chunks(jsonl(parent / source_file), config.get("shard_size", 4096))):
        path = directory / "shards" / f"{number:06d}"
        shards.append({"path": str(path), "count": len(records)})
        if shard_id is not None and number % num_shards != shard_id:
            continue
        with Artifact(path, {"encoder": config["encoder"], "side": side, "shard": number,
                             "maximum_cached_support": cache_cap,
                             "ids": digest([r.get("doc_id", r.get("query_id")) for r in records])},
                      [parent, fitted]) as work:
            if work.reused:
                continue
            stop.check()
            if encoder is None:
                encoder = make_encoder(config["encoder"])
                if encoder.revision != basis.fit_metadata["model_revision"]:
                    raise ValueError("Encoder revision changed since fitting; pin the resolved model revision")
            matrices, ids, measures = [], [], []
            for batch in chunks(records, config["encoder"].get("batch_size", 16)):
                stop.check()
                texts = [Document(**r).content if side == "documents" else r["text"] for r in batch]
                encoded = encoder.encode(texts, "document" if side == "documents" else "query")
                matrix = basis.aggregate(encoded.pooled, encoded.tokens, granularity)
                matrices.append(sparsify(matrix, cache_cap))
                for i, r in enumerate(batch):
                    item_id = r.get("doc_id", r.get("query_id"))
                    ids.append(item_id)
                    measures.append({"id": item_id, "truncated": encoded.truncated[i],
                                     "content_tokens": encoded.token_counts[i], "native_nonzeros": int(matrix[i].nnz)})
            sparse.save_npz(path / "vectors.npz", sparse.vstack(matrices, format="csr"))
            atomic_json(path / "ids.json", ids)
            write_jsonl(path / "encoding.jsonl", measures)
            work.complete(["vectors.npz", "ids.json", "encoding.jsonl"], items=len(ids),
                          maximum_cached_support=cache_cap,
                          model_revision=encoder.revision, truncated=sum(r["truncated"] for r in measures),
                          content_tokens=sum(r["content_tokens"] for r in measures),
                          native_nonzeros=sum(r["native_nonzeros"] for r in measures))
    # A final unsharded invocation joins already completed shards without another encoder pass.
    if shard_id is None:
        with Artifact(directory, {"side": side, "shard_size": config.get("shard_size", 4096)},
                      [Path(s["path"]) for s in shards]) as work:
            if not work.reused:
                atomic_json(directory / "shards.json", shards)
                work.complete(["shards.json"], items=sum(s["count"] for s in shards))
    return directory


def encode_corpus(config, dataset, name, granularity, seed, stop, shard_id=None, num_shards=1):
    if num_shards < 1 or shard_id is not None and not 0 <= shard_id < num_shards:
        raise ValueError("Invalid deterministic shard assignment")
    return [_encode_side(config, dataset, name, granularity, seed, side, stop, shard_id, num_shards)
            for side in ("documents", "queries")]


def build_index(config, dataset, name, granularity, seed, budget_name, stop):
    budget = named(config["budgets"], budget_name)
    encoded = encoded_path(config, dataset, name, granularity, seed, "documents")
    directory = index_path(config, dataset, name, granularity, seed, budget_name)
    with Artifact(directory, {"budget": budget}, [encoded]) as work:
        if work.reused:
            return directory
        stats, shards, outputs = None, [], []
        for number, record in enumerate(read_json(encoded / "shards.json")):
            stop.check()
            values = sparsify(sparse.load_npz(Path(record["path"]) / "vectors.npz"), budget["document"])
            if stats is None:
                stats = CollectionStats.empty(values.shape[1])
            stats.update(values)
            filename = f"{number:06d}.npz"
            sparse.save_npz(directory / filename, values)
            outputs.append(filename)
            shards.append({"vectors": str(directory / filename), "ids": str(Path(record["path"]) / "ids.json")})
        if stats is None:
            raise ValueError("Empty document corpus")
        atomic_json(directory / "statistics.json", stats.to_dict())
        atomic_json(directory / "diagnostics.json", collection_diagnostics(stats))
        atomic_json(directory / "shards.json", shards)
        work.complete(outputs + ["statistics.json", "diagnostics.json", "shards.json"],
                      document_count=stats.n_items, nonzeros=stats.nnz_sum,
                      index_bytes=sum((directory / p).stat().st_size for p in outputs))
    return directory


def evaluate(config, dataset, name, granularity, seed, budget_name, stop, k1=1.2, b=0.75,
             control=None):
    budget = named(config["budgets"], budget_name)
    spec = spec_for(config, dataset)
    index = index_path(config, dataset, name, granularity, seed, budget_name)
    queries = encoded_path(config, dataset, name, granularity, seed, "queries")
    collection, _ = prepared_paths(config, dataset)
    representation = f"{name}__{granularity}__s{seed}"
    directory = root(config) / "evaluations" / dataset / representation / budget_name / f"k{k1}_b{b}_{control or 'primary'}"
    identity = {"dataset": dataset, "representation": representation, "budget": budget,
                "k1": k1, "b": b, "control": control, "analysis": config.get("analysis", {})}
    with Artifact(directory, identity, [index, queries, collection]) as work:
        if work.reused:
            return directory
        stats = CollectionStats.from_dict(read_json(index / "statistics.json"))
        if control not in (None, "shuffled_idf"):
            raise ValueError("Unsupported retrieval control")
        if control == "shuffled_idf":
            # Permuting df permutes IDF, without changing feature values, lengths, or support.
            stats.df = np.random.default_rng(seed).permutation(stats.df)
        judgments = qrels_mapping(jsonl(collection / "qrels.jsonl"))
        query_ids, matrices = [], []
        for shard in read_json(queries / "shards.json"):
            path = Path(shard["path"])
            ids = read_json(path / "ids.json")
            keep = [i for i, qid in enumerate(ids) if qid in judgments]
            query_ids.extend(ids[i] for i in keep)
            matrices.append(sparse.load_npz(path / "vectors.npz")[keep])
        query_matrix = sparsify(sparse.vstack(matrices, format="csr"), budget["query"])
        shards = read_json(index / "shards.json")
        outputs, outcomes, summaries = [], np.zeros((len(query_ids), 8)), {}
        metric_fn = reference_metrics if config.get("fixture", False) else standard_metrics
        cache_root = os.environ.get("CSX_POSTINGS_CACHE")
        cache_identity = read_json(index / "manifest.json")
        gpu_scoring = os.environ.get("CSX_EVALUATION_BACKEND", "cpu") == "cuda"
        if gpu_scoring:
            from src.retrieval.gpu_scoring import BatchedScorer
            document_ids = np.concatenate([np.asarray(read_json(s["ids"]), dtype=str) for s in shards])
            checkpoint_size = config.get("evaluation_batch_size", 32)
            gpu_queries = int(os.environ.get("CSX_GPU_QUERY_BATCH", "512"))
            if gpu_queries < checkpoint_size:
                raise ValueError("GPU query batch must cover at least one checkpoint batch")
        for column, switches in enumerate(configurations(k1, b)):
            cache = (PostingsCache(cache_root, cache_identity, stats, switches,
                                  int(float(os.environ.get("CSX_POSTINGS_CACHE_GIB", "192")) * 2**30))
                     if cache_root else None)
            all_records = []
            if gpu_scoring:
                pending = [start for start in range(0, len(query_ids), checkpoint_size)
                           if not (directory / f"{switches.key}/{start:08d}.json").exists()]
                for group_start in range(0, len(pending), gpu_queries // checkpoint_size):
                    group = pending[group_start:group_start + gpu_queries // checkpoint_size]
                    rows = [row for start in group for row in range(start, min(len(query_ids), start + checkpoint_size))]
                    stop.check()
                    tick = time.perf_counter()
                    scorer = BatchedScorer(query_matrix[rows], document_ids,
                        excluded=[query_ids[row] for row in rows] if spec.get("exclude_identical_ids", False) else None)
                    offset = 0
                    for shard in shards:
                        stop.check()
                        postings = (cache.get(shard) if cache else
                                    InvertedIndex(sparse.load_npz(shard["vectors"]), read_json(shard["ids"]), stats, switches))
                        scorer.add(postings, offset)
                        offset += len(postings.doc_ids)
                    rankings, visits = scorer.finish()
                    elapsed = time.perf_counter() - tick
                    records = []
                    for i, row in enumerate(rows):
                        qid = query_ids[row]
                        values = (metric_fn(rankings[i], judgments[qid]) if config.get("fixture", False)
                                  else metric_fn(qid, rankings[i], judgments[qid]))
                        records.append({"query_id": qid, "ranking": rankings[i], "metrics": values,
                            "posting_visits": int(visits[i]), "query_nonzeros": int(query_matrix[row].nnz),
                            "amortized_search_seconds": elapsed / len(rows),
                            "execution": {"backend": "cuda_ordered_fp32", "query_group_size": len(rows),
                                "postings_cache": bool(cache), "worker_slots": int(os.environ.get("CSX_EVALUATION_SLOTS", "1"))}})
                    position = 0
                    for start in group:
                        count = min(checkpoint_size, len(query_ids) - start)
                        atomic_json(directory / f"{switches.key}/{start:08d}.json", records[position:position+count])
                        position += count
                    del scorer
            for start in range(0, len(query_ids), config.get("evaluation_batch_size", 32)):
                stop.check()
                end = min(len(query_ids), start + config.get("evaluation_batch_size", 32))
                filename = f"{switches.key}/{start:08d}.json"
                path = directory / filename
                if not path.exists():
                    rankings = [[] for _ in range(end - start)]
                    visits = np.zeros(end - start, dtype=np.int64)
                    tick = time.perf_counter()
                    for shard in shards:
                        stop.check()
                        postings = (cache.get(shard) if cache else
                                    InvertedIndex(sparse.load_npz(shard["vectors"]),
                                                  read_json(shard["ids"]), stats, switches))
                        for i, row in enumerate(range(start, end)):
                            ranking, count = postings.search(query_matrix[row], 1000,
                                query_ids[row] if spec.get("exclude_identical_ids", False) else None)
                            rankings[i] = merge_rankings([rankings[i], ranking], 1000)
                            visits[i] += count
                    elapsed = time.perf_counter() - tick
                    records = []
                    for i, qid in enumerate(query_ids[start:end]):
                        values = (metric_fn(rankings[i], judgments[qid]) if config.get("fixture", False)
                                  else metric_fn(qid, rankings[i], judgments[qid]))
                        records.append({"query_id": qid, "ranking": rankings[i], "metrics": values,
                                        "posting_visits": int(visits[i]), "query_nonzeros": int(query_matrix[start+i].nnz),
                                        "amortized_search_seconds": elapsed / (end - start),
                                        "execution": {"postings_cache": bool(cache),
                                            "worker_slots": int(os.environ.get("CSX_EVALUATION_SLOTS", "1"))}})
                    atomic_json(path, records)
                records = read_json(path)
                all_records.extend(records)
                outcomes[start:end, column] = [r["metrics"]["ndcg@10"] for r in records]
                outputs.append(filename)
            summaries[switches.key] = {
                metric: float(np.mean([r["metrics"][metric] for r in all_records]))
                for metric in ("ndcg@10", "mrr@10", "recall@100", "recall@1000")}
            summaries[switches.key]["amortized_search_seconds"] = float(np.mean([
                r["amortized_search_seconds"] for r in all_records]))
        effect = paired_effect(outcomes, config.get("analysis", {}).get("bootstrap_samples", 2000), seed)
        encoding_metrics = {}
        for side in ("documents", "queries"):
            side_path = encoded_path(config, dataset, name, granularity, seed, side)
            measures = [read_json(Path(s["path"]) / "manifest.json")["metadata"]
                        for s in read_json(side_path / "shards.json")]
            count = sum(m["items"] for m in measures)
            encoding_metrics[side] = {"items": count,
                "truncation_fraction": sum(m["truncated"] for m in measures)/count,
                "mean_content_tokens": sum(m["content_tokens"] for m in measures)/count,
                "mean_native_nonzeros": sum(m["native_nonzeros"] for m in measures)/count}
        encoding_metrics["mean_retained_query_nonzeros"] = query_matrix.nnz / len(query_ids)
        encoding_metrics["mean_retained_document_nonzeros"] = stats.nnz_sum / stats.n_items
        result = {"row_id": digest(identity), "collection": dataset, "corpus_id": spec["corpus_id"],
                  "language": spec["language"], "family": Basis.load(basis_path(config, name, granularity, seed)).family,
                  "basis": name, "granularity": granularity, "seed": seed,
                  "representation_id": f"{name}__{granularity}", "budget": budget_name,
                  "primary_budget": budget.get("primary", False), "primary_parameters": k1 == 1.2 and b == 0.75,
                  "k1": k1, "b": b, "control": control or ("shared_dictionary" if named(config["bases"], name).get("secondary") else None),
                  "fixture": config.get("fixture", False), "encoding": encoding_metrics,
                  "diagnostics": read_json(index / "diagnostics.json"), "metrics": summaries,
                  "index": read_json(index / "manifest.json")["metadata"],
                  "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  **effect}
        np.save(directory / "outcomes.npy", outcomes)
        atomic_json(directory / "query_ids.json", query_ids)
        atomic_json(directory / "result.json", result)
        work.complete(outputs + ["outcomes.npy", "query_ids.json", "result.json"])
    return directory


def collect_results(config):
    rows = []
    for path in sorted((root(config) / "evaluations").glob("*/*/*/*/result.json")):
        if read_json(path.parent / "manifest.json")["status"] == "complete":
            rows.append(read_json(path))
    primary = [r for r in rows if r["seed"] == 0 and r["primary_budget"] and r["primary_parameters"]
               and not r["control"] and r["p_value"] is not None]
    for row, p in zip(primary, holm([r["p_value"] for r in primary])):
        row["holm_p"] = p
        row["comparison_family"] = "primary_all_on_vs_all_off"
    return rows


def diagnose(config):
    rows = collect_results(config)
    jobs, inputs = [], []
    samples = config.get("analysis", {}).get("bootstrap_samples", 2000)
    for path in sorted((root(config) / "evaluations").glob("*/*/*/*/result.json")):
        result = read_json(path)
        if not (result["seed"] == 0 and result["primary_budget"] and result["primary_parameters"] and not result["control"]):
            continue
        if read_json(path.parent / "manifest.json")["status"] != "complete":
            continue
        outcomes = path.parent / "outcomes.npy"
        inputs.append({"path": str(outcomes), "sha256": file_hash(outcomes), "row_id": result["row_id"]})
        jobs.append((str(outcomes), result["row_id"], samples))
    identity = {"rows_hash": digest(rows), "outcomes": inputs, "analysis": config.get("analysis", {})}
    directory = root(config) / "analysis" / "diagnostics" / digest({"identity": identity, "code": provenance()})
    with Artifact(directory, identity) as work:
        if work.reused:
            _publish_diagnostics(directory, root(config) / "analysis")
            return read_json(directory / "rows.json")
        missing, filenames = [], []
        for job in jobs:
            filename = f"ablation_rows/{job[1]}.json"
            filenames.append(filename)
            path = directory / filename
            if path.exists():
                if read_json(path)["fingerprint"] != work.fingerprint:
                    raise ValueError(f"Analysis checkpoint identity changed: {path}")
            else:
                missing.append(job)
        for job, effects in zip(missing, parallel_map(ablation_task, missing)):
            atomic_json(directory / f"ablation_rows/{job[1]}.json",
                        {"fingerprint": work.fingerprint, "effects": effects})
        ablations = [effect for filename in filenames for effect in read_json(directory / filename)["effects"]]
        for component in ("idf", "saturation", "length"):
            group = [r for r in ablations if r["component"] == component and r["p_value"] is not None]
            for record, p in zip(group, holm([r["p_value"] for r in group])):
                record["holm_p"] = p
        atomic_json(directory / "rows.json", rows)
        atomic_json(directory / "component_ablations.json", ablations)
        work.complete(filenames + ["rows.json", "component_ablations.json"],
                      execution={"workers": int(os.environ.get("CSX_ANALYSIS_WORKERS", "1"))})
    _publish_diagnostics(directory, root(config) / "analysis")
    return rows


def _publish_diagnostics(directory, destination):
    for name in ("rows.json", "component_ablations.json"):
        source, target = directory / name, destination / name
        if not target.exists() or file_hash(source) != file_hash(target):
            atomic_json(target, read_json(source))


def fit_predictor(config):
    rows = [r for r in diagnose(config) if r["primary_parameters"] and not r["control"]]
    directory = root(config) / "predictor"
    with Artifact(directory, {"rows_hash": digest(rows), "analysis": config.get("analysis", {})}) as work:
        if not work.reused:
            fit_predictors(rows, directory, config.get("analysis", {}).get("bootstrap_samples", 2000))
            work.complete([p.name for p in directory.glob("*.json") if p.name != "manifest.json"])
    return directory


def run_retrieval(config, stop):
    prepare(config, fit_dataset(config), stop)
    encode_sample(config, stop)
    for basis in config["bases"]:
        if basis.get("secondary"):
            continue
        for granularity in config.get("granularities", ["pooled", "token"]):
            for seed in config.get("seeds", [0]):
                fit_basis(config, basis["name"], granularity, seed, stop)
                for dataset, spec in config["datasets"].items():
                    if spec["role"] != "evaluate" or not spec.get("enabled", True):
                        continue
                    prepare(config, dataset, stop)
                    encode_corpus(config, dataset, basis["name"], granularity, seed, stop)
                    for budget in config["budgets"]:
                        if budget.get("collections") and dataset not in budget["collections"]:
                            continue
                        build_index(config, dataset, basis["name"], granularity, seed, budget["name"], stop)
                        evaluate(config, dataset, basis["name"], granularity, seed, budget["name"], stop)
    return diagnose(config)
