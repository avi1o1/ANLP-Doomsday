"""Seven serializable nonnegative bases; fitting never uses evaluation labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from src.artifacts import atomic_json, read_json
from src.scoring import nonnegative_csr

FAMILIES = {"identity": "linear", "random": "linear", "pca": "linear", "kmeans": "clustered",
            "pq": "quantized", "rq": "quantized", "sae": "learned"}


def split_rectify(x):
    return np.concatenate((np.maximum(x, 0), np.maximum(-x, 0)), axis=-1).astype(np.float32)


def nearest(x, centers, k=1):
    # Chunk callers to bound the batch x codebook distance matrix.
    distances = np.maximum((x * x).sum(1, keepdims=True) + (centers * centers).sum(1)[None]
                           - 2 * x @ centers.T, 0)
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(distances, order, axis=1)


def fit_centers(x, k, seed, backend="faiss", iterations=30):
    if len(x) < k:
        raise ValueError(f"Need at least {k} fitting vectors; found {len(x)}")
    x = np.ascontiguousarray(x, dtype=np.float32)
    if backend == "faiss":
        import faiss

        model = faiss.Kmeans(x.shape[1], k, niter=iterations, seed=seed, verbose=False,
                             max_points_per_centroid=max(256, (len(x) + k - 1) // k))
        model.train(x)
        return np.asarray(model.centroids, dtype=np.float32)
    if backend == "sklearn":
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(k, random_state=seed, n_init=3, max_iter=iterations,
                                batch_size=min(4096, len(x)))
        return model.fit(x).cluster_centers_.astype(np.float32)
    raise ValueError(f"Unknown clustering backend: {backend}")


@dataclass
class Basis:
    spec: dict
    dimension: int
    seed: int = 0
    scale: float = 1.0
    arrays: dict = field(default_factory=dict)
    fit_metadata: dict = field(default_factory=dict)

    @property
    def kind(self):
        return self.spec["kind"]

    @property
    def family(self):
        return FAMILIES[self.kind]

    @property
    def vocabulary_size(self):
        kind = self.kind
        if kind == "identity":
            return 2 * self.dimension
        if kind in {"random", "pca"}:
            return 2 * self.spec["components"]
        if kind == "kmeans":
            return self.spec["clusters"]
        if kind == "pq":
            return self.spec["subspaces"] * self.spec["codes"]
        if kind == "rq":
            return self.spec["stages"] * self.spec["codes"]
        if kind == "sae":
            return self.spec["features"]
        raise ValueError(kind)

    def feature_metadata(self, feature_id):
        if not 0 <= feature_id < self.vocabulary_size:
            raise ValueError("Feature ID is outside the fitted vocabulary")
        info = {"id": feature_id, "kind": self.kind}
        if self.kind in {"identity", "random", "pca"}:
            width = self.vocabulary_size // 2
            info.update(coordinate=feature_id % width, sign="positive" if feature_id < width else "negative")
        elif self.kind in {"pq", "rq"}:
            info.update(code=feature_id % self.spec["codes"],
                        group=feature_id // self.spec["codes"],
                        group_kind="subspace" if self.kind == "pq" else "residual_stage")
        else:
            info["component"] = feature_id
        return info

    def fit(self, x, validation=None, checkpoint_dir=None, stop=None, device="cpu"):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dimension or not len(x) or not np.isfinite(x).all():
            raise ValueError("Invalid fitting activations")
        kind = self.kind
        rng = np.random.default_rng(self.seed)
        backend = self.spec.get("backend", "faiss")
        if kind == "identity":
            pass
        elif kind == "random":
            n = self.spec["components"]
            self.arrays["projection"] = (rng.normal(size=(self.dimension, n)) / np.sqrt(n)).astype(np.float32)
        elif kind == "pca":
            from sklearn.decomposition import IncrementalPCA

            n = self.spec["components"]
            if n > min(x.shape):
                raise ValueError("PCA components exceed sample size or input dimension")
            pca = IncrementalPCA(n_components=n, batch_size=max(4096, n))
            pca.fit(x)
            self.arrays.update(mean=pca.mean_.astype(np.float32), projection=pca.components_.T.astype(np.float32))
            self.fit_metadata["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        elif kind == "kmeans":
            centers = fit_centers(x, self.spec["clusters"], self.seed, backend)
            self.arrays["centers"] = centers
            _, distances = nearest(x[:min(4096, len(x))], centers, self.spec.get("assignments", 4))
            positive = distances[distances > 0]
            self.arrays["temperature"] = np.asarray(np.median(positive) if positive.size else 1.0)
        elif kind == "pq":
            m = self.spec["subspaces"]
            if self.dimension % m:
                raise ValueError("PQ input dimension must be divisible by subspaces")
            width = self.dimension // m
            for i in range(m):
                self.arrays[f"centers_{i}"] = fit_centers(x[:, i * width:(i + 1) * width],
                    self.spec["codes"], self.seed + i, backend)
        elif kind == "rq":
            residual = x.copy()
            for i in range(self.spec["stages"]):
                centers = fit_centers(residual, self.spec["codes"], self.seed + i, backend)
                self.arrays[f"centers_{i}"] = centers
                for start in range(0, len(x), 4096):
                    selected, _ = nearest(residual[start:start + 4096], centers)
                    residual[start:start + 4096] -= centers[selected[:, 0]]
        elif kind == "sae":
            from src.retrieval.sae import fit_sae

            if validation is None or not len(validation):
                raise ValueError("SAE requires document-disjoint validation activations")
            self.arrays, self.fit_metadata = fit_sae(x, validation, self.spec, self.seed,
                                                    checkpoint_dir, stop, device)
        else:
            raise ValueError(f"Unknown basis kind: {kind}")
        self.fit_metadata.update(fitting_vectors=len(x), dimension=self.dimension, seed=self.seed)
        if validation is not None and len(validation) and kind != "sae":
            error, active = 0.0, 0
            live = np.zeros(self.vocabulary_size, dtype=bool)
            if kind == "random":
                self.arrays["reconstruction"] = np.linalg.pinv(self.arrays["projection"]).astype(np.float32)
            for start in range(0, len(validation), 1024):
                batch = np.asarray(validation[start:start+1024], dtype=np.float32)
                codes = self.transform(batch, raw=True)
                reconstructed = self.reconstruct(codes)
                error += float(np.square(reconstructed.astype(np.float64) - batch).sum())
                active += codes.nnz
                live[codes.indices] = True
            self.fit_metadata.update(validation_mse=error / (len(validation)*self.dimension),
                validation_dead_feature_fraction=float(1-live.mean()),
                validation_active_fraction=active / (len(validation)*self.vocabulary_size))
        return self

    def reconstruct(self, raw_codes):
        codes = raw_codes.toarray()
        if self.kind == "identity":
            return codes[:, :self.dimension] - codes[:, self.dimension:]
        if self.kind in {"random", "pca"}:
            n = self.spec["components"]
            linear = codes[:, :n] - codes[:, n:]
            inverse = self.arrays["reconstruction"] if self.kind == "random" else self.arrays["projection"].T
            return linear @ inverse + self.arrays.get("mean", 0)
        if self.kind == "kmeans":
            return codes @ self.arrays["centers"]
        if self.kind in {"pq", "rq"}:
            n = self.spec["codes"]
            stages = self.spec.get("subspaces", self.spec.get("stages"))
            parts = [codes[:, i*n:(i+1)*n] @ self.arrays[f"centers_{i}"] for i in range(stages)]
            return np.concatenate(parts, axis=1) if self.kind == "pq" else sum(parts)
        if self.kind == "sae":
            return codes @ self.arrays["decoder"].T + self.arrays["center"]
        raise ValueError(self.kind)

    def transform(self, x, raw=False, chunk_size=1024):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError("Invalid transform activations")
        blocks = [self._transform(x[i:i + chunk_size]) for i in range(0, len(x), chunk_size)]
        result = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, self.vocabulary_size))
        if not raw:
            if not np.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("Basis activation scale must be positive")
            result.data /= self.scale
        return nonnegative_csr(result)

    def _transform(self, x):
        kind = self.kind
        if kind == "identity":
            return sparse.csr_matrix(split_rectify(x))
        if kind in {"random", "pca"}:
            centered = x - self.arrays.get("mean", 0)
            return sparse.csr_matrix(split_rectify(centered @ self.arrays["projection"]))
        if kind == "kmeans":
            selected, distance = nearest(x, self.arrays["centers"], self.spec.get("assignments", 4))
            logits = -distance / float(self.arrays["temperature"])
            weights = np.exp(logits - logits.max(1, keepdims=True))
            weights /= weights.sum(1, keepdims=True)
        elif kind in {"pq", "rq"}:
            m = self.spec["subspaces"] if kind == "pq" else self.spec["stages"]
            width = self.dimension // m if kind == "pq" else self.dimension
            residual = x.copy()
            columns = []
            for i in range(m):
                values = x[:, i * width:(i + 1) * width] if kind == "pq" else residual
                centers = self.arrays[f"centers_{i}"]
                ids, _ = nearest(values, centers)
                columns.append(ids[:, 0] + i * self.spec["codes"])
                if kind == "rq":
                    residual -= centers[ids[:, 0]]
            selected = np.stack(columns, axis=1)
            weights = np.ones_like(selected, dtype=np.float32)
        elif kind == "sae":
            centered = x - self.arrays["center"]
            activation = np.maximum(centered @ self.arrays["encoder"].T + self.arrays["bias"], 0)
            # Stable argsort fixes feature-ID tie order in the sparse code.
            selected = np.argsort(-activation, axis=1, kind="stable")[:, :self.spec["active"]]
            weights = np.take_along_axis(activation, selected, axis=1)
        else:
            raise ValueError(kind)
        rows = np.repeat(np.arange(len(x)), selected.shape[1])
        result = sparse.csr_matrix((weights.ravel(), (rows, selected.ravel())),
                                   shape=(len(x), self.vocabulary_size), dtype=np.float32)
        result.eliminate_zeros()
        return result

    def aggregate(self, pooled, tokens, granularity, raw=False):
        if granularity == "pooled":
            return self.transform(pooled, raw=raw)
        if granularity != "token":
            raise ValueError("Granularity must be pooled or token")
        rows = []
        for states in tokens:
            codes = self.transform(states, raw=raw)
            rows.append(sparse.csr_matrix(codes.sum(axis=0)))
        return sparse.vstack(rows, format="csr") if rows else sparse.csr_matrix((0, self.vocabulary_size))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "weights.npz", **self.arrays)
        atomic_json(directory / "basis.json", {"spec": self.spec, "dimension": self.dimension,
                    "seed": self.seed, "scale": self.scale, "fit_metadata": self.fit_metadata})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        info = read_json(directory / "basis.json")
        basis = cls(**info)
        with np.load(directory / "weights.npz", allow_pickle=False) as archive:
            basis.arrays = {key: archive[key] for key in archive.files}
        return basis
