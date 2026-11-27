from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def canonical_edges(pairs: np.ndarray, n: int, self_loops: bool = True) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size:
        if pairs.min() < 0 or pairs.max() >= n:
            raise ValueError("Graph edge index is out of bounds")
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    if self_loops:
        pairs = np.concatenate((pairs, np.column_stack((np.arange(n), np.arange(n)))), axis=0)
    # Row-major order matches np.nonzero of the original spatial adjacency.
    return np.unique(pairs, axis=0).T.copy()


def spatial_edges(coords: np.ndarray, radius: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Expected finite two-dimensional spatial coordinates")
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    # Original STAGATE excludes zero-distance pairs other than the added self-loop.
    if len(pairs):
        pairs = pairs[np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1) > 0]
    return canonical_edges(pairs, len(coords))


def expression_edges(pca: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k < len(pca):
        raise ValueError("expression_k must be between 1 and n_spots - 1")
    indices = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(pca).kneighbors(pca, return_distance=False)
    # Explicitly remove self: tied distances do not guarantee self is first.
    pairs = [(i, int(j)) for i, row in enumerate(indices) for j in row if j != i]
    per_node = [[] for _ in range(len(pca))]
    for i, j in pairs:
        if len(per_node[i]) < k:
            per_node[i].append(j)
    return canonical_edges(np.array([(i, j) for i, row in enumerate(per_node) for j in row]), len(pca))


def local_expression_edges(pca: np.ndarray, spatial: np.ndarray, k: int) -> np.ndarray:
    """Cosine KNN restricted to spatial graph distance <= 2; no distant fallback."""
    from scipy import sparse
    n = len(pca)
    adjacency = sparse.csr_matrix((np.ones(spatial.shape[1], dtype=np.float32), (spatial[0], spatial[1])), shape=(n, n))
    candidates = (adjacency @ adjacency).tocsr()
    normalized = pca / np.maximum(np.linalg.norm(pca, axis=1, keepdims=True), 1e-12)
    pairs = []
    for i in range(n):
        indices = candidates.indices[candidates.indptr[i]:candidates.indptr[i + 1]]
        indices = indices[indices != i]
        similarities = normalized[indices] @ normalized[i]
        chosen = indices[np.argsort(-similarities, kind="stable")[:k]]
        pairs.extend((i, int(j)) for j in chosen)
    return canonical_edges(np.asarray(pairs), n)


def apply_boundary_control(probability: np.ndarray, mode="expression", seed=17) -> np.ndarray:
    if mode == "expression":
        return probability.copy()
    if mode == "constant":
        return np.full_like(probability, probability.mean())
    if mode == "shuffled":
        return np.random.default_rng(seed).permutation(probability)
    raise ValueError(f"Unknown boundary mode: {mode}")


def structural_edges(edge_index: np.ndarray) -> np.ndarray:
    # Each undirected spatial edge is used once. The averaged symmetric loss is unchanged.
    return edge_index[:, edge_index[0] < edge_index[1]].copy()


def boundary_prior(pca: np.ndarray, edges: np.ndarray, slope: float = 2.0, offset: float = 1.0) -> np.ndarray:
    from scipy.special import expit
    pca = np.asarray(pca, dtype=np.float64)
    pca = pca / np.maximum(np.linalg.norm(pca, axis=1, keepdims=True), 1e-12)
    delta = np.linalg.norm(pca[edges[0]] - pca[edges[1]], axis=1)
    if not len(delta):
        raise ValueError("Boundary loss needs at least one non-self spatial edge")
    standardized = (delta - delta.mean()) / (delta.std() + 1e-8)
    return expit(slope * (standardized - offset)).astype(np.float32)


def graph_summary(edges: np.ndarray, n: int) -> dict:
    nonself = edges[0] != edges[1]
    degree = np.bincount(edges[1, nonself], minlength=n)
    return {"nodes": n, "directed_nonself_edges": int(nonself.sum()), "self_loops": int((~nonself).sum()),
            "mean_degree": float(degree.mean()), "isolated_without_self": int((degree == 0).sum())}
