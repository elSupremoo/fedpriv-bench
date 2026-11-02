"""Client partitioning: IID and Dirichlet label skew (Hsu et al., 2019)."""

from __future__ import annotations

import numpy as np


def iid_partition(n: int, num_clients: int, seed: int) -> list[np.ndarray]:
    """Uniform random split of indices 0..n-1 into near-equal shards."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(s) for s in np.array_split(perm, num_clients)]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 100,
    max_tries: int = 1000,
) -> list[np.ndarray]:
    """Label-skewed split: for every class c, draw p ~ Dir(alpha * 1_K) and
    deal that class's (shuffled) samples to clients in proportions p.

    Small alpha => each client sees mostly one or two classes. The draw is
    repeated until every client has at least ``min_size`` samples so that
    local training (and DP sampling) is well defined. Every index is assigned
    to exactly one client.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_tries):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                buckets[k].extend(part.tolist())
        sizes = [len(b) for b in buckets]
        if min(sizes) >= min_size:
            return [np.sort(np.asarray(b, dtype=np.int64)) for b in buckets]
    raise RuntimeError(
        f"could not reach min_size={min_size} with alpha={alpha} in {max_tries} tries"
    )


def make_partition(labels: np.ndarray, num_clients: int, alpha, seed: int, min_size: int = 100):
    """alpha=None (or 'iid') -> IID; otherwise Dirichlet(alpha)."""
    if alpha is None or alpha == "iid":
        return iid_partition(len(labels), num_clients, seed)
    return dirichlet_partition(labels, num_clients, float(alpha), seed, min_size=min_size)


def class_histogram(labels: np.ndarray, parts: list[np.ndarray], num_classes: int) -> np.ndarray:
    return np.stack([np.bincount(labels[p], minlength=num_classes) for p in parts])
