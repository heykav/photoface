"""Synthetic face-embedding generators and clustering-quality metrics.

These stand in for SFace output so the clustering algorithm (not the model)
is what the tests exercise. Embeddings are 128-d like SFace's.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from clustering import FaceRecord

DIM = 128


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def blobs(sizes: Sequence[int], sigma: float, seed: int = 0,
          centroid_sim: float = 0.0) -> Tuple[List[FaceRecord], Dict[int, int]]:
    """`len(sizes)` identities. Members are unit(centroid + sigma * noise / sqrt(DIM)),
    so two members of one identity have expected cosine similarity ~ 1/(1+sigma^2).
    `centroid_sim` > 0 pulls every centroid toward a shared direction, making
    identities overlap. Returns (records with person_id=None, {face_id: true_label}).
    """
    rng = np.random.RandomState(seed)
    shared = unit(rng.normal(size=DIM))
    cents = []
    for _ in sizes:
        c = unit(rng.normal(size=DIM))
        c = unit(np.sqrt(1 - centroid_sim) * c + np.sqrt(centroid_sim) * shared)
        cents.append(c)
    faces, truth = [], {}
    fid = 1
    for label, (c, n) in enumerate(zip(cents, sizes)):
        for _ in range(n):
            e = unit(c + sigma * rng.normal(size=DIM) / np.sqrt(DIM))
            faces.append(FaceRecord(fid, e.astype(np.float32)))
            truth[fid] = label
            fid += 1
    return faces, truth


def purity_completeness(assign: Dict[int, int], truth: Dict[int, int]) -> Tuple[float, float]:
    """purity: share of faces whose cluster's majority true label is their own.
    completeness: share of faces whose true identity's majority cluster is their own.
    Both 1.0 for a perfect clustering."""
    def majority_share(key_of, val_of) -> float:
        groups: Dict[int, Dict[int, int]] = {}
        for f in assign:
            groups.setdefault(key_of(f), {}).setdefault(val_of(f), 0)
            groups[key_of(f)][val_of(f)] += 1
        return sum(max(g.values()) for g in groups.values()) / len(assign)

    return (majority_share(lambda f: assign[f], lambda f: truth[f]),
            majority_share(lambda f: truth[f], lambda f: assign[f]))


def partition(assign: Dict[int, int]) -> frozenset:
    """Cluster ids are arbitrary; compare partitions as sets of face-id sets."""
    groups: Dict[int, set] = {}
    for f, c in assign.items():
        groups.setdefault(c, set()).add(f)
    return frozenset(frozenset(g) for g in groups.values())
