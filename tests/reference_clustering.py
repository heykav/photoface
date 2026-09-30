"""Reference copy of the pre-optimisation `clustering.recluster`.

`clustering.recluster` now caches each row's best partner instead of taking
an argmax over a second masked n x n matrix on every merge. This copy of the
old code (O(n^3) time, 16 n^2 bytes) is kept only so the tests can check the
fast version returns exactly the same assignment, placeholder ids included.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from clustering import DEFAULT_THRESHOLD, FaceRecord


def reference_recluster(faces: List[FaceRecord],
                        threshold: float = DEFAULT_THRESHOLD) -> Dict[int, int]:
    """The previous full-matrix implementation, verbatim apart from this
    docstring: every merge rescans the whole masked n x n matrix."""
    n = len(faces)
    if n == 0:
        return {}

    emb = np.nan_to_num(
        np.stack([np.asarray(f.embedding, dtype=np.float64).ravel() for f in faces]),
        nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sim = unit @ unit.T  # average cluster-to-cluster similarity, initially pairwise

    has_anchor = np.array([f.pinned and f.person_id is not None for f in faces])
    anchor_id = np.array([f.person_id if h else 0 for f, h in zip(faces, has_anchor)],
                         dtype=np.int64)
    size = np.ones(n)
    active = np.ones(n, dtype=bool)
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}

    def compatible(i: int) -> np.ndarray:
        """Which clusters may merge with cluster i (not itself, still active,
        not anchored to a different pinned person)."""
        ok = active.copy()
        ok[i] = False
        if has_anchor[i]:
            ok &= ~(has_anchor & (anchor_id != anchor_id[i]))
        return ok

    work = np.full((n, n), -np.inf)
    for i in range(n):
        ok = compatible(i)
        work[i, ok] = sim[i, ok]

    while True:
        flat = int(np.argmax(work))  # first maximum -> lowest (i, j): deterministic
        i, j = divmod(flat, n)
        best = work[i, j]
        if not np.isfinite(best) or best < threshold - 1e-12:
            break
        if i > j:
            i, j = j, i
        # merge j into i (the lower index stays the representative)
        total = size[i] + size[j]
        merged = (size[i] * sim[i] + size[j] * sim[j]) / total
        sim[i, :] = merged
        sim[:, i] = merged
        size[i] = total
        members[i].extend(members.pop(j))
        active[j] = False
        if not has_anchor[i] and has_anchor[j]:
            has_anchor[i], anchor_id[i] = True, anchor_id[j]
        work[j, :] = -np.inf
        work[:, j] = -np.inf
        ok = compatible(i)
        row = np.where(ok, sim[i], -np.inf)
        work[i, :] = row
        work[:, i] = row

    result: Dict[int, int] = {}
    for root, idxs in members.items():
        pid = int(anchor_id[root]) if has_anchor[root] else -(min(idxs) + 1)
        for i in idxs:
            result[faces[i].face_id] = pid
    return result
