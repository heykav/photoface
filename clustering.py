"""Face clustering into persons.

Two passes, mirroring the reference project's design:

- `greedy_assign`: a fast incremental pass usable while analysis is still
  running - each new face is compared to the running mean embedding of every
  known person and joined to the closest one above `threshold`, else it
  starts a new person.
- `recluster`: a full average-linkage agglomerative re-clustering of every
  *unpinned* face once a batch of analysis is done, for a cleaner final
  grouping than the greedy pass alone produces. Faces the user has pinned
  (hand-assigned, or confirmed by a manual edit) are treated as fixed anchors
  for their person and are never moved or reassigned by this function -
  clusters anchored to two different pinned persons are never merged
  together, and a cluster not touching any pinned anchor becomes a new
  person.

Embeddings are compared with cosine similarity (SFace's own recommended
metric); `threshold` is a similarity cutoff in [-1, 1], not a distance.

Thresholds (all cosine similarity):
- `DEFAULT_THRESHOLD` = 0.363 is the cutoff the OpenCV Zoo publishes for
  SFace ("same person" if similarity >= 0.363). Raise it for fewer false
  merges (more, smaller clusters); lower it for fewer split identities.
- `greedy_assign` joins a face to a person when similarity is strictly
  greater than the threshold; `recluster` merges two clusters while their
  *average* pairwise similarity is at least the threshold.

Determinism and stability:
- Both functions are pure and deterministic for a fixed input order (no
  randomness anywhere; ties break toward the lowest input index).
- `recluster` returns the same partition for any permutation of the input
  whenever no two candidate merges tie exactly.
- `stabilize_assignment` maps the anonymous clusters `recluster` finds back
  onto existing person ids, so reclustering twice does not churn people
  (and renamed people keep their names).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

DEFAULT_THRESHOLD = 0.363  # OpenCV Zoo's published SFace cosine-similarity cutoff


def _unit_rows(vectors: List[np.ndarray]) -> np.ndarray:
    """Stack embeddings as float64 rows scaled to unit length. NaN/inf
    components become 0 and an all-zero row stays all-zero (similarity 0 to
    everything), so a corrupt embedding can never poison a comparison."""
    emb = np.nan_to_num(
        np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors]),
        nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class FaceRecord:
    face_id: int
    embedding: np.ndarray
    person_id: Optional[int] = None
    pinned: bool = False


def greedy_assign(new_faces: List[FaceRecord], existing: List[FaceRecord],
                  threshold: float = DEFAULT_THRESHOLD) -> Dict[int, int]:
    """Assign each face in `new_faces` (person_id currently None) to the
    closest person found in `existing` (already-assigned faces), by cosine
    similarity to the mean of the person's *unit-length* embeddings, or give
    it a tentative id <= -1000000 (caller creates a new person) if no person
    is strictly more similar than `threshold`. Faces in one batch are
    processed in input order and each assigned face updates its person's
    mean, so the result depends on that order (`recluster` does not).
    Returns {face_id: person_id_or_tentative_id}. Does not mutate its inputs.

    Embeddings are normalized first (and NaN/inf zeroed, as in `recluster`),
    so the outcome does not depend on the arbitrary scale of each raw
    embedding and one corrupt face cannot make its person unmatchable.
    """
    if not new_faces:
        return {}
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    known = [f for f in existing if f.person_id is not None]
    if known:
        for f, u in zip(known, _unit_rows([f.embedding for f in known]), strict=True):
            sums[f.person_id] = sums.get(f.person_id, 0.0) + u
            counts[f.person_id] = counts.get(f.person_id, 0) + 1

    assignments: Dict[int, int] = {}
    new_unit = _unit_rows([f.embedding for f in new_faces])
    next_tentative = -1000000
    for face, u in zip(new_faces, new_unit, strict=True):
        best_pid, best_sim = -1, threshold
        for pid, total in sums.items():  # insertion order: ties -> first seen
            sim = _cosine_sim(u, total / counts[pid])
            if sim > best_sim:
                best_pid, best_sim = pid, sim
        if best_pid == -1:
            # tentative new person within this batch, so later faces in the
            # batch can still cluster with it; caller maps these to real
            # person rows after the pass completes.
            best_pid = next_tentative
            next_tentative -= 1
            sums[best_pid] = np.zeros_like(u)
            counts[best_pid] = 0
        # fold this face into the running mean so later faces in the same
        # batch can also match against it
        sums[best_pid] = sums[best_pid] + u
        counts[best_pid] += 1
        assignments[face.face_id] = best_pid
    return assignments


def recluster(faces: List[FaceRecord],
              threshold: float = DEFAULT_THRESHOLD) -> Dict[int, int]:
    """Full average-linkage re-clustering over ALL given faces (pinned and
    unpinned together, so unpinned faces can still join a pinned person's
    cluster). Returns {face_id: person_id}, where person_id is either a real,
    existing person id (a cluster anchored to a pinned face keeps that
    person's id) or a negative placeholder id (-(lowest input index in the
    cluster + 1)) shared by every face in a cluster that has no pinned
    anchor - the caller maps each distinct negative id to a person row (see
    `stabilize_assignment`). Pinned faces always keep their existing
    person_id, and two clusters anchored to different pinned persons are
    never merged.

    Algorithm: repeatedly merge the most similar pair of compatible clusters
    while its average pairwise cosine similarity is >= `threshold`. Ties
    break toward the lowest (row, column) cluster index. Average linkage on
    unit vectors needs no pairwise loops: one n x n cluster-similarity matrix
    is kept and updated with the Lance-Williams rule
    sim(i+j, k) = (|i| sim(i, k) + |j| sim(j, k)) / (|i| + |j|).
    Each row caches its best compatible partner, so a merge only rescans the
    rows whose cached partner was one of the two merged clusters, instead of
    the whole matrix.

    Cost: 8 n^2 bytes (one float64 matrix; about 0.8 GB at n = 10,000 faces)
    and O(n^2) time for typical inputs (O(n^3) worst case).
    """
    n = len(faces)
    if n == 0:
        return {}

    unit = _unit_rows([f.embedding for f in faces])
    sim = unit @ unit.T  # average cluster-to-cluster similarity, initially pairwise
    del unit
    # make it exactly symmetric (BLAS need not be), in row blocks so no second
    # n x n temporary is allocated
    block = 256
    for r0 in range(block, n, block):
        r1 = min(n, r0 + block)
        sim[r0:r1, :r0] = sim[:r0, r0:r1].T

    has_anchor = np.array([f.pinned and f.person_id is not None for f in faces])
    anchor_id = np.array([f.person_id if h else 0 for f, h in zip(faces, has_anchor, strict=True)],
                         dtype=np.int64)
    size = np.ones(n)
    active = np.ones(n, dtype=bool)
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    best_val = np.full(n, -np.inf)
    best_arg = np.zeros(n, dtype=np.int64)

    def compatible_with(i: int) -> np.ndarray:
        """Which clusters may merge with cluster i (not itself, still active,
        not anchored to a different pinned person). Symmetric in i and k."""
        ok = active.copy()
        ok[i] = False
        if has_anchor[i]:
            ok &= ~(has_anchor & (anchor_id != anchor_id[i]))
        return ok

    def rescan(rows: np.ndarray) -> None:
        """Recompute the cached best partner (first maximum) of each row."""
        for c0 in range(0, len(rows), block):
            r = rows[c0:c0 + block]
            ok = np.broadcast_to(active, (len(r), n)).copy()
            ok[np.arange(len(r)), r] = False
            anchored = has_anchor[r]
            if anchored.any():
                clash = (has_anchor[None, :]
                         & (anchor_id[None, :] != anchor_id[r][:, None])
                         & anchored[:, None])
                ok &= ~clash
            vals = np.where(ok, sim[r], -np.inf)
            arg = np.argmax(vals, axis=1)
            best_arg[r] = arg
            best_val[r] = vals[np.arange(len(r)), arg]

    rescan(np.arange(n))

    while True:
        # lowest row holding the global maximum, then its lowest column: the
        # same pair a row-major argmax over the full masked matrix would pick
        i = int(np.argmax(best_val))
        best = best_val[i]
        if not np.isfinite(best) or best < threshold - 1e-12:
            break
        j = int(best_arg[i])
        if i > j:  # cannot happen for a symmetric matrix; kept as a guard
            i, j = j, i
        # merge j into i (the lower index stays the representative)
        total = size[i] + size[j]
        merged = (size[i] * sim[i] + size[j] * sim[j]) / total
        sim[i, :] = merged
        sim[:, i] = merged
        size[i] = total
        members[i].extend(members.pop(j))
        active[j] = False
        best_val[j] = -np.inf
        if not has_anchor[i] and has_anchor[j]:
            has_anchor[i], anchor_id[i] = True, anchor_id[j]

        # column i changed for every row; column j is gone
        col = np.where(compatible_with(i), merged, -np.inf)
        stale = active & ((best_arg == i) | (best_arg == j))
        stale[i] = True
        better = active & ~stale & ((col > best_val)
                                    | ((col == best_val) & (i < best_arg)))
        best_val[better] = col[better]
        best_arg[better] = i
        rescan(np.flatnonzero(stale))

    result: Dict[int, int] = {}
    for root, idxs in members.items():
        pid = int(anchor_id[root]) if has_anchor[root] else -(min(idxs) + 1)
        for i in idxs:
            result[faces[i].face_id] = pid
    return result


def stabilize_assignment(faces: List[FaceRecord],
                         assignment: Dict[int, int]) -> Dict[int, int]:
    """Rewrite the negative placeholder ids in a `recluster` result to
    existing person ids where the cluster overlaps a person the faces were
    previously assigned to, so reclustering does not create new people for
    groups that already had one (and renamed people keep their names).

    Each previous person id is claimed by at most one cluster, and never by a
    cluster if the id is already used by a pinned-anchored cluster. Claims are
    resolved greedily by overlap size (ties: lowest person id, then lowest
    face id in the cluster), so the outcome is deterministic. Clusters that
    claim nothing keep their negative placeholder (caller creates a person).
    Applying this to its own output is a fixed point.
    """
    prev = {f.face_id: f.person_id for f in faces}
    taken = {pid for pid in assignment.values() if pid >= 0}
    overlap: Dict[tuple, int] = {}
    cluster_min_face: Dict[int, int] = {}
    for fid, cid in assignment.items():
        if cid >= 0:
            continue
        cluster_min_face[cid] = min(fid, cluster_min_face.get(cid, fid))
        p = prev.get(fid)
        if p is not None and p not in taken:
            overlap[(cid, p)] = overlap.get((cid, p), 0) + 1
    ranked = sorted(overlap.items(),
                    key=lambda kv: (-kv[1], kv[0][1], cluster_min_face[kv[0][0]]))
    mapping: Dict[int, int] = {}
    used: set = set()
    for (cid, p), _count in ranked:
        if cid in mapping or p in used:
            continue
        mapping[cid] = p
        used.add(p)
    return {fid: mapping.get(cid, cid) if cid < 0 else cid
            for fid, cid in assignment.items()}
