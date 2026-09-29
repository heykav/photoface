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
    similarity of the person's mean embedding, or leave it as -1 (caller
    creates a new person) if nothing is close enough. Returns
    {face_id: person_id_or_-1}. Does not mutate its inputs."""
    means: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for f in existing:
        if f.person_id is None:
            continue
        means[f.person_id] = means.get(f.person_id, np.zeros_like(f.embedding)) + f.embedding
        counts[f.person_id] = counts.get(f.person_id, 0) + 1
    for pid in means:
        means[pid] = means[pid] / counts[pid]

    assignments: Dict[int, int] = {}
    for face in new_faces:
        best_pid, best_sim = -1, threshold
        for pid, mean in means.items():
            sim = _cosine_sim(face.embedding, mean)
            if sim > best_sim:
                best_pid, best_sim = pid, sim
        assignments[face.face_id] = best_pid
        if best_pid != -1:
            # fold this face into the running mean so later faces in the
            # same batch can also match against it
            n = counts.get(best_pid, 0)
            means[best_pid] = (means[best_pid] * n + face.embedding) / (n + 1)
            counts[best_pid] = n + 1
        else:
            # tentative new person within this batch, keyed by negative id
            # so later faces in the batch can still cluster with it; caller
            # maps these to real person rows after the pass completes.
            tentative_id = -(len(means) + 1000000)
            means[tentative_id] = face.embedding
            counts[tentative_id] = 1
            assignments[face.face_id] = tentative_id
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

    Average linkage on unit vectors needs no pairwise loops: the mean cosine
    similarity of clusters A and B is (sum_A . sum_B) / (|A||B|), so a
    cluster-similarity matrix is kept and updated with the Lance-Williams
    rule when two clusters merge. Cost is O(n^2) memory and O(n^2) per merge.
    """
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
