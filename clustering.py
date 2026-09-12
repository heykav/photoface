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


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def recluster(faces: List[FaceRecord],
             threshold: float = DEFAULT_THRESHOLD) -> Dict[int, int]:
    """Full average-linkage re-clustering over ALL given faces (pinned and
    unpinned together, so unpinned faces can still join a pinned person's
    cluster). Returns {face_id: person_id}, where person_id is either a real,
    existing person id (a cluster anchored to a pinned face keeps that
    person's id) or a negative placeholder id shared by every face in a
    cluster that has no pinned anchor - the caller maps each distinct
    negative id to one newly-created person row (mirrors the tentative-id
    convention `greedy_assign` uses). Pinned faces always keep their existing
    person_id; this function only decides where *unpinned* faces land, and
    never merges two clusters that are each anchored to a different pinned
    person."""
    n = len(faces)
    if n == 0:
        return {}
    if n == 1:
        f = faces[0]
        pid = f.person_id if f.pinned and f.person_id is not None else -1
        return {f.face_id: pid}

    emb = np.stack([f.embedding for f in faces])
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sim = unit @ unit.T  # cosine similarity matrix
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)

    uf = _UnionFind(n)
    # cluster -> pinned person_id it's anchored to (None if none yet)
    anchor: Dict[int, Optional[int]] = {}
    for i, f in enumerate(faces):
        anchor[i] = f.person_id if f.pinned else None

    members: Dict[int, List[int]] = {i: [i] for i in range(n)}

    def cluster_of(i: int) -> int:
        return uf.find(i)

    def avg_linkage(a_members: List[int], b_members: List[int]) -> float:
        sub = dist[np.ix_(a_members, b_members)]
        finite = sub[np.isfinite(sub)]
        return float(finite.mean()) if finite.size else np.inf

    threshold_dist = 1.0 - threshold
    changed = True
    while changed:
        changed = False
        roots = sorted(set(uf.find(i) for i in range(n)))
        best = (None, None, np.inf)
        for ai in range(len(roots)):
            for bi in range(ai + 1, len(roots)):
                ra, rb = roots[ai], roots[bi]
                aa, ab = anchor.get(ra), anchor.get(rb)
                if aa is not None and ab is not None and aa != ab:
                    continue  # never merge two different pinned identities
                d = avg_linkage(members[ra], members[rb])
                if d < best[2]:
                    best = (ra, rb, d)
        ra, rb, d = best
        if ra is not None and d <= threshold_dist:
            uf.union(ra, rb)
            new_root = uf.find(ra)
            old_root = rb if new_root == ra else ra
            merged_members = members[ra] + members[rb]
            aa, ab = anchor.get(ra), anchor.get(rb)
            merged_anchor = aa if aa is not None else ab
            for r in (ra, rb):
                if r != new_root:
                    del members[r]
                    del anchor[r]
            members[new_root] = merged_members
            anchor[new_root] = merged_anchor
            changed = True

    result: Dict[int, int] = {}
    for root, idxs in members.items():
        pid = anchor.get(root)
        if pid is None:
            pid = -(root + 1)  # negative placeholder, unique per unanchored cluster
        for i in idxs:
            result[faces[i].face_id] = pid
    return result
