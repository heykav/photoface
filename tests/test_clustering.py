import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from clustering import FaceRecord, greedy_assign, recluster, DEFAULT_THRESHOLD


def _vec(seed: int, noise: float = 0.0, dim: int = 8) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.normal(size=dim)
    if noise:
        base = base + np.random.RandomState(seed + 999).normal(scale=noise, size=dim)
    return base.astype(np.float32)


class TestGreedyAssign:
    def test_new_face_matches_existing_person(self):
        person_face = _vec(1)
        existing = [FaceRecord(1, person_face, person_id=10, pinned=True)]
        new_face = FaceRecord(2, person_face + 0.001, person_id=None)
        result = greedy_assign([new_face], existing, threshold=0.5)
        assert result[2] == 10

    def test_dissimilar_face_gets_tentative_id(self):
        existing = [FaceRecord(1, _vec(1), person_id=10, pinned=True)]
        new_face = FaceRecord(2, _vec(999), person_id=None)  # very different seed
        result = greedy_assign([new_face], existing, threshold=0.95)
        assert result[2] != 10
        assert result[2] <= -1000000

    def test_two_similar_new_faces_share_tentative_id(self):
        base = _vec(42)
        f1 = FaceRecord(1, base, person_id=None)
        f2 = FaceRecord(2, base + 0.0001, person_id=None)
        result = greedy_assign([f1, f2], [], threshold=0.5)
        assert result[1] == result[2]

    def test_no_existing_persons_all_new(self):
        f1 = FaceRecord(1, _vec(1), person_id=None)
        result = greedy_assign([f1], [], threshold=0.5)
        assert result[1] <= -1000000


class TestRecluster:
    def test_single_pinned_face_keeps_person(self):
        f = FaceRecord(1, _vec(1), person_id=5, pinned=True)
        result = recluster([f])
        assert result[1] == 5

    def test_single_unpinned_face_gets_placeholder(self):
        f = FaceRecord(1, _vec(1), person_id=None, pinned=False)
        result = recluster([f])
        assert result[1] < 0

    def test_similar_faces_cluster_together(self):
        base = _vec(7)
        faces = [
            FaceRecord(1, base, person_id=None),
            FaceRecord(2, base + 0.001, person_id=None),
            FaceRecord(3, base - 0.001, person_id=None),
        ]
        result = recluster(faces, threshold=0.5)
        assert result[1] == result[2] == result[3]

    def test_dissimilar_faces_stay_separate(self):
        faces = [
            FaceRecord(1, _vec(1), person_id=None),
            FaceRecord(2, _vec(500), person_id=None),
        ]
        result = recluster(faces, threshold=0.98)
        assert result[1] != result[2]

    def test_pinned_face_anchors_its_cluster(self):
        base = _vec(3)
        faces = [
            FaceRecord(1, base, person_id=99, pinned=True),
            FaceRecord(2, base + 0.001, person_id=None, pinned=False),
        ]
        result = recluster(faces, threshold=0.5)
        assert result[2] == 99

    def test_two_pinned_persons_never_merge(self):
        base = _vec(10)
        faces = [
            FaceRecord(1, base, person_id=1, pinned=True),
            FaceRecord(2, base + 0.0001, person_id=2, pinned=True),
        ]
        result = recluster(faces, threshold=0.999999)
        assert result[1] == 1
        assert result[2] == 2

    def test_pinned_face_never_reassigned_by_caller_semantics(self):
        # recluster() itself always returns the pinned face's own id for a
        # pinned face (the anchor), even if - due to threshold quirks - it
        # ends up alone; the invariant callers rely on is exercised here.
        f = FaceRecord(1, _vec(1), person_id=42, pinned=True)
        result = recluster([f])
        assert result[1] == 42

    def test_empty_input(self):
        assert recluster([]) == {}
