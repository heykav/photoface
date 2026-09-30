"""Invariants of the row-cached `recluster` and the normalized `greedy_assign`,
checked on synthetic embeddings (no ONNX models involved)."""
import random
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from clustering import FaceRecord, greedy_assign, recluster
from reference_clustering import reference_recluster
from synth import DIM, blobs, partition


def _pin(faces, rng, n_pins, person_ids):
    for f in rng.sample(faces, n_pins):
        f.person_id, f.pinned = rng.choice(person_ids), True
    return faces


def _refines(fine, coarse) -> bool:
    """Every cluster of `fine` lies inside one cluster of `coarse`."""
    where = {f: c for c, group in enumerate(coarse) for f in group}
    return all(len({where[f] for f in group}) == 1 for group in fine)


class TestMatchesReferenceImplementation:
    """The optimized loop must make exactly the merges the old full-matrix
    loop made - same partition AND same placeholder / person ids."""

    @pytest.mark.parametrize("seed", range(12))
    def test_random_libraries_with_pins(self, seed):
        rng = random.Random(seed)
        sizes = [rng.randint(1, 25) for _ in range(rng.randint(1, 8))]
        faces, _ = blobs(sizes, sigma=rng.choice([0.4, 0.9, 1.3]), seed=seed,
                         centroid_sim=rng.choice([0.0, 0.3, 0.6]))
        _pin(faces, rng, min(len(faces), rng.randint(0, 6)), [11, 22, 33])
        rng.shuffle(faces)
        for thr in (-1.0, 0.0, 0.2, 0.363, 0.6, 0.95):
            assert recluster(faces, thr) == reference_recluster(faces, thr)

    def test_exact_ties_break_the_same_way(self):
        # many bit-identical embeddings -> many exactly tied candidate merges
        rng = np.random.RandomState(0)
        protos = [rng.normal(size=DIM).astype(np.float32) for _ in range(4)]
        faces = [FaceRecord(i + 1, protos[i % 4].copy()) for i in range(40)]
        faces[3].person_id, faces[3].pinned = 7, True
        faces[7].person_id, faces[7].pinned = 8, True  # same proto, other person
        for thr in (-1.0, 0.363, 0.999):
            assert recluster(faces, thr) == reference_recluster(faces, thr)

    def test_degenerate_embeddings(self):
        faces = [FaceRecord(1, np.zeros(DIM, np.float32)),
                 FaceRecord(2, np.full(DIM, np.nan, np.float32)),
                 FaceRecord(3, np.ones(DIM, np.float32)),
                 FaceRecord(4, np.ones(DIM, np.float32) * 1e-30),
                 FaceRecord(5, -np.ones(DIM, np.float32))]
        for thr in (-1.0, 0.0, 0.5):
            assert recluster(faces, thr) == reference_recluster(faces, thr)

    def test_larger_library(self):
        faces, _ = blobs([10] * 60, sigma=1.0, seed=3, centroid_sim=0.4)
        _pin(faces, random.Random(3), 20, [1, 2, 3, 4])
        assert recluster(faces, 0.363) == reference_recluster(faces, 0.363)


class TestProperties:
    @pytest.mark.parametrize("seed", range(6))
    def test_higher_threshold_gives_a_refinement(self, seed):
        """The merge sequence does not depend on the threshold, which only
        decides where it stops - so a stricter threshold can only split
        clusters, never regroup faces differently."""
        faces, _ = blobs([12, 9, 7, 3, 1], sigma=1.0, seed=seed, centroid_sim=0.4)
        _pin(faces, random.Random(seed), 4, [5, 6])
        parts = [partition(recluster(faces, t)) for t in (-0.2, 0.1, 0.363, 0.5, 0.8)]
        for coarse, fine in zip(parts, parts[1:], strict=False):
            assert _refines(fine, coarse)

    @pytest.mark.parametrize("seed", range(6))
    def test_each_cluster_holds_at_most_one_pinned_person(self, seed):
        rng = random.Random(seed)
        faces, _ = blobs([10, 10, 10], sigma=1.1, seed=seed, centroid_sim=0.5)
        _pin(faces, rng, 9, [1, 2, 3])
        result = recluster(faces, -1.0)  # everything that may merge, merges
        for f in faces:
            if f.pinned:
                assert result[f.face_id] == f.person_id
        for group in partition(result):
            pinned_people = {f.person_id for f in faces if f.face_id in group and f.pinned}
            assert len(pinned_people) <= 1
        # at threshold -1 every unanchored cluster can join an anchored one
        assert set(result.values()) == {f.person_id for f in faces if f.pinned}

    @pytest.mark.parametrize("seed", range(4))
    def test_rescaling_embeddings_changes_nothing(self, seed):
        """Cosine similarity ignores each embedding's length, so both passes
        must too (raw SFace features are not unit length)."""
        faces, _ = blobs([8, 8, 8, 2], sigma=0.7, seed=seed, centroid_sim=0.3)
        rng = np.random.RandomState(seed)
        scaled = [FaceRecord(f.face_id, f.embedding * np.float32(rng.uniform(0.01, 100)))
                  for f in faces]
        assert partition(recluster(scaled)) == partition(recluster(faces))
        assert partition(greedy_assign(scaled, [])) == partition(greedy_assign(faces, []))

    def test_permutation_with_pins_keeps_partition(self):
        faces, _ = blobs([10, 10, 10], sigma=0.5, seed=12)
        faces[0].person_id, faces[0].pinned = 1, True
        faces[15].person_id, faces[15].pinned = 2, True
        base = recluster(faces)
        rng = random.Random(1)
        for _ in range(5):
            shuffled = faces[:]
            rng.shuffle(shuffled)
            again = recluster(shuffled)
            assert partition(again) == partition(base)
            assert again[faces[0].face_id] == 1 and again[faces[15].face_id] == 2

    def test_placeholder_ids_are_lowest_input_index(self):
        faces, _ = blobs([3, 2], sigma=0.1, seed=0)
        result = recluster(faces)
        assert sorted(set(result.values()), reverse=True) == [-1, -4]


class TestGreedy:
    def test_scale_of_one_face_does_not_dominate_its_person(self):
        """A person's centre is the mean of unit vectors, so one face with a
        huge raw norm cannot drag it away from that person's other faces."""
        rng = np.random.RandomState(0)
        a, b = rng.normal(size=DIM), rng.normal(size=DIM)
        existing = [FaceRecord(1, (a * 1000).astype(np.float32), 5),
                    FaceRecord(2, b.astype(np.float32), 5),
                    FaceRecord(3, b.astype(np.float32), 5)]
        new = FaceRecord(4, b.astype(np.float32))
        # similarity to mean of units ~ (0 + 1 + 1) / 3 / |mean| ~ 0.94
        assert greedy_assign([new], existing, threshold=0.5) == {4: 5}

    def test_corrupt_face_does_not_make_person_unmatchable(self):
        e = np.ones(DIM, np.float32)
        existing = [FaceRecord(1, e, 9),
                    FaceRecord(2, np.full(DIM, np.nan, np.float32), 9)]
        assert greedy_assign([FaceRecord(3, e)], existing, threshold=0.5) == {3: 9}

    def test_only_new_faces_are_returned_and_inputs_untouched(self):
        faces, _ = blobs([5, 5], sigma=0.3, seed=1)
        existing = [FaceRecord(f.face_id, f.embedding, 1, True) for f in faces[:3]]
        before = [f.embedding.copy() for f in faces]
        out = greedy_assign(faces[3:], existing)
        assert set(out) == {f.face_id for f in faces[3:]}
        assert all(np.array_equal(b, f.embedding) for b, f in zip(before, faces, strict=True))
        assert all(f.person_id is None for f in faces[3:])

    def test_tentative_ids_are_distinct_and_below_the_sentinel(self):
        rng = np.random.RandomState(2)
        faces = [FaceRecord(i, rng.normal(size=DIM).astype(np.float32)) for i in range(20)]
        out = greedy_assign(faces, [], threshold=0.99)
        assert len(set(out.values())) == 20
        assert all(v <= -1000000 for v in out.values())

    def test_tie_goes_to_first_seen_person(self):
        e = np.ones(DIM, np.float32)
        existing = [FaceRecord(1, e, 30), FaceRecord(2, e, 20)]
        assert greedy_assign([FaceRecord(3, e)], existing, threshold=0.5) == {3: 30}


def test_memory_is_one_float64_matrix():
    """Peak traced allocation stays near 8 n^2 bytes (the old loop kept two
    n x n float64 matrices: ~17 n^2 measured)."""
    faces, _ = blobs([10] * 300, sigma=0.9, seed=0, centroid_sim=0.3)
    n = len(faces)
    tracemalloc.start()
    try:
        recluster(faces)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 11 * n * n
