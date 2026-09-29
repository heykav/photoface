"""Property and adversarial tests for clustering.py using synthetic
embeddings (no ONNX models involved)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from clustering import FaceRecord, greedy_assign, recluster, stabilize_assignment
from synth import DIM, blobs, partition, purity_completeness


def _apply(faces, assign):
    """Simulate what the DB does: unpinned faces take the result."""
    out = []
    for f in faces:
        out.append(FaceRecord(f.face_id, f.embedding,
                              f.person_id if f.pinned else assign[f.face_id], f.pinned))
    return out


class TestQualityOnSyntheticData:
    def test_well_separated_blobs_are_recovered_exactly(self):
        faces, truth = blobs([20, 15, 10, 5], sigma=0.5, seed=1)
        result = recluster(faces, threshold=0.363)
        assert partition(result) == partition(truth)

    def test_overlapping_blobs_report(self, capsys):
        # Honest measurement, seed 3, 3x30 faces, threshold 0.363:
        #   sigma=1.2 centroid_sim=0.5 -> purity 1.00, completeness 0.98
        #   sigma=1.0 centroid_sim=0.8 -> purity 0.33, completeness 1.00 (all
        #     three identities collapse into one: no threshold can fix data
        #     whose identities are that close)
        #   sigma=1.5 centroid_sim=0.6 -> purity 1.00, completeness 0.34 (very
        #     noisy identities shatter into many small clusters)
        for sigma, cs, min_purity, min_compl in [(1.2, 0.5, 0.95, 0.9),
                                                 (1.0, 0.8, 0.0, 0.95),
                                                 (1.5, 0.6, 0.95, 0.0)]:
            faces, truth = blobs([30, 30, 30], sigma=sigma, seed=3, centroid_sim=cs)
            purity, completeness = purity_completeness(recluster(faces, 0.363), truth)
            print(f"sigma={sigma} overlap={cs} purity={purity:.3f} "
                  f"completeness={completeness:.3f}")
            assert purity >= min_purity and completeness >= min_compl

    def test_matches_naive_reference_implementation(self):
        """The optimized Lance-Williams version must agree with a literal
        O(n^3) average-linkage (mean of all pairwise distances)."""
        def naive(faces, thr):
            clusters = [[i] for i in range(len(faces))]
            unit = np.stack([f.embedding / np.linalg.norm(f.embedding) for f in faces])
            sim = unit @ unit.T
            anchor = lambda c: {faces[i].person_id for i in c if faces[i].pinned}  # noqa: E731
            while True:
                best, pair = -2.0, None
                for a in range(len(clusters)):
                    for b in range(a + 1, len(clusters)):
                        if len(anchor(clusters[a]) | anchor(clusters[b])) > 1:
                            continue
                        s = np.mean([sim[i, j] for i in clusters[a] for j in clusters[b]])
                        if s > best + 1e-12:
                            best, pair = s, (a, b)
                if pair is None or best < thr:
                    return {faces[i].face_id: k for k, c in enumerate(clusters) for i in c}
                a, b = pair
                clusters[a] += clusters.pop(b)

        for seed in range(6):
            faces, _ = blobs([6, 5, 5, 4], sigma=0.9, seed=seed, centroid_sim=0.4)
            faces[0].person_id, faces[0].pinned = 1, True
            faces[7].person_id, faces[7].pinned = 2, True
            for thr in (0.2, 0.363, 0.5):
                assert partition(recluster(faces, thr)) == partition(naive(faces, thr))

    @pytest.mark.parametrize("seed", range(5))
    def test_noise_sweep_never_crashes_and_purity_high_when_separable(self, seed):
        faces, truth = blobs([12] * 6, sigma=0.6, seed=seed)
        purity, completeness = purity_completeness(recluster(faces, 0.363), truth)
        assert purity >= 0.99 and completeness >= 0.99

    def test_higher_threshold_never_merges_more(self):
        faces, _ = blobs([15, 15, 15], sigma=0.9, seed=7, centroid_sim=0.3)
        counts = [len(set(recluster(faces, t).values())) for t in (0.2, 0.4, 0.6, 0.8)]
        assert counts == sorted(counts)


class TestEdgeCases:
    def test_empty_library(self):
        assert recluster([]) == {}
        assert greedy_assign([], []) == {}
        assert stabilize_assignment([], {}) == {}

    def test_single_member_clusters_stay_singletons(self):
        faces, _ = blobs([1, 1, 1, 1], sigma=0.1, seed=2)
        result = recluster(faces, 0.363)
        assert len(set(result.values())) == 4

    def test_zero_and_nan_embeddings_do_not_crash(self):
        faces = [FaceRecord(1, np.zeros(DIM, np.float32)),
                 FaceRecord(2, np.full(DIM, np.nan, np.float32)),
                 FaceRecord(3, np.ones(DIM, np.float32))]
        result = recluster(faces, 0.363)
        assert set(result) == {1, 2, 3}
        assert result[1] != result[3] and result[2] != result[3]

    def test_identical_embeddings_merge(self):
        e = np.ones(DIM, np.float32)
        result = recluster([FaceRecord(i, e) for i in range(1, 6)], 0.9)
        assert len(set(result.values())) == 1


class TestDeterminism:
    def test_same_input_same_output(self):
        faces, _ = blobs([10, 10, 10], sigma=0.9, seed=4, centroid_sim=0.4)
        assert recluster(faces, 0.363) == recluster(faces, 0.363)

    def test_input_order_does_not_change_partition(self):
        faces, _ = blobs([12, 12, 12], sigma=0.5, seed=5)
        base = partition(recluster(faces, 0.363))
        rng = random.Random(0)
        for _ in range(5):
            shuffled = faces[:]
            rng.shuffle(shuffled)
            assert partition(recluster(shuffled, 0.363)) == base

    def test_greedy_deterministic(self):
        faces, _ = blobs([8, 8], sigma=0.5, seed=6)
        assert greedy_assign(faces, []) == greedy_assign(faces, [])


class TestPinning:
    def _scenario(self, seed=8):
        faces, truth = blobs([10, 10, 10], sigma=0.8, seed=seed, centroid_sim=0.4)
        # pin a few faces, deliberately onto the WRONG person for some
        for f in faces[:3]:
            f.person_id, f.pinned = 100, True
        for f in faces[10:12]:
            f.person_id, f.pinned = 200, True
        faces[15].person_id, faces[15].pinned = 100, True  # cross-identity pin
        return faces

    def test_pinned_faces_never_change_person(self):
        faces = self._scenario()
        for thr in (-0.5, 0.0, 0.363, 0.7, 0.99):
            result = recluster(faces, thr)
            for f in faces:
                if f.pinned:
                    assert result[f.face_id] == f.person_id

    def test_different_pinned_persons_never_share_a_cluster(self):
        faces = self._scenario()
        result = recluster(faces, -1.0)  # threshold so low everything wants to merge
        assert result[faces[0].face_id] == 100
        assert result[faces[10].face_id] == 200
        assert 100 in result.values() and 200 in result.values()
        # a cluster id maps to exactly one person: 100 and 200 groups are disjoint
        p100 = {f for f, p in result.items() if p == 100}
        p200 = {f for f, p in result.items() if p == 200}
        assert p100.isdisjoint(p200)

    def test_pinned_invariant_survives_repeated_reclusters(self):
        faces = self._scenario()
        for _ in range(3):
            assign = stabilize_assignment(faces, recluster(faces, 0.363))
            faces = _apply(faces, {k: (v if v >= 0 else v) for k, v in assign.items()})
            for f in faces:
                if f.pinned:
                    assert assign[f.face_id] == f.person_id

    def test_pinned_without_person_is_treated_as_unanchored(self):
        f = FaceRecord(1, np.ones(DIM, np.float32), person_id=None, pinned=True)
        assert recluster([f])[1] < 0


class TestIdempotence:
    def test_recluster_is_idempotent_via_stabilize(self):
        faces, _ = blobs([10, 10, 10, 4], sigma=0.8, seed=9, centroid_sim=0.3)
        first = stabilize_assignment(faces, recluster(faces, 0.363))
        # give placeholders real ids (as the DB would), then run again
        ids = {}
        real = {}
        for fid, cid in first.items():
            if cid < 0:
                ids.setdefault(cid, 1000 + len(ids))
                cid = ids[cid]
            real[fid] = cid
        again_faces = _apply(faces, real)
        second = stabilize_assignment(again_faces, recluster(again_faces, 0.363))
        assert second == real

    def test_stabilize_reuses_existing_person_ids(self):
        faces, _ = blobs([6, 6], sigma=0.4, seed=10)
        for f in faces[:6]:
            f.person_id = 7
        for f in faces[6:]:
            f.person_id = 8
        out = stabilize_assignment(faces, recluster(faces, 0.363))
        assert {out[f.face_id] for f in faces[:6]} == {7}
        assert {out[f.face_id] for f in faces[6:]} == {8}

    def test_stabilize_never_gives_one_person_to_two_clusters(self):
        # a person whose faces split into two clusters: bigger cluster keeps id
        faces, _ = blobs([8, 3], sigma=0.4, seed=11)
        for f in faces:
            f.person_id = 5
        out = stabilize_assignment(faces, recluster(faces, 0.363))
        assert {out[f.face_id] for f in faces[:8]} == {5}
        assert out[faces[8].face_id] < 0
