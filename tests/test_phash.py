import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from analyzer import compute_phash


def _save(tmp: Path, name: str, arr: np.ndarray) -> Path:
    path = tmp / name
    Image.fromarray(arr).save(path)
    return path


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class TestComputePhash:
    def test_same_image_same_hash(self, tmp_path=Path(tempfile.mkdtemp())):
        rng = np.random.RandomState(1)
        arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        p1 = _save(tmp_path, "a.png", arr)
        p2 = _save(tmp_path, "b.png", arr)
        assert compute_phash(p1) == compute_phash(p2)

    def test_hash_is_16_hex_chars(self, tmp_path=Path(tempfile.mkdtemp())):
        arr = np.random.RandomState(2).randint(0, 255, (64, 64, 3), dtype=np.uint8)
        p = _save(tmp_path, "a.png", arr)
        h = compute_phash(p)
        assert h is not None
        assert len(h) == 16
        int(h, 16)  # must parse as hex

    def test_very_different_images_far_apart(self, tmp_path=Path(tempfile.mkdtemp())):
        # This dHash only compares horizontally-adjacent pixels, so it's
        # deliberately tested here with two unrelated photos (independent
        # random noise) rather than a crafted pattern - a pure vertical
        # gradient, for instance, is a genuine blind spot for a
        # horizontal-only dHash (every horizontally-adjacent pair is equal),
        # which would be the wrong thing for this test to assert against.
        # Two unrelated real-world photos reliably land far apart (expected
        # Hamming distance ~32 of 64 bits for independent images).
        img1 = np.random.RandomState(10).randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.RandomState(20).randint(0, 255, (64, 64, 3), dtype=np.uint8)
        p1 = _save(tmp_path, "img1.png", img1)
        p2 = _save(tmp_path, "img2.png", img2)
        h1, h2 = compute_phash(p1), compute_phash(p2)
        assert _hamming(h1, h2) > 20

    def test_slightly_modified_image_close_hash(self, tmp_path=Path(tempfile.mkdtemp())):
        rng = np.random.RandomState(3)
        base = rng.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        modified = base.copy()
        modified[0:3, 0:3] = 255  # a tiny localized change, like a small crop/edit
        p1 = _save(tmp_path, "base.png", base)
        p2 = _save(tmp_path, "modified.png", modified)
        h1, h2 = compute_phash(p1), compute_phash(p2)
        assert _hamming(h1, h2) <= 6

    def test_missing_file_returns_none(self):
        assert compute_phash(Path("/nonexistent/path/x.jpg")) is None
