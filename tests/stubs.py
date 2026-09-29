"""Test doubles: a fake face engine and synthetic photos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

from analyzer import DetectedFace
from synth import DIM, unit


class StubEngine:
    """Deterministic 'model': the image's red channel picks the identity."""

    def __init__(self):
        rng = np.random.RandomState(0)
        self.centroids = [unit(rng.normal(size=DIM)) for _ in range(256)]
        self.calls = 0
        self.fail_on_call = None

    def detect_and_embed(self, image_bgr):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("detector blew up")
        ident = int(image_bgr[..., 2].mean()) // 32
        rng = np.random.RandomState(self.calls)
        emb = unit(self.centroids[ident] + 0.3 * rng.normal(size=DIM) / np.sqrt(DIM))
        return [DetectedFace(2.0, 2.0, 10.0, 10.0, emb.astype(np.float32), 0.99)]


def make_photos(folder: Path, reds=(10, 10, 10, 200, 200, 90)):
    folder.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(reds):
        Image.new("RGB", (32, 32), (r, 50 + i, 60)).save(folder / f"p{i}.jpg", "JPEG",
                                                          quality=95)
    return sorted(folder.glob("*.jpg"))
