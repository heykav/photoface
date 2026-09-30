"""Deterministic synthetic benchmarks behind the README's measured numbers.

Nothing here touches a real photo, a real face or an ONNX model:

* clustering(): identities are Gaussian blobs of 128-d unit vectors
  (tests/synth.py), scored with purity / completeness for both passes the
  app runs - `greedy_assign` during analysis (fed in a shuffled order, since
  a real folder is not sorted by person) and the final `recluster`.
* duplicates(): procedurally drawn "scenes" (smooth colour blobs plus sensor
  noise), each transformed the way copies of a photo usually differ, and
  hashed with the app's own `compute_phash`.

`scripts/make_benchmarks.py` writes the results to docs/benchmarks.json, the
README tables and docs/img/clustering-benchmark-*.svg;
tests/test_benchmarks.py re-runs both and fails if any of those artefacts no
longer match the code.
"""
from __future__ import annotations

import io
import random
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance

from analyzer import compute_phash
from clustering import DEFAULT_THRESHOLD, greedy_assign, recluster
from synth import blobs, purity_completeness

SEEDS = range(5)
EVEN = (30, 30, 30)
# name, identity sizes, sigma (within-identity noise), centroid_sim (overlap)
CLUSTER_SCENARIOS = [
    ("separated", EVEN, 0.5, 0.0),
    ("mild overlap", EVEN, 0.9, 0.3),
    ("unbalanced sizes", (60, 20, 5, 2, 1, 1, 1), 0.9, 0.3),
    ("moderate overlap", EVEN, 1.2, 0.5),
    ("near-coincident", EVEN, 1.0, 0.8),
    ("very noisy", EVEN, 1.5, 0.6),
]
GREEDY_ORDERS = 4  # shuffled input orders per seed for the order-dependent greedy pass


def _summary(runs: List[tuple]) -> Dict[str, float]:
    p = [r[0] for r in runs]
    c = [r[1] for r in runs]
    k = [r[2] for r in runs]
    return {"purity": round(float(np.mean(p)), 3), "purity_min": round(float(min(p)), 3),
            "completeness": round(float(np.mean(c)), 3),
            "completeness_min": round(float(min(c)), 3),
            "clusters": round(float(np.mean(k)), 1)}


def clustering(threshold: float = DEFAULT_THRESHOLD) -> List[dict]:
    rows = []
    for name, sizes, sigma, overlap in CLUSTER_SCENARIOS:
        rec, gre = [], []
        for seed in SEEDS:
            faces, truth = blobs(sizes, sigma=sigma, seed=seed, centroid_sim=overlap)
            a = recluster(faces, threshold)
            rec.append((*purity_completeness(a, truth), len(set(a.values()))))
            for k in range(GREEDY_ORDERS):
                order = faces[:]
                random.Random(seed * 100 + k).shuffle(order)
                g = greedy_assign(order, [], threshold)
                gre.append((*purity_completeness(g, truth), len(set(g.values()))))
        rows.append({"scenario": name, "identities": len(sizes), "faces": sum(sizes),
                     "sigma": sigma, "overlap": overlap,
                     "recluster": _summary(rec), "greedy": _summary(gre)})
    return rows


# ---------------------------------------------------------------------- #
# Duplicate detection
# ---------------------------------------------------------------------- #
N_SCENES = 60
HAMMING_LIMITS = (2, 4, 6, 8)


def scene(seed: int, w: int = 320, h: int = 240) -> Image.Image:
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3)) + rng.uniform(0, 255, 3)
    for _ in range(rng.randint(3, 8)):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(20, 160)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        img += g[..., None] * rng.uniform(-150, 150, 3)
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _encode(img: Image.Image, fmt: str = "BMP", **kw) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    buf.seek(0)
    return buf


def _crop(frac: float) -> Callable[[Image.Image], Image.Image]:
    def f(im: Image.Image) -> Image.Image:
        dx, dy = int(im.width * frac), int(im.height * frac)
        return im.crop((dx, dy, im.width - dx, im.height - dy))
    return f


# name -> (image transform, encoder)
TRANSFORMS: Dict[str, tuple] = {
    "resize to 50%": (lambda im: im.resize((im.width // 2, im.height // 2), Image.LANCZOS), "BMP"),
    "resize to 200%": (lambda im: im.resize((im.width * 2, im.height * 2), Image.BICUBIC), "BMP"),
    "JPEG quality 90": (lambda im: im, 90),
    "JPEG quality 70": (lambda im: im, 70),
    "JPEG quality 40": (lambda im: im, 40),
    "crop 2% per side": (_crop(0.02), "BMP"),
    "crop 5% per side": (_crop(0.05), "BMP"),
    "crop 10% per side": (_crop(0.10), "BMP"),
    "brightness -20%": (lambda im: ImageEnhance.Brightness(im).enhance(0.8), "BMP"),
    "brightness +20%": (lambda im: ImageEnhance.Brightness(im).enhance(1.2), "BMP"),
    "contrast +30%": (lambda im: ImageEnhance.Contrast(im).enhance(1.3), "BMP"),
    "mirrored": (lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), "BMP"),
}


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _rates(dists: List[int]) -> dict:
    d = np.array(dists)
    return {"median": float(np.median(d)), "max": int(d.max()),
            **{f"within_{k}": round(float(np.mean(d <= k)), 3) for k in HAMMING_LIMITS}}


def duplicates() -> dict:
    base, per = [], {name: [] for name in TRANSFORMS}
    for s in range(N_SCENES):
        im = scene(s)
        h0 = compute_phash(_encode(im))
        base.append(h0)
        for name, (fn, enc) in TRANSFORMS.items():
            buf = _encode(im, "JPEG", quality=enc) if isinstance(enc, int) else _encode(fn(im))
            per[name].append(_hamming(h0, compute_phash(buf)))
    unrelated = [_hamming(base[a], base[b])
                 for a in range(N_SCENES) for b in range(a + 1, N_SCENES)]
    return {"scenes": N_SCENES, "pairs": len(unrelated),
            "transforms": {k: _rates(v) for k, v in per.items()},
            "unrelated": _rates(unrelated)}


def run() -> dict:
    return {"threshold": DEFAULT_THRESHOLD, "seeds": len(SEEDS),
            "greedy_orders": GREEDY_ORDERS, "clustering": clustering(),
            "duplicates": duplicates()}
