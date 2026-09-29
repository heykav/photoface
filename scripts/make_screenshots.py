#!/usr/bin/env python3
"""Regenerate the README screenshots from a SYNTHETIC demo library.

    QT_QPA_PLATFORM=offscreen python scripts/make_screenshots.py [--out docs/img]

Nothing here uses a real person's photo, a face model, or the network:

* "Photos" are procedurally drawn faceless avatar tokens (a coloured head with
  a geometric emblem) on abstract backgrounds, each stamped "SYNTHETIC DEMO".
* A stub engine (same idea as tests/stubs.py) stands in for YuNet + SFace. It
  returns the known avatar box and a synthetic 128-d embedding (identity
  centroid + noise), so the real code paths run for everything else:
  Analyzer.analyze_folder (scan, EXIF/GPS parsing, dHash, greedy clustering),
  Analyzer.recluster_all (average-linkage), the SQLite layer, and the real
  MainWindow / Sidebar / PhotoCard / LightboxDialog widgets.
* Everything is seeded and fixed-date, so a rerun on the same machine (same Qt
  and fonts) reproduces the same images. Text is rendered by the local Qt font
  stack, so other machines may differ in glyph shapes.

The output shows how the UI behaves; it says nothing about real-model accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance, ImageFont  # noqa: E402

import analyzer as analyzer_mod  # noqa: E402
from analyzer import Analyzer, DetectedFace  # noqa: E402
from database import Database  # noqa: E402

WATERMARK = "SYNTHETIC DEMO"
DIM = 128
SIGMA = 0.30

# name, base colour, emblem - fictional people, faceless avatar tokens
IDENTITIES = [
    ("Avery", (233, 96, 118), "ring"),
    ("Blair", (46, 176, 168), "triangle"),
    ("Casey", (240, 172, 48), "square"),
    ("Drew", (142, 104, 224), "diamond"),
    ("Emery", (76, 158, 236), "star"),
    ("Finley", (132, 190, 72), "plus"),
]
SCENES = [((38, 52, 96), (222, 122, 96)), ((24, 70, 84), (232, 200, 132)),
          ((70, 44, 96), (240, 150, 134)), ((30, 60, 44), (176, 214, 150)),
          ((52, 60, 110), (150, 190, 236)), ((88, 50, 60), (244, 196, 150))]


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _emblem(d: ImageDraw.ImageDraw, kind: str, cx: float, cy: float, s: float, fill):
    if kind == "ring":
        d.ellipse([cx - s, cy - s, cx + s, cy + s], outline=fill, width=max(3, int(s * .28)))
    elif kind == "triangle":
        d.polygon([(cx, cy - s), (cx + s, cy + s * .8), (cx - s, cy + s * .8)], fill=fill)
    elif kind == "square":
        d.rectangle([cx - s * .8, cy - s * .8, cx + s * .8, cy + s * .8], fill=fill)
    elif kind == "diamond":
        d.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=fill)
    elif kind == "star":
        pts = []
        for k in range(10):
            r = s if k % 2 == 0 else s * .45
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d.polygon(pts, fill=fill)
    else:  # plus
        t = s * .32
        d.rectangle([cx - s, cy - t, cx + s, cy + t], fill=fill)
        d.rectangle([cx - t, cy - s, cx + t, cy + s], fill=fill)


def draw_photo(size, people, scene_idx, rng: random.Random):
    """people: [(identity_index, cx_frac, cy_frac, r_frac)] -> (image, boxes)."""
    w, h = size
    top, bottom = SCENES[scene_idx % len(SCENES)]
    img = Image.new("RGB", size)
    px = ImageDraw.Draw(img)
    for y in range(h):
        px.line([(0, y), (w, y)], fill=_mix(top, bottom, y / (h - 1)))
    d = ImageDraw.Draw(img, "RGBA")
    sx, sy = rng.uniform(.15, .85) * w, rng.uniform(.1, .4) * h
    sr = rng.uniform(.07, .16) * min(w, h)
    d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 244, 214, 200))
    for k in range(3):
        base = h * (.62 + .1 * k)
        amp = h * rng.uniform(.04, .12)
        ph = rng.uniform(0, 6.28)
        pts = [(x, base + amp * math.sin(x / w * 5 + ph)) for x in range(0, w + 8, 8)]
        pts += [(w, h), (0, h)]
        d.polygon(pts, fill=_mix(top, (10, 14, 24), .25 + .2 * k) + (255,))
    boxes = []
    for ident, cxf, cyf, rf in people:
        _name, col, emblem = IDENTITIES[ident]
        cx, cy, r = cxf * w, cyf * h, rf * min(w, h)
        dark = _mix(col, (10, 10, 20), .45)
        d.ellipse([cx - r * 1.9, cy + r * 1.15, cx + r * 1.9, cy + r * 5.2], fill=dark + (255,))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_mix(col, (255, 255, 255), .12) + (255,))
        d.pieslice([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=dark + (255,))
        _emblem(d, emblem, cx, cy + r * .12, r * .42, (255, 255, 255, 235))
        boxes.append((cx - r, cy - r, 2 * r, 2 * r))
    try:
        font = ImageFont.load_default(size=max(12, w // 42))
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    d.text((w * .03, h - h * .045), WATERMARK, fill=(255, 255, 255, 190), font=font, anchor="ls")
    return img, boxes


def _gps_ifd(exif: Image.Exif, lat: float, lon: float) -> None:
    def dms(v):
        v = abs(v)
        d_, m_ = int(v), int((v - int(v)) * 60)
        return (float(d_), float(m_), round((v - d_ - m_ / 60) * 3600, 2))
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2] = ("N" if lat >= 0 else "S"), dms(lat)
    gps[3], gps[4] = ("E" if lon >= 0 else "W"), dms(lon)


class DemoEngine:
    """Stand-in for FaceEngine: looks the image up in a registry of the boxes
    and identities the generator drew, and emits synthetic embeddings."""

    def __init__(self):
        rng = np.random.RandomState(7)
        c = rng.normal(size=(len(IDENTITIES), DIM))
        self.centroids = c / np.linalg.norm(c, axis=1, keepdims=True)
        self.registry: dict[str, list[tuple[int, tuple]]] = {}
        self.noise = np.random.RandomState(11)

    def register(self, path: Path, faces):
        key = hashlib.md5(cv2.imread(str(path)).tobytes()).hexdigest()
        self.registry[key] = faces

    def detect_and_embed(self, image_bgr):
        out = []
        for ident, (x, y, w, h) in self.registry.get(
                hashlib.md5(image_bgr.tobytes()).hexdigest(), []):
            e = self.centroids[ident] + SIGMA * self.noise.normal(size=DIM) / math.sqrt(DIM)
            out.append(DetectedFace(x, y, w, h, (e / np.linalg.norm(e)).astype(np.float32), 0.99))
        return out


def build_library(folder: Path, engine: DemoEngine):
    """Write the synthetic photos; returns {filename: {"tags": [...]}}."""
    rng = random.Random(2025)
    folder.mkdir(parents=True, exist_ok=True)
    singles = [0] * 6 + [1] * 4 + [2] * 4 + [3] * 3 + [4] * 2 + [5] * 2
    specs = [[i] for i in singles] + [[0, 1], [2, 3], [0, 2], [1, 3], [4, 5]] + [[], [], [], []]
    rng.shuffle(specs)
    # every 4th photo plus the first group shot is geotagged (fictional trip)
    first_pair = next(i for i, ids in enumerate(specs, start=1) if len(ids) == 2)
    geotag = {n: (46.50 + 0.004 * n, 6.60 + 0.006 * n)
              for n in range(1, len(specs) + 1) if n % 4 == 2 or n == first_pair}
    meta: dict[str, dict] = {}
    for n, idents in enumerate(specs, start=1):
        landscape = len(idents) == 2 or (rng.random() < .25)
        size = (960, 720) if landscape else (720, 960)
        if len(idents) == 2:
            people = [(idents[0], .32, .34, .085), (idents[1], .68, .38, .085)]
        elif len(idents) == 1:
            people = [(idents[0], rng.uniform(.4, .6), rng.uniform(.3, .42),
                       rng.uniform(.10, .15))]
        else:
            people = []
        img, boxes = draw_photo(size, people, scene_idx=n + rng.randrange(6), rng=rng)
        exif = Image.Exif()
        exif[0x0132] = f"2025:0{1 + n % 9}:{1 + (n * 3) % 27:02d} {9 + n % 10}:{(n * 7) % 60:02d}:00"
        if n in geotag:
            _gps_ifd(exif, *geotag[n])
        path = folder / f"IMG_{n:04d}.jpg"
        img.save(path, "JPEG", quality=92, exif=exif.tobytes())
        engine.register(path, [(p[0], b) for p, b in zip(people, boxes)])
        tags = []
        if n in geotag:
            tags.append("trip")
        if len(idents) == 2:
            tags.append("together")
        meta[path.name] = {"tags": tags, "idents": idents}
        if n in (3, 8, 15):   # re-exports of the same shot -> possible duplicates
            for suffix, q, br in (("-edit", 60, 1.05),) + ((("-edit2", 45, .96),) if n == 15 else ()):
                dup = ImageEnhance.Brightness(img).enhance(br)
                dpath = folder / f"IMG_{n:04d}{suffix}.jpg"
                dup.save(dpath, "JPEG", quality=q, exif=exif.tobytes())
                engine.register(dpath, [(p[0], b) for p, b in zip(people, boxes)])
                meta[dpath.name] = {"tags": tags, "idents": idents}
    for p in folder.glob("*.jpg"):   # fixed mtimes: stable thumbnail-cache keys
        os.utime(p, (1_750_000_000, 1_750_000_000))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    tmp = Path(tempfile.mkdtemp(prefix="photoface-demo-"))
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp / "cfg"))
    app = QApplication.instance() or QApplication([])
    from gui import main_window, thumbnails
    from gui.lightbox import LightboxDialog
    from gui.theme import STYLESHEET
    app.setStyleSheet(STYLESHEET)
    cache = tmp / "cache"
    cache.mkdir()
    thumbnails.thumb_cache_dir = lambda: cache          # noqa: E731
    dbfile = tmp / "demo.db"
    main_window.db_path = lambda: dbfile                # noqa: E731

    # deterministic scan order (os.walk order is filesystem-dependent)
    real_iter = analyzer_mod.iter_image_files
    analyzer_mod.iter_image_files = lambda root: iter(sorted(real_iter(root)))

    engine = DemoEngine()
    photos = tmp / "photos"
    meta = build_library(photos, engine)

    db = Database(dbfile)
    an = Analyzer(db, engine)
    an.analyze_folder(photos)      # scan -> detect -> embed -> greedy clustering
    an.recluster_all()             # average-linkage recluster

    # name people after the identity of their faces; check the clustering
    by_person: dict[int, set[int]] = {}
    for photo in db.all_photos():
        idents = meta[Path(photo["path"]).name]["idents"]
        faces = sorted(db.faces_for_photo(photo["id"]), key=lambda f: f["x"])
        for f, ident in zip(faces, idents):
            by_person.setdefault(f["person_id"], set()).add(ident)
    assert len(by_person) == len(IDENTITIES) and all(len(v) == 1 for v in by_person.values()), \
        f"synthetic clusters not recovered: {by_person}"
    for pid, (ident,) in by_person.items():
        db.rename_person(pid, IDENTITIES[ident][0])
        # match the box colour to the avatar colour so the demo reads clearly
        db.conn.execute("UPDATE persons SET color = ? WHERE id = ?",
                        ("#%02x%02x%02x" % IDENTITIES[ident][1], pid))
    db.conn.commit()
    for photo in db.all_photos():
        for tag in meta[Path(photo["path"]).name]["tags"]:
            db.tag_photo(photo["id"], tag)
    dup_sizes = sorted(len(g) for g in db.duplicate_groups())
    assert dup_sizes == [2, 2, 3], f"unexpected duplicate groups {dup_sizes}"
    db.close()

    win = main_window.MainWindow()
    win.resize(1300, 830)
    win.show()

    def shot(widget, name):
        for _ in range(3):
            app.processEvents()
        widget.grab().save(str(args.out / name), "PNG")
        print("wrote", args.out / name)

    shot(win, "people-overview.png")

    people = sorted(win.db.all_persons(), key=lambda p: -p["face_count"])
    row = next(i for i in range(win.sidebar.people_list.count())
               if win.sidebar.people_list.item(i).data(0x0100) == people[0]["id"])
    win.sidebar.people_list.item(row).setSelected(True)   # real sidebar filter
    shot(win, "person-grid.png")
    win.sidebar.people_list.clearSelection()

    win.sidebar.duplicates_check.setChecked(True)
    shot(win, "duplicates.png")
    win.sidebar.duplicates_check.setChecked(False)

    items = [{"photo_row": p, "faces": win._faces_with_display(p["id"])}
             for p in win._current_photos()]
    idx = max((i for i, it in enumerate(items) if it["photo_row"]["exif_lat"] is not None),
              key=lambda i: (len(items[i]["faces"]), -i))
    dlg = LightboxDialog(win.db, items, idx, win)
    dlg.resize(1100, 760)
    dlg.show()
    app.processEvents()
    dlg._render_current()          # re-fit the photo to the shown dialog size
    shot(dlg, "lightbox-exif.png")
    dlg.close()
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
