"""Folder scanning, face detection/embedding, and tying clustering to the DB.

Detection: OpenCV YuNet (`cv2.FaceDetectorYN`), run on a downscaled copy of
each photo for speed, with the resulting boxes/landmarks scaled back to full
resolution before alignment/embedding - so stored face boxes stay accurate
even on very large photos.

Embedding: OpenCV SFace (`cv2.FaceRecognizerSF`), 128-d L2-normalizable
feature vector via `alignCrop` + `feature` on the full-resolution image.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from PIL import Image

from clustering import (FaceRecord, DEFAULT_THRESHOLD, greedy_assign, recluster,
                        stabilize_assignment)
from database import Database
from exif_utils import extract_exif, upright  # noqa: F401  (extract_exif re-exported)
from model_files import require_installed
from paths import models_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
_DETECT_MAX_DIM = 1600

# a fixed, readable palette; persons cycle through it and keep whichever
# color they're assigned for as long as they exist
_AUTO_NAME = re.compile(r"^Persona \d+$")

PERSON_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
]


def _raise(err: OSError) -> None:
    raise err


def iter_image_files(root: Path):
    """Yield every image under `root`. Unreadable directories raise instead
    of being skipped: a silently-skipped folder (e.g. an unmounted drive)
    would otherwise look like 'those photos were deleted'."""
    for dirpath, _dirnames, filenames in os.walk(root, onerror=_raise):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / name


def compute_phash(path: Path) -> Optional[str]:
    """Difference hash (dHash) of the photo as displayed (EXIF orientation
    applied): resize to 9x8 grayscale, compare each pixel to its right
    neighbor -> 64 bits -> 16 hex chars. Near-duplicates (resized, JPEG
    recompressed, slightly cropped or darkened) usually land a few bits
    apart and unrelated photos about 32 apart; see tests/dupbench.py and
    the README for measured rates on synthetic images. A mirrored copy is
    NOT a near-duplicate under this hash. Returns None if unreadable."""
    try:
        with Image.open(path) as raw:
            img = upright(raw).convert("L").resize((9, 8), Image.LANCZOS)
    except Exception:  # noqa: BLE001
        return None
    pixels = img.tobytes()  # mode "L": one byte per pixel
    bits = 0
    for row in range(8):
        for col in range(8):
            bits <<= 1
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            if left > right:
                bits |= 1
    return f"{bits:016x}"


@dataclass
class DetectedFace:
    x: float
    y: float
    w: float
    h: float
    embedding: np.ndarray
    confidence: float


class FaceEngine:
    """Owns the YuNet detector + SFace recognizer. Not thread-safe - use one
    instance per worker thread."""

    def __init__(self, score_threshold: float = 0.7):
        yunet = models_dir() / "face_detection_yunet_2023mar.onnx"
        sface = models_dir() / "face_recognition_sface_2021dec.onnx"
        require_installed()  # raises ModelError (a FileNotFoundError) with next steps
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (0, 0), score_threshold=score_threshold
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect_and_embed(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        h, w = image_bgr.shape[:2]
        scale = min(1.0, _DETECT_MAX_DIM / max(h, w)) if max(h, w) > 0 else 1.0
        if scale < 1.0:
            small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        else:
            small = image_bgr
        sh, sw = small.shape[:2]
        self._detector.setInputSize((sw, sh))
        _, faces = self._detector.detect(small)
        if faces is None:
            return []

        out: List[DetectedFace] = []
        inv = 1.0 / scale
        for row in faces:
            scaled = row.copy()
            scaled[:14] *= inv  # bbox (4) + 5 landmarks (10) all scale linearly
            aligned = self._recognizer.alignCrop(image_bgr, scaled)
            feature = self._recognizer.feature(aligned)
            out.append(DetectedFace(
                x=float(scaled[0]), y=float(scaled[1]),
                w=float(scaled[2]), h=float(scaled[3]),
                embedding=feature.flatten().astype(np.float32),
                confidence=float(row[14]),
            ))
        return out


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_pinned(old_faces, detections, min_iou: float = 0.5) -> dict:
    """When a photo is re-analyzed (its mtime/size changed), keep the user's
    manual corrections: each previously *pinned* face is matched to the
    re-detected face it overlaps most (IoU >= min_iou, one-to-one). Returns
    {detection_index: old_face_row}. Unmatched pinned faces are dropped with
    the old detections - the image changed under them."""
    pinned = [f for f in old_faces if f["pinned"] and f["person_id"] is not None]
    pairs = []
    for f in pinned:
        for k, d in enumerate(detections):
            iou = _iou((f["x"], f["y"], f["w"], f["h"]), (d.x, d.y, d.w, d.h))
            if iou >= min_iou:
                pairs.append((-iou, f["id"], k, f))
    pairs.sort(key=lambda t: t[:3])
    out, used_old = {}, set()
    for _neg, fid, k, f in pairs:
        if k not in out and fid not in used_old:
            out[k] = f
            used_old.add(fid)
    return out


ProgressCB = Callable[[int, int, str], None]


class Analyzer:
    def __init__(self, db: Database, engine: Optional[FaceEngine] = None,
                threshold: float = DEFAULT_THRESHOLD):
        self.db = db
        self._engine = engine
        self.threshold = threshold

    @property
    def engine(self) -> FaceEngine:
        """Created on first use so clustering-only work (recluster_all) does
        not need the ONNX models."""
        if self._engine is None:
            self._engine = FaceEngine()
        return self._engine

    def _next_color(self) -> str:
        used = {p["color"] for p in self.db.all_persons()}
        for c in PERSON_COLORS:
            if c not in used:
                return c
        # palette exhausted - derive a further color deterministically
        idx = len(used)
        hue = (idx * 47) % 360
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.65, 0.85)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    def analyze_folder(self, folder: Path, progress: Optional[ProgressCB] = None,
                       cancel_check: Optional[Callable[[], bool]] = None) -> None:
        if not Path(folder).is_dir():
            # never treat a missing/unmounted folder as "every photo was deleted"
            raise FileNotFoundError(f"Photo folder not found: {folder}")
        engine = self.engine  # fail fast (missing models) before touching anything
        files = list(iter_image_files(folder))
        total = len(files)
        self.db.delete_photos_not_in(str(f) for f in files)

        touched_new_face_ids: List[int] = []

        for i, path in enumerate(files):
            if cancel_check and cancel_check():
                break
            if progress:
                progress(i + 1, total, str(path))
            try:
                st = path.stat()
            except OSError:
                continue
            existing = self.db.get_photo_by_path(str(path))
            if existing is not None and existing["mtime"] == st.st_mtime and \
               existing["size"] == st.st_size and existing["analyzed_at"] is not None:
                # unchanged - skip re-analysis, but fill in a hash that a
                # schema migration dropped (or an older version never made)
                if existing["phash"] is None:
                    phash = compute_phash(path)
                    if phash is not None:
                        self.db.set_photo_phash(existing["id"], phash)
                continue

            image = cv2.imread(str(path))
            if image is None:
                continue
            h, w = image.shape[:2]
            date, lat, lon = extract_exif(path)
            phash = compute_phash(path)

            try:
                detections = engine.detect_and_embed(image)
            except Exception:  # noqa: BLE001 - one bad image shouldn't abort the run
                # not recorded, so it is retried next run instead of being
                # marked "analyzed, no faces" forever
                continue

            # One transaction per photo: a crash can never leave a photo
            # marked analyzed (and so skipped next time) without its faces.
            with self.db.transaction():
                old_faces = self.db.faces_for_photo(existing["id"]) if existing else []
                photo_id = self.db.upsert_photo(
                    str(path), st.st_mtime, st.st_size, w, h, date, lat, lon,
                    time.time(), phash=phash,
                )
                if existing is not None:
                    self.db.delete_photo_faces(photo_id)
                inherited = _match_pinned(old_faces, detections)
                for k, det in enumerate(detections):
                    prev = inherited.get(k)
                    face_id = self.db.add_face(
                        photo_id, det.x, det.y, det.w, det.h, det.embedding,
                        det.confidence,
                        person_id=prev["person_id"] if prev else None,
                        pinned=bool(prev),
                    )
                    if prev is None:
                        touched_new_face_ids.append(face_id)

        if touched_new_face_ids:
            self._greedy_pass(touched_new_face_ids)

    def _row_to_record(self, row) -> FaceRecord:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        return FaceRecord(row["id"], emb, row["person_id"], bool(row["pinned"]))

    def _greedy_pass(self, new_face_ids: List[int]) -> None:
        all_faces = [self._row_to_record(r) for r in self.db.all_faces()]
        by_id = {f.face_id: f for f in all_faces}
        new_faces = [by_id[i] for i in new_face_ids if i in by_id]
        new_set = set(new_face_ids)
        existing = [f for f in all_faces if f.face_id not in new_set]

        assignments = greedy_assign(new_faces, existing, self.threshold)
        tentative_to_person: dict[int, int] = {}
        with self.db.transaction():
            self._apply_greedy(assignments, tentative_to_person)

    def _apply_greedy(self, assignments, tentative_to_person) -> None:
        for face_id, pid in assignments.items():
            if pid == -1 or pid <= -1000000:
                if pid not in tentative_to_person:
                    name = f"Persona {len(self.db.all_persons()) + 1}"
                    tentative_to_person[pid] = self.db.create_person(name, self._next_color())
                real_pid = tentative_to_person[pid]
            else:
                real_pid = pid
            self.db.set_face_person(face_id, real_pid, pinned=False)

    def recluster_all(self) -> None:
        """Full re-clustering pass over every face - call after a batch of
        analysis, or on demand from a 'Recluster now' menu action.

        Runs as ONE database transaction: a crash or error part-way leaves
        the previous assignment untouched. Unpinned faces keep their existing
        person when their cluster overlaps one (so a renamed person stays
        renamed and running this twice changes nothing); auto-named people
        left with no faces are removed."""
        with self.db.transaction():
            records = [self._row_to_record(r) for r in self.db.all_faces()]
            if not records:
                return
            by_id = {r.face_id: r for r in records}
            assignment = stabilize_assignment(records, recluster(records, self.threshold))

            # Every face sharing the same negative placeholder id is one
            # unanchored cluster - map each such id to one new person row.
            placeholder_to_person: dict[int, int] = {}
            for face_id in sorted(assignment):
                pid = assignment[face_id]
                if pid < 0:
                    if pid not in placeholder_to_person:
                        name = f"Persona {len(self.db.all_persons()) + 1}"
                        placeholder_to_person[pid] = self.db.create_person(
                            name, self._next_color()
                        )
                    pid = placeholder_to_person[pid]
                rec = by_id[face_id]
                if rec.pinned or rec.person_id == pid:
                    continue  # never move a pinned face; skip no-op writes
                self.db.set_face_person(face_id, pid, pinned=False)

            for p in self.db.all_persons():
                if p["face_count"] == 0 and _AUTO_NAME.match(p["name"]):
                    self.db.delete_person(p["id"])
