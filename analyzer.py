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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

from clustering import FaceRecord, DEFAULT_THRESHOLD, greedy_assign, recluster
from database import Database
from paths import models_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
_DETECT_MAX_DIM = 1600

# a fixed, readable palette; persons cycle through it and keep whichever
# color they're assigned for as long as they exist
PERSON_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
]


def iter_image_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / name


def _dms_to_decimal(dms, ref: str) -> Optional[float]:
    try:
        deg, minutes, seconds = dms
        value = float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if ref in ("S", "W"):
        value = -value
    return value


def extract_exif(path: Path) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Returns (capture_date_iso_or_None, lat_or_None, lon_or_None)."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None, None, None
    except Exception:  # noqa: BLE001
        return None, None, None

    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    date = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if isinstance(date, str):
        date = date.replace(":", "-", 2).replace(" ", "T", 1)

    lat = lon = None
    try:
        gps_ifd = exif.get_ifd(0x8825)  # GPS IFD
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        if "GPSLatitude" in gps and "GPSLongitude" in gps:
            lat = _dms_to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
            lon = _dms_to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    except Exception:  # noqa: BLE001
        pass
    return date, lat, lon


def compute_phash(path: Path) -> Optional[str]:
    """Difference hash (dHash): resize to 9x8 grayscale, compare each pixel to
    its right neighbor -> 64 bits -> 16 hex chars. Near-duplicate photos
    (recompressed, lightly cropped/resized, minor edits) land a small Hamming
    distance apart; unrelated photos land far apart. Cheap enough to compute
    for every photo during analysis."""
    try:
        img = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    except Exception:  # noqa: BLE001
        return None
    pixels = list(img.getdata())
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
        if not yunet.exists() or not sface.exists():
            raise FileNotFoundError(
                "Face models not found - run `python scripts/download_models.py` first."
            )
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


ProgressCB = Callable[[int, int, str], None]


class Analyzer:
    def __init__(self, db: Database, engine: Optional[FaceEngine] = None,
                threshold: float = DEFAULT_THRESHOLD):
        self.db = db
        self.engine = engine or FaceEngine()
        self.threshold = threshold

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
                continue  # unchanged - skip re-analysis

            image = cv2.imread(str(path))
            if image is None:
                continue
            h, w = image.shape[:2]
            date, lat, lon = extract_exif(path)
            phash = compute_phash(path)

            photo_id = self.db.upsert_photo(
                str(path), st.st_mtime, st.st_size, w, h, date, lat, lon,
                time.time(), phash=phash,
            )
            if existing is not None:
                self.db.delete_photo_faces(photo_id)

            try:
                detections = self.engine.detect_and_embed(image)
            except Exception:  # noqa: BLE001 - a single bad image shouldn't abort the run
                detections = []

            for det in detections:
                face_id = self.db.add_face(
                    photo_id, det.x, det.y, det.w, det.h, det.embedding, det.confidence
                )
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
        existing = [f for f in all_faces if f.face_id not in set(new_face_ids)]

        assignments = greedy_assign(new_faces, existing, self.threshold)
        tentative_to_person: dict[int, int] = {}
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
        analysis, or on demand from a 'Recluster' menu action."""
        records = [self._row_to_record(r) for r in self.db.all_faces()]
        if not records:
            return
        by_id = {r.face_id: r for r in records}
        assignment = recluster(records, self.threshold)

        # Every face sharing the same negative placeholder id is one
        # unanchored cluster - map each such id to one new person row.
        placeholder_to_person: dict[int, int] = {}
        for face_id, pid in assignment.items():
            if pid < 0:
                if pid not in placeholder_to_person:
                    name = f"Persona {len(self.db.all_persons()) + 1}"
                    placeholder_to_person[pid] = self.db.create_person(
                        name, self._next_color()
                    )
                real_pid = placeholder_to_person[pid]
            else:
                real_pid = pid

            row = by_id.get(face_id)
            if row is not None and row.pinned:
                continue  # never move a pinned face
            self.db.set_face_person(face_id, real_pid, pinned=False)
