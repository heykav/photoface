"""Disk-cached thumbnails, so a large library re-opens instantly instead of
re-decoding every full-size photo on every gallery paint."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap

from paths import thumb_cache_dir

MAX_DIM = 480


def _cache_key(path: str, mtime: float) -> str:
    h = hashlib.sha1(f"{path}:{mtime}:{MAX_DIM}".encode()).hexdigest()
    return h


def get_thumbnail(path: str, mtime: float) -> Optional[QPixmap]:
    key = _cache_key(path, mtime)
    cache_file = thumb_cache_dir() / f"{key}.jpg"
    if cache_file.exists():
        pix = QPixmap(str(cache_file))
        if not pix.isNull():
            return pix

    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        img.save(cache_file, "JPEG", quality=85)
    except Exception:  # noqa: BLE001
        return None

    return QPixmap(str(cache_file))


def scale_bbox_to_thumb(x: float, y: float, w: float, h: float,
                       orig_w: int, orig_h: int,
                       thumb_w: int, thumb_h: int) -> tuple[float, float, float, float]:
    sx = thumb_w / orig_w if orig_w else 1.0
    sy = thumb_h / orig_h if orig_h else 1.0
    return x * sx, y * sy, w * sx, h * sy
