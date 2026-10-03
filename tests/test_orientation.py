"""EXIF orientation: face boxes come from cv2.imread (which applies the
Orientation tag), so everything that draws or hashes a photo must use the
same upright pixels. Synthetic images only."""
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pytest
from PIL import Image

from analyzer import Analyzer, compute_phash
from database import SCHEMA_VERSION, Database
from exif_utils import upright
from stubs import StubEngine, make_photos

QUAD = [(230, 20, 20), (20, 230, 20), (20, 20, 230), (230, 230, 20)]  # RGB TL TR BL BR


def _quadrants(w=96, h=64) -> Image.Image:
    arr = np.zeros((h, w, 3), np.uint8)
    arr[: h // 2, : w // 2] = QUAD[0]
    arr[: h // 2, w // 2:] = QUAD[1]
    arr[h // 2:, : w // 2] = QUAD[2]
    arr[h // 2:, w // 2:] = QUAD[3]
    return Image.fromarray(arr)


def _tagged(path: Path, orientation: int) -> Path:
    img = _quadrants()
    exif = img.getexif()
    exif[0x0112] = orientation
    img.save(path, "JPEG", quality=98, exif=exif.tobytes())
    return path


def _corner_colours(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    pts = [(h // 4, w // 4), (h // 4, 3 * w // 4), (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4)]
    return [tuple(int(c) for c in rgb[y, x]) for y, x in pts]


def _close(a, b, tol=24):
    return all(all(abs(x - y) <= tol for x, y in zip(p, q, strict=True)) for p, q in zip(a, b, strict=True))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_upright_matches_cv2_imread(tmp_path, orientation):
    path = _tagged(tmp_path / "o.jpg", orientation)
    cv = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    with Image.open(path) as raw:
        pil = np.asarray(upright(raw).convert("RGB"))
    assert pil.shape == cv.shape
    assert _close(_corner_colours(pil), _corner_colours(cv))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_qt_reader_with_autotransform_matches_cv2(tmp_path, orientation):
    pytest.importorskip("PySide6.QtGui", exc_type=ImportError)
    from PySide6.QtGui import QImageReader
    path = _tagged(tmp_path / "o.jpg", orientation)
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    img = reader.read()
    h, w = cv2.imread(str(path)).shape[:2]
    assert (img.width(), img.height()) == (w, h)


def test_hash_of_tagged_photo_equals_hash_of_rotated_copy(tmp_path):
    rng = np.random.RandomState(0)
    arr = (np.add.outer(np.arange(64), np.arange(96)) * 2 % 256).astype(np.uint8)
    arr = np.stack([arr, arr[::-1], rng.randint(0, 50, arr.shape).astype(np.uint8)], -1)
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6  # display rotated 90 degrees clockwise
    img.save(tmp_path / "tagged.png", exif=exif.tobytes())
    img.transpose(Image.ROTATE_270).save(tmp_path / "rotated.png")
    assert compute_phash(tmp_path / "tagged.png") == compute_phash(tmp_path / "rotated.png")


def test_thumbnail_is_upright(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from gui import thumbnails
    monkeypatch.setattr(thumbnails, "thumb_cache_dir", lambda: tmp_path)
    path = _tagged(tmp_path / "o.jpg", 6)  # stored 96x64, displayed 64x96
    pix = thumbnails.get_thumbnail(str(path), path.stat().st_mtime)
    assert (pix.width(), pix.height()) == (64, 96)
    assert not list(tmp_path.glob("*.tmp"))


def test_migration_to_v2_drops_old_hashes_and_backs_up(tmp_path):
    p = tmp_path / "v1.db"
    d = Database(p)
    d.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0, phash="ffff000012345678")
    d.conn.execute("PRAGMA user_version = 1")
    d.conn.commit()
    d.close()
    d = Database(p)
    assert d.schema_version() == SCHEMA_VERSION >= 2
    assert d.get_photo_by_path("/a.jpg")["phash"] is None
    d.close()
    backup = sqlite3.connect(tmp_path / "v1.db.bak-v1")
    assert backup.execute("SELECT phash FROM photos").fetchone()[0] == "ffff000012345678"
    backup.close()


def test_unchanged_photos_get_missing_hash_back_without_redetection(tmp_path):
    photos = make_photos(tmp_path / "photos", reds=(10, 200, 90))
    d = Database(tmp_path / "t.db")
    engine = StubEngine()
    an = Analyzer(d, engine)
    an.analyze_folder(tmp_path / "photos")
    before = {r["path"]: r["phash"] for r in d.all_photos()}
    calls = engine.calls
    d.conn.execute("UPDATE photos SET phash = NULL")
    d.conn.commit()
    an.analyze_folder(tmp_path / "photos")
    assert engine.calls == calls  # nothing re-detected
    assert {r["path"]: r["phash"] for r in d.all_photos()} == before
    assert all(before[str(p)] for p in photos)
    d.close()


def test_lightbox_shows_upright_pixels(tmp_path):
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from gui.lightbox import LightboxDialog
    path = _tagged(tmp_path / "o.jpg", 8)  # stored 96x64, displayed 64x96
    d = Database(tmp_path / "t.db")
    d.upsert_photo(str(path), 1.0, 1, 64, 96, None, None, None, 1.0)
    dlg = LightboxDialog(d, [{"photo_row": d.get_photo_by_path(str(path)), "faces": []}], 0)
    size = dlg._pixmap_item.pixmap().size()
    assert (size.width(), size.height()) == (64, 96)
    dlg.close()
    d.close()
