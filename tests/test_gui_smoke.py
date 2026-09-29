"""Headless (QT_QPA_PLATFORM=offscreen) smoke tests of the real MainWindow
over a temp database populated by a stub engine - no models, no display."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from analyzer import Analyzer  # noqa: E402
from database import Database  # noqa: E402
from stubs import StubEngine, make_photos  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path / "cfg"))
    from gui import main_window, thumbnails
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(thumbnails, "thumb_cache_dir", lambda: cache)
    dbfile = tmp_path / "gui.db"
    monkeypatch.setattr(main_window, "db_path", lambda: dbfile)

    photos = tmp_path / "photos"
    make_photos(photos, reds=(10, 10, 10, 200, 200, 90))
    db = Database(dbfile)
    an = Analyzer(db, StubEngine())
    an.analyze_folder(photos)
    an.recluster_all()
    db.close()

    win = main_window.MainWindow()
    yield win
    win.close()


def test_gallery_shows_every_photo(window):
    assert len(window.gallery._cards) == 6
    assert window.status.text() == "6 photo(s)"


def test_sidebar_lists_three_people(window):
    assert window.sidebar.people_list.count() == 3


def test_person_filter_and_clear(window):
    people = window.db.all_persons()
    biggest = max(people, key=lambda p: p["face_count"])
    assert biggest["face_count"] == 3
    window._on_persons_filter([biggest["id"]])
    assert len(window.gallery._cards) == 3
    window._on_persons_filter([])
    assert len(window.gallery._cards) == 6


def test_recluster_button_works_without_models(window):
    before = [(r["id"], r["person_id"]) for r in window.db.all_faces()]
    window._recluster_now()
    assert [(r["id"], r["person_id"]) for r in window.db.all_faces()] == before


def test_tagging_and_geotag_filter(window):
    pid = window.db.all_photos()[0]["id"]
    window.db.tag_photo(pid, "trip")
    window._reload_all()
    tag_id = window.db.all_tags()[0]["id"]
    window._on_tags_filter([tag_id])
    assert len(window.gallery._cards) == 1
    window._on_tags_filter([])
    window._on_geotag_toggle(True)
    assert len(window.gallery._cards) == 0   # synthetic photos carry no GPS


def test_lightbox_constructs(window):
    from gui.lightbox import LightboxDialog
    items = [{"photo_row": p, "faces": window._faces_with_display(p["id"])}
             for p in window._current_photos()]
    dlg = LightboxDialog(window.db, items, 0, window)
    dlg.close()
