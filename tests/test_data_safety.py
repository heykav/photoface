"""Migrations, crash safety, and 'never touch the user's photos' tests. The
face engine is a stub returning synthetic embeddings - no ONNX models."""
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import analyzer as analyzer_mod
from analyzer import Analyzer
from database import SCHEMA_VERSION, Database, DatabaseVersionError
from stubs import StubEngine, make_photos  # noqa: F401


def snapshot(folder: Path):
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns,
                     p.stat().st_size, p.stat().st_mode)
            for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture
def db(tmp_path):
    (tmp_path / "data").mkdir()
    d = Database(tmp_path / "data" / "t.db")
    yield d
    d.close()


class TestOriginalsNeverModified:
    def test_analysis_and_recluster_leave_photos_untouched(self, tmp_path, db):
        photos = tmp_path / "photos"
        make_photos(photos)
        (photos / "sub").mkdir()
        make_photos(photos / "sub", reds=(10, 200))
        os.chmod(photos / "p0.jpg", 0o444)  # read-only must not matter/change
        before = snapshot(photos)
        an = Analyzer(db, StubEngine())
        an.analyze_folder(photos)
        an.recluster_all()
        an.analyze_folder(photos)  # second run: skip path
        assert snapshot(photos) == before
        assert sorted(p.name for p in photos.rglob("*") if p.is_file()) == \
            sorted(Path(k).name for k in before)  # no stray files created

    def test_analysis_works_on_a_readonly_folder(self, tmp_path, db):
        photos = tmp_path / "photos"
        make_photos(photos)
        os.chmod(photos, 0o555)
        try:
            Analyzer(db, StubEngine()).analyze_folder(photos)
            assert len(db.all_photos()) == 6
        finally:
            os.chmod(photos, 0o755)

    def test_thumbnail_cache_never_written_beside_originals(self, tmp_path, db, monkeypatch):
        pytest.importorskip("PySide6.QtGui", exc_type=ImportError)
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from gui import thumbnails
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(thumbnails, "thumb_cache_dir", lambda: cache)
        photos = tmp_path / "photos"
        files = make_photos(photos)
        before = snapshot(photos)
        assert thumbnails.get_thumbnail(str(files[0]), files[0].stat().st_mtime) is not None
        assert snapshot(photos) == before
        assert list(cache.glob("*.jpg"))


class TestAnalysisCrashSafety:
    def test_crash_mid_photo_leaves_no_half_analyzed_photo(self, tmp_path, db, monkeypatch):
        photos = tmp_path / "photos"
        make_photos(photos)
        an = Analyzer(db, StubEngine())
        real_add = db.add_face
        state = {"n": 0}

        def crashing_add(*a, **k):
            state["n"] += 1
            if state["n"] == 3:
                raise KeyboardInterrupt("power cut")  # BaseException, like a kill
            return real_add(*a, **k)

        monkeypatch.setattr(db, "add_face", crashing_add)
        with pytest.raises(KeyboardInterrupt):
            an.analyze_folder(photos)
        monkeypatch.undo()
        # 2 photos fully committed, the third rolled back entirely (no row that
        # would be skipped forever as "already analyzed" yet have no faces)
        assert len(db.all_photos()) == 2
        assert len(db.all_faces()) == 2

        Analyzer(db, StubEngine()).analyze_folder(photos)   # resume
        assert len(db.all_photos()) == 6 and len(db.all_faces()) == 6

    def test_detector_error_does_not_mark_photo_analyzed(self, tmp_path, db):
        photos = tmp_path / "photos"
        make_photos(photos)
        eng = StubEngine()
        eng.fail_on_call = 2
        Analyzer(db, eng).analyze_folder(photos)
        assert len(db.all_photos()) == 5          # the failed one is not recorded
        Analyzer(db, StubEngine()).analyze_folder(photos)
        assert len(db.all_photos()) == 6          # retried and picked up

    def test_missing_folder_does_not_wipe_library(self, tmp_path, db):
        photos = tmp_path / "photos"
        make_photos(photos)
        an = Analyzer(db, StubEngine())
        an.analyze_folder(photos)
        an.recluster_all()
        n_faces = len(db.all_faces())
        with pytest.raises(FileNotFoundError):
            an.analyze_folder(tmp_path / "unmounted")
        assert len(db.all_photos()) == 6 and len(db.all_faces()) == n_faces

    def test_deleted_files_are_dropped(self, tmp_path, db):
        photos = tmp_path / "photos"
        files = make_photos(photos)
        an = Analyzer(db, StubEngine())
        an.analyze_folder(photos)
        files[0].unlink()
        an.analyze_folder(photos)
        assert len(db.all_photos()) == 5


class TestPinnedSurvivesReanalysis:
    def test_pinned_correction_kept_when_photo_changes(self, tmp_path, db):
        photos = tmp_path / "photos"
        files = make_photos(photos, reds=(10, 200))
        an = Analyzer(db, StubEngine())
        an.analyze_folder(photos)
        an.recluster_all()
        pid = db.get_photo_by_path(str(files[0]))["id"]
        face = db.faces_for_photo(pid)[0]
        target = db.create_person("Manually chosen", "#123456")
        db.set_face_person(face["id"], target, pinned=True)

        # the file is re-saved (new mtime) -> re-analysis re-detects the face
        os.utime(files[0], (1_700_000_000, 1_700_000_000))
        an.analyze_folder(photos)
        an.recluster_all()

        faces = db.faces_for_photo(pid)
        assert len(faces) == 1
        assert faces[0]["person_id"] == target and faces[0]["pinned"] == 1


class TestMigrations:
    def _legacy_db(self, path):
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE photos (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE NOT NULL,
              mtime REAL NOT NULL, size INTEGER NOT NULL, width INTEGER, height INTEGER,
              exif_date TEXT, exif_lat REAL, exif_lon REAL, analyzed_at REAL);
            INSERT INTO photos (path, mtime, size) VALUES ('/keep.jpg', 1.0, 2);
        """)
        conn.commit()
        conn.close()

    def test_fresh_db_is_current_version(self, tmp_path):
        d = Database(tmp_path / "n.db")
        assert d.schema_version() == SCHEMA_VERSION
        d.close()

    def test_legacy_unversioned_db_migrates_keeps_data_and_is_backed_up(self, tmp_path):
        p = tmp_path / "old.db"
        self._legacy_db(p)
        d = Database(p)
        assert d.schema_version() == SCHEMA_VERSION
        assert d.get_photo_by_path("/keep.jpg")["size"] == 2
        assert "phash" in d.get_photo_by_path("/keep.jpg").keys()
        d.close()
        backup = tmp_path / "old.db.bak-v0"
        assert backup.exists()
        assert "phash" not in {r[1] for r in sqlite3.connect(backup).execute(
            "PRAGMA table_info(photos)")}

    def test_reopen_is_a_noop(self, tmp_path):
        p = tmp_path / "old.db"
        self._legacy_db(p)
        Database(p).close()
        Database(p).close()
        assert [f.name for f in tmp_path.glob("old.db.bak*")] == ["old.db.bak-v0"]

    def test_newer_database_is_refused_and_untouched(self, tmp_path):
        p = tmp_path / "future.db"
        conn = sqlite3.connect(p)
        conn.execute("CREATE TABLE marker (x)")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
        conn.commit()
        conn.close()
        with pytest.raises(DatabaseVersionError):
            Database(p)
        conn = sqlite3.connect(p)
        assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master")} == {"marker"}
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION + 5

    def test_failed_migration_rolls_back(self, tmp_path, monkeypatch):
        import database
        p = tmp_path / "old.db"
        self._legacy_db(p)

        def boom(conn):
            conn.execute("ALTER TABLE photos ADD COLUMN phash TEXT")
            raise RuntimeError("migration died")

        monkeypatch.setitem(database.MIGRATIONS, 1, boom)
        with pytest.raises(RuntimeError):
            Database(p)
        conn = sqlite3.connect(p)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        assert "phash" not in {r[1] for r in conn.execute("PRAGMA table_info(photos)")}
        assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 1


class TestDuplicates:
    def test_solid_colour_images_are_not_duplicates_of_each_other(self, db):
        for i in range(3):
            db.upsert_photo(f"/black{i}.jpg", 1.0, 1, 1, 1, None, None, None, 1.0,
                            phash="0" * 16)
        assert db.duplicate_groups() == []

    def test_close_hashes_group(self, db):
        db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0, phash="ffff000012345678")
        db.upsert_photo("/b.jpg", 1.0, 1, 1, 1, None, None, None, 1.0, phash="ffff000012345679")
        db.upsert_photo("/c.jpg", 1.0, 1, 1, 1, None, None, None, 1.0, phash="0123456789abcdef")
        groups = db.duplicate_groups()
        assert [sorted(r["path"] for r in g) for g in groups] == [["/a.jpg", "/b.jpg"]]


def test_analyzer_module_exports_extract_exif():
    assert analyzer_mod.extract_exif is not None
