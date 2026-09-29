"""SQLite schema and query helpers for photoface.

Tables:
  photos(id, path UNIQUE, mtime, size, width, height, exif_date, exif_lat, exif_lon)
  persons(id, name, color)
  faces(id, photo_id, person_id NULLABLE, x, y, w, h, embedding BLOB, pinned, confidence)
  tags(id, name UNIQUE)
  photo_tags(photo_id, tag_id)  -- many-to-many

Schema versioning: `PRAGMA user_version` holds the schema version. Migrations
are forward-only, run in one transaction each, and an existing database is
copied to `<name>.bak-v<old>` first. A database written by a NEWER photoface
is refused (DatabaseVersionError) instead of being opened and corrupted.

`embedding` stores a 128-float SFace embedding as raw bytes (float32).
`pinned` marks a face whose person assignment was made or confirmed by hand -
reclustering must never move a pinned face to a different person.
"""
from __future__ import annotations

import functools
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    exif_date TEXT,
    exif_lat REAL,
    exif_lon REAL,
    phash TEXT,
    analyzed_at REAL
);

CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    color TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
    embedding BLOB NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS photo_tags (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (photo_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_photo_tags_tag ON photo_tags(tag_id);
"""


def _locked(fn):
    """Serialize a mutating method against `transaction()` blocks running on
    another thread (the connection is shared with the analysis worker), so one
    thread's commit can never land in the middle of another's transaction."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


SCHEMA_VERSION = 1


class DatabaseVersionError(RuntimeError):
    """The database was created by a newer version of photoface."""


def _m1_add_phash(conn: sqlite3.Connection) -> None:
    # v0 (unversioned) databases created before duplicate detection
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    if "phash" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN phash TEXT")


# version -> migration that upgrades (version - 1) to version
MIGRATIONS = {1: _m1_add_phash}


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._had_data = Path(path).exists() and Path(path).stat().st_size > 0
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        try:
            if self.schema_version() <= SCHEMA_VERSION:
                self.conn.executescript(SCHEMA)
            self._migrate()
        except BaseException:
            self.conn.close()
            raise
        self.conn.commit()

    def schema_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        """Bring the schema up to SCHEMA_VERSION (see module docstring)."""
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise DatabaseVersionError(
                f"{self.path} has schema version {current}, but this photoface "
                f"only understands up to {SCHEMA_VERSION}. Update photoface.")
        if current == SCHEMA_VERSION:
            return
        if self._had_data:
            backup = self.path.with_name(f"{self.path.name}.bak-v{current}")
            if not backup.exists():
                dest = sqlite3.connect(str(backup))
                try:
                    self.conn.backup(dest)
                finally:
                    dest.close()
        for version in range(current + 1, SCHEMA_VERSION + 1):
            self.conn.execute("BEGIN")
            try:
                MIGRATIONS[version](self.conn)
                self.conn.execute(f"PRAGMA user_version = {version}")
            except BaseException:
                self.conn.rollback()
                raise
            self.conn.commit()

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()

    @contextmanager
    def transaction(self):
        """Group several writes into one atomic unit: all are committed
        together, or (on any exception, e.g. a crash mid-recluster) none are.
        Nestable; only the outermost block commits."""
        with self._lock:
            self._tx_depth += 1
            try:
                yield self
            except BaseException:
                if self._tx_depth == 1:
                    self.conn.rollback()
                raise
            else:
                if self._tx_depth == 1:
                    self.conn.commit()
            finally:
                self._tx_depth -= 1

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # Photos
    # ------------------------------------------------------------------ #
    def get_photo_by_path(self, path: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM photos WHERE path = ?", (path,)
        ).fetchone()

    @_locked
    def upsert_photo(self, path: str, mtime: float, size: int,
                     width: int, height: int, exif_date: Optional[str],
                     exif_lat: Optional[float], exif_lon: Optional[float],
                     analyzed_at: float, phash: Optional[str] = None) -> int:
        self.conn.execute(
            """INSERT INTO photos (path, mtime, size, width, height,
                                    exif_date, exif_lat, exif_lon, phash, analyzed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                 mtime=excluded.mtime, size=excluded.size,
                 width=excluded.width, height=excluded.height,
                 exif_date=excluded.exif_date, exif_lat=excluded.exif_lat,
                 exif_lon=excluded.exif_lon, phash=excluded.phash,
                 analyzed_at=excluded.analyzed_at
            """,
            (path, mtime, size, width, height, exif_date, exif_lat, exif_lon,
             phash, analyzed_at),
        )
        self._commit()
        row = self.get_photo_by_path(path)
        return row["id"]

    def duplicate_groups(self, max_hamming_distance: int = 4) -> list[list[sqlite3.Row]]:
        """Groups of photos whose perceptual hash (`phash`, a 64-bit dHash as
        a 16-char hex string) differ by at most `max_hamming_distance` bits.
        O(n^2) over photos that have a phash - fine for a personal library;
        would want a proper nearest-neighbor index for a huge one."""
        rows = [r for r in self.conn.execute(
            "SELECT * FROM photos WHERE phash IS NOT NULL ORDER BY path"
        )]
        n = len(rows)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        hashes = [int(r["phash"], 16) for r in rows]
        # An all-zero dHash means "no gradients at all" (solid-colour frame):
        # every such image would match every other, so it is never a duplicate
        # signal on its own.
        usable = [i for i in range(n) if hashes[i] != 0]
        for a in range(len(usable)):
            i = usable[a]
            for b in range(a + 1, len(usable)):
                j = usable[b]
                if bin(hashes[i] ^ hashes[j]).count("1") <= max_hamming_distance:
                    union(i, j)

        groups: dict[int, list[sqlite3.Row]] = {}
        for i, row in enumerate(rows):
            groups.setdefault(find(i), []).append(row)
        return [g for g in groups.values() if len(g) > 1]

    @_locked
    def delete_photo_faces(self, photo_id: int) -> None:
        self.conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        self._commit()

    @_locked
    def delete_photos_not_in(self, paths: Iterable[str]) -> None:
        paths = list(paths)
        existing = [r["path"] for r in self.conn.execute("SELECT path FROM photos")]
        gone = set(existing) - set(paths)
        if gone:
            self.conn.executemany("DELETE FROM photos WHERE path = ?",
                                  [(p,) for p in gone])
            self._commit()

    def all_photos(self, order_by: str = "path") -> list[sqlite3.Row]:
        col = "exif_date" if order_by == "date" else "path"
        return self.conn.execute(
            f"SELECT * FROM photos ORDER BY {col} IS NULL, {col}"
        ).fetchall()

    # ------------------------------------------------------------------ #
    # Faces
    # ------------------------------------------------------------------ #
    @_locked
    def add_face(self, photo_id: int, x: float, y: float, w: float, h: float,
                embedding: np.ndarray, confidence: float,
                person_id: Optional[int] = None, pinned: bool = False) -> int:
        cur = self.conn.execute(
            """INSERT INTO faces (photo_id, person_id, x, y, w, h, embedding,
                                   pinned, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (photo_id, person_id, x, y, w, h,
             embedding.astype(np.float32).tobytes(), int(pinned), confidence),
        )
        self._commit()
        return cur.lastrowid

    def all_faces(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM faces").fetchall()

    def faces_for_photo(self, photo_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM faces WHERE photo_id = ?", (photo_id,)
        ).fetchall()

    @_locked
    def set_face_person(self, face_id: int, person_id: Optional[int],
                        pinned: bool = True) -> None:
        self.conn.execute(
            "UPDATE faces SET person_id = ?, pinned = ? WHERE id = ?",
            (person_id, int(pinned), face_id),
        )
        self._commit()

    def unpinned_faces(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM faces WHERE pinned = 0").fetchall()

    @_locked
    def delete_face(self, face_id: int) -> None:
        self.conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self._commit()

    # ------------------------------------------------------------------ #
    # Persons
    # ------------------------------------------------------------------ #
    @_locked
    def create_person(self, name: str, color: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO persons (name, color) VALUES (?, ?)", (name, color)
        )
        self._commit()
        return cur.lastrowid

    def all_persons(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT p.*, COUNT(f.id) as face_count
               FROM persons p LEFT JOIN faces f ON f.person_id = p.id
               GROUP BY p.id ORDER BY p.name"""
        ).fetchall()

    @_locked
    def rename_person(self, person_id: int, name: str) -> None:
        self.conn.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id))
        self._commit()

    @_locked
    def merge_persons(self, src_id: int, dst_id: int) -> None:
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE person_id = ?", (dst_id, src_id)
        )
        self.conn.execute("DELETE FROM persons WHERE id = ?", (src_id,))
        self._commit()

    @_locked
    def delete_person(self, person_id: int) -> None:
        self.conn.execute(
            "UPDATE faces SET person_id = NULL, pinned = 0 WHERE person_id = ?",
            (person_id,),
        )
        self.conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self._commit()

    # ------------------------------------------------------------------ #
    # Tags
    # ------------------------------------------------------------------ #
    @_locked
    def get_or_create_tag(self, name: str) -> int:
        row = self.conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
        self._commit()
        return cur.lastrowid

    def all_tags(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT t.*, COUNT(pt.photo_id) as photo_count
               FROM tags t LEFT JOIN photo_tags pt ON pt.tag_id = t.id
               GROUP BY t.id ORDER BY t.name"""
        ).fetchall()

    @_locked
    def tag_photo(self, photo_id: int, tag_name: str) -> None:
        tag_id = self.get_or_create_tag(tag_name)
        self.conn.execute(
            "INSERT OR IGNORE INTO photo_tags (photo_id, tag_id) VALUES (?, ?)",
            (photo_id, tag_id),
        )
        self._commit()

    @_locked
    def untag_photo(self, photo_id: int, tag_id: int) -> None:
        self.conn.execute(
            "DELETE FROM photo_tags WHERE photo_id = ? AND tag_id = ?",
            (photo_id, tag_id),
        )
        self._commit()

    def tags_for_photo(self, photo_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT t.* FROM tags t
               JOIN photo_tags pt ON pt.tag_id = t.id
               WHERE pt.photo_id = ?""",
            (photo_id,),
        ).fetchall()

    # ------------------------------------------------------------------ #
    # Filtered gallery query
    # ------------------------------------------------------------------ #
    def filtered_photos(self, person_ids: Optional[list[int]] = None,
                        tag_ids: Optional[list[int]] = None,
                        geotagged_only: bool = False,
                        order_by: str = "path") -> list[sqlite3.Row]:
        col = "exif_date" if order_by == "date" else "path"
        query = "SELECT DISTINCT p.* FROM photos p"
        joins = []
        wheres = []
        params: list[Any] = []

        if person_ids:
            for i, pid in enumerate(person_ids):
                alias = f"f{i}"
                joins.append(
                    f"JOIN faces {alias} ON {alias}.photo_id = p.id AND {alias}.person_id = ?"
                )
                params.append(pid)
        if tag_ids:
            for i, tid in enumerate(tag_ids):
                alias = f"pt{i}"
                joins.append(
                    f"JOIN photo_tags {alias} ON {alias}.photo_id = p.id AND {alias}.tag_id = ?"
                )
                params.append(tid)
        if geotagged_only:
            wheres.append("p.exif_lat IS NOT NULL AND p.exif_lon IS NOT NULL")

        query += " " + " ".join(joins)
        if wheres:
            query += " WHERE " + " AND ".join(wheres)
        query += f" ORDER BY p.{col} IS NULL, p.{col}"
        return self.conn.execute(query, params).fetchall()
