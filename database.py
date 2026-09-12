"""SQLite schema and query helpers for photoface.

Tables:
  photos(id, path UNIQUE, mtime, size, width, height, exif_date, exif_lat, exif_lon)
  persons(id, name, color)
  faces(id, photo_id, person_id NULLABLE, x, y, w, h, embedding BLOB, pinned, confidence)
  tags(id, name UNIQUE)
  photo_tags(photo_id, tag_id)  -- many-to-many

`embedding` stores a 128-float SFace embedding as raw bytes (float32).
`pinned` marks a face whose person assignment was made or confirmed by hand -
reclustering must never move a pinned face to a different person.
"""
from __future__ import annotations

import sqlite3
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


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Lightweight forward-only migrations for columns added after a
        database may already have been created."""
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(photos)")}
        if "phash" not in cols:
            self.conn.execute("ALTER TABLE photos ADD COLUMN phash TEXT")

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # Photos
    # ------------------------------------------------------------------ #
    def get_photo_by_path(self, path: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM photos WHERE path = ?", (path,)
        ).fetchone()

    def upsert_photo(self, path: str, mtime: float, size: int,
                     width: int, height: int, exif_date: Optional[str],
                     exif_lat: Optional[float], exif_lon: Optional[float],
                     analyzed_at: float, phash: Optional[str] = None) -> int:
        cur = self.conn.execute(
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
        self.conn.commit()
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
        for i in range(n):
            for j in range(i + 1, n):
                if bin(hashes[i] ^ hashes[j]).count("1") <= max_hamming_distance:
                    union(i, j)

        groups: dict[int, list[sqlite3.Row]] = {}
        for i, row in enumerate(rows):
            groups.setdefault(find(i), []).append(row)
        return [g for g in groups.values() if len(g) > 1]

    def delete_photo_faces(self, photo_id: int) -> None:
        self.conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        self.conn.commit()

    def delete_photos_not_in(self, paths: Iterable[str]) -> None:
        paths = list(paths)
        existing = [r["path"] for r in self.conn.execute("SELECT path FROM photos")]
        gone = set(existing) - set(paths)
        if gone:
            self.conn.executemany("DELETE FROM photos WHERE path = ?",
                                  [(p,) for p in gone])
            self.conn.commit()

    def all_photos(self, order_by: str = "path") -> list[sqlite3.Row]:
        col = "exif_date" if order_by == "date" else "path"
        return self.conn.execute(
            f"SELECT * FROM photos ORDER BY {col} IS NULL, {col}"
        ).fetchall()

    # ------------------------------------------------------------------ #
    # Faces
    # ------------------------------------------------------------------ #
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
        self.conn.commit()
        return cur.lastrowid

    def all_faces(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM faces").fetchall()

    def faces_for_photo(self, photo_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM faces WHERE photo_id = ?", (photo_id,)
        ).fetchall()

    def set_face_person(self, face_id: int, person_id: Optional[int],
                        pinned: bool = True) -> None:
        self.conn.execute(
            "UPDATE faces SET person_id = ?, pinned = ? WHERE id = ?",
            (person_id, int(pinned), face_id),
        )
        self.conn.commit()

    def unpinned_faces(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM faces WHERE pinned = 0").fetchall()

    def delete_face(self, face_id: int) -> None:
        self.conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Persons
    # ------------------------------------------------------------------ #
    def create_person(self, name: str, color: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO persons (name, color) VALUES (?, ?)", (name, color)
        )
        self.conn.commit()
        return cur.lastrowid

    def all_persons(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT p.*, COUNT(f.id) as face_count
               FROM persons p LEFT JOIN faces f ON f.person_id = p.id
               GROUP BY p.id ORDER BY p.name"""
        ).fetchall()

    def rename_person(self, person_id: int, name: str) -> None:
        self.conn.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id))
        self.conn.commit()

    def merge_persons(self, src_id: int, dst_id: int) -> None:
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE person_id = ?", (dst_id, src_id)
        )
        self.conn.execute("DELETE FROM persons WHERE id = ?", (src_id,))
        self.conn.commit()

    def delete_person(self, person_id: int) -> None:
        self.conn.execute(
            "UPDATE faces SET person_id = NULL, pinned = 0 WHERE person_id = ?",
            (person_id,),
        )
        self.conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Tags
    # ------------------------------------------------------------------ #
    def get_or_create_tag(self, name: str) -> int:
        row = self.conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
        self.conn.commit()
        return cur.lastrowid

    def all_tags(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT t.*, COUNT(pt.photo_id) as photo_count
               FROM tags t LEFT JOIN photo_tags pt ON pt.tag_id = t.id
               GROUP BY t.id ORDER BY t.name"""
        ).fetchall()

    def tag_photo(self, photo_id: int, tag_name: str) -> None:
        tag_id = self.get_or_create_tag(tag_name)
        self.conn.execute(
            "INSERT OR IGNORE INTO photo_tags (photo_id, tag_id) VALUES (?, ?)",
            (photo_id, tag_id),
        )
        self.conn.commit()

    def untag_photo(self, photo_id: int, tag_id: int) -> None:
        self.conn.execute(
            "DELETE FROM photo_tags WHERE photo_id = ? AND tag_id = ?",
            (photo_id, tag_id),
        )
        self.conn.commit()

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
        query = f"SELECT DISTINCT p.* FROM photos p"
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
