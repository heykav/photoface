import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from database import Database


@pytest.fixture
def db():
    tmp = Path(tempfile.mkdtemp()) / "test.db"
    database = Database(tmp)
    yield database
    database.close()


class TestPhotos:
    def test_upsert_and_get(self, db):
        pid = db.upsert_photo("/a.jpg", 123.0, 100, 800, 600, None, None, None, 1.0)
        row = db.get_photo_by_path("/a.jpg")
        assert row["id"] == pid
        assert row["width"] == 800

    def test_upsert_updates_existing(self, db):
        db.upsert_photo("/a.jpg", 123.0, 100, 800, 600, None, None, None, 1.0)
        db.upsert_photo("/a.jpg", 456.0, 200, 800, 600, None, None, None, 2.0)
        row = db.get_photo_by_path("/a.jpg")
        assert row["mtime"] == 456.0
        assert row["size"] == 200

    def test_delete_photos_not_in(self, db):
        db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        db.upsert_photo("/b.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        db.delete_photos_not_in(["/a.jpg"])
        assert db.get_photo_by_path("/a.jpg") is not None
        assert db.get_photo_by_path("/b.jpg") is None


class TestFacesAndPersons:
    def test_add_face_and_read_back_embedding(self, db):
        pid = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fid = db.add_face(pid, 0, 0, 10, 10, emb, 0.9)
        row = db.faces_for_photo(pid)[0]
        assert row["id"] == fid
        back = np.frombuffer(row["embedding"], dtype=np.float32)
        np.testing.assert_array_equal(back, emb)

    def test_create_and_assign_person(self, db):
        pid = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        person_id = db.create_person("Alice", "#ff0000")
        fid = db.add_face(pid, 0, 0, 10, 10, np.zeros(3, dtype=np.float32), 0.9,
                          person_id=person_id)
        persons = db.all_persons()
        assert persons[0]["face_count"] == 1
        assert persons[0]["name"] == "Alice"

    def test_merge_persons(self, db):
        photo_id = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        p1 = db.create_person("A", "#fff")
        p2 = db.create_person("B", "#000")
        db.add_face(photo_id, 0, 0, 1, 1, np.zeros(2, dtype=np.float32), 1.0, person_id=p1)
        db.add_face(photo_id, 0, 0, 1, 1, np.zeros(2, dtype=np.float32), 1.0, person_id=p2)
        db.merge_persons(p1, p2)
        persons = {p["id"]: p for p in db.all_persons()}
        assert p1 not in persons
        assert persons[p2]["face_count"] == 2

    def test_delete_person_unassigns_faces(self, db):
        photo_id = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        p1 = db.create_person("A", "#fff")
        fid = db.add_face(photo_id, 0, 0, 1, 1, np.zeros(2, dtype=np.float32), 1.0, person_id=p1)
        db.delete_person(p1)
        row = db.faces_for_photo(photo_id)[0]
        assert row["person_id"] is None
        assert row["pinned"] == 0

    def test_set_face_person_pins_by_default(self, db):
        photo_id = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        fid = db.add_face(photo_id, 0, 0, 1, 1, np.zeros(2, dtype=np.float32), 1.0)
        p1 = db.create_person("A", "#fff")
        db.set_face_person(fid, p1)
        row = db.faces_for_photo(photo_id)[0]
        assert row["pinned"] == 1


class TestTags:
    def test_tag_and_untag_photo(self, db):
        photo_id = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        db.tag_photo(photo_id, "vacation")
        tags = db.tags_for_photo(photo_id)
        assert tags[0]["name"] == "vacation"
        db.untag_photo(photo_id, tags[0]["id"])
        assert db.tags_for_photo(photo_id) == []

    def test_get_or_create_tag_is_idempotent(self, db):
        t1 = db.get_or_create_tag("beach")
        t2 = db.get_or_create_tag("beach")
        assert t1 == t2


class TestFilteredQuery:
    def test_filter_by_person(self, db):
        p_id = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        p2_id = db.upsert_photo("/b.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        person = db.create_person("Alice", "#fff")
        db.add_face(p_id, 0, 0, 1, 1, np.zeros(2, dtype=np.float32), 1.0, person_id=person)
        result = db.filtered_photos(person_ids=[person])
        assert [r["id"] for r in result] == [p_id]

    def test_filter_by_geotag(self, db):
        db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        geo_id = db.upsert_photo("/b.jpg", 1.0, 1, 1, 1, None, 40.0, -70.0, 1.0)
        result = db.filtered_photos(geotagged_only=True)
        assert [r["id"] for r in result] == [geo_id]

class TestDuplicateGroups:
    def _photo(self, db, path, phash):
        return db.upsert_photo(path, 1.0, 1, 1, 1, None, None, None, 1.0, phash=phash)

    def test_identical_hashes_grouped(self, db):
        self._photo(db, "/a.jpg", "0" * 16)
        self._photo(db, "/b.jpg", "0" * 16)
        groups = db.duplicate_groups()
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_close_hashes_grouped_within_threshold(self, db):
        self._photo(db, "/a.jpg", "0000000000000000")
        self._photo(db, "/b.jpg", "0000000000000003")  # 2 bits different
        groups = db.duplicate_groups(max_hamming_distance=4)
        assert len(groups) == 1

    def test_far_hashes_not_grouped(self, db):
        self._photo(db, "/a.jpg", "0000000000000000")
        self._photo(db, "/b.jpg", "ffffffffffffffff")  # maximally different
        groups = db.duplicate_groups(max_hamming_distance=4)
        assert groups == []

    def test_photos_without_phash_ignored(self, db):
        self._photo(db, "/a.jpg", None)
        self._photo(db, "/b.jpg", None)
        assert db.duplicate_groups() == []

    def test_singletons_not_returned_as_groups(self, db):
        self._photo(db, "/a.jpg", "0000000000000000")
        self._photo(db, "/b.jpg", "ffffffffffffffff")
        groups = db.duplicate_groups(max_hamming_distance=4)
        assert groups == []

    def test_transitive_chain_groups_all_three(self, db):
        # a<->b close, b<->c close, a<->c not directly close enough alone -
        # union-find should still merge all three into one group.
        self._photo(db, "/a.jpg", "0000000000000000")
        self._photo(db, "/b.jpg", "0000000000000003")
        self._photo(db, "/c.jpg", "000000000000000f")
        groups = db.duplicate_groups(max_hamming_distance=2)
        assert len(groups) == 1
        assert len(groups[0]) == 3


class TestFilteredQueryByTag:
    def test_filter_by_tag(self, db):
        p1 = db.upsert_photo("/a.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        p2 = db.upsert_photo("/b.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
        db.tag_photo(p1, "beach")
        tag_id = db.tags_for_photo(p1)[0]["id"]
        result = db.filtered_photos(tag_ids=[tag_id])
        assert [r["id"] for r in result] == [p1]
