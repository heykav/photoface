"""Analyzer.recluster_all against a real (temp) SQLite DB with synthetic
embeddings - no models, no images."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from analyzer import Analyzer
from database import Database
from synth import blobs


@pytest.fixture
def db():
    d = Database(Path(tempfile.mkdtemp()) / "t.db")
    yield d
    d.close()


def _load(db, sizes=(6, 5, 4), person_of=None, **kw):
    faces, truth = blobs(list(sizes), sigma=0.5, seed=kw.get("seed", 1))
    pid = db.upsert_photo("/p.jpg", 1.0, 1, 1, 1, None, None, None, 1.0)
    ids = []
    for f in faces:
        ids.append(db.add_face(pid, 0, 0, 1, 1, f.embedding, 0.9))
    return ids, [truth[f.face_id] for f in faces]


def _assignment(db):
    return {r["id"]: r["person_id"] for r in db.all_faces()}


def test_first_recluster_creates_one_person_per_identity(db):
    ids, labels = _load(db)
    Analyzer(db).recluster_all()
    a = _assignment(db)
    assert len({a[i] for i in ids}) == 3
    for lab in range(3):
        assert len({a[i] for i, lbl in zip(ids, labels, strict=True) if lbl == lab}) == 1


def test_recluster_is_idempotent_and_does_not_churn_people(db):
    _load(db)
    an = Analyzer(db)
    an.recluster_all()
    before = (_assignment(db), [(p["id"], p["name"]) for p in db.all_persons()])
    an.recluster_all()
    an.recluster_all()
    assert (_assignment(db), [(p["id"], p["name"]) for p in db.all_persons()]) == before


def test_renamed_person_survives_recluster(db):
    """Regression: renaming did not pin, so recluster orphaned the named
    person and created a fresh 'Persona N'."""
    ids, _ = _load(db)
    an = Analyzer(db)
    an.recluster_all()
    person = _assignment(db)[ids[0]]
    db.rename_person(person, "Alice")
    an.recluster_all()
    assert _assignment(db)[ids[0]] == person
    assert {p["name"] for p in db.all_persons()} >= {"Alice"}
    assert len(db.all_persons()) == 3


def test_pinned_faces_never_move(db):
    ids, _ = _load(db)
    an = Analyzer(db)
    an.recluster_all()
    a = _assignment(db)
    other = a[ids[-1]]                      # someone from a different identity
    db.set_face_person(ids[0], other, pinned=True)   # user says: this is them
    for thr in (0.0, 0.363, 0.9):
        an.threshold = thr
        an.recluster_all()
        assert _assignment(db)[ids[0]] == other
        assert bool(db.conn.execute("SELECT pinned FROM faces WHERE id=?",
                                    (ids[0],)).fetchone()[0])


def test_empty_auto_named_persons_are_cleaned_but_custom_kept(db):
    _load(db)
    an = Analyzer(db)
    an.recluster_all()
    db.create_person("Persona 99", "#000000")
    db.create_person("Grandma", "#111111")
    an.recluster_all()
    names = {p["name"] for p in db.all_persons()}
    assert "Persona 99" not in names and "Grandma" in names


def test_empty_library(db):
    Analyzer(db).recluster_all()
    assert db.all_persons() == []


def test_recluster_is_atomic_when_it_fails_midway(db, monkeypatch):
    ids, _ = _load(db)
    an = Analyzer(db)
    an.recluster_all()
    snapshot = (_assignment(db), [tuple(p)[:3] for p in db.all_persons()])
    # make the next recluster want to change things, then crash part-way
    an.threshold = 0.99
    calls = {"n": 0}
    real = db.set_face_person

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash")
        return real(*a, **k)

    monkeypatch.setattr(db, "set_face_person", flaky)
    with pytest.raises(RuntimeError):
        an.recluster_all()
    monkeypatch.undo()
    assert calls["n"] >= 3  # it really did write before crashing
    assert (_assignment(db), [tuple(p)[:3] for p in db.all_persons()]) == snapshot
