# photoface

A desktop app (PySide6) that scans a folder (recursively) for photos, detects
faces with OpenCV, clusters them into persons, and lets you browse and filter
your library by person, tag, location, or possible duplicates.

## Features

- Recursively scans a folder for jpg/jpeg/png/webp/bmp/tiff images.
- Face detection with OpenCV **YuNet**, face embeddings with **SFace**
  (both fetched from the OpenCV Zoo).
- Two-stage clustering: a fast greedy incremental pass while analysis runs,
  then a full average-linkage re-clustering pass (`Recluster now` in the
  sidebar, or automatically after each analysis run).
- Manual face edits (move to another person, split off a new person, delete
  a false detection) are **pinned** - reclustering never moves a pinned
  face to a different person again, and two different pinned identities are
  never merged into each other.
- People sidebar: rename, merge, or delete a person; multi-select filters
  the gallery with AND semantics (photos where all selected people appear).
- **Tags**: right-click any photo (in the gallery) to add a tag or remove an
  existing one; each tag gets its own stable color (derived from its name)
  shown as a dot on the card; the tags sidebar list filters the gallery
  alongside the person filter, combined with AND.
- **Possible-duplicates detection**: every photo gets a perceptual hash
  (dHash) at analysis time; a sidebar toggle filters the gallery down to
  photos whose hash is within a small distance of another photo's - re-saves,
  light edits, or re-exports of the same shot - grouped with a union-find
  pass so a whole chain of near-duplicates surfaces together, not just pairs.
- Full-size lightbox viewer: mouse-wheel zoom centered on the cursor,
  click-drag pan, ←/→ or on-screen buttons to move through the *current
  filtered* set, Esc/background-click/✕ to close.
- **Gallery keyboard navigation**: arrow keys move the selection (grid-aware,
  using the current column count), Enter opens the selected photo, Esc clears
  the person filter.
- EXIF: capture date shown in the lightbox and used for date sorting; GPS
  coordinates shown, with copy-to-clipboard and "open in OpenStreetMap";
  a sidebar toggle filters the gallery to geotagged photos only.
- **Folder watching**: optionally watches the analyzed folder (and every
  subfolder) and automatically re-analyzes, debounced 2s after changes go
  quiet so a big copy operation triggers one re-scan, not dozens.
- Thumbnails are cached on disk (keyed by path + mtime), so a large library
  re-opens fast instead of re-decoding every photo.
- Settings dialog: preview aspect (3:4 or 4:3), face-box visibility on
  thumbnails, sort order, clustering similarity threshold, folder watching,
  log verbosity - all persisted between launches via `QSettings`.
- SQLite persistence; unchanged files (same mtime + size) are skipped on
  re-analysis instead of being re-detected from scratch. Schema changes
  (e.g. adding the duplicate-detection hash column) migrate forward
  automatically for an existing database.

## Run from source

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py   # fetches the YuNet + SFace ONNX models
python main.py
```

Type or browse to a folder in the top bar and press **Analyze**.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

36 tests cover the clustering algorithm (greedy assignment, average-linkage
re-clustering, the "never merge two different pinned identities" invariant),
the perceptual-hash function (determinism, near-duplicate closeness,
unrelated-image distance), and the full database layer (photos/faces/
persons/tags CRUD, filtered queries, duplicate grouping) - no display or
model files required, and CI (`.github/workflows/tests.yml`) runs the suite
on Python 3.10-3.12 on every push and pull request. The GUI itself was
verified interactively under `QT_QPA_PLATFORM=offscreen` - gallery rendering,
sidebar person/tag lists, lightbox face overlays and geolocation, tagging,
duplicate filtering, keyboard navigation, folder-watcher arming, and
filter/rename/merge/recluster round-trips - rather than covered by automated
GUI tests in this v1.

## Project layout

| Path                         | Purpose                                          |
|------------------------------|---------------------------------------------------|
| `main.py`                    | Desktop entry point (QApplication + theme)        |
| `gui/`                       | PySide6 UI: main window, gallery, sidebar, lightbox, settings |
| `analyzer.py`                | Folder scanning, EXIF, perceptual hashing, face detection/embedding, ties clustering to the DB |
| `clustering.py`              | Greedy incremental assignment + average-linkage re-clustering |
| `database.py`                | SQLite schema, migrations, and query helpers      |
| `paths.py`                   | Path resolution (source vs. a future frozen bundle) |
| `scripts/download_models.py` | Downloads YuNet and SFace ONNX models from the OpenCV Zoo |
| `tests/`                     | Pytest suite for `clustering.py`, `database.py`, and perceptual hashing |
| `.github/workflows/tests.yml`| CI: runs the test suite on Python 3.10-3.12       |

## Storage

- `photoface.db` next to the code when running from source (see `paths.py`
  for where a packaged build would put it instead).
- Thumbnail cache and downloaded models live in `thumbcache/` and `models/`.
- View settings (aspect, face-box visibility, sort order, clustering
  threshold, folder watching, log verbosity) are stored with Qt `QSettings`
  under the `photoface` organization/app name.

## Packaged executables

`.github/workflows/build.yml` builds macOS/Linux/Windows executables (via
PyInstaller, `photoface.spec`) and attaches them as a draft GitHub release
whenever a `v*` tag is pushed. Build locally with:

```bash
pip install pyinstaller
python scripts/download_models.py
pyinstaller photoface.spec
```

The macOS `.app` path was built and launch-tested locally (starts, detects
it's frozen, writes its database under `~/.photoface/` as designed, finds its
bundled model files under `Contents/Resources/models/`). The plain onedir
output used for Windows/Linux follows standard PyInstaller practice for
those platforms but is only actually exercised by CI, on real Windows/Linux
runners, before anything is attached to a release.

## Not yet built (ideas for a v2)

- A dedicated "review duplicates" flow (e.g. pick-one-to-keep, bulk delete)
  beyond just filtering the gallery down to the duplicate groups.
- Automated GUI interaction tests (the GUI is currently verified manually
  under an offscreen Qt platform, screenshotted and inspected, rather than
  covered by `pytest`).

---

Made with ❤️ in India by Krishna Anubhav.
