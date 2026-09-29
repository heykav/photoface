"""Locating, downloading and verifying the YuNet / SFace ONNX models.

Integrity policy
----------------
Each model has an expected SHA-256 in `MODELS`. A downloaded file is written
to `<name>.part`, hashed, and only moved into place (atomically) if the hash
matches - a truncated or tampered download can never be mistaken for a model.

`sha256=None` means "no authoritative checksum pinned yet". The checksums are
published as Git LFS object ids in the opencv/opencv_zoo repository; they were
NOT pinned here because that repository could not be reached to read them
when this mechanism was written, and inventing hashes would be worse than
having none. Until someone pins them (run `python scripts/download_models.py`
on a trusted network, compare the printed digests with the LFS pointers at
https://github.com/opencv/opencv_zoo, and paste them below):

- default: trust-on-first-use. The digest of the first download is printed
  and stored in `<model>.sha256`; every later start verifies against it, so
  corruption or later tampering is detected.
- `--strict` (download script): refuse any model that has no pinned hash.

URLs point at the `main` branch of opencv_zoo; pin them to a commit SHA
together with the hashes.
"""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from paths import models_dir

_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

# name -> {"url": ..., "sha256": hex digest or None (see module docstring)}
MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "face_detection_yunet_2023mar.onnx": {
        "url": f"{_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": None,
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": f"{_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": None,
    },
}

_CHUNK = 1 << 20
_TIMEOUT = 30


class ModelError(FileNotFoundError):
    """Models missing, unreadable, corrupt or failing verification."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + ".sha256")


def expected_hash(name: str, dest: Path) -> Optional[str]:
    """Pinned hash if there is one, else the trust-on-first-use record."""
    pinned = MODELS[name]["sha256"]
    if pinned:
        return pinned.lower()
    side = _sidecar(dest)
    if side.exists():
        txt = side.read_text().strip().split()
        return txt[0].lower() if txt else None
    return None


def check_installed(directory: Optional[Path] = None) -> List[str]:
    """Problems with the installed models (empty list = all good)."""
    directory = directory or models_dir()
    problems = []
    for name in MODELS:
        dest = directory / name
        if not dest.exists() or dest.stat().st_size == 0:
            problems.append(f"{name}: missing")
            continue
        want = expected_hash(name, dest)
        if want and sha256_file(dest) != want:
            problems.append(f"{name}: checksum mismatch (corrupt or modified)")
    return problems


def require_installed(directory: Optional[Path] = None) -> None:
    problems = check_installed(directory)
    if problems:
        raise ModelError(
            "Face models are not usable:\n  " + "\n  ".join(problems) +
            "\nRun `python scripts/download_models.py` (needs internet once), or "
            "copy the two .onnx files into the models/ folder by hand."
        )


def _open(url: str):
    return urllib.request.urlopen(url, timeout=_TIMEOUT)  # noqa: S310 - fixed https URLs


def download_model(name: str, directory: Optional[Path] = None, strict: bool = False,
                   force: bool = False, log=print) -> Path:
    """Fetch one model, verify it, and install it atomically. Raises
    ModelError with a human-readable reason on any failure (offline, HTTP
    error, checksum mismatch, strict mode without a pinned hash)."""
    directory = directory or models_dir()
    spec = MODELS[name]
    pinned = spec["sha256"]
    dest = directory / name

    if strict and not pinned:
        raise ModelError(f"{name}: no pinned SHA-256 in model_files.MODELS (--strict)")

    if dest.exists() and not force:
        want = expected_hash(name, dest)
        if dest.stat().st_size > 0 and (want is None or sha256_file(dest) == want):
            log(f"already have {name} ({dest.stat().st_size} bytes)"
                + ("" if want else " - unverified, no checksum on record"))
            return dest
        log(f"{name}: existing file is empty or fails its checksum, re-downloading")

    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        log(f"downloading {name} ...")
        with _open(spec["url"]) as resp, open(part, "wb") as out:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        part.unlink(missing_ok=True)
        raise ModelError(
            f"Could not download {name}: {e}\n"
            f"Are you offline or behind a proxy? Fetch it from\n  {spec['url']}\n"
            f"and place it at\n  {dest}\n(then re-run to verify it)."
        ) from e
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    got = digest.hexdigest()
    if part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise ModelError(f"{name}: downloaded file is empty")
    if pinned and got != pinned.lower():
        part.unlink(missing_ok=True)
        raise ModelError(f"{name}: SHA-256 mismatch - expected {pinned}, got {got}. "
                         "Not installing it.")
    os.replace(part, dest)
    if pinned:
        log(f"  verified sha256 {got}")
    else:
        _sidecar(dest).write_text(f"{got}  {name}\n")
        log(f"  WARNING: no pinned checksum for {name}; recorded first-use "
            f"sha256 {got}. Compare it with the opencv_zoo LFS pointer.")
    return dest
