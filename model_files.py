"""Locating, downloading and verifying the YuNet / SFace ONNX models.

Integrity policy (fail closed)
------------------------------
Each model in `MODELS` can carry a pinned SHA-256 and byte size. A download
is written to a temporary `*.part` file in the models folder, hashed and
checked, and only then moved into place with an atomic rename. On any
failure (network error, empty file, size or hash mismatch) the temporary
file is deleted, so a truncated, tampered or unverified download is never
left where the app would load it.

A model is only accepted when its expected hash is known, from either

1. a SHA-256 pinned in `MODELS` (authoritative), or
2. a trust-on-first-use (TOFU) record `models/<name>.sha256`, which is only
   written when the user explicitly opted in (see below).

With neither, the download is refused before any network access and
`check_installed()` reports the model as unusable. This is the default.

Trust-on-first-use is an explicit opt-in, via either

- `python scripts/download_models.py --trust-on-first-use`, or
- the environment variable `PHOTOFACE_TRUST_ON_FIRST_USE=1`.

In that mode an unpinned model is downloaded (or an existing hand-copied
file is hashed), its SHA-256 and size are printed and the hash is recorded
in `models/<name>.sha256`; every later start verifies against that record.
Compare the printed digest with the Git LFS pointer of the file in
https://github.com/opencv/opencv_zoo and paste it into `MODELS` to pin it.
`strict=True` (`--strict`) refuses unpinned models even with TOFU enabled.

Current status: the hashes below are NOT pinned. They are published as Git
LFS object ids in opencv/opencv_zoo, which could not be reached when this
was written, and inventing them would be worse than having none. The URLs
point at the `main` branch; pin them to a commit SHA together with the
hashes.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Union

from paths import models_dir

_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

TOFU_ENV = "PHOTOFACE_TRUST_ON_FIRST_USE"

# name -> {"url": str, "sha256": hex digest or None, "size": bytes or None}
MODELS: Dict[str, Dict[str, Union[str, int, None]]] = {
    "face_detection_yunet_2023mar.onnx": {
        "url": f"{_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": None,
        "size": None,
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": f"{_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": None,
        "size": None,
    },
}

_CHUNK = 1 << 20
_TIMEOUT = 30


class ModelError(FileNotFoundError):
    """Models missing, unreadable, corrupt or failing verification."""


def tofu_enabled() -> bool:
    """True if the user opted in to trust-on-first-use via the environment."""
    return os.environ.get(TOFU_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + ".sha256")


def _pinned(name: str) -> Optional[str]:
    pinned = MODELS[name].get("sha256")
    return str(pinned).lower() if pinned else None


def _pinned_size(name: str) -> Optional[int]:
    size = MODELS[name].get("size")
    return int(size) if size else None


def expected_hash(name: str, dest: Path) -> Optional[str]:
    """Pinned hash if there is one, else the trust-on-first-use record, else
    None (unverifiable, which every caller treats as a failure)."""
    pinned = _pinned(name)
    if pinned:
        return pinned
    side = _sidecar(dest)
    if side.exists():
        txt = side.read_text().strip().split()
        return txt[0].lower() if txt else None
    return None


def _problem(name: str, path: Path, want: Optional[str]) -> Optional[str]:
    """Why `path` is not an acceptable copy of model `name`, or None if it is."""
    size = path.stat().st_size
    if size == 0:
        return "file is empty"
    want_size = _pinned_size(name)
    if want_size is not None and size != want_size:
        return f"size mismatch (corrupt or modified): expected {want_size} bytes, got {size}"
    if want is None:
        return ("no pinned or recorded SHA-256, cannot verify it (see 'Model integrity' "
                "in README; --trust-on-first-use records its hash)")
    got = sha256_file(path)
    if got != want:
        return f"checksum mismatch (corrupt or modified): expected {want}, got {got}"
    return None


def check_installed(directory: Optional[Path] = None) -> List[str]:
    """Problems with the installed models (empty list = all good). A model
    whose hash is neither pinned nor recorded counts as a problem."""
    directory = directory or models_dir()
    problems = []
    for name in MODELS:
        dest = directory / name
        if not dest.exists():
            problems.append(f"{name}: missing")
            continue
        reason = _problem(name, dest, expected_hash(name, dest))
        if reason:
            problems.append(f"{name}: {reason}")
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


def _no_hash_message(name: str) -> str:
    return (f"{name}: no pinned SHA-256 in model_files.MODELS and no recorded hash, "
            f"so it cannot be verified. Refusing to install it.\n"
            f"To accept it anyway and record its hash (trust on first use), run\n"
            f"  python scripts/download_models.py --trust-on-first-use\n"
            f"or set {TOFU_ENV}=1; then compare the printed SHA-256 with the Git LFS "
            f"pointer in opencv/opencv_zoo and pin it in model_files.py.")


def _record_tofu(name: str, dest: Path, digest: str, log) -> None:
    _sidecar(dest).write_text(f"{digest}  {name}\n")
    log(f"  WARNING: {name} has no pinned checksum; trusting it on first use.")
    log(f"  sha256 {digest}")
    log(f"  size   {dest.stat().st_size} bytes")
    log(f"  recorded in {_sidecar(dest)}. Compare the sha256 with the opencv_zoo "
        "Git LFS pointer, then pin it in model_files.MODELS.")


def download_model(name: str, directory: Optional[Path] = None, strict: bool = False,
                   force: bool = False, log=print,
                   trust_on_first_use: Optional[bool] = None) -> Path:
    """Fetch one model, verify it, and install it atomically. Raises
    ModelError with a human-readable reason on any failure (offline, HTTP
    error, size/checksum mismatch, no known hash without the TOFU opt-in).

    `trust_on_first_use=None` reads the opt-in from the environment
    (`PHOTOFACE_TRUST_ON_FIRST_USE`); without it, unpinned models are refused.
    """
    directory = directory or models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    spec = MODELS[name]
    dest = directory / name
    tofu = tofu_enabled() if trust_on_first_use is None else bool(trust_on_first_use)

    if strict and not _pinned(name):
        raise ModelError(f"{name}: no pinned SHA-256 in model_files.MODELS (--strict)")

    want = expected_hash(name, dest)
    if want is None and not tofu:
        raise ModelError(_no_hash_message(name))

    if dest.exists() and not force:
        if want is None and dest.stat().st_size > 0:
            # hand-copied file and the user opted in: record what is there
            _record_tofu(name, dest, sha256_file(dest), log)
            return dest
        reason = _problem(name, dest, want)
        if reason is None:
            log(f"already have {name} ({dest.stat().st_size} bytes), sha256 verified")
            return dest
        log(f"{name}: existing file rejected ({reason}); removing it and re-downloading")
        dest.unlink()

    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=directory)
    part = Path(tmp)
    digest = hashlib.sha256()
    try:
        log(f"downloading {name} ...")
        with os.fdopen(fd, "wb") as out, _open(str(spec["url"])) as resp:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
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

    try:
        got = digest.hexdigest()
        size = part.stat().st_size
        want_size = _pinned_size(name)
        if size == 0:
            raise ModelError(f"{name}: downloaded file is empty. Not installing it.")
        if want_size is not None and size != want_size:
            raise ModelError(f"{name}: size mismatch - expected {want_size} bytes, "
                             f"got {size}. Not installing it.")
        if want is not None and got != want:
            raise ModelError(f"{name}: SHA-256 mismatch - expected {want}, got {got}. "
                             "Not installing it.")
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    if want is not None:
        log(f"  verified sha256 {got}")
    else:
        _record_tofu(name, dest, got, log)
    return dest
