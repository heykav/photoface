"""Path resolution: works the same whether run from source or as a frozen
(PyInstaller) executable, and keeps user data out of the source tree when
running as a packaged app."""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Directory containing the source/bundle - used to find bundled assets
    (models/) that ship alongside the code."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Where the database and log live. Next to the source when running from
    source (matches the reference project); under ~/.photoface for a frozen
    build so a packaged app never writes into its own bundle."""
    if is_frozen():
        d = Path.home() / ".photoface"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return app_root()


def db_path() -> Path:
    return user_data_dir() / "photoface.db"


def log_path() -> Path:
    return user_data_dir() / "photoface.log"


def models_dir() -> Path:
    d = app_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumb_cache_dir() -> Path:
    d = user_data_dir() / "thumbcache"
    d.mkdir(parents=True, exist_ok=True)
    return d
