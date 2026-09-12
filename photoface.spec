# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for photoface. Build with:

    pip install pyinstaller
    python scripts/download_models.py
    pyinstaller photoface.spec

Bundles the downloaded ONNX models into the executable itself (via `datas`)
so a packaged app needs no separate model-download step - `paths.models_dir()`
resolves to the PyInstaller extraction dir (`sys._MEIPASS`) at runtime, where
this places them under `models/`.

Verification status: the macOS .app path (COLLECT + BUNDLE below) was built
and launch-tested locally - it starts, detects it's frozen, and writes its
database under ~/.photoface/ as designed. The plain onedir COLLECT output
used for Windows/Linux follows standard PyInstaller practice for those
platforms but has not been built or run on either in this repo - CI's
release workflow is what actually exercises them, on real Windows/Linux
runners, before anything is attached to a release.
"""

import sys

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[("models", "models")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="photoface",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# onedir, not onefile: a macOS .app bundle needs its resources (including the
# bundled `models/` datas) laid out on disk under Contents/, which only the
# COLLECT + BUNDLE combination produces correctly - onefile mode packs
# everything into a single self-extracting binary instead, which is why an
# earlier version of this spec built successfully but silently shipped an
# app with no model files inside it at all.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="photoface",
)

if sys.platform == "darwin":
    BUNDLE(
        coll,
        name="photoface.app",
        icon=None,
        bundle_identifier="com.photoface.app",
    )
# Windows/Linux: the COLLECT step above already produced dist/photoface/ as
# a plain onedir folder (photoface.exe + its libraries on Windows, photoface
# + its libraries on Linux) - that folder is the distributable as-is, no
# further bundling step is applicable on those platforms.
