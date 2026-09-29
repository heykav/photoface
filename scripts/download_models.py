#!/usr/bin/env python3
"""Download the YuNet (face detection) and SFace (face embedding) ONNX
models from the OpenCV Zoo into models/, verifying integrity.

This is the ONLY network access photoface performs (the app itself never
uploads anything). Usage:

    python scripts/download_models.py            # trust-on-first-use if no pinned hash
    python scripts/download_models.py --strict   # fail unless a SHA-256 is pinned
    python scripts/download_models.py --force    # re-download even if present

See model_files.py for the checksum policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_files import MODELS, ModelError, download_model  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="refuse models that have no pinned SHA-256")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args(argv)
    for name in MODELS:
        try:
            download_model(name, strict=args.strict, force=args.force)
        except ModelError as e:
            print(f"FAILED: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
