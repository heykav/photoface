#!/usr/bin/env python3
"""Download the YuNet (face detection) and SFace (face embedding) ONNX
models from the OpenCV Zoo into models/, verifying integrity.

This is the ONLY network access photoface performs (the app itself never
uploads anything). Usage:

    python scripts/download_models.py          # needs a pinned SHA-256, else refuses
    python scripts/download_models.py --trust-on-first-use
                                               # accept unpinned; print + record hash
    python scripts/download_models.py --strict # refuse unpinned even with TOFU
    python scripts/download_models.py --force  # re-download even if present

--trust-on-first-use can also be enabled with PHOTOFACE_TRUST_ON_FIRST_USE=1.
See model_files.py for the checksum policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_files import MODELS, TOFU_ENV, ModelError, download_model  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trust-on-first-use", action="store_true",
                    help="accept a model that has no pinned SHA-256, print its hash and "
                         f"record it for later verification (same as {TOFU_ENV}=1)")
    ap.add_argument("--strict", action="store_true",
                    help="refuse models that have no pinned SHA-256, even with "
                         "--trust-on-first-use")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args(argv)
    for name in MODELS:
        try:
            download_model(name, strict=args.strict, force=args.force,
                           trust_on_first_use=True if args.trust_on_first_use else None)
        except ModelError as e:
            print(f"FAILED: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
