#!/usr/bin/env python3
"""Download the YuNet (face detection) and SFace (face embedding) ONNX
models from the OpenCV Zoo into models/, if not already present."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import models_dir  # noqa: E402

_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx":
        f"{_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        f"{_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main() -> int:
    dest_dir = models_dir()
    for filename, url in MODELS.items():
        dest = dest_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already have {filename} ({dest.stat().st_size} bytes)")
            continue
        print(f"downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {filename}: {e}", file=sys.stderr)
            return 1
        print(f"  -> {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
