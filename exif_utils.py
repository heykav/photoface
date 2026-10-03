"""Defensive EXIF parsing: capture date and GPS position.

Camera and phone EXIF is frequently incomplete or malformed (missing
hemisphere refs, zero denominators, junk dates, GPS at 0,0 when there was no
fix). Everything here returns None for "unknown" rather than guessing or
raising, because a wrong location or date is worse than none.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from PIL import ExifTags, Image, ImageOps

_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_DATE_RE = re.compile(r"^\s*(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


def _to_float(v: Any) -> Optional[float]:
    """float from a Pillow IFDRational, an int/float, or a (numerator,
    denominator) pair; None for anything non-finite (e.g. x/0)."""
    try:
        if isinstance(v, (tuple, list)) and len(v) == 2:
            num, den = float(v[0]), float(v[1])
            if den == 0:
                return None
            f = num / den
        else:
            f = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if math.isfinite(f) else None


def dms_to_decimal(dms: Any, ref: Any, positive: str, negative: str) -> Optional[float]:
    """Degrees/minutes/seconds (1-3 components) + hemisphere ref -> signed
    decimal degrees. `ref` may be str or bytes, any case; it must be one of
    `positive`/`negative` (a missing or unrecognised ref returns None rather
    than silently assuming N/E)."""
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "ignore")
    if not isinstance(ref, str):
        return None
    ref = ref.strip("\x00 ").upper()[:1]
    if ref not in (positive, negative):
        return None
    if not isinstance(dms, (tuple, list)) or not 1 <= len(dms) <= 3:
        return None
    parts = [_to_float(p) for p in dms]
    if any(p is None or p < 0 for p in parts):
        return None
    parts += [0.0] * (3 - len(parts))
    deg, minutes, seconds = parts
    if minutes >= 60 or seconds >= 60:
        return None
    value = deg + minutes / 60.0 + seconds / 3600.0
    return -value if ref == negative else value


def parse_gps(gps: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """(lat, lon) from a GPS IFD keyed by tag *name*. Both or neither: out of
    range values, a half-valid pair, and exactly (0, 0) - the value phones
    write when there was no fix - all give (None, None)."""
    lat = dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"), "N", "S")
    lon = dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"), "E", "W")
    if lat is None or lon is None:
        return None, None
    if abs(lat) > 90 or abs(lon) > 180 or (lat == 0.0 and lon == 0.0):
        return None, None
    return lat, lon


def parse_exif_date(value: Any) -> Optional[str]:
    """'YYYY:MM:DD HH:MM:SS' -> 'YYYY-MM-DDTHH:MM:SS', or None if it is not a
    real calendar date/time (camera-unset '0000:00:00 00:00:00', garbage)."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    if not isinstance(value, str):
        return None
    m = _DATE_RE.match(value.strip("\x00"))
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups())).isoformat()
    except ValueError:
        return None


def upright(img: Image.Image) -> Image.Image:
    """`img` rotated/flipped as its EXIF Orientation tag says it should be
    displayed (what `cv2.imread` does by default, so face boxes, thumbnails
    and hashes all use the same coordinates). Returned unchanged if the tag
    is missing or unreadable."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - a broken tag must not hide the photo
        return img


def extract_exif(path: Path) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Returns (capture_date_iso_or_None, lat_or_None, lon_or_None). Never
    raises and never writes to the file."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None, None, None
            # DateTimeOriginal lives in the Exif sub-IFD, not in IFD0 (which
            # only has the file-modification DateTime).
            candidates = []
            try:
                sub = exif.get_ifd(_EXIF_IFD)
                candidates.append(sub.get(0x9003))   # DateTimeOriginal
                candidates.append(sub.get(0x9004))   # DateTimeDigitized
            except Exception:  # noqa: BLE001
                pass
            candidates.append(exif.get(0x0132))      # DateTime (IFD0)
            date = next((d for d in map(parse_exif_date, candidates) if d), None)

            lat = lon = None
            try:
                gps_ifd = exif.get_ifd(_GPS_IFD)
                gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                lat, lon = parse_gps(gps)
            except Exception:  # noqa: BLE001
                pass
            return date, lat, lon
    except Exception:  # noqa: BLE001
        return None, None, None
