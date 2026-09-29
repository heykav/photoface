import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image

from exif_utils import dms_to_decimal, extract_exif, parse_exif_date, parse_gps


class TestDms:
    @pytest.mark.parametrize("ref,sign", [("N", 1), ("S", -1), ("n", 1), ("s", -1),
                                          (b"N\x00", 1), (b"S", -1), (" S ", -1)])
    def test_latitude_hemisphere(self, ref, sign):
        assert dms_to_decimal((10.0, 30.0, 0.0), ref, "N", "S") == sign * 10.5

    @pytest.mark.parametrize("ref,sign", [("E", 1), ("W", -1), (b"W", -1)])
    def test_longitude_hemisphere(self, ref, sign):
        assert dms_to_decimal((122, 15, 0), ref, "E", "W") == sign * 122.25

    def test_rational_pairs_and_seconds(self):
        # (num, den) pairs as older Pillow returns them
        v = dms_to_decimal(((37, 1), (25, 1), (1919, 100)), "N", "N", "S")
        assert v == pytest.approx(37 + 25 / 60 + 19.19 / 3600)

    @pytest.mark.parametrize("bad", [
        None, "x", (), (1, 2, 3, 4), (float("nan"), 0, 0), (float("inf"), 0, 0),
        ((1, 0), 0, 0), (-1, 0, 0), (10, 60, 0), (10, 0, 61), ("a", "b", "c")])
    def test_malformed_values_give_none(self, bad):
        assert dms_to_decimal(bad, "N", "N", "S") is None

    @pytest.mark.parametrize("ref", [None, "", "X", "E", 5])
    def test_missing_or_wrong_ref_gives_none(self, ref):
        assert dms_to_decimal((10, 0, 0), ref, "N", "S") is None

    def test_short_tuples_are_padded(self):
        assert dms_to_decimal((45,), "N", "N", "S") == 45.0
        assert dms_to_decimal((45, 30), "S", "N", "S") == -45.5


class TestParseGps:
    def _g(self, lat, lref, lon, lonref):
        return {"GPSLatitude": lat, "GPSLatitudeRef": lref,
                "GPSLongitude": lon, "GPSLongitudeRef": lonref}

    def test_southern_western_hemispheres(self):
        lat, lon = parse_gps(self._g((33, 51, 0), "S", (151, 12, 0), "E"))
        assert lat == pytest.approx(-33.85) and lon == pytest.approx(151.2)
        lat, lon = parse_gps(self._g((40, 0, 0), "N", (74, 0, 0), "W"))
        assert (lat, lon) == (40.0, -74.0)

    def test_half_valid_pair_is_dropped(self):
        assert parse_gps(self._g((40, 0, 0), "N", None, "W")) == (None, None)
        assert parse_gps({"GPSLatitude": (1, 0, 0)}) == (None, None)

    @pytest.mark.parametrize("lat,lon", [((91, 0, 0), (10, 0, 0)), ((10, 0, 0), (181, 0, 0))])
    def test_out_of_range(self, lat, lon):
        assert parse_gps(self._g(lat, "N", lon, "E")) == (None, None)

    def test_null_island_is_treated_as_no_fix(self):
        assert parse_gps(self._g((0, 0, 0), "N", (0, 0, 0), "E")) == (None, None)

    def test_boundaries_allowed(self):
        assert parse_gps(self._g((90, 0, 0), "N", (180, 0, 0), "W")) == (90.0, -180.0)


class TestDate:
    def test_valid(self):
        assert parse_exif_date("2021:07:04 15:30:59") == "2021-07-04T15:30:59"
        assert parse_exif_date(b"2021:07:04 15:30:59\x00") == "2021-07-04T15:30:59"

    @pytest.mark.parametrize("bad", ["0000:00:00 00:00:00", "2021:13:01 00:00:00",
                                     "2021:02:30 00:00:00", "garbage", "", None, 5,
                                     "2021:07:04 25:00:00"])
    def test_invalid(self, bad):
        assert parse_exif_date(bad) is None


def _save(path, exif=None, size=(8, 8)):
    img = Image.new("RGB", size, (10, 20, 30))
    if exif is None:
        img.save(path, "JPEG")
    else:
        img.save(path, "JPEG", exif=exif)
    return path


class TestExtractExif:
    def test_no_exif(self, tmp_path):
        assert extract_exif(_save(tmp_path / "a.jpg")) == (None, None, None)

    def test_missing_file_and_corrupt_file(self, tmp_path):
        assert extract_exif(tmp_path / "nope.jpg") == (None, None, None)
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"\xff\xd8\xff\xe1not really exif" * 3)
        assert extract_exif(bad) == (None, None, None)

    def test_date_original_preferred_over_modification_date(self, tmp_path):
        """Regression: DateTimeOriginal is in the Exif sub-IFD, which the old
        code never read, so it always reported the (edit-time) DateTime."""
        ex = Image.Exif()
        ex[0x0132] = "2024:01:01 00:00:00"                       # DateTime (IFD0)
        ex.get_ifd(0x8769)[0x9003] = "2019:05:06 07:08:09"       # DateTimeOriginal
        d, _, _ = extract_exif(_save(tmp_path / "a.jpg", ex))
        assert d == "2019-05-06T07:08:09"

    def test_invalid_original_falls_back(self, tmp_path):
        ex = Image.Exif()
        ex[0x0132] = "2024:01:01 10:00:00"
        ex.get_ifd(0x8769)[0x9003] = "0000:00:00 00:00:00"
        d, _, _ = extract_exif(_save(tmp_path / "a.jpg", ex))
        assert d == "2024-01-01T10:00:00"

    def test_gps_south_west_round_trip(self, tmp_path):
        ex = Image.Exif()
        gps = ex.get_ifd(0x8825)
        gps[1], gps[2] = "S", (33.0, 51.0, 0.0)
        gps[3], gps[4] = "W", (70.0, 30.0, 0.0)
        _, lat, lon = extract_exif(_save(tmp_path / "a.jpg", ex))
        assert lat == pytest.approx(-33.85) and lon == pytest.approx(-70.5)

    def test_gps_without_ref_is_ignored(self, tmp_path):
        ex = Image.Exif()
        gps = ex.get_ifd(0x8825)
        gps[2], gps[4] = (33.0, 51.0, 0.0), (70.0, 30.0, 0.0)
        assert extract_exif(_save(tmp_path / "a.jpg", ex))[1:] == (None, None)
