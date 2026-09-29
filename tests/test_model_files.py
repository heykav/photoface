import hashlib
import io
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import model_files
from model_files import ModelError, download_model

NAME = "face_detection_yunet_2023mar.onnx"
PAYLOAD = b"pretend-onnx-bytes" * 1000
GOOD = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def net(monkeypatch):
    state = {"payload": PAYLOAD, "urls": []}

    def fake_open(url):
        state["urls"].append(url)
        if isinstance(state["payload"], Exception):
            raise state["payload"]
        return FakeResp(state["payload"])

    monkeypatch.setattr(model_files, "_open", fake_open)
    return state


@pytest.fixture(autouse=True)
def no_tofu_env(monkeypatch):
    # the default policy is fail-closed; never let the caller's shell leak in
    monkeypatch.delenv(model_files.TOFU_ENV, raising=False)


@pytest.fixture
def tofu(monkeypatch):
    monkeypatch.setenv(model_files.TOFU_ENV, "1")


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setitem(model_files.MODELS, NAME,
                        {**model_files.MODELS[NAME], "sha256": GOOD})


def test_pinned_hash_match_installs_atomically(tmp_path, net, pinned):
    dest = download_model(NAME, tmp_path, log=lambda *_: None)
    assert dest.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.part"))


def test_pinned_hash_mismatch_installs_nothing(tmp_path, net, pinned):
    net["payload"] = PAYLOAD + b"tampered"
    with pytest.raises(ModelError, match="SHA-256 mismatch"):
        download_model(NAME, tmp_path, log=lambda *_: None)
    assert list(tmp_path.iterdir()) == []


def test_offline_gives_actionable_error_and_no_files(tmp_path, net, pinned):
    net["payload"] = urllib.error.URLError("Name or service not known")
    with pytest.raises(ModelError) as e:
        download_model(NAME, tmp_path, log=lambda *_: None)
    msg = str(e.value)
    assert "offline" in msg and NAME in msg and "https://" in msg
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_leaves_no_model(tmp_path, monkeypatch, pinned):
    class Dies(FakeResp):
        def read(self, n=-1):
            raise ConnectionResetError("cut")
    monkeypatch.setattr(model_files, "_open", lambda url: Dies(b"x"))
    with pytest.raises(ModelError):
        download_model(NAME, tmp_path, log=lambda *_: None)
    assert list(tmp_path.iterdir()) == []


def test_strict_refuses_unpinned(tmp_path, net):
    with pytest.raises(ModelError, match="no pinned"):
        download_model(NAME, tmp_path, strict=True, log=lambda *_: None)
    assert net["urls"] == []  # did not even hit the network


def test_first_use_records_hash_and_later_corruption_is_detected(tmp_path, net, tofu):
    dest = download_model(NAME, tmp_path, log=lambda *_: None)
    assert (tmp_path / (NAME + ".sha256")).read_text().split()[0] == GOOD
    dest.write_bytes(b"corrupted")
    net["urls"].clear()
    download_model(NAME, tmp_path, log=lambda *_: None)   # notices, re-downloads
    assert dest.read_bytes() == PAYLOAD and len(net["urls"]) == 1


def test_existing_good_file_not_redownloaded(tmp_path, net, pinned):
    (tmp_path / NAME).write_bytes(PAYLOAD)
    download_model(NAME, tmp_path, log=lambda *_: None)
    assert net["urls"] == []


def test_empty_leftover_file_is_replaced(tmp_path, net, tofu):
    (tmp_path / NAME).write_bytes(b"")
    download_model(NAME, tmp_path, log=lambda *_: None)
    assert (tmp_path / NAME).read_bytes() == PAYLOAD


def test_check_installed_reports_missing_and_corrupt(tmp_path, pinned):
    problems = model_files.check_installed(tmp_path)
    assert any("missing" in p and NAME in p for p in problems)
    (tmp_path / NAME).write_bytes(b"junk")
    assert any("checksum mismatch" in p for p in model_files.check_installed(tmp_path))


def test_require_installed_error_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        model_files.require_installed(tmp_path)


def test_urls_are_https_and_hashes_are_not_invented():
    for spec in model_files.MODELS.values():
        assert spec["url"].startswith("https://")
        assert spec["sha256"] is None or len(spec["sha256"]) == 64


# --- fail-closed policy -----------------------------------------------------

def test_correct_hash_accepted_and_reported_installed(tmp_path, net, pinned):
    download_model(NAME, tmp_path, log=lambda *_: None)
    problems = model_files.check_installed(tmp_path)
    assert not any(NAME in p for p in problems)
    assert not (tmp_path / (NAME + ".sha256")).exists()  # pinned: no TOFU record


def test_wrong_hash_rejected_and_existing_bad_file_removed(tmp_path, net, pinned):
    (tmp_path / NAME).write_bytes(b"tampered model")
    net["payload"] = PAYLOAD + b"also tampered"
    with pytest.raises(ModelError, match="SHA-256 mismatch"):
        download_model(NAME, tmp_path, log=lambda *_: None)
    assert list(tmp_path.iterdir()) == []  # bad file gone, no .part left


def test_pinned_size_mismatch_rejected(tmp_path, net, monkeypatch):
    monkeypatch.setitem(model_files.MODELS, NAME, {**model_files.MODELS[NAME],
                        "sha256": GOOD, "size": len(PAYLOAD) + 1})
    with pytest.raises(ModelError, match="size mismatch"):
        download_model(NAME, tmp_path, log=lambda *_: None)
    assert list(tmp_path.iterdir()) == []


def test_missing_hash_fails_closed_without_network(tmp_path, net):
    with pytest.raises(ModelError, match="--trust-on-first-use"):
        download_model(NAME, tmp_path, log=lambda *_: None)
    assert net["urls"] == []
    assert list(tmp_path.iterdir()) == []


def test_missing_hash_hand_copied_file_is_not_usable(tmp_path):
    (tmp_path / NAME).write_bytes(PAYLOAD)
    problems = model_files.check_installed(tmp_path)
    assert any(NAME in p and "cannot verify" in p for p in problems)
    with pytest.raises(ModelError):
        download_model(NAME, tmp_path, log=lambda *_: None)
    assert (tmp_path / NAME).read_bytes() == PAYLOAD  # user's file left alone


def test_tofu_opt_in_prints_hash_and_records_it(tmp_path, net):
    lines = []
    download_model(NAME, tmp_path, log=lines.append, trust_on_first_use=True)
    assert any(GOOD in line for line in lines)
    assert (tmp_path / (NAME + ".sha256")).read_text().split()[0] == GOOD
    assert not any(NAME in p for p in model_files.check_installed(tmp_path))


def test_tofu_env_opt_in_records_hand_copied_file(tmp_path, net, tofu):
    (tmp_path / NAME).write_bytes(PAYLOAD)
    download_model(NAME, tmp_path, log=lambda *_: None)
    assert net["urls"] == []
    assert (tmp_path / (NAME + ".sha256")).read_text().split()[0] == GOOD


def test_tofu_redownload_must_match_record(tmp_path, net, tofu):
    download_model(NAME, tmp_path, log=lambda *_: None)
    net["payload"] = PAYLOAD + b"changed upstream or tampered"
    with pytest.raises(ModelError, match="SHA-256 mismatch"):
        download_model(NAME, tmp_path, force=True, log=lambda *_: None)
    assert (tmp_path / NAME).read_bytes() == PAYLOAD  # verified copy untouched
    assert not list(tmp_path.glob("*.part"))


def test_strict_refuses_unpinned_even_with_tofu(tmp_path, net, tofu):
    with pytest.raises(ModelError, match="no pinned"):
        download_model(NAME, tmp_path, strict=True, log=lambda *_: None)
    assert net["urls"] == []


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("yes", True),
                                             ("", False), ("0", False), ("no", False)])
def test_tofu_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(model_files.TOFU_ENV, value)
    assert model_files.tofu_enabled() is expected


def test_download_script_flags(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import download_models
    calls = []
    monkeypatch.setattr(download_models, "download_model",
                        lambda name, **kw: calls.append(kw))
    assert download_models.main(["--trust-on-first-use"]) == 0
    assert calls and all(kw["trust_on_first_use"] is True for kw in calls)
    calls.clear()
    assert download_models.main([]) == 0
    assert all(kw["trust_on_first_use"] is None for kw in calls)  # env decides
