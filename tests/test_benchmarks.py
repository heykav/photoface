"""The README's measured numbers, its benchmark figure and docs/benchmarks.json
must all be what the current code produces (regenerate with
`python scripts/make_benchmarks.py`). Synthetic data only."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import pytest

import benchmarks

# Floating-point BLAS kernels differ slightly between CPUs; allow a sliver of
# slack on rates so a rounding-level difference is not a failure.
RATE_TOL = 0.02


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_benchmarks", ROOT / "scripts" / "make_benchmarks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fresh():
    return benchmarks.run()


@pytest.fixture(scope="module")
def committed():
    return json.loads((ROOT / "docs" / "benchmarks.json").read_text(encoding="utf-8"))


def _assert_close(a, b, path="result"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        tol = RATE_TOL if abs(a) <= 1 and abs(b) <= 1 else 1.0
        assert abs(a - b) <= tol, f"{path}: committed {a}, fresh {b}"
    else:
        assert a == b, f"{path}: committed {a}, fresh {b}"


def test_committed_results_match_a_fresh_run(committed, fresh):
    _assert_close(committed, fresh)


def test_readme_tables_are_generated_from_committed_results(committed):
    mod = _load_script()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert mod.render_readme(text, committed) == text


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_figure_is_generated_from_committed_results(committed, theme):
    mod = _load_script()
    svg = (ROOT / "docs" / "img" / f"clustering-benchmark-{theme}.svg").read_text(encoding="utf-8")
    assert mod.svg(committed, theme) == svg


class TestClaimsInTheReadme:
    def _row(self, fresh, name):
        return next(r for r in fresh["clustering"] if r["scenario"] == name)

    @pytest.mark.parametrize("name", ["separated", "mild overlap", "unbalanced sizes"])
    def test_easy_scenarios_recovered_exactly(self, fresh, name):
        r = self._row(fresh, name)["recluster"]
        assert r["purity_min"] == r["completeness_min"] == 1.0

    def test_near_coincident_collapses(self, fresh):
        r = self._row(fresh, "near-coincident")
        assert r["recluster"]["clusters"] == 1.0 and r["recluster"]["purity"] < 0.4

    def test_very_noisy_shatters_under_average_linkage_but_not_greedy(self, fresh):
        r = self._row(fresh, "very noisy")
        assert r["recluster"]["completeness"] < 0.5 < r["greedy"]["completeness"]
        assert r["recluster"]["clusters"] > 3 * r["greedy"]["clusters"]

    def test_greedy_worst_order_is_worse_than_its_mean(self, fresh):
        r = self._row(fresh, "moderate overlap")["greedy"]
        assert r["purity_min"] < r["purity"] - 0.1

    def test_duplicates(self, fresh):
        t = fresh["duplicates"]["transforms"]
        for name in ("resize to 50%", "resize to 200%", "JPEG quality 90", "crop 2% per side"):
            assert t[name]["within_4"] >= 0.9, name
        for name in ("JPEG quality 70", "JPEG quality 40"):
            assert t[name]["within_4"] >= 0.85, name
        for name in ("crop 10% per side", "brightness +20%", "contrast +30%"):
            assert t[name]["within_4"] <= 0.7, name
        assert t["mirrored"]["within_4"] <= 0.05
        assert fresh["duplicates"]["unrelated"]["within_4"] <= 0.01
