#!/usr/bin/env python3
"""Re-run the synthetic benchmarks and regenerate everything quoting them.

    python scripts/make_benchmarks.py

Runs tests/benchmarks.py (synthetic embeddings and procedurally drawn images
only - no real photos, faces or models) and writes:

* docs/benchmarks.json - the raw results;
* docs/img/clustering-benchmark-{light,dark}.svg - the README figure;
* the README sections between the `<!-- benchmark:... -->` markers.

tests/test_benchmarks.py checks all three still match a fresh run, so the
numbers in the README cannot drift from the code or be edited by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]

from make_docs_art import FONT, THEMES, esc  # noqa: E402

JSON_PATH = ROOT / "docs" / "benchmarks.json"
README = ROOT / "README.md"
SVG_PATH = ROOT / "docs" / "img" / "clustering-benchmark-{theme}.svg"

# two-series colours, checked for colour-vision-deficiency separation and
# lightness/contrast against each theme's background
SERIES = {"light": {"recluster": "#2f5fd6", "greedy": "#b35309"},
          "dark": {"recluster": "#5b8cff", "greedy": "#d46f28"}}
LABELS = {"recluster": "final: average-linkage recluster",
          "greedy": "during analysis: greedy pass (shuffled order)"}


def _f(x: float) -> str:
    return f"{x:.2f}"


def clustering_table(res: dict) -> str:
    lines = [
        f"Threshold {res['threshold']}, {res['seeds']} seeds per scenario (greedy pass: "
        f"{res['greedy_orders']} shuffled input orders per seed). "
        "Mean over runs, worst run in parentheses.",
        "",
        "| Scenario | faces / identities | noise σ | overlap | recluster purity "
        "| recluster completeness | recluster clusters | greedy purity "
        "| greedy completeness | greedy clusters |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["clustering"]:
        a, g = r["recluster"], r["greedy"]
        lines.append(
            f"| {r['scenario']} | {r['faces']} / {r['identities']} | {r['sigma']} | "
            f"{r['overlap']} | {_f(a['purity'])} ({_f(a['purity_min'])}) | "
            f"{_f(a['completeness'])} ({_f(a['completeness_min'])}) | {a['clusters']} | "
            f"{_f(g['purity'])} ({_f(g['purity_min'])}) | "
            f"{_f(g['completeness'])} ({_f(g['completeness_min'])}) | {g['clusters']} |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%" if 0 < x < 0.1 else f"{x * 100:.0f}%"


def duplicates_table(res: dict) -> str:
    d = res["duplicates"]
    lines = [
        f"{d['scenes']} synthetic scenes; share of copies whose dHash is within "
        "k bits of the original (the app groups at k <= 4). Last row: share of "
        f"the {d['pairs']} pairs of *different* scenes within k bits (false positives).",
        "",
        "| Copy made by | median bits | worst | k <= 2 | k <= 4 | k <= 6 | k <= 8 |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = list(d["transforms"].items()) + [("*unrelated scene (false positive)*", d["unrelated"])]
    for name, v in rows:
        lines.append(f"| {name} | {v['median']:g} | {v['max']} | {_pct(v['within_2'])} | "
                     f"{_pct(v['within_4'])} | {_pct(v['within_6'])} | {_pct(v['within_8'])} |")
    return "\n".join(lines)


def svg(res: dict, theme: str) -> str:
    t, col = THEMES[theme], SERIES[theme]
    rows = res["clustering"]
    label_w, panel_w, gap = 236, 300, 56
    top, row_h, bar_h = 134, 46, 13
    w = 40 + label_w + 2 * panel_w + gap + 60
    h = top + row_h * len(rows) + 58
    title = "Clustering on synthetic embeddings"
    desc = ("Mean purity and completeness per synthetic scenario, for the final "
            "average-linkage recluster and for the greedy pass. " +
            "; ".join(f"{r['scenario']}: recluster {_f(r['recluster']['purity'])} / "
                      f"{_f(r['recluster']['completeness'])}, greedy "
                      f"{_f(r['greedy']['purity'])} / {_f(r['greedy']['completeness'])}"
                      for r in rows))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">'
         f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
         f'<rect width="{w}" height="{h}" rx="12" fill="{t["bg"]}"/>',
         f'<text x="40" y="44" font-family="{FONT}" font-size="21" font-weight="700" '
         f'fill="{t["text"]}">{esc(title)}</text>',
         f'<text x="40" y="68" font-family="{FONT}" font-size="13" fill="{t["muted"]}">'
         f'Synthetic 128-d blobs, threshold {res["threshold"]}, mean of runs. '
         'Not a measurement of real SFace accuracy.</text>']
    # legend
    lx = 40
    for key in ("recluster", "greedy"):
        o.append(f'<rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{col[key]}"/>'
                 f'<text x="{lx + 18}" y="95" font-family="{FONT}" font-size="13" '
                 f'fill="{t["text"]}">{esc(LABELS[key])}</text>')
        lx += 18 + 7.1 * len(LABELS[key]) + 28
    x0s = [40 + label_w, 40 + label_w + panel_w + gap]
    for p, (metric, name) in enumerate([("purity", "Purity"), ("completeness", "Completeness")]):
        x0 = x0s[p]
        o.append(f'<text x="{x0}" y="{top - 6}" font-family="{FONT}" font-size="13" '
                 f'font-weight="700" fill="{t["text"]}">{name}</text>')
        for tick in (0, 0.5, 1.0):
            x = x0 + tick * panel_w
            o.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + row_h * len(rows)}" '
                     f'stroke="{t["line"]}" stroke-width="1"/>'
                     f'<text x="{x}" y="{top + row_h * len(rows) + 18}" text-anchor="middle" '
                     f'font-family="{FONT}" font-size="12" fill="{t["muted"]}">{tick:g}</text>')
        for i, r in enumerate(rows):
            y = top + i * row_h + 8
            for k, key in enumerate(("recluster", "greedy")):
                v = r[key][metric]
                by = y + k * (bar_h + 2)
                bw = max(2.0, v * panel_w)
                o.append(f'<rect x="{x0}" y="{by}" width="{bw:.1f}" height="{bar_h}" rx="4" '
                         f'fill="{col[key]}"/>'
                         f'<text x="{x0 + bw + 5:.1f}" y="{by + 11}" font-family="{FONT}" '
                         f'font-size="11.5" fill="{t["text"]}">{_f(v)}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h + 8
        o.append(f'<text x="40" y="{y + 12}" font-family="{FONT}" font-size="13.5" '
                 f'font-weight="600" fill="{t["text"]}">{esc(r["scenario"])}</text>'
                 f'<text x="40" y="{y + 27}" font-family="{FONT}" font-size="11.5" '
                 f'fill="{t["muted"]}">σ {r["sigma"]}, overlap {r["overlap"]}, '
                 f'{r["faces"]} faces / {r["identities"]} ids</text>')
    o.append(f'<text x="40" y="{h - 14}" font-family="{FONT}" font-size="12" fill="{t["muted"]}">'
             'Generated by scripts/make_benchmarks.py from tests/benchmarks.py; '
             'checked by tests/test_benchmarks.py.</text>')
    o.append("</svg>")
    return "".join(o) + "\n"


def replace_block(text: str, name: str, body: str) -> str:
    start, end = f"<!-- benchmark:{name}:start -->", f"<!-- benchmark:{name}:end -->"
    a, b = text.index(start) + len(start), text.index(end)
    return text[:a] + "\n" + body + "\n" + text[b:]


def render_readme(text: str, res: dict) -> str:
    text = replace_block(text, "clustering", clustering_table(res))
    return replace_block(text, "duplicates", duplicates_table(res))


def main() -> None:
    import benchmarks
    res = benchmarks.run()
    JSON_PATH.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for theme in ("light", "dark"):
        Path(str(SVG_PATH).format(theme=theme)).write_text(svg(res, theme), encoding="utf-8")
    README.write_text(render_readme(README.read_text(encoding="utf-8"), res), encoding="utf-8")
    print(f"wrote {JSON_PATH.relative_to(ROOT)}, 2 SVGs and the README tables")


if __name__ == "__main__":
    main()
