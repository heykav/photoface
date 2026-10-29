#!/usr/bin/env python3
"""Generate the README artwork in docs/img (no new dependencies).

    QT_QPA_PLATFORM=offscreen python scripts/make_docs_art.py [--out docs/img]

Writes hero-{dark,light}.svg and pipeline-{dark,light}.svg (plain SVG text,
hand-templated below), and social-preview.png (1280x640, GitHub's recommended
size for a repository social preview). The PNG is rendered with Qt (already a
dependency) and embeds docs/img/people-overview.png, so run
scripts/make_screenshots.py first.

The avatars in the artwork are faceless geometric tokens - no real people. The
pipeline diagram restates analyzer.py / clustering.py / database.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent

SERIF = "Georgia, 'DejaVu Serif', serif"
MONO = "'DejaVu Sans Mono', Menlo, Consolas, monospace"
FONT = MONO  # body text in the diagrams is monospace; titles use SERIF

THEMES = {
    "dark": dict(bg="#090A09", panel="#0E100E", line="#1F231F", text="#E9ECE6",
                 muted="#8A918A", accent="#00FF66", accent2="#B6FF2E", warn="#F5A524",
                 ok="#00FF66", badge_text="#04120A", token_bg="#151815", red="#FF6B5E"),
    "light": dict(bg="#F4F5F2", panel="#FFFFFF", line="#D7DAD3", text="#14170F",
                  muted="#5B635B", accent="#0B8A3E", accent2="#5E8A00", warn="#B35309",
                  ok="#0B8A3E", badge_text="#FFFFFF", token_bg="#ECEEE8", red="#C23B2E"),
}
# same fictional avatar colours as scripts/make_screenshots.py
TOKENS = [("#e96076", "ring"), ("#2eb0a8", "triangle"), ("#f0ac30", "square"),
          ("#8e68e0", "diamond"), ("#4c9eec", "star"), ("#84be48", "plus")]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def emblem(kind: str, cx: float, cy: float, s: float) -> str:
    f = 'fill="#ffffff"'
    if kind == "ring":
        return f'<circle cx="{cx}" cy="{cy}" r="{s * .78}" fill="none" stroke="#fff" stroke-width="{s * .28}"/>'
    if kind == "triangle":
        return f'<polygon {f} points="{cx},{cy - s} {cx + s},{cy + s * .8} {cx - s},{cy + s * .8}"/>'
    if kind == "square":
        return f'<rect {f} x="{cx - s * .8}" y="{cy - s * .8}" width="{s * 1.6}" height="{s * 1.6}"/>'
    if kind == "diamond":
        return f'<polygon {f} points="{cx},{cy - s} {cx + s},{cy} {cx},{cy + s} {cx - s},{cy}"/>'
    if kind == "star":
        import math
        pts = []
        for k in range(10):
            r = s if k % 2 == 0 else s * .45
            a = -math.pi / 2 + k * math.pi / 5
            pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
        return f'<polygon {f} points="{" ".join(pts)}"/>'
    t = s * .32
    return (f'<rect {f} x="{cx - s}" y="{cy - t}" width="{s * 2}" height="{s * 2 * .32}"/>'
            f'<rect {f} x="{cx - t}" y="{cy - s}" width="{s * .64}" height="{s * 2}"/>')


def token_card(x: float, y: float, w: float, h: float, color: str, kind: str, t: dict) -> str:
    """A faceless avatar card: head + shoulders + face box + emblem."""
    cx, r = x + w / 2, w * .2
    cy = y + h * .36
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{t["token_bg"]}" '
        f'stroke="{t["line"]}"/>'
        f'<path d="M{cx - w * .36} {y + h - 1} A{w * .36} {h * .3} 0 0 1 {cx + w * .36} {y + h - 1} Z" '
        f'fill="{color}" opacity=".55"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'
        f'<rect x="{cx - r - 5}" y="{cy - r - 5}" width="{2 * r + 10}" height="{2 * r + 10}" rx="4" '
        f'fill="none" stroke="{color}" stroke-width="2.5"/>'
        + emblem(kind, round(cx, 1), round(cy, 1), r * .42)
    )


def svg_open(w: int, h: int, title: str, desc: str, t: dict) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">'
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
            f'<rect width="{w}" height="{h}" fill="{t["bg"]}"/>')


def chrome(w: int, h: int, title: str, t: dict) -> str:
    """Terminal-window frame: rounded panel, title bar and three dots."""
    return (f'<rect x="24" y="24" width="{w - 48}" height="{h - 48}" rx="12" fill="{t["panel"]}" stroke="{t["line"]}"/>'
            f'<path d="M24 64H{w - 24}" stroke="{t["line"]}"/>'
            f'<circle cx="46" cy="44" r="5.5" fill="{t["red"]}"/><circle cx="66" cy="44" r="5.5" fill="{t["accent2"]}"/>'
            f'<circle cx="86" cy="44" r="5.5" fill="{t["accent"]}"/>'
            f'<text x="{w / 2}" y="49" text-anchor="middle" font-family="{MONO}" font-size="12" fill="{t["muted"]}">{esc(title)}</text>')


NO_MOTION = "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"


def hero(t: dict) -> str:
    w, h = 1280, 440
    out = [svg_open(w, h, "photoface",
                    "photoface: a terminal-style banner. Six faceless avatar tokens, an illustration "
                    "with no real people, regroup into three people. Tagline: find the people in "
                    "your photo library, without uploading a single photo.", t)]
    out.append(chrome(w, h, "photoface ~ analyze ./photos", t))
    out.append(f'<text x="72" y="118" font-family="{MONO}" font-size="14" fill="{t["accent"]}">$ photoface analyze ./photos'
               f'</text><rect class="cur" x="320" y="105" width="9" height="16" fill="{t["accent"]}"/>')
    out.append(f'<text x="72" y="204" font-family="{SERIF}" font-size="78" font-weight="700" fill="{t["text"]}">photoface</text>')
    out.append(f'<text x="74" y="256" font-family="{SERIF}" font-size="28" fill="{t["text"]}">Find the people in your photo library.</text>')
    out.append(f'<text x="74" y="296" font-family="{SERIF}" font-size="28" font-style="italic" fill="{t["accent"]}">Without uploading a single photo.</text>')
    out.append(f'<text x="74" y="350" font-family="{MONO}" font-size="13" fill="{t["muted"]}">PySide6 desktop app | YuNet + SFace | SQLite | runs locally</text>')
    out.append(f'<text x="74" y="372" font-family="{MONO}" font-size="12" fill="{t["muted"]}">Illustration: synthetic avatar tokens, not real people.</text>')
    # three people, two photos each. Resting state is grouped; the loop scatters and regroups.
    cw, ch, gx, gy = 96, 116, 22, 18
    x0, y0 = 760, 112
    scatter = [(2, 1), (0, 0), (1, 1), (2, 0), (0, 1), (1, 0)]   # interleaved slots, (col,row)
    css = []
    for i in range(6):
        person, copy = i // 2, i % 2
        col, kind = TOKENS[person]
        gxp, gyp = x0 + person * (cw + gx), y0 + copy * (ch + gy)
        sc, sr = scatter[i]
        dx, dy = (x0 + sc * (cw + gx)) - gxp, (y0 + sr * (ch + gy)) - gyp
        css.append(f'.t{i}{{animation:m{i} 9s ease-in-out infinite}}'
                   f'@keyframes m{i}{{0%,58%{{transform:translate(0,0)}}66%,78%{{transform:translate({dx}px,{dy}px)}}90%,100%{{transform:translate(0,0)}}}}')
        out.append(f'<g class="t{i}">{token_card(gxp, gyp, cw, ch, col, kind, t)}</g>')
    for person in range(3):
        out.append(f'<text x="{x0 + person * (cw + gx) + cw / 2}" y="{y0 + 2 * ch + gy + 30}" text-anchor="middle" '
                   f'font-family="{MONO}" font-size="12" fill="{t["muted"]}">person {person + 1}</text>')
    out.append(f'<rect x="1090" y="80" width="132" height="28" rx="14" fill="none" stroke="{t["ok"]}" stroke-width="1.5"/>'
               f'<text x="1156" y="99" text-anchor="middle" font-family="{MONO}" font-size="13" font-weight="700" fill="{t["ok"]}">100% local</text>')
    css.append(f'.cur{{animation:bl 1.2s steps(1) infinite}}@keyframes bl{{50%{{opacity:0}}}}{NO_MOTION}')
    out.insert(1, f'<style>{"".join(css)}</style>')
    out.append("</svg>")
    return "".join(out)


STAGES = [
    ("1", "Scan", "analyzer.py",
     ["Walk the folder recursively", "Skip unchanged (mtime + size)", "Read EXIF date/GPS, dHash"]),
    ("2", "Detect", "YuNet",
     ["cv2.FaceDetectorYN", "Runs on a copy up to 1600 px", "Boxes scaled back to full size"]),
    ("3", "Embed", "SFace",
     ["cv2.FaceRecognizerSF", "Align + crop each face", "One 128-d vector per face"]),
    ("4", "Greedy online|clustering", "clustering.py",
     ["Cosine sim. to each person's", "running mean: join if > 0.363", "else start a new person"]),
    ("5", "Average-linkage|recluster", "clustering.py",
     ["Re-cluster every face together", "Merge while avg. sim. >= t", "One transaction, stable ids"]),
    ("6", "Pinned corrections|preserved", "clustering.py",
     ["Hand-edited faces are anchors", "Never moved by reclustering", "Two pinned people never merge"]),
    ("7", "SQLite", "database.py",
     ["photoface.db on your disk", "photos, persons, faces, tags", "Embeddings stored as float32"]),
]


def pipeline(t: dict) -> str:
    w, h = 1280, 780
    bw, bh, gap = 250, 200, 50
    xs = [65 + i * (bw + gap) for i in range(4)]
    y1, y2 = 160, 420
    out = [svg_open(w, h, "photoface analysis pipeline",
                    "Scan folder, detect faces with YuNet, embed with SFace, greedy online "
                    "clustering, average-linkage reclustering, pinned corrections preserved, "
                    "stored in SQLite. Everything runs locally.", t)]
    out.insert(1, f'<style>.fl{{stroke-dasharray:7 6;animation:fl 1.4s linear infinite}}@keyframes fl{{to{{stroke-dashoffset:-13}}}}{NO_MOTION}</style>')
    out.append(chrome(w, h, "photoface ~ pipeline", t))
    out.append('<g transform="translate(0,44)">')
    out.append(f'<text x="40" y="52" font-family="{SERIF}" font-size="32" font-weight="700" '
               f'fill="{t["text"]}">How photoface analyzes a folder</text>')
    out.append(f'<rect x="20" y="84" width="1240" height="566" rx="18" fill="none" '
               f'stroke="{t["ok"]}" stroke-width="1.5" stroke-dasharray="10 7" opacity=".7"/>')
    out.append(f'<text x="44" y="122" font-family="{FONT}" font-size="15" font-weight="700" '
               f'letter-spacing="1.5" fill="{t["ok"]}">EVERYTHING BELOW RUNS ON YOUR MACHINE</text>')

    def box(i: int, x: int, y: int) -> None:
        num, title, sub, lines = STAGES[i]
        color = t["warn"] if num == "6" else t["accent"]
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="14" fill="{t["panel"]}" '
                   f'stroke="{color if num == "6" else t["line"]}" stroke-width="{2 if num == "6" else 1}"/>')
        out.append(f'<circle cx="{x + 28}" cy="{y + 30}" r="15" fill="{color}"/>'
                   f'<text x="{x + 28}" y="{y + 36}" text-anchor="middle" font-family="{FONT}" '
                   f'font-size="16" font-weight="700" fill="{t["badge_text"]}">{num}</text>')
        tl = title.split("|")
        for j, line in enumerate(tl):
            out.append(f'<text x="{x + 52}" y="{y + 36 + j * 22}" font-family="{SERIF}" font-size="19" '
                       f'font-weight="700" fill="{t["text"]}">{esc(line)}</text>')
        off = 22 * (len(tl) - 1)
        out.append(f'<text x="{x + 18}" y="{y + 68 + off}" font-family="{MONO}" font-size="13" '
                   f'fill="{color}">{esc(sub)}</text>')
        for k, line in enumerate(lines):
            out.append(f'<text x="{x + 18}" y="{y + 104 + off + k * 26}" font-family="{FONT}" '
                       f'font-size="14" fill="{t["muted"]}">{esc(line)}</text>')

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        c = t["muted"]
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2.5" class="fl"/>')
        if x2 != x1:   # horizontal
            d = 1 if x2 > x1 else -1
            out.append(f'<polygon fill="{c}" points="{x2},{y2} {x2 - 11 * d},{y2 - 7} {x2 - 11 * d},{y2 + 7}"/>')
        else:
            out.append(f'<polygon fill="{c}" points="{x2},{y2} {x2 - 7},{y2 - 11} {x2 + 7},{y2 - 11}"/>')

    for i in range(4):
        box(i, xs[i], y1)
    for i in range(3):
        arrow(xs[i] + bw + 4, y1 + bh / 2, xs[i + 1] - 4, y1 + bh / 2)
    # second row runs right-to-left under row one: 5 under 4, 6 under 3, 7 under 2
    box(4, xs[3], y2)
    box(5, xs[2], y2)
    box(6, xs[1], y2)
    arrow(xs[3] + bw / 2, y1 + bh + 4, xs[3] + bw / 2, y2 - 4)
    arrow(xs[3] - 4 + 0, y2 + bh / 2, xs[2] + bw + 4, y2 + bh / 2)
    arrow(xs[2] - 4, y2 + bh / 2, xs[1] + bw + 4, y2 + bh / 2)
    # privacy note in the free slot
    x, y = xs[0], y2
    out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="14" fill="none" '
               f'stroke="{t["ok"]}" stroke-width="1.5"/>')
    out.append(f'<text x="{x + 18}" y="{y + 36}" font-family="{SERIF}" font-size="20" '
               f'font-weight="700" fill="{t["ok"]}">Private by design</text>')
    for k, line in enumerate(["Photos, embeddings and the", "database stay on this disk.",
                              "No photoface server exists.", "Only network use: the one-", "time model download."]):
        out.append(f'<text x="{x + 18}" y="{y + 68 + k * 24}" font-family="{FONT}" font-size="12.5" '
                   f'fill="{t["muted"]}">{esc(line)}</text>')
    out.append(f'<text x="40" y="680" font-family="{FONT}" font-size="13" fill="{t["muted"]}">'
               'Read from analyzer.py, clustering.py and database.py. t = similarity threshold '
               '(default 0.363, adjustable in Settings).</text>')
    out.append("</g></svg>")
    return "".join(out)


def social_bg_svg() -> str:
    t = THEMES["dark"]
    w, h = 1280, 640
    o = [svg_open(w, h, "photoface social preview", "photoface", t)]
    o.append(chrome(w, h, "photoface ~ analyze ./photos", t))
    o.append(f'<text x="72" y="124" font-family="{MONO}" font-size="14" fill="{t["accent"]}">$ photoface analyze ./photos</text>')
    o.append(f'<text x="70" y="236" font-family="{SERIF}" font-size="88" font-weight="700" fill="{t["text"]}">photoface</text>')
    for k, line in enumerate(["Find the people in your", "photo library. Without", "uploading a photo."]):
        o.append(f'<text x="72" y="{306 + k * 42}" font-family="{SERIF}" font-size="31" '
                 f'font-style="{"italic" if k == 2 else "normal"}" fill="{t["accent"] if k == 2 else t["text"]}">{line}</text>')
    o.append(f'<text x="72" y="470" font-family="{MONO}" font-size="14" fill="{t["muted"]}">Desktop app | YuNet + SFace | SQLite</text>')
    o.append(f'<rect x="72" y="500" width="150" height="36" rx="18" fill="none" '
             f'stroke="{t["ok"]}" stroke-width="1.6"/><text x="147" y="523" text-anchor="middle" '
             f'font-family="{MONO}" font-size="14" font-weight="700" fill="{t["ok"]}">100% local</text>')
    o.append(f'<text x="640" y="604" font-family="{MONO}" font-size="12" fill="{t["muted"]}">'
             'Screenshot of the real app on synthetic demo data, not real people.</text>')
    o.append("</svg>")
    return "".join(o)


def render_social(out: Path) -> None:
    from PySide6.QtCore import QByteArray, QRectF
    from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap
    from PySide6.QtSvg import QSvgRenderer

    img = QImage(1280, 640, QImage.Format_RGB32)
    p = QPainter(img)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
    QSvgRenderer(QByteArray(social_bg_svg().encode())).render(p)
    shot = QPixmap(str(out / "people-overview.png"))
    if shot.isNull():
        raise SystemExit("run scripts/make_screenshots.py first (people-overview.png missing)")
    rect = QRectF(640, 92, 560, shot.height() * 560 / shot.width())
    path = QPainterPath()
    path.addRoundedRect(rect, 14, 14)
    p.fillPath(path.translated(0, 6), QColor(0, 0, 0, 90))
    p.setClipPath(path)
    p.drawPixmap(rect.toRect(), shot)
    p.end()
    if not img.save(str(out / "social-preview.png"), "PNG"):
        raise SystemExit("could not write social-preview.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        (args.out / f"hero-{name}.svg").write_text(hero(t) + "\n", encoding="utf-8")
        (args.out / f"pipeline-{name}.svg").write_text(pipeline(t) + "\n", encoding="utf-8")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv)
    render_social(args.out)
    print("wrote art to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
