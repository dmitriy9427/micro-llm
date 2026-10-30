"""Рисует кривые обучения из out/log.jsonl в SVG (без matplotlib).

    python scripts/plot_log.py out/log.jsonl out/loss.svg

Синяя линия — ошибка на обучении, оранжевая — на проверке. Если они расходятся, это переобучение.
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "out/log.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "out/loss.svg"
    rows = [json.loads(l) for l in open(src) if l.strip()]
    if not rows:
        raise SystemExit("лог пуст")
    W, H, P = 640, 360, 48
    steps = [r["step"] for r in rows]
    vals = [r[k] for r in rows for k in ("train", "val")]
    x0, x1 = min(steps), max(steps) or 1
    y0, y1 = min(vals) * 0.95, max(vals) * 1.02

    def px(s):
        return P + (s - x0) / (x1 - x0 or 1) * (W - 2 * P)

    def py(v):
        return H - P - (v - y0) / (y1 - y0 or 1) * (H - 2 * P)

    def line(key, color):
        pts = " ".join(f"{px(r['step']):.1f},{py(r[key]):.1f}" for r in rows)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'

    ticks = "".join(
        f'<text x="8" y="{py(v)+4:.0f}" font-size="11" fill="#888">{v:.1f}</text>'
        f'<line x1="{P}" x2="{W-P}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="#eee"/>'
        for v in [y0 + (y1 - y0) * k / 4 for k in range(5)]
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui">'
        f'<rect width="100%" height="100%" fill="#fff"/>{ticks}'
        f'{line("train", "#2f6bff")}{line("val", "#ff8a1f")}'
        f'<text x="{P}" y="20" font-size="13" fill="#2f6bff">обучение</text>'
        f'<text x="{P+80}" y="20" font-size="13" fill="#ff8a1f">проверка</text>'
        f'<text x="{W-P-40}" y="{H-14}" font-size="11" fill="#888">шаг {x1}</text></svg>'
    )
    with open(dst, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"{len(rows)} точек → {dst}")


if __name__ == "__main__":
    main()
