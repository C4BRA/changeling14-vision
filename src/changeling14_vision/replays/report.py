"""A page that draws each recorded picture's answer key on top of it.

Open report.html next to the frames folder to check that the boxes sit on the
sprites: green boxes are seen, grey dashed boxes are behind a wall, and the
followed player is outlined in yellow.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


def write_report(recording: Path, limit: int = 200) -> Path:
    recording = Path(recording).expanduser()
    frames = recording / "frames"
    keys = sorted(frames.glob("*.json"))[:limit]
    parts = []
    for key_path in keys:
        key = json.loads(key_path.read_text(encoding="utf-8"))
        w, h = key["width"], key["height"]
        boxes = []
        for e in sorted(key["entities"], key=lambda e: e["draw_depth"]):
            x0, y0, x1, y1 = e["box"]
            if e["subject"]:
                style = 'stroke="#ffd400" stroke-width="2"'
            elif e["blocked"]:
                style = 'stroke="#999" stroke-width="1" stroke-dasharray="3 3"'
            else:
                style = 'stroke="#3c3" stroke-width="1"'
            label = html.escape(f"{e['name']} ({e['prototype']})")
            boxes.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" {style}>'
                         f"<title>{label}</title></rect>")
        seen = sum(1 for e in key["entities"] if not e["blocked"])
        parts.append(
            f'<figure><div class="pic" style="aspect-ratio:{w}/{h}">'
            f'<img src="frames/{html.escape(key["picture"])}">'
            f'<svg viewBox="0 0 {w} {h}">{"".join(boxes)}</svg></div>'
            f'<figcaption>{html.escape(key["picture"])}: {w}×{h}, {len(key["entities"])} sprites, '
            f'{seen} not behind walls, {key["time"]:.0f} s into the round</figcaption></figure>')
    page = f"""<!doctype html><meta charset="utf-8"><title>Replay answer keys</title>
<style>
body{{font:14px system-ui;background:#111;color:#ddd;margin:16px}}
figure{{margin:0 0 24px}} .pic{{position:relative;max-width:100%;width:960px}}
.pic img,.pic svg{{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}}
</style>
<h1>{html.escape(recording.name)}</h1>
<p>Green: seen. Grey dashed: behind a wall. Yellow: the followed player. Hover a box for its name.</p>
{"".join(parts)}"""
    out = recording / "report.html"
    out.write_text(page, encoding="utf-8")
    return out
