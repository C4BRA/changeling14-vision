"""duplicates.html: what dedup grouped in a model's picture list (data.py).

Written once, when a new model folder's index is built with dedup, so each
codebase's catalogue gets its own report: the biggest groups of identical
pictures, copies whose folders disagreed (and the folder they now share),
held-out pictures that moved into training with their copy, and the biggest
animations, with thumbnails.
"""

from __future__ import annotations

import base64
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SHOW = 24


def _thumb(path: Path, px: int = 64) -> str:
    try:
        with Image.open(path) as img:
            img = img.convert("RGBA")
    except OSError:
        return ""
    scale = max(1, px // max(img.size))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.thumbnail((px, px), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _names(dataset: Path) -> dict[str, str]:
    names = {}
    path = dataset / "catalogue" / "entities.jsonl"
    if path.is_file():
        for line in path.open(encoding="utf-8"):
            e = json.loads(line)
            if (e.get("name") or "").strip():
                names[e["id"]] = e["name"].strip()
    return names


def write_report(index: dict, dataset: Path, path: Path) -> None:
    dataset = Path(dataset)
    items, cats = index["items"], index["categories"]
    names = _names(dataset)
    e = html.escape

    def img(it, size=64):
        return f'<img src="{_thumb(dataset / it["path"])}" width="{size}" height="{size}" alt="">'

    def where(it):
        return e(f'{it["sheet"].removesuffix(".rsi")} · {it["state"]}')

    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if "pixels" in it:
            groups[it["pixels"]].append(it)
    multi = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
    extra = sum(len(g) - 1 for g in multi)
    train = [it for it in items if it["split"] == "train"]
    distinct = len({it["pixels"] for it in train if "pixels" in it})
    moved = [it for it in items if it.get("moved")]
    refiled = [g for g in multi if any("own_category" in it for it in g)]
    frames = defaultdict(list)
    for it in items:
        frames[(it["sheet"], it["state"], it["direction"])].append(it)
    anims = sorted((v for v in frames.values() if len(v) > 1), key=len, reverse=True)

    def group_card(g, refile=False):
        sheets = Counter(it["sheet"].removesuffix(".rsi") for it in g)
        states = Counter(it["state"] for it in g)
        found = sorted({names.get(x, x) for x in g[0].get("entities", []) if x in names})
        rows = [("Sheets", "; ".join(f"{e(s)} ×{n}" for s, n in sheets.most_common(3))
                 + (f" +{len(sheets) - 3} more" if len(sheets) > 3 else "")),
                ("States", ", ".join(e(s) for s, _ in states.most_common(4))
                 + (f" +{len(states) - 4}" if len(states) > 4 else "")),
                ("Folder", e(cats[g[0]["category"]])),
                ("Names", ", ".join(e(n) for n in found[:6]) + (f" +{len(found) - 6}" if len(found) > 6 else "")
                 if found else "none")]
        if refile:
            was = Counter(cats[it.get("own_category", it["category"])] for it in g)
            rows.append(("Was", " / ".join(f"{e(c)} ×{n}" for c, n in was.most_common())))
        dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
        return f'<article class="g">{img(g[0])}<div><b>{len(g)} copies</b><dl>{dl}</dl></div></article>'

    def anim_card(v):
        strip = "".join(img(it, 48) for it in v[:8])
        more = f" +{len(v) - 8} frames" if len(v) > 8 else ""
        return (f'<article><div class="strip">{strip}<span class="m">{more}</span></div>'
                f'<b>{where(v[0])}</b><br><span class="m">counts as one picture, not {len(v)}</span></article>')

    def moved_card(it):
        return f'<article class="g">{img(it)}<div>{where(it)}<br><span class="m">held out before; its copy trains</span></div></article>'

    n = len(items)
    body = f"""
<h1>Duplicate pictures</h1>
<p class="m">{e(str(dataset))} · {n:,} pictures. Identical means every pixel and the size match.</p>
<section class="stats">
<div><b>{extra:,}</b><span>extra copies of an identical picture ({extra / max(1, n):.0%})</span></div>
<div><b>{distinct:,}</b><span>distinct training pictures out of {len(train):,}</span></div>
<div><b>{len(moved):,}</b><span>held-out pictures moved into training with their copy</span></div>
<div><b>{len(refiled):,}</b><span>groups whose copies sat in different folders</span></div>
</section>
<p>Each group counts as one picture and keeps every item name of its copies. Its folder is the one most copies
have (ties: alphabetical). Each animated sprite counts once per direction. Recolours (same shape, other
colours) are never grouped.</p>
<h2>Biggest groups</h2><div class="grid">{"".join(group_card(g) for g in multi[:SHOW])}</div>
<h2>Copies that sat in different folders</h2><div class="grid">{"".join(group_card(g, True) for g in refiled[:SHOW // 2]) or "<p>None.</p>"}</div>
<h2>Held-out pictures moved into training</h2><div class="grid">{"".join(moved_card(it) for it in moved[:SHOW // 2]) or "<p>None.</p>"}</div>
<h2>Biggest animations</h2><div class="list">{"".join(anim_card(v) for v in anims[:8])}</div>
"""
    css = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duplicate pictures</title><style>
:root{--bg:#f3f4f6;--panel:#fff;--fg:#1d2330;--muted:#5d6676;--line:#d9dde4;--tile:#2a2f3a;--accent:#1f7a8c}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--panel:#1c2028;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2e3440;--tile:#0e1015;--accent:#5fc4d6;color-scheme:dark}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1040px;margin:0 auto;padding:24px 16px}
h2{margin-top:32px} .m{color:var(--muted)} p{max-width:70ch}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stats div{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;display:flex;flex-direction:column}
.stats b{font-size:1.6rem;color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}.list{display:flex;flex-direction:column;gap:10px}
article{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;min-width:0;overflow-wrap:anywhere}
.g{display:flex;gap:12px} dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:4px 0 0;font-size:.85rem}
dt{color:var(--muted)} dd{margin:0} .strip{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px}
img{image-rendering:pixelated;background:var(--tile);border-radius:4px;object-fit:contain;flex:none}
</style>"""
    path.write_text(css + body, encoding="utf-8")
