"""Turn replay recordings into a training dataset.

Reads recording folders (from `replays record`) and writes:

  <out>/dataset.json                 the frames, their split and their sprites
  <out>/frames/<recording>/<name>.png

Each picture is shrunk back to the game's own 32 pixels per tile (the client
draws at a multiple of that). For every sprite it keeps the box in the shrunk
picture, its centre (the middle of the box), its origin (where the entity
stands, which differs for wall-mounted things) and three flags:

  blocked   a wall is in the way (from the recorder)
  dark      the box is almost black in the picture: unlit, or outside the
            player's view cone
  visible   neither of the above: the player can actually see it

Nothing is dropped for being blocked or dark, so the model also learns what
it can't see. Pictures that are nearly the same as the last kept picture from
the same player are dropped, and so are pictures that are almost entirely
black (a player in a locker or a dark room with no light). Whole rounds are held out for testing, never
single frames, because frames from one round look alike.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 32
DARK = 12 / 255           # mean brightness below this counts as dark
REPEAT = 4 / 255          # mean difference below this counts as a repeat
BLACK = 0.95              # pictures with more near-black pixels than this show nothing to learn


def _round_of(recording: Path, library_rows: dict[str, dict]) -> str:
    row = library_rows.get(recording.name)
    if row and row.get("roundId") is not None:
        return str(row["roundId"])
    return recording.name


def _scale(key: dict) -> int:
    """How many picture pixels the client drew per game pixel."""
    sizes = sorted(round(e["box"][2] - e["box"][0]) for e in key["entities"]
                   if e["prototype"].startswith(("Wall", "Floor", "Grille")) or e["draw_depth"] <= -10)
    tile = sizes[len(sizes) // 2] if sizes else TILE
    return max(1, round(tile / TILE))


def _thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize((64, 48), Image.BILINEAR), dtype=np.float32) / 255


def pick_holdout(rounds: list[str], holdout: list[str] | None) -> set[str]:
    if holdout is not None:
        return set(holdout)
    if len(rounds) < 2:
        return set()
    # The same rounds are held out however many rounds are added later.
    ranked = sorted(rounds, key=lambda r: hashlib.sha256(r.encode()).hexdigest())
    return set(ranked[:max(1, len(rounds) // 5)])


def build_dataset(recordings: list[Path], out: Path, library_rows: dict[str, dict] | None = None,
                  holdout: list[str] | None = None, progress=None) -> dict:
    out = Path(out).expanduser()
    library_rows = library_rows or {}
    recordings = [Path(r).expanduser() for r in recordings]
    rounds = sorted({_round_of(r, library_rows) for r in recordings})
    test_rounds = pick_holdout(rounds, holdout)

    frames_out = out / "frames"
    shutil.rmtree(frames_out, ignore_errors=True)
    frames, repeats, blacks = [], 0, 0
    keys = [(rec, k) for rec in recordings for k in sorted((rec / "frames").glob("*.json"))]
    last: dict[tuple[str, int], np.ndarray] = {}
    for n, (rec, key_path) in enumerate(keys):
        if progress:
            progress.update("build dataset", n, len(keys), key_path.name)
            if progress.stop_requested:
                break
        key = json.loads(key_path.read_text(encoding="utf-8"))
        img = Image.open(key_path.with_name(key["picture"])).convert("RGB")
        s = _scale(key)
        if s > 1:
            img = img.reduce(s)
        if float((np.asarray(img.convert("L")) < 10).mean()) > BLACK:
            blacks += 1
            continue
        thumb = _thumb(img)
        who = (rec.name, key["subject"])
        if who in last and float(np.abs(thumb - last[who]).mean()) < REPEAT:
            repeats += 1
            continue
        last[who] = thumb

        pixels = np.asarray(img, dtype=np.float32) / 255
        h, w = pixels.shape[:2]
        sprites = []
        for e in key["entities"]:
            x0, y0, x1, y1 = (v / s for v in e["box"])
            cx0, cy0 = max(0, int(x0)), max(0, int(y0))
            cx1, cy1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
            if cx1 - cx0 < 2 or cy1 - cy0 < 2:       # (almost) outside the picture
                continue
            dark = float(pixels[cy0:cy1, cx0:cx1].mean()) < DARK
            sprites.append({
                "prototype": e["prototype"], "name": e["name"],
                "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                # The middle of what is drawn. Wall-mounted things (lights, signs) are
                # drawn on the wall but stand on the floor tile next to it (origin).
                "center": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
                "origin": [round(e["center"][0] / s, 1), round(e["center"][1] / s, 1)],
                "draw_depth": e["draw_depth"], "blocked": e["blocked"], "dark": dark,
                "visible": not e["blocked"] and not dark,
                "humanoid": e["humanoid"], "alive": e["alive"], "subject": e["subject"],
            })
        rel = Path("frames") / rec.name / key["picture"]
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(out / rel)
        round_id = _round_of(rec, library_rows)
        frames.append({"picture": str(rel), "recording": rec.name, "round": round_id,
                       "split": "test" if round_id in test_rounds else "train",
                       "time": key["time"], "tick": key["tick"], "width": w, "height": h,
                       "eye_rotation": key["eye_rotation"], "sprites": sprites})

    summary = {
        "frames": len(frames), "repeats_dropped": repeats, "black_dropped": blacks,
        "rounds": rounds, "test_rounds": sorted(test_rounds),
        "train_frames": sum(f["split"] == "train" for f in frames),
        "test_frames": sum(f["split"] == "test" for f in frames),
        "sprites": sum(len(f["sprites"]) for f in frames),
        "visible_sprites": sum(sum(s["visible"] for s in f["sprites"]) for f in frames),
    }
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "dataset.json.tmp"
    tmp.write_text(json.dumps({"version": 1, "tile": TILE, "summary": summary, "frames": frames}),
                   encoding="utf-8")
    tmp.replace(out / "dataset.json")
    write_dataset_report(out)
    if progress:
        progress.update("build dataset", len(keys), len(keys), f"{len(frames)} frames", force=True)
    return summary


def write_dataset_report(out: Path, limit: int = 120) -> Path:
    import html

    data = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    figs = []
    for f in data["frames"][:limit]:
        rects = []
        for s in sorted(f["sprites"], key=lambda s: s["draw_depth"]):
            x0, y0, x1, y1 = s["box"]
            colour = "#ffd400" if s["subject"] else "#3c3" if s["visible"] else "#38f" if s["dark"] else "#999"
            dash = "" if s["visible"] or s["subject"] else ' stroke-dasharray="2 2"'
            title = html.escape(f"{s['name']} ({s['prototype']})")
            rects.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" '
                         f'stroke="{colour}" stroke-width="0.7"{dash}><title>{title}</title></rect>')
        seen = sum(s["visible"] for s in f["sprites"])
        figs.append(
            f'<figure><div class="pic" style="aspect-ratio:{f["width"]}/{f["height"]}">'
            f'<img src="{html.escape(f["picture"])}"><svg viewBox="0 0 {f["width"]} {f["height"]}">'
            f'{"".join(rects)}</svg></div><figcaption>{html.escape(f["picture"])} ({f["split"]}): '
            f'{seen} of {len(f["sprites"])} sprites visible</figcaption></figure>')
    s = data["summary"]
    page = f"""<!doctype html><meta charset="utf-8"><title>Replay dataset</title>
<style>
body{{font:14px system-ui;background:#111;color:#ddd;margin:16px}}
figure{{margin:0 0 24px}} .pic{{position:relative;max-width:100%;width:960px}}
.pic img,.pic svg{{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}}
</style>
<h1>Replay dataset</h1>
<p>{s["frames"]} frames ({s["train_frames"]} train, {s["test_frames"]} test from rounds
{html.escape(", ".join(s["test_rounds"]) or "none")}), {s["repeats_dropped"]} near-repeats and {s.get("black_dropped", 0)} black pictures dropped,
{s["visible_sprites"]} of {s["sprites"]} sprites visible.</p>
<p>Green: visible. Blue dashed: dark. Grey dashed: behind a wall. Yellow: the followed player.</p>
{"".join(figs)}"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
