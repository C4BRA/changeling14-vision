"""Turn replay recordings into a training dataset.

Reads recording folders (from `replays record`) and writes:

  <out>/dataset.json                 the frames and their split (labels.py)
  <out>/sprites.npz                  every frame's sprites, as compact arrays
  <out>/floor.npz                    floor tiles, gas and decals (floor.py)
  <out>/frames/<recording>/<name>.png

The new dataset is built in .<name>.building next to <out> and only then
swapped in; the dataset it replaces is kept as .<name>.previous (one
generation), so a failed or stopped build never leaves a half-written dataset
behind. Both are hidden, so lists of datasets can skip them.

Each picture is shrunk back to the game's own 32 pixels per tile (the client
draws at a multiple of that, divided by the eye's zoom: a zoomed-out view
shows more, smaller tiles, and comes out bigger than 672 x 480). For every sprite it keeps the box in the shrunk
picture, its centre (the middle of the box), its origin (where the entity
stands, which differs for wall-mounted things) and three flags:

  blocked   a wall is in the way (from the recorder)
  dark      the box is almost black in the picture: unlit, or outside the
            player's view cone
  faint     a floor-level sprite (puddle, cable, pipe, decal) that is under a
            floor tile or looks just like the floor around it
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

from .floor import FloorWriter
from .labels import LabelWriter, load

# NPC factions that are hostile to the crew (NanoTrasen's "hostile" list in
# Resources/Prototypes/ai_factions.yml): a non-humanoid mob in one of them is
# "hostile", other mobs are "animal".
HOSTILE_FACTIONS = {"SimpleHostile", "Syndicate", "Xeno", "Zombie", "Revolutionary", "Dragon", "AllHostile",
                    "Wizard", "Xenoborg"}


def mob_class(e: dict) -> str:
    if e.get("humanoid"):
        return "humanoid"
    if e.get("borg"):
        return "borg"
    if not e.get("mob"):
        return "none"
    factions = set(filter(None, (e.get("factions") or "").split(",")))
    return "hostile" if factions & HOSTILE_FACTIONS else "animal"

TILE = 32
DARK = 12 / 255           # mean brightness below this counts as dark
REPEAT = 4 / 255          # mean difference below this counts as a repeat
FAINT = 12 / 255          # floor-level sprites differing less than this from around them can't be seen,
FAINT_PEAK = 22 / 255     # unless their most different pixels (95th percentile) differ more than this
FLOOR_LEVEL = -11         # draw depths up to HighFloorObjects (puddles, pipes, cables, decals)
BLACK = 0.95              # pictures with more near-black pixels than this show nothing to learn


def _round_of(recording: Path, library_rows: dict[str, dict]) -> str:
    row = library_rows.get(recording.name)
    if row and row.get("roundId") is not None:
        return str(row["roundId"])
    return recording.name


VIEW = (672, 480)         # the game view at 32 px per tile, eye zoom 1


def _scale(key: dict) -> float | None:
    """Picture pixels per game pixel at 32 px tiles: the client's render scale
    divided by the eye's zoom. None for a picture that isn't the normal game
    view."""
    w, h = key["width"], key["height"]
    if w % VIEW[0] or w // VIEW[0] * VIEW[1] != h:
        return None
    zoom = float(key.get("zoom", 1))
    if not 0.25 <= zoom <= 4:
        return None
    return (w // VIEW[0]) / zoom


def _check_scale(key: dict, k: float) -> float:
    """Walls are exactly one tile wide: if the walls in view say the tiles are
    another size than k gives (more than 10% off), trust the walls."""
    widths = [e["box"][2] - e["box"][0] for e in key["entities"] if e["draw_depth"] == -8 and e.get("rotation", 0) % 90 == 0]
    if len(widths) >= 5:
        seen = float(np.median(widths)) / TILE
        if abs(seen / k - 1) > 0.1:
            return seen
    return k


def _looks_faint(pixels: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                 same: list[tuple[int, int, int, int]] = ()) -> bool:
    """Whether a box looks just like the floor around it: on average it differs
    by less than FAINT from the median colour of the tile-wide ring, and no
    real line or outline stands out either (a thin pipe or cable has a low
    average but a few strongly different pixels). Parts of the ring covered by
    the same kind of sprite (a carpet next to carpet) are left out; with too
    little ring left there is nothing to compare against, so it isn't faint."""
    h, w = pixels.shape[:2]
    rx0, ry0, rx1, ry1 = max(0, x0 - TILE), max(0, y0 - TILE), min(w, x1 + TILE), min(h, y1 + TILE)
    ring = np.ones((ry1 - ry0, rx1 - rx0), bool)
    for sx0, sy0, sx1, sy1 in same:
        ring[max(0, sy0 - ry0):max(0, sy1 - ry0), max(0, sx0 - rx0):max(0, sx1 - rx0)] = False
    ring[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
    full = (ry1 - ry0) * (rx1 - rx0) - (y1 - y0) * (x1 - x0)
    if ring.sum() < 0.25 * max(1, full):
        return False
    around = np.median(pixels[ry0:ry1, rx0:rx1][ring], axis=0)
    diff = np.abs(pixels[y0:y1, x0:x1] - around).mean(axis=2)
    return float(diff.mean()) < FAINT and float(np.percentile(diff, 95)) < FAINT_PEAK


def _decals(key: dict, s: float, pixels: np.ndarray) -> list[dict]:
    """The decals of a picture in frame pixels, marked dark, faint or blocked like sprites."""
    h, w = pixels.shape[:2]
    names = key.get("decal_names", [])
    boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for d in key.get("decals", []):
        b = [v / s for v in d["box"]]
        boxes.setdefault(d["id"], []).append((int(b[0]), int(b[1]), int(np.ceil(b[2])), int(np.ceil(b[3]))))
    out = []
    for d in key.get("decals", []):
        x0, y0, x1, y1 = (v / s for v in d["box"])
        cx0, cy0 = max(0, int(x0)), max(0, int(y0))
        cx1, cy1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        dark = float(pixels[cy0:cy1, cx0:cx1].mean()) < DARK
        faint = _looks_faint(pixels, cx0, cy0, cx1, cy1, boxes[d["id"]])
        out.append({"id": names[d["id"]], "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    "dark": dark, "faint": faint, "blocked": d["blocked"],
                    "visible": not (dark or faint or d["blocked"])})
    return out


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

    final = out
    out = final.with_name(f".{final.name}.building")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    writer = LabelWriter()
    floors = FloorWriter()
    stopped = False
    frames, repeats, blacks, other_view, zoomed, rescaled = [], 0, 0, 0, 0, 0
    sprites_total = visible_total = faint_total = 0
    keys = [(rec, k) for rec in recordings for k in sorted((rec / "frames").glob("*.json"))]
    last: dict[tuple[str, int], np.ndarray] = {}
    for n, (rec, key_path) in enumerate(keys):
        if progress:
            progress.update("build dataset", n, len(keys), key_path.name)
            if progress.stop_requested:
                stopped = True
                break
        key = json.loads(key_path.read_text(encoding="utf-8"))
        img = Image.open(key_path.with_name(key["picture"])).convert("RGB")
        s = _scale(key)
        if s is None:
            other_view += 1
            continue
        k = _check_scale(key, s)
        rescaled += k != s
        s = k
        zoomed += abs(float(key.get("zoom", 1)) - 1) > 0.01
        if abs(s - round(s)) < 1e-6 and round(s) >= 1:
            if round(s) > 1:
                img = img.reduce(int(round(s)))
        else:
            img = img.resize((max(1, round(img.width / s)), max(1, round(img.height / s))), Image.BOX)
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
        boxes: dict[str, list[tuple[int, int, int, int]]] = {}
        for e in key["entities"]:
            bx = [v / s for v in e["box"]]
            boxes.setdefault(e["prototype"], []).append(
                (int(bx[0]), int(bx[1]), int(np.ceil(bx[2])), int(np.ceil(bx[3]))))
        for e in key["entities"]:
            x0, y0, x1, y1 = (v / s for v in e["box"])
            cx0, cy0 = max(0, int(x0)), max(0, int(y0))
            cx1, cy1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
            if cx1 - cx0 < 2 or cy1 - cy0 < 2:       # (almost) outside the picture
                continue
            dark = float(pixels[cy0:cy1, cx0:cx1].mean()) < DARK
            # Floor-level things (puddles, cables, pipes, decals) can be drawn but
            # not seen: under a floor tile, dried to a speck, or floor-coloured.
            faint = bool(e.get("under_floor")) or (
                e["draw_depth"] <= FLOOR_LEVEL and _looks_faint(pixels, cx0, cy0, cx1, cy1, boxes[e["prototype"]]))
            sprites.append({
                "prototype": e["prototype"], "name": e["name"],
                "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                # The middle of what is drawn. Wall-mounted things (lights, signs) are
                # drawn on the wall but stand on the floor tile next to it (origin).
                "center": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
                "origin": [round(e["center"][0] / s, 1), round(e["center"][1] / s, 1)],
                "draw_depth": e["draw_depth"], "blocked": e["blocked"], "dark": dark,
                "faint": faint, "visible": not e["blocked"] and not dark and not faint,
                "humanoid": e["humanoid"], "alive": e["alive"], "subject": e["subject"],
                "mob_class": mob_class(e),
            })
        rel = Path("frames") / rec.name / key["picture"]
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(out / rel)
        round_id = _round_of(rec, library_rows)
        first, count = writer.add(sprites)
        sprites_total += count
        visible_total += sum(sp["visible"] for sp in sprites)
        faint_total += sum(sp["faint"] for sp in sprites)
        frame = {"picture": str(rel), "recording": rec.name, "round": round_id,
                 "split": "test" if round_id in test_rounds else "train",
                 "time": key["time"], "tick": key["tick"], "width": w, "height": h,
                 "eye_rotation": key["eye_rotation"], "zoom": key.get("zoom", 1), "first": first, "count": count}
        if "grids" in key:
            frame["floor"] = floors.add_maps(key, s, w, h)
            frame["decals"] = floors.add_decals(_decals(key, s, pixels))
        frames.append(frame)

    summary = {
        "frames": len(frames), "repeats_dropped": repeats, "black_dropped": blacks, "other_view_dropped": other_view,
        "zoomed_frames": zoomed, "scale_from_walls": rescaled,
        "floor_frames": sum("floor" in f for f in frames), "decals": floors.n_decals,
        "rounds": rounds, "test_rounds": sorted(test_rounds),
        "train_frames": sum(f["split"] == "train" for f in frames),
        "test_frames": sum(f["split"] == "test" for f in frames),
        "sprites": sprites_total, "visible_sprites": visible_total, "faint_sprites": faint_total,
    }
    if stopped:
        # The dataset already in place (if any) stays as it was.
        shutil.rmtree(out, ignore_errors=True)
        return {**summary, "stopped": True}
    writer.labels().save(out / "sprites.npz")
    if summary["floor_frames"]:
        floors.save(out / "floor.npz")
    (out / "dataset.json").write_text(json.dumps({"version": 2, "tile": TILE, "summary": summary,
                                                  "frames": frames}), encoding="utf-8")
    write_dataset_report(out)
    previous = final.with_name(f".{final.name}.previous")
    if final.exists():
        shutil.rmtree(previous, ignore_errors=True)
        final.rename(previous)
    out.rename(final)
    if progress:
        progress.update("build dataset", len(keys), len(keys), f"{len(frames)} frames", force=True)
    return summary


def write_dataset_report(out: Path, limit: int = 120) -> Path:
    import html

    from .floor import DECAL_BITS, Floor

    data, labels = load(out)
    floor = Floor.load(out)
    figs = []
    for f in data["frames"][:limit]:
        rects = []
        if floor is not None and "decals" in f:
            r = floor.decals(f)
            for d, (x0, y0, x1, y1), fl in zip(floor.decal_id[r], floor.decal_box[r], floor.decal_flags[r]):
                dash = "" if fl & DECAL_BITS["visible"] else ' stroke-dasharray="2 2"'
                rects.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
                             f'fill="none" stroke="#c6f" stroke-width="0.5"{dash}>'
                             f'<title>decal {html.escape(floor.decal_names[d])}</title></rect>')
        sprites = labels.dicts(f)
        for s in sorted(sprites, key=lambda s: s["draw_depth"]):
            x0, y0, x1, y1 = s["box"]
            colour = "#ffd400" if s["subject"] else "#3c3" if s["visible"] else "#38f" if s["dark"] else "#999"
            dash = "" if s["visible"] or s["subject"] else ' stroke-dasharray="2 2"'
            title = html.escape(f"{s['name']} ({s['prototype']})")
            rects.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" '
                         f'stroke="{colour}" stroke-width="0.7"{dash}><title>{title}</title></rect>')
        seen = sum(s["visible"] for s in sprites)
        figs.append(
            f'<figure><div class="pic" style="aspect-ratio:{f["width"]}/{f["height"]}">'
            f'<img src="{html.escape(f["picture"])}"><svg viewBox="0 0 {f["width"]} {f["height"]}">'
            f'{"".join(rects)}</svg></div><figcaption>{html.escape(f["picture"])} ({f["split"]}): '
            f'{seen} of {len(sprites)} sprites visible</figcaption></figure>')
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
<p>Green: visible. Blue dashed: dark. Grey dashed: behind a wall. Yellow: the followed player.
Purple: decals (dashed when they can't be seen).</p>
{"".join(figs)}"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
