"""Use a replay-trained model: decode its guesses, compare them with answer
keys, or draw them live over the game.

  predict_report   held-out frames with the model's boxes and names next to
                   the true ones (predictions.html in the model folder)
  live             capture part of the screen, run the model on it, and show
                   the boxes and names in a window
"""

from __future__ import annotations

import html
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..model.encoder import Encoder
from ..pretrain.text import TextTable
from ..pretrain.train import load_checkpoint, pick_device
from .detect import CELL, DetectModel, Frames, collate, load_matching
from .floor import Floor
from .labels import MOB_CLASSES, load


@dataclass
class Guess:
    box: tuple[float, float, float, float]      # x0, y0, x1, y1 in frame pixels (32 px tiles)
    name: str
    score: float                                # how sure that something is there, 0..1
    names: list[tuple[str, float]]              # top names with their match scores
    player: float = 0.0                         # how sure that a player stands here, 0..1
    mob: str = "none"                           # the square's mob class (labels.MOB_CLASSES), first slot only
    slot: int = 0                               # 0 for the topmost sprite of its square, then 1, 2, ...


@dataclass
class FloorGuess:
    tiles: list[list[str]]                      # floor tile per square, rows of the frame
    gas: dict[tuple[int, int], list[str]]       # (row, column) -> gases there
    decals: dict[tuple[int, int], list[str]]    # (row, column) -> decals centred there


def load_detect_model(folder: Path, device: str = "auto") -> tuple[DetectModel, TextTable, torch.device]:
    folder = Path(folder).expanduser()
    ckpt = load_checkpoint(folder / "latest.pt")
    if ckpt.get("kind") != "detect":
        raise SystemExit(f"{folder} is not a replay-trained model (train one with 'replays train')")
    table = TextTable(folder / "text.npz")
    model = DetectModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}), table.dim,
                        ckpt.get("vocab"), ckpt.get("slots", 1))
    # The weight average is what was scored; models from before it have none.
    load_matching(model, ckpt.get("ema") or ckpt["model"])
    dev = pick_device(device)
    model.to(dev).eval()
    model.set_text(table)
    return model, table, dev


def _names(table: TextTable) -> list[str]:
    return [t.split(". ", 1)[0] for t in table.texts]


@torch.no_grad()
def guesses(model: DetectModel, names: list[str], image: torch.Tensor, threshold: float = 0.5,
            top: int = 3, all_t: torch.Tensor | None = None, out: dict | None = None) -> list[list[Guess]]:
    """image (B, 3, H, W) in [0, 1] at 32 px per tile, H and W multiples of 32.
    A square's first box shows when "something" passes threshold, each further
    one when "more" does too."""
    out = model(image) if out is None else out
    gh, gw = out["grid"]
    b, cells, k = image.shape[0], gh * gw, model.slots
    all_t = model.texts() if all_t is None else all_t
    p = torch.sigmoid(out["something"]).view(b, cells).cpu()
    more = torch.sigmoid(out["more"]).view(b, cells, k - 1).cpu()
    pl = torch.sigmoid(out["player"]).view(b, cells).cpu()
    mob = out["mob"].argmax(1).view(b, cells).cpu()
    sims = torch.einsum("nke,te->nkt", out["names"], all_t) * model.scale.exp() + model.bias
    best = torch.sigmoid(sims).topk(top, dim=2)
    vals, rows = best.values.view(b, cells, k, top).cpu(), best.indices.view(b, cells, k, top).cpu()
    box = out["slot_box"].view(b, cells, k, 4).cpu()
    result = []
    for i in range(b):
        found = []
        for cell in torch.nonzero(p[i] > threshold).flatten().tolist():
            cy, cx = divmod(cell, gw)
            for j in range(k):
                if j and more[i, cell, j - 1] <= threshold:
                    break
                ox, oy, lw, lh = box[i, cell, j].tolist()
                x, y = (cx + ox) * CELL, (cy + oy) * CELL
                w, h = float(np.exp(lw)) * CELL, float(np.exp(lh)) * CELL
                top_names = [(names[r], float(v)) for r, v in zip(rows[i, cell, j].tolist(), vals[i, cell, j].tolist())]
                sure = float(p[i, cell]) if j == 0 else float(more[i, cell, j - 1])
                found.append(Guess((x - w / 2, y - h / 2, x + w / 2, y + h / 2), top_names[0][0], sure, top_names,
                                   float(pl[i, cell]), MOB_CLASSES[int(mob[i, cell])] if j == 0 else "none", j))
        result.append(found)
    return result


@torch.no_grad()
def guesses_fast(model: DetectModel, names: list[str], out: dict, threshold: float = 0.5, top: int = 3,
                 all_t: torch.Tensor | None = None, scale: float | None = None, bias: float | None = None) -> list[Guess]:
    """guesses() for one image, faster: only squares and slots that pass the
    threshold are compared with the entity texts, the box decoding is
    vectorised, and only the small result is copied off the device. Same
    Guess list, in the same order. out may come from the torch model or from
    the Core ML export (tensors on any device); scale and bias default to the
    model's."""
    gh, gw = out["grid"]
    k = model.slots
    all_t = model.texts() if all_t is None else all_t
    dev = all_t.device
    p = torch.sigmoid(out["something"].reshape(-1).to(dev))
    cells = torch.nonzero(p > threshold).flatten()
    if not len(cells):
        return []
    more = torch.sigmoid(out["more"].reshape(gh * gw, k - 1).to(dev))[cells]
    # Slot j shows only if every slot before it did (guesses() breaks at the first miss).
    ok = torch.ones(len(cells), k, dtype=torch.bool, device=dev)
    if k > 1:
        ok[:, 1:] = torch.cumprod((more > threshold).to(torch.int8), dim=1).bool()
    ci, sj = torch.nonzero(ok, as_tuple=True)                    # cell-major, then slot: guesses()'s order
    cell = cells[ci]
    q = out["names"].reshape(gh * gw, k, -1).to(dev)[cell, sj]   # (M, E)
    s_ = scale if scale is not None else float(model.scale.exp())
    b_ = bias if bias is not None else float(model.bias)
    vals, rows = (q @ all_t.T * s_ + b_).topk(top, dim=1)
    vals = torch.sigmoid(vals)                                   # sigmoid keeps the order, so topk first
    sure = torch.where(sj == 0, p[cell], more[ci, (sj - 1).clamp(min=0)])
    box = out["slot_box"].reshape(gh * gw, k, 4).to(dev)[cell, sj]
    cy, cx = cell // gw, cell % gw
    x = (cx + box[:, 0]) * CELL
    y = (cy + box[:, 1]) * CELL
    w, h = torch.exp(box[:, 2]) * CELL, torch.exp(box[:, 3]) * CELL
    xyxy = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
    pl = torch.sigmoid(out["player"].reshape(-1).to(dev))[cell]
    mob = out["mob"].reshape(gh * gw, -1).to(dev).argmax(1)[cell]
    xyxy, vals, rows, sure, pl, mob, sj = (t.cpu().tolist() for t in (xyxy, vals, rows, sure, pl, mob, sj))
    found = []
    for i in range(len(sj)):
        top_names = [(names[r], v) for r, v in zip(rows[i], vals[i])]
        found.append(Guess(tuple(xyxy[i]), top_names[0][0], sure[i], top_names, pl[i],
                           MOB_CLASSES[mob[i]] if sj[i] == 0 else "none", sj[i]))
    return found


@torch.no_grad()
def floor_guesses(model: DetectModel, out: dict, threshold: float = 0.5) -> list[FloorGuess] | None:
    """The floor tile, gases and decals the model sees in each square, or None
    for a model without floor heads."""
    if "tile" not in out:
        return None
    gh, gw = out["grid"]
    v = model.vocab
    tiles = out["tile"].argmax(1).view(-1, gh, gw).cpu()
    gas = (torch.sigmoid(out["gas"]) > threshold).view(-1, gh * gw, out["gas"].shape[1]).cpu() if "gas" in out else None
    decal = (torch.sigmoid(out["decal"]) > threshold).view(-1, gh * gw, out["decal"].shape[1]).cpu() \
        if "decal" in out else None
    result = []
    for i in range(tiles.shape[0]):
        found_gas, found_decals = {}, {}
        for table, found, names in ((gas, found_gas, v["gas_names"]), (decal, found_decals, v["decal_names"])):
            if table is None:
                continue
            for cell, j in torch.nonzero(table[i]).tolist():
                found.setdefault(divmod(cell, gw), []).append(names[j])
        result.append(FloorGuess([[v["tile_names"][t] for t in row] for row in tiles[i].tolist()],
                                 found_gas, found_decals))
    return result


def _true_name(sprite: dict, table: TextTable, names: list[str]) -> str:
    """The catalogue name the model is taught for this sprite: its prototype's
    text, not the in-game name (a player's character name can't be seen)."""
    row = table.row_of_entity.get(sprite["prototype"])
    return (names[row] if row is not None else sprite["name"]).lower()


def predict_report(folder: Path, dataset: Path, limit: int = 60, threshold: float = 0.5,
                   device: str = "auto", progress=None) -> Path:
    """predictions.html: held-out frames, true visible sprites (green) beside
    the model's guesses (orange), plus the names it confuses most."""
    folder, dataset = Path(folder).expanduser(), Path(dataset).expanduser()
    model, table, dev = load_detect_model(folder, device)
    names = _names(table)
    source = load(dataset)
    labels = source[1]
    frames = Frames(dataset, "test", table, None, source=source, floor=Floor.load(dataset), vocab=model.vocab)
    if not len(frames):
        raise SystemExit(f"{dataset} has no held-out frames")
    step = max(1, len(frames) // limit)
    picks = list(range(0, len(frames), step))[:limit]
    all_t = model.texts()
    confused: Counter = Counter()
    tile_confused: Counter = Counter()
    right = wrong = tile_right = tile_seen = 0
    figs = []
    for n, i in enumerate(picks):
        if progress:
            progress.update("predict", n, len(picks), frames.frames[i]["picture"])
            if progress.stop_requested:
                break
        f = frames.frames[i]
        item = collate([frames[i]])
        with torch.no_grad():
            out = model(item["image"].to(dev))
        found = guesses(model, names, item["image"].to(dev), threshold, all_t=all_t, out=out)[0]
        floor = floor_guesses(model, out, threshold)
        floor_note = ""
        if floor is not None:
            truth_tiles = item["tile"].tolist()
            gwid = len(floor[0].tiles[0])
            here_right = here_seen = 0
            for cell, t in enumerate(truth_tiles):
                if t < 0:
                    continue
                guess_tile = floor[0].tiles[cell // gwid][cell % gwid]
                true_tile = model.vocab["tile_names"][t]
                here_seen += 1
                here_right += guess_tile == true_tile
                if guess_tile != true_tile:
                    tile_confused[(true_tile, guess_tile)] += 1
            tile_right += here_right
            tile_seen += here_seen
            floor_note = f"; floor tiles {here_right} of {here_seen} right"
        gh, gw = f["height"] // CELL, f["width"] // CELL
        truth = [s for s in labels.dicts({"first": f["dataset_first"], "count": f["count"]}) if s["visible"]]
        # Name check per square: the guessed name against the names truly centred there.
        by_cell: dict[int, set[str]] = {}
        for s in truth:
            cx, cy = (s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2
            if 0 <= cx < gw * CELL and 0 <= cy < gh * CELL:
                by_cell.setdefault(int(cy // CELL) * gw + int(cx // CELL), set()).add(_true_name(s, table, names))
        rects = []
        for s in sorted(truth, key=lambda s: s["draw_depth"]):
            x0, y0, x1, y1 = s["box"]
            rects.append(f'<rect class="t" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}">'
                         f"<title>{html.escape(_true_name(s, table, names))} ({html.escape(s['name'])})</title></rect>")
        for g in found:
            x0, y0, x1, y1 = g.box
            cell = int(((y0 + y1) / 2) // CELL) * gw + int(((x0 + x1) / 2) // CELL)
            ok = g.name.lower() in by_cell.get(cell, set())
            right += ok
            wrong += not ok
            if not ok and by_cell.get(cell):
                confused[(sorted(by_cell[cell])[0], g.name)] += 1
            tip = ", ".join(f"{nm} {v:.2f}" for nm, v in g.names)
            if g.mob != "none":
                tip += f"; mob: {g.mob}"
            rects.append(f'<rect class="{"g" if ok else "x"}{" s" if g.slot else ""}" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
                         f'height="{y1 - y0:.1f}"><title>{html.escape(tip)} (sure {g.score:.2f})</title></rect>')
        pic = html.escape(f["picture"])
        figs.append(
            f'<figure><div class="pic" style="aspect-ratio:{f["width"]}/{f["height"]}"><img src="{pic}">'
            f'<svg viewBox="0 0 {f["width"]} {f["height"]}">{"".join(rects)}</svg></div>'
            f"<figcaption>{pic}: {len(truth)} visible sprites, {len(found)} guesses{floor_note}</figcaption></figure>")
    rows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{c}</td></tr>"
                   for (a, b), c in confused.most_common(40))
    tile_rows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{c}</td></tr>"
                        for (a, b), c in tile_confused.most_common(20))
    floor_part = (f"<h2>Floor tiles</h2><p>{tile_right} of {tile_seen} visible squares' floor named right "
                  f"({tile_right / max(1, tile_seen):.0%}).</p><table>{tile_rows}</table>") if tile_seen else ""
    page = f"""<!doctype html><meta charset="utf-8"><title>Replay predictions</title>
<base href="{dataset.as_uri()}/">
<style>
body{{font:14px system-ui;background:#111;color:#ddd;margin:16px}}
figure{{margin:0 0 24px}} .pic{{position:relative;max-width:100%;width:960px}}
.pic img,.pic svg{{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}}
rect{{fill:none;stroke-width:.8}} .t{{stroke:#3c3;stroke-dasharray:2 2}} .g{{stroke:#3af}} .x{{stroke:#f80}}
.s{{stroke-dasharray:1 1}}
td{{padding:2px 10px}}
</style>
<h1>{html.escape(folder.name)} on held-out frames</h1>
<p>Green dashed: what is really there and visible. Blue: a guess with the right name for its square.
Orange: a guess whose name is wrong (or nothing is centred there). Dotted: a second or third sprite in
the same square. Hover a box for the top names and mob class.</p>
<p>{right} of {right + wrong} guesses named right ({right / max(1, right + wrong):.0%}) on {len(figs)} frames.</p>
<h2>Most common mix-ups (really there, guessed)</h2><table>{rows}</table>
{floor_part}
{"".join(figs)}"""
    out = folder / "predictions.html"
    out.write_text(page, encoding="utf-8")
    return out


MOB_COLOURS = {"humanoid": (255, 0, 255), "borg": (255, 255, 0), "animal": (0, 220, 0), "hostile": (0, 0, 255)}


def _pick_region(screen, cv2) -> tuple[int, int, int, int]:
    """Let the user drag a box around the game view on a screenshot."""
    mon = screen.monitors[1]
    shot = np.asarray(screen.grab(mon))[:, :, :3]
    ratio = shot.shape[1] / mon["width"]             # Retina screens grab 2 pixels per point
    small = cv2.resize(shot, (shot.shape[1] // 2, shot.shape[0] // 2))
    x, y, w, h = cv2.selectROI("drag a box around the game view, then press Enter", small, False)
    cv2.destroyAllWindows()
    if w < 10 or h < 10:
        raise SystemExit("no region picked")
    k = 2 / ratio
    return int(mon["left"] + x * k), int(mon["top"] + y * k), int(w * k), int(h * k)


def _to_model(shot: np.ndarray, tile_px: float, cv2) -> np.ndarray:
    scale = CELL / tile_px
    frame = cv2.resize(shot, (max(CELL, int(shot.shape[1] * scale)), max(CELL, int(shot.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    return frame[:frame.shape[0] // CELL * CELL, :frame.shape[1] // CELL * CELL]


def _tensor(frame: np.ndarray, dev) -> torch.Tensor:
    rgb = frame[:, :, ::-1].astype(np.float32) / 255
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(dev)


@torch.no_grad()
def _guess_tile(model: DetectModel, shot: np.ndarray, dev, cv2) -> float:
    """The screen size of one game tile: the scale at which the model is most
    sure about the most squares."""
    best, best_px = -1.0, 64.0
    for px in np.arange(24, 160, 4):
        frame = _to_model(shot, float(px), cv2)
        if frame.shape[0] < 3 * CELL or frame.shape[1] < 3 * CELL:
            continue
        p = torch.sigmoid(model(_tensor(frame, dev))["something"])
        score = float((p > 0.5).float().mean() * p[p > 0.5].mean()) if (p > 0.5).any() else 0.0
        if score > best:
            best, best_px = score, float(px)
    return best_px


def live(folder: Path, region: tuple[int, int, int, int] | None, tile_px: float | None,
         threshold: float = 0.5, device: str = "auto", fps: float = 10, labels: bool = True,
         name_threshold: float = 0.6, show_floor: bool = False) -> None:
    """Capture region (left, top, width, height in screen points; None: drag a
    box) of the game view, where one game tile is tile_px captured pixels
    (None: worked out from the first picture), and show the model's boxes and
    names in a window; a name the model is less sure of than name_threshold
    shows as "?". Mobs are coloured by class: players magenta, borgs cyan,
    animals green, hostile mobs red. show_floor also writes each square's
    floor tile and tints squares with gas. Press q in the window to stop.
    Nothing is sent to the game."""
    try:
        import cv2
        import mss
    except ImportError as exc:
        raise SystemExit('the live view needs: pip install -e ".[live]"') from exc
    model, table, dev = load_detect_model(folder, device)
    names = _names(table)
    all_t = model.texts()
    show = 2                                   # draw at 2x the model's size so names are readable
    with mss.mss() as screen:
        if region is None:
            region = _pick_region(screen, cv2)
        left, top, width, height = region
        box = {"left": left, "top": top, "width": width, "height": height}
        if tile_px is None:
            tile_px = _guess_tile(model, np.asarray(screen.grab(box))[:, :, :3], dev, cv2)
        print(f"live: region {left},{top} {width}x{height} points, one tile = {tile_px:.0f} captured pixels "
              "(pass --tile-px to fix it). Press q in the window to stop.", flush=True)
        last = 0.0
        while True:
            wait = 1 / fps - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            frame = _to_model(np.asarray(screen.grab(box))[:, :, :3], tile_px, cv2)
            t0 = time.time()
            x = _tensor(frame, dev)
            with torch.no_grad():
                out = model(x)
            found = guesses(model, names, x, threshold, all_t=all_t, out=out)[0]
            floor = floor_guesses(model, out) if show_floor else None
            ms = (time.time() - t0) * 1000
            h32, w32 = frame.shape[:2]
            view = cv2.resize(frame, (w32 * show, h32 * show), interpolation=cv2.INTER_NEAREST)
            if floor is not None:
                # Gas tints its squares; each square's floor tile is written small in its corner.
                for (r, c), gases in floor[0].gas.items():
                    tint = view[r * CELL * show:(r + 1) * CELL * show, c * CELL * show:(c + 1) * CELL * show]
                    tint[:] = (tint * 0.6 + np.array((200, 0, 200)) * 0.4).astype(np.uint8)
                for r, row in enumerate(floor[0].tiles):
                    for c, tile in enumerate(row):
                        cv2.putText(view, tile[:10], (c * CELL * show + 2, (r + 1) * CELL * show - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (160, 160, 160), 1, cv2.LINE_AA)
            for g in found:
                x0, y0, x1, y1 = (int(v * show) for v in g.box)
                is_player = g.player >= 0.5
                colour = MOB_COLOURS.get(g.mob) or ((255, 0, 255) if is_player else (0, 200, 255))
                cv2.rectangle(view, (x0, y0), (x1, y1), colour, 2 if g.mob != "none" or is_player else 1)
                if labels:
                    label = g.name[:18] if g.names[0][1] >= name_threshold else "?"
                    if g.mob not in ("none", "humanoid"):
                        label = f"{g.mob}: {label}"
                    elif is_player:
                        label = f"player: {label}"
                    cv2.putText(view, label, (x0, max(10, y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(view, f"{len(found)} things, model {ms:.0f} ms", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("changeling14 live", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
