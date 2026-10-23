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
from .detect import CELL, DetectModel, Frames, collate
from .labels import load


@dataclass
class Guess:
    box: tuple[float, float, float, float]      # x0, y0, x1, y1 in frame pixels (32 px tiles)
    name: str
    score: float                                # how sure that something is there, 0..1
    names: list[tuple[str, float]]              # top names with their match scores
    player: float = 0.0                         # how sure that a player stands here, 0..1


def load_detect_model(folder: Path, device: str = "auto") -> tuple[DetectModel, TextTable, torch.device]:
    folder = Path(folder).expanduser()
    ckpt = load_checkpoint(folder / "latest.pt")
    if ckpt.get("kind") != "detect":
        raise SystemExit(f"{folder} is not a replay-trained model (train one with 'replays train')")
    table = TextTable(folder / "text.npz")
    model = DetectModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}), table.dim)
    # The weight average is what was scored; models from before it have none.
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=False)
    dev = pick_device(device)
    model.to(dev).eval()
    model.set_text(table)
    return model, table, dev


def _names(table: TextTable) -> list[str]:
    return [t.split(". ", 1)[0] for t in table.texts]


@torch.no_grad()
def guesses(model: DetectModel, names: list[str], image: torch.Tensor, threshold: float = 0.5,
            top: int = 3, all_t: torch.Tensor | None = None) -> list[list[Guess]]:
    """image (B, 3, H, W) in [0, 1] at 32 px per tile, H and W multiples of 32."""
    out = model(image)
    gh, gw = out["grid"]
    b = image.shape[0]
    all_t = model.texts() if all_t is None else all_t
    p = torch.sigmoid(out["something"]).view(b, gh * gw)
    pl = torch.sigmoid(out["player"]).view(b, gh * gw).cpu()
    sims = out["names"] @ all_t.T * model.scale.exp() + model.bias
    best = torch.sigmoid(sims).topk(top, dim=1)
    vals, rows = best.values.view(b, gh * gw, top).cpu(), best.indices.view(b, gh * gw, top).cpu()
    box = out["box"].view(b, gh * gw, 4).cpu()
    p = p.cpu()
    result = []
    for i in range(b):
        found = []
        for cell in torch.nonzero(p[i] > threshold).flatten().tolist():
            cy, cx = divmod(cell, gw)
            ox, oy, lw, lh = box[i, cell].tolist()
            x, y = (cx + ox) * CELL, (cy + oy) * CELL
            w, h = float(np.exp(lw)) * CELL, float(np.exp(lh)) * CELL
            top_names = [(names[r], float(v)) for r, v in zip(rows[i, cell].tolist(), vals[i, cell].tolist())]
            found.append(Guess((x - w / 2, y - h / 2, x + w / 2, y + h / 2), top_names[0][0],
                               float(p[i, cell]), top_names, float(pl[i, cell])))
        result.append(found)
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
    frames = Frames(dataset, "test", table, None, source=source)
    if not len(frames):
        raise SystemExit(f"{dataset} has no held-out frames")
    step = max(1, len(frames) // limit)
    picks = list(range(0, len(frames), step))[:limit]
    all_t = model.texts()
    confused: Counter = Counter()
    right = wrong = 0
    figs = []
    for n, i in enumerate(picks):
        if progress:
            progress.update("predict", n, len(picks), frames.frames[i]["picture"])
            if progress.stop_requested:
                break
        f = frames.frames[i]
        item = collate([frames[i]])
        found = guesses(model, names, item["image"].to(dev), threshold, all_t=all_t)[0]
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
            rects.append(f'<rect class="{"g" if ok else "x"}" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
                         f'height="{y1 - y0:.1f}"><title>{html.escape(tip)} (sure {g.score:.2f})</title></rect>')
        pic = html.escape(f["picture"])
        figs.append(
            f'<figure><div class="pic" style="aspect-ratio:{f["width"]}/{f["height"]}"><img src="{pic}">'
            f'<svg viewBox="0 0 {f["width"]} {f["height"]}">{"".join(rects)}</svg></div>'
            f"<figcaption>{pic}: {len(truth)} visible sprites, {len(found)} guesses</figcaption></figure>")
    rows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{c}</td></tr>"
                   for (a, b), c in confused.most_common(40))
    page = f"""<!doctype html><meta charset="utf-8"><title>Replay predictions</title>
<base href="{dataset.as_uri()}/">
<style>
body{{font:14px system-ui;background:#111;color:#ddd;margin:16px}}
figure{{margin:0 0 24px}} .pic{{position:relative;max-width:100%;width:960px}}
.pic img,.pic svg{{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}}
rect{{fill:none;stroke-width:.8}} .t{{stroke:#3c3;stroke-dasharray:2 2}} .g{{stroke:#3af}} .x{{stroke:#f80}}
td{{padding:2px 10px}}
</style>
<h1>{html.escape(folder.name)} on held-out frames</h1>
<p>Green dashed: what is really there and visible. Blue: a guess with the right name for its square.
Orange: a guess whose name is wrong (or nothing is centred there). Hover a box for the top names.</p>
<p>{right} of {right + wrong} guesses named right ({right / max(1, right + wrong):.0%}) on {len(figs)} frames.</p>
<h2>Most common mix-ups (really there, guessed)</h2><table>{rows}</table>
{"".join(figs)}"""
    out = folder / "predictions.html"
    out.write_text(page, encoding="utf-8")
    return out


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
         name_threshold: float = 0.6) -> None:
    """Capture region (left, top, width, height in screen points; None: drag a
    box) of the game view, where one game tile is tile_px captured pixels
    (None: worked out from the first picture), and show the model's boxes and
    names in a window; a name the model is less sure of than name_threshold
    shows as "?". Press q in the window to stop. Nothing is sent to the game."""
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
            found = guesses(model, names, _tensor(frame, dev), threshold, all_t=all_t)[0]
            ms = (time.time() - t0) * 1000
            h32, w32 = frame.shape[:2]
            view = cv2.resize(frame, (w32 * show, h32 * show), interpolation=cv2.INTER_NEAREST)
            for g in found:
                x0, y0, x1, y1 = (int(v * show) for v in g.box)
                is_player = g.player >= 0.5
                cv2.rectangle(view, (x0, y0), (x1, y1), (255, 0, 255) if is_player else (0, 200, 255),
                              2 if is_player else 1)
                if labels:
                    label = g.name[:18] if g.names[0][1] >= name_threshold else "?"
                    if is_player:
                        label = f"player: {label}"
                    cv2.putText(view, label, (x0, max(10, y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(view, f"{len(found)} things, model {ms:.0f} ms", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("changeling14 live", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
