"""Watch set: the same held-out sprites shown with the model's guesses at every
checkpoint, so progress (and obvious failures) show at a glance in the app.

The sprites come from sheets the model never trains on, spread over kinds,
preferring things a player handles or walks past (objects, clothing,
structures) over overlays and markings. Each checkpoint folder gets the
plain 64x64 inputs at 3x and predictions.json, which the app turns into
cards: the real name, and whether the model named it, nearly named it (in
its top 5) or thought it was something else. With captions=True (the
benchmark's mistakes gallery, opened as plain files) the same verdict is
written under each picture instead.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .data import CANVAS, Augment, floor_tiles, render

WATCHSET_SIZE = 24
SCALE = 3
CAPTION_H = 42
GOOD, CLOSE, BAD = (70, 190, 90), (220, 160, 40), (215, 70, 70)
TOP = 5
PREDICTIONS = "predictions.json"
BACKGROUND = (30, 32, 37)
# Sprites a player handles or walks past. Markings, species parts, effects,
# decals, tiles, interface and the like only fill in when these run short
# (small catalogues), as do pipes and walls, which barely differ.
PREFERRED = ("Clothing/", "Objects/", "Structures/", "Mobs/Animals", "Mobs/Pets")
NOT_PREFERRED = ("Objects/Tiles", "Structures/Piping", "Structures/Walls")


def preferred(category: str) -> bool:
    return category.startswith(PREFERRED) and not category.startswith(NOT_PREFERRED)


def pick_watchset(index: dict, seed: int, n: int = WATCHSET_SIZE, named: bool = False) -> list[dict]:
    """Up to n unseen sprites (first direction, first frame), spread over categories,
    from the PREFERRED top folders first.
    With named=True, only sprites an entity draws, so each has a true name to show.
    Falls back to held-out views when a small catalogue has no unseen sheets."""
    rng = random.Random(seed)
    split = "unseen" if index["counts"].get("unseen") else "newview"
    by_cat: dict[int, list[dict]] = {}
    for it in index["items"]:
        if (it["split"] == split and it["direction"] <= 0 and it["path"].endswith("_f0.png")
                and not it["worn"] and (not named or it.get("entities"))):
            by_cat.setdefault(it["category"], []).append(it)
    for items in by_cat.values():
        rng.shuffle(items)
    picked: list[dict] = []
    first = [c for c in sorted(by_cat) if preferred(index["categories"][c])]
    others = [c for c in sorted(by_cat) if c not in first]
    for cats in (first, others):
        rng.shuffle(cats)
        while len(picked) < n and any(by_cat[c] for c in cats):
            for c in cats:
                if by_cat[c] and len(picked) < n:
                    picked.append(by_cat[c].pop())
    return picked


def _font():
    try:
        return ImageFont.load_default(size=11)
    except TypeError:          # Pillow < 10.1
        return ImageFont.load_default()


def _short(text: str, width: int = 30, keep_start: bool = False) -> str:
    if len(text) <= width:
        return text
    return text[:width - 1] + "…" if keep_start else "…" + text[-(width - 1):]


def _top_names(sims: torch.Tensor, names: list[str], k: int = TOP) -> list[tuple[str, float]]:
    """The k best distinct names: several entity texts can share one name."""
    out: list[tuple[str, float]] = []
    values, rows = sims.topk(min(len(names), k * 8))
    for value, row in zip(values.tolist(), rows.tolist()):
        if names[row].lower() not in (n.lower() for n, _ in out):
            out.append((names[row], value))
            if len(out) == k:
                break
    return out


def _verdict(rank: int | None, guesses: list[dict]) -> tuple[str, tuple]:
    if rank == 1:
        return "Got it", GOOD
    if rank:
        return f"Close: right name was guess {rank}", CLOSE
    return f"Thought it was: {guesses[0]['name']}" if guesses else "No guess", BAD


@torch.no_grad()
def predict_watchset(model, dataset: Path, index: dict, items: list[dict], device, seed: int = 14,
                     text=None) -> tuple[list[np.ndarray], list[dict]]:
    """The input canvases and, per sprite, its real name(s), the model's top
    names and whether the real one is among them (rank 1-5, or None). Names
    match case-insensitively, so any entity sharing the sprite's name counts.
    Without text matching (or a name), kind is "group" and the folder group
    stands in for the name."""
    plain = Augment(offset=False, zoom=0.0, rotate=0.0, floor=0.0, background=BACKGROUND)
    canvases = [render(it, dataset, random.Random(seed + i), [], plain) for i, it in enumerate(items)]
    if not canvases:
        return [], []
    was_training = model.training
    model.eval()
    out = model(torch.from_numpy(np.stack(canvases)).to(device))
    if was_training:
        model.train()
    cat_p = out["category"].softmax(1).cpu()
    names = getattr(model, "text_names", None)
    sims = None
    if names and model.text_dim and len(model.text_vectors):
        g = torch.nn.functional.normalize(out["global"], dim=-1)
        sims = torch.sigmoid((g @ model.text_embeddings().T) * model.logit_scale.exp() + model.logit_bias).cpu()
    top_cats = cat_p.topk(min(TOP, cat_p.shape[1]), dim=1)
    rows = []
    for i, it in enumerate(items):
        true = sorted({names[r] for r in (text.rows_for(it.get("entities", [])) if text else [])}) \
            if sims is not None else []
        kind = "name" if true else "group"
        if true:
            guesses = [{"name": n, "score": round(v, 4)} for n, v in _top_names(sims[i], names)]
            wanted = {n.lower() for n in true}
        else:
            # No name to check against: the folder group is the answer.
            true = [index["categories"][it["category"]]]
            guesses = [{"name": index["categories"][c], "score": round(v, 4)}
                       for v, c in zip(top_cats.values[i].tolist(), top_cats.indices[i].tolist())]
            wanted = {true[0].lower()}
        rank = next((k + 1 for k, g in enumerate(guesses) if g["name"].lower() in wanted), None)
        rows.append({"kind": kind, "name": true[0], "names": true, "guesses": guesses, "rank": rank,
                     "group": index["categories"][it["category"]], "sheet": it["sheet"], "state": it["state"]})
    return canvases, rows


def draw_watchset(model, dataset: Path, index: dict, items: list[dict], device, folder: Path,
                  seed: int = 14, text=None, captions: bool = False) -> list[Path]:
    """Write the watch set into folder: one picture per sprite plus
    predictions.json. With captions=True each picture also gets its verdict
    written underneath, for viewing outside the app."""
    folder.mkdir(parents=True, exist_ok=True)
    canvases, rows = predict_watchset(model, dataset, index, items, device, seed, text)
    font = _font() if captions else None
    paths = []
    for i, (it, arr, row) in enumerate(zip(items, canvases, rows)):
        img = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB")
        img = img.resize((CANVAS * SCALE, CANVAS * SCALE), Image.NEAREST)
        if captions:
            verdict, colour = _verdict(row["rank"], row["guesses"])
            out = Image.new("RGB", (img.width + 4, img.height + 4 + CAPTION_H), (24, 24, 28))
            d = ImageDraw.Draw(out)
            d.rectangle((0, 0, img.width + 3, img.height + 3), fill=colour)
            out.paste(img, (2, 2))
            y = img.height + 6
            d.text((4, y), _short(row["name"], keep_start=True), fill=(220, 220, 220), font=font)
            d.text((4, y + 12), _short(verdict, keep_start=True), fill=colour, font=font)
            d.text((4, y + 24), _short(row["group"], keep_start=True), fill=(150, 150, 160), font=font)
            img = out
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(it["sheet"]).stem + "-" + it["state"])
        path = folder / f"{i:02d}-{name[:40]}.png"
        img.save(path)
        paths.append(path)
        row["image"] = path.name
    tmp = folder / (PREDICTIONS + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "top": TOP, "items": rows}, indent=1))
    tmp.replace(folder / PREDICTIONS)
    return paths
