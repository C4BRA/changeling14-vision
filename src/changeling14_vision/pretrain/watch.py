"""Watch set: the same held-out sprites drawn with the model's guesses at every
checkpoint, so progress (and obvious failures) show at a glance in the app.

The sprites come from sheets the model never trains on, one or two from each
category. Each image is the 96x96 input at 2x, with a caption: the true
category, the predicted one with its confidence, and the training sprite the
model thinks it looks most like. The frame is green when the category is right.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .data import CANVAS, Augment, floor_tiles, render

WATCHSET_SIZE = 24
SCALE = 2
CAPTION_H = 42
GOOD, BAD = (70, 190, 90), (215, 70, 70)
# Worn and in-hand overlays are hard to judge alone; the watch set shows the objects themselves.
WORN = ("inhand-", "equipped-", "wielded-")


def pick_watchset(index: dict, seed: int, n: int = WATCHSET_SIZE) -> list[dict]:
    """Up to n unseen sprites (first direction, first frame), spread over categories.
    Falls back to held-out views when a small catalogue has no unseen sheets."""
    rng = random.Random(seed)
    split = "unseen" if index["counts"].get("unseen") else "newview"
    by_cat: dict[int, list[dict]] = {}
    for it in index["items"]:
        if (it["split"] == split and it["direction"] <= 0 and it["path"].endswith("_f0.png")
                and not it["state"].startswith(WORN)):
            by_cat.setdefault(it["category"], []).append(it)
    for items in by_cat.values():
        rng.shuffle(items)
    picked: list[dict] = []
    cats = sorted(by_cat)
    rng.shuffle(cats)
    while len(picked) < n and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop())
    return picked


def _font():
    try:
        return ImageFont.load_default(size=11)
    except TypeError:          # Pillow < 10.1
        return ImageFont.load_default()


def _short(text: str, width: int = 30) -> str:
    return text if len(text) <= width else "…" + text[-(width - 1):]


@torch.no_grad()
def draw_watchset(model, dataset: Path, index: dict, items: list[dict], device, folder: Path,
                  seed: int = 14) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    floors = floor_tiles(dataset, index)
    canvases = [render(it, dataset, random.Random(seed + i), floors, Augment(zoom=0.0, rotate=0.0))
                for i, it in enumerate(items)]
    if not canvases:
        return []
    was_training = model.training
    model.eval()
    cat_logits, state_logits, _ = model(torch.from_numpy(np.stack(canvases)).to(device))
    if was_training:
        model.train()
    cat_p = cat_logits.softmax(1).cpu()
    state_best = state_logits.argmax(1).cpu()
    font = _font()
    paths = []
    for i, (it, arr) in enumerate(zip(items, canvases)):
        conf, pred = cat_p[i].max(0)
        ok = int(pred) == it["category"]
        img = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB")
        img = img.resize((CANVAS * SCALE, CANVAS * SCALE), Image.NEAREST)
        out = Image.new("RGB", (img.width + 4, img.height + 4 + CAPTION_H), (24, 24, 28))
        ImageDraw.Draw(out).rectangle((0, 0, img.width + 3, img.height + 3), fill=GOOD if ok else BAD)
        out.paste(img, (2, 2))
        d = ImageDraw.Draw(out)
        like = index["states"][int(state_best[i])].replace(".rsi|", "/") if index["states"] else "?"
        y = img.height + 6
        d.text((4, y), _short(f"is   {index['categories'][it['category']]}"), fill=(200, 200, 200), font=font)
        d.text((4, y + 12), _short(f"says {index['categories'][int(pred)]} {float(conf):.0%}"),
               fill=GOOD if ok else BAD, font=font)
        d.text((4, y + 24), _short(f"like {like.split('/', 1)[-1]}"), fill=(150, 150, 160), font=font)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(it["sheet"]).stem + "-" + it["state"])
        path = folder / f"{i:02d}-{name[:40]}.png"
        out.save(path)
        paths.append(path)
    return paths
