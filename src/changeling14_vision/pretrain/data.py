"""Single-sprite training samples from the M0 sprite catalogue.

Each sample is one sprite cell drawn onto a 64x64 canvas (2x2 tiles) at a
random offset, so the encoder learns that objects need not sit centred on a
tile. Backgrounds are plain colours, noise or real floor tiles. Zoom is by
whole numbers and rotation is nearest-neighbour, so pixel art stays crisp.

Labels, all from the catalogue, never hand-made:
  components what the entities that draw the sprite are made of (Item,
             Clothing, MeleeWeapon, Door, Edible...): several per sprite,
             only for sprites an entity draws
  text       the name and description of an entity that draws the sprite,
             matched through a frozen sentence model (pretrain/text.py)
  category   coarse group from the sheet's folder, e.g. Objects/Tools
  state      the (sheet, state) pair: the sprite's exact visual identity
  direction  0-3 for four-direction states, else ignored

Splits hold out whole sheets, never single images:
  unseen   10% of sheets, never trained on; scored on category
  newview  one image held out of each multi-image training state (another
           direction or animation frame); scored on state

Held and worn overlays (inhand-*, equipped-*) are a third of the catalogue
but alone are small fragments of a mob. They train at a quarter of the
weight and are left out of the scores.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CANVAS = 64
MAX_SPRITE = 64
MIN_SPRITE = 8
MIN_CATEGORY_SHEETS = 8
# Screen HUD and editor-only art: out of scope for an in-world encoder.
SKIP_PREFIXES = ("Interface/", "Markers/", "Actions/", "Parallaxes/", "LobbyScreens/", "Logo/",
                 "Tips/", "deprecated")
INDEX_VERSION = 5
MAX_TEXTS = 8           # text rows kept per sprite for scoring
# A component becomes a label when at least this many linked sprites have it,
# and at most this share of them (Sprite, Physics and the like say nothing).
MIN_COMPONENT_STATES = 30
MAX_COMPONENT_SHARE = 0.6
# Overlays drawn on a mob (held or worn). Alone they are small fragments, so
# they are sampled less often and left out of the headline scores.
WORN = ("inhand-", "equipped-", "wielded-")
WORN_WEIGHT = 0.25


class DataError(Exception):
    pass


def category_of(sheet: str) -> str:
    parts = sheet.split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def is_worn(state: str) -> bool:
    return state.startswith(WORN) or "-inhand-" in state


def _holdout(sheet: str, seed: int, fraction: float) -> bool:
    h = hashlib.sha256(f"{seed}:{sheet}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < fraction


def build_index(dataset: Path, seed: int = 14, unseen_fraction: float = 0.1, progress=None) -> dict:
    """Read catalogue/images.jsonl, drop empty and out-of-scope images, assign
    labels and splits. Opens every image once to skip fully transparent ones."""
    images_file = dataset / "catalogue" / "images.jsonl"
    if not images_file.is_file():
        raise DataError(f"{dataset} has no catalogue/images.jsonl; build the sprite catalogue (M0) first")
    rows = [json.loads(line) for line in images_file.open(encoding="utf-8")]
    if not (dataset / "images").is_dir():
        raise DataError(f"{dataset} has no images/ folder; rebuild M0 without --no-images")
    sheet_dirs: dict[str, int] = {}
    kept = []
    opaque: set[str] = set()
    for i, r in enumerate(rows):
        if progress and i % 500 == 0:
            progress.update("index sprites", i, len(rows), f"{len(kept):,} usable")
        if r["sheet"].startswith(SKIP_PREFIXES):
            continue
        if not (MIN_SPRITE <= r["width"] <= MAX_SPRITE and MIN_SPRITE <= r["height"] <= MAX_SPRITE):
            continue
        try:
            with Image.open(dataset / r["path"]) as img:
                alpha = img.getchannel("A") if img.mode == "RGBA" else None
                if alpha is not None and alpha.getbbox() is None:
                    continue
                if r["sheet"].startswith("Tiles/") and (alpha is None or alpha.getextrema()[0] == 255):
                    opaque.add(r["path"])
        except OSError:
            continue
        sheet_dirs[f"{r['sheet']}|{r['state']}"] = max(sheet_dirs.get(f"{r['sheet']}|{r['state']}", 0),
                                                        r["direction"] + 1)
        kept.append(r)
    if not kept:
        raise DataError(f"no usable sprites in {dataset}")

    sheets = sorted({r["sheet"] for r in kept})
    raw_cat = {s: category_of(s) for s in sheets}
    cat_sheets = Counter(raw_cat.values())
    # Small groups fold into their top folder, e.g. Objects/Vehicles -> Objects.
    cat_of = {s: (c if cat_sheets[c] >= MIN_CATEGORY_SHEETS else s.split("/")[0]) for s, c in raw_cat.items()}
    categories = sorted(set(cat_of.values()))
    unseen = {s for s in sheets if _holdout(s, seed, unseen_fraction)}

    train_states = sorted({f"{r['sheet']}|{r['state']}" for r in kept if r["sheet"] not in unseen})
    state_ids = {k: i for i, k in enumerate(train_states)}
    by_state: dict[str, list[int]] = {}
    items = []
    for r in kept:
        key = f"{r['sheet']}|{r['state']}"
        directions = sheet_dirs[key]
        split = "unseen" if r["sheet"] in unseen else "train"
        items.append({"path": r["path"], "sheet": r["sheet"], "state": r["state"],
                      "direction": r["direction"] if directions == 4 else -1,
                      "category": categories.index(cat_of[r["sheet"]]),
                      "state_id": state_ids.get(key, -1), "split": split,
                      "w": r["width"], "h": r["height"], "worn": is_worn(r["state"]),
                      "opaque": r["path"] in opaque})
        if split == "train":
            by_state.setdefault(key, []).append(len(items) - 1)
    rng = random.Random(seed)
    for idxs in by_state.values():
        if len(idxs) >= 2:
            items[rng.choice(idxs)]["split"] = "newview"
    components = _attach_components(dataset, items)
    counts = Counter(it["split"] for it in items)
    return {"version": INDEX_VERSION, "dataset": str(dataset), "seed": seed,
            "categories": categories, "states": train_states, "components": components,
            "counts": dict(counts), "items": items}


def _attach_components(dataset: Path, items: list[dict]) -> list[str]:
    """Give each sprite the components of the entities that draw it (M0's
    sprite_links and entities), as indices into the returned vocabulary.
    Sprites no entity draws get None and are left out of that loss."""
    cat = dataset / "catalogue"
    if not (cat / "sprite_links.jsonl").is_file() or not (cat / "entities.jsonl").is_file():
        for it in items:
            it["components"] = None
            it["entities"] = []
        return []
    entity_components = {}
    for line in (cat / "entities.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        entity_components[e["id"]] = e.get("components") or []
    per_state: dict[tuple[str, str], set[str]] = {}
    users: dict[tuple[str, str], list[str]] = {}
    for line in (cat / "sprite_links.jsonl").open(encoding="utf-8"):
        link = json.loads(line)
        comps: set[str] = set()
        for user in link["entities"]:
            comps.update(entity_components.get(user["entity"], ()))
        users[(link["sheet"], link["state"])] = sorted({u["entity"] for u in link["entities"]})
        if comps:
            per_state[(link["sheet"], link["state"])] = comps
    linked = {(it["sheet"], it["state"]) for it in items} & per_state.keys()
    freq = Counter(c for key in linked for c in per_state[key])
    vocab = sorted(c for c, n in freq.items()
                   if n >= MIN_COMPONENT_STATES and n <= MAX_COMPONENT_SHARE * max(1, len(linked)))
    ids = {c: i for i, c in enumerate(vocab)}
    for it in items:
        comps = per_state.get((it["sheet"], it["state"]))
        it["components"] = sorted(ids[c] for c in comps if c in ids) if comps is not None else None
        it["entities"] = users.get((it["sheet"], it["state"]), [])
    return vocab


def load_or_build_index(dataset: Path, out: Path, seed: int = 14, progress=None) -> dict:
    """The index lives in the training folder (never inside the M0 output) so a
    resumed run keeps the same label numbering even if M0 is rebuilt."""
    path = out / "index.json"
    if path.is_file():
        index = json.loads(path.read_text())
        if index.get("version") == INDEX_VERSION:
            return index
    index = build_index(dataset, seed=seed, progress=progress)
    out.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("index.json.tmp")
    tmp.write_text(json.dumps(index))
    tmp.replace(path)
    return index


def floor_tiles(dataset: Path, index: dict, limit: int = 200) -> list[str]:
    """Paths of fully opaque 32x32 floor tile sprites to use as backgrounds."""
    seen = [it["path"] for it in index["items"] if it.get("opaque") and it["w"] == 32 and it["h"] == 32]
    return seen[:limit]


@dataclass
class Augment:
    offset: bool = True        # random placement anywhere on the canvas
    zoom: float = 0.15         # chance of a 2x whole-number zoom for small sprites
    rotate: float = 0.25       # chance of a random nearest-neighbour rotation (one-direction sprites only)
    floor: float = 0.35        # chance of a real floor-tile background


def _background(rng: random.Random, dataset: Path, floors: list[str], aug: Augment) -> Image.Image:
    if floors and rng.random() < aug.floor:
        with Image.open(dataset / rng.choice(floors)) as tile:
            tile = tile.convert("RGBA")
        bg = Image.new("RGBA", (CANVAS, CANVAS))
        for y in range(0, CANVAS, 32):
            for x in range(0, CANVAS, 32):
                bg.paste(tile, (x, y))
        return bg
    base = [rng.randint(0, 110) for _ in range(3)]
    if rng.random() < 0.5:
        return Image.new("RGBA", (CANVAS, CANVAS), (*base, 255))
    np_rng = np.random.default_rng(rng.getrandbits(32))
    noise = np_rng.integers(-12, 13, size=(CANVAS, CANVAS, 1))
    arr = np.clip(np.array(base)[None, None, :] + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def render(item: dict, dataset: Path, rng: random.Random, floors: list[str],
           aug: Augment | None = None) -> np.ndarray:
    """One training canvas as a (3, 64, 64) float32 array in [0, 1]."""
    aug = aug or Augment()
    with Image.open(dataset / item["path"]) as img:
        sprite = img.convert("RGBA")
    if aug.zoom and max(sprite.size) <= CANVAS // 2 and rng.random() < aug.zoom:
        sprite = sprite.resize((sprite.width * 2, sprite.height * 2), Image.NEAREST)
    if aug.rotate and item["direction"] < 0 and rng.random() < aug.rotate:
        sprite = sprite.rotate(rng.uniform(0, 360), resample=Image.NEAREST, expand=True)
        box = sprite.getchannel("A").getbbox()
        if box:
            sprite = sprite.crop(box)
        if max(sprite.size) > CANVAS:
            sprite = sprite.crop((0, 0, min(sprite.width, CANVAS), min(sprite.height, CANVAS)))
    canvas = _background(rng, dataset, floors, aug)
    if aug.offset:
        x = rng.randint(0, CANVAS - sprite.width)
        y = rng.randint(0, CANVAS - sprite.height)
    else:
        x, y = (CANVAS - sprite.width) // 2, (CANVAS - sprite.height) // 2
    canvas.alpha_composite(sprite, (x, y))
    arr = np.asarray(canvas.convert("RGB"), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1).copy()


class SpriteDataset:
    """Map-style dataset usable by torch's DataLoader (no torch import here)."""

    def __init__(self, dataset: Path, index: dict, split: str, train: bool, seed: int = 14,
                 worn: bool = True, text=None):
        self.dataset = Path(dataset)
        self.items = [it for it in index["items"] if it["split"] == split and (worn or not it["worn"])]
        self.floors = floor_tiles(self.dataset, index)
        self.n_components = len(index.get("components", ()))
        # Rows of the text table (pretrain/text.py) for the entities that draw each sprite.
        self.text_rows = [text.rows_for(it.get("entities", [])) if text else [] for it in self.items]
        self.train = train
        self.seed = seed
        self.epoch = 0

    def weights(self) -> list[float]:
        """Sampling weights for training: worn overlays less often."""
        return [WORN_WEIGHT if it["worn"] else 1.0 for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        if self.train:
            rng = random.Random()
            aug = Augment()
        else:
            rng = random.Random(self.seed * 1_000_003 + i)   # same canvas every evaluation
            aug = Augment(zoom=0.0, rotate=0.0)
        x = render(item, self.dataset, rng, self.floors, aug)
        comps = np.zeros(max(1, self.n_components), dtype=np.float32)
        if item.get("components"):
            comps[item["components"]] = 1.0
        rows = self.text_rows[i]
        all_rows = np.full(MAX_TEXTS, -1, dtype=np.int64)
        all_rows[:len(rows[:MAX_TEXTS])] = rows[:MAX_TEXTS]
        text = (rng.choice(rows) if self.train else rows[0]) if rows else -1
        return {"x": x, "category": item["category"], "state": item["state_id"], "direction": item["direction"],
                "components": comps, "has_components": item.get("components") is not None,
                "text": text, "text_rows": all_rows}
