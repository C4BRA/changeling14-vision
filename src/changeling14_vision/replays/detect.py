"""Stage 2: teach a sprite model to find and name things in real game frames.

Starts from a stage-1 model folder (its latest.pt and text.npz) and trains on
a replay dataset (dataset.py). The encoder gives one vector per 32 px square
of the frame; for each square the heads learn:

  boxes      up to SLOTS sprites centred in the square, topmost first (a gun
             on a table, a player on an item): for each, where its centre sits
             in the square and how big its box is. "something" says whether
             the first slot is filled, "more" whether each further one is.
  names      one vector per slot, compared to every entity text: the slot's
             sprite should match its own text. Sprites the player can't see
             (dark, behind a wall) are neither right nor wrong answers there,
             and neither are the square's other sprites.
  player     whether a visible player or other humanoid stands in the square
  mob        the topmost visible mob's class: none, humanoid, borg, animal,
             hostile (labels.MOB_CLASSES)
  tile       the floor tile under the square's centre (floor.py), or space
  gas        which visible gases (and fire) are on that tile
  decals     which decals (warning stripes, markings, dirt) are centred in it

Floor, gas and decal heads exist only when the dataset has floors (recordings
made with the current recorder); the tile under a dark square, or under a
wall, is neither right nor wrong.

Training crops random 480x480 windows at any pixel offset, so the model does
not learn that sprites line up with its squares. Scores come from whole
frames of rounds held out by the dataset (split "test"); see evaluate().

Only names that some training frame shows are pushed away as wrong answers.
Training also varies the lighting and, slightly, the tile size; batch-norm
statistics stay at stage 1's; a moving average of the weights is what gets
scored and saved ("ema").

Writes into --out:
  latest.pt              model, weight average, optimiser, step, the floor,
                         gas and decal names its heads use, and the dataset's
                         fingerprint (resume loads it, and refuses another
                         dataset or stage-1 model unless told to)
  checkpoints/step-N.pt  weights at the newest checkpoints and the best one
  text.npz               the entity texts (copied from the stage-1 folder)
  metrics.jsonl          losses and held-out scores per checkpoint
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .. import __version__
from ..model.encoder import EMBED_DIM, Encoder
from ..pretrain.text import TextTable
from ..pretrain.train import WeightAverage, load_checkpoint, pick_device, prune_checkpoints
from ..progress import Progress
from .floor import CELL as FLOOR_CELL
from .floor import DECAL_BITS, Floor
from .labels import BIT, GROUPS, MOB_CLASSES, Labels, dataset_hash, group_of, load

CELL = 32
CROP = 480
SLOTS = 3                 # sprites per square that get their own box and name
NEGATIVES = 2048          # random wrong texts per step, besides every text in the batch
DARK = 12 / 255           # squares darker than this on average: their floor can't be seen
MOB_WEIGHTS = (1.0, 4.0, 4.0, 4.0, 4.0)   # mobs are rare squares, so their misses count more


@dataclass
class DetectOptions:
    dataset: Path
    start: Path               # stage-1 model folder
    out: Path
    steps: int = 6000
    batch: int = 8
    lr: float = 3e-4          # heads; the encoder gets encoder_lr
    encoder_lr: float = 1e-4
    weight_decay: float = 0.05
    warmup: int = 300
    checkpoint_every: int = 500
    workers: int = 2
    device: str = "auto"
    seed: int = 14
    minutes: float | None = None
    ema: float = 0.998        # decay of the weight average that is scored and saved
    keep_checkpoints: int | None = None   # newest checkpoints kept, plus the best; 0 keeps all.
                                          # None: 5 for a new model, the model's own setting when resuming
    allow_new_data: bool = False          # resume on a dataset other than the one the model trained on
    allow_different_start: bool = False   # resume although --from names another stage-1 model


def floor_vocab(floor: Floor | None) -> dict:
    """The floor, gas and decal names a model's heads use."""
    if floor is None:
        return {"tile_names": [], "gas_names": [], "decal_names": []}
    return {"tile_names": floor.tile_names, "gas_names": floor.gas_names, "decal_names": floor.decal_names}


class Frames(torch.utils.data.Dataset):
    """Frames of one split with per-square targets. Only this split's sprites
    are kept, as arrays (labels.py), so loader processes get a small copy.
    vocab: the floor, gas and decal names of the model (floor_vocab); the
    dataset's own are mapped onto them by name, and unknown ones are left out."""

    def __init__(self, dataset: Path, split: str, text: TextTable, crop: int | None, seed: int = 0,
                 augment: bool = False, source: tuple[dict, Labels] | None = None,
                 floor: Floor | None = None, vocab: dict | None = None):
        data, labels = source or load(dataset)
        floor = floor if floor is not None else Floor.load(dataset)
        self.root = dataset
        frames = [f for f in data["frames"] if f["split"] == split]
        if crop:
            # Frames smaller than a crop (zoomed-in views) are padded with black.
            frames = [f for f in frames if f["width"] >= CELL * 4 and f["height"] >= CELL * 4]
        idx = np.concatenate([np.arange(f["first"], f["first"] + f["count"]) for f in frames]) \
            if frames else np.zeros(0, np.int64)
        self.frames, n = [], 0
        for f in frames:
            self.frames.append({"picture": f["picture"], "width": f["width"], "height": f["height"],
                                "first": n, "count": f["count"], "dataset_first": f["first"],
                                **{k: f[k] for k in ("floor", "decals") if k in f and floor is not None}})
            n += f["count"]
        proto_row = np.array([text.row_of_entity.get(p, -1) for p in labels.prototypes] or [-1], np.int64)
        self.row = proto_row[labels.proto[idx]] if len(idx) else np.zeros(0, np.int64)
        self.box = labels.box[idx]
        self.depth = labels.depth[idx]
        self.flags = labels.flags[idx]
        self.mob = labels.mob[idx]
        self.group = np.array([group_of(int(d), int(fl)) for d, fl in zip(self.depth, self.flags)], np.int64)

        vocab = vocab if vocab is not None else floor_vocab(floor)
        self.n_tiles, self.n_gas, self.n_decals = (len(vocab[k]) for k in ("tile_names", "gas_names", "decal_names"))
        self.floor = None
        if floor is not None and self.n_tiles:
            # Only this split's floor maps and decals, renumbered like the sprites.
            tiles, gas, ids, boxes, dflags, off, dn = [], [], [], [], [], 0, 0
            for f in self.frames:
                if "floor" not in f:
                    continue
                t, g = floor.maps(f)
                tiles.append(t.ravel())
                gas.append(g.ravel())
                f["floor"] = [off, t.shape[0], t.shape[1]]
                off += t.size
                r = floor.decals(f)
                ids.append(floor.decal_id[r])
                boxes.append(floor.decal_box[r])
                dflags.append(floor.decal_flags[r])
                f["decals"] = [dn, r.stop - r.start]
                dn += r.stop - r.start
            remap = lambda names, ours: np.array([ours.index(x) if x in ours else -1 for x in names] or [-1], np.int64)
            tile_map = remap(floor.tile_names, vocab["tile_names"])
            gas_map = remap(floor.gas_names, vocab["gas_names"])
            decal_map = remap(floor.decal_names, vocab["decal_names"])
            cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype)
            self.floor = {"tiles": tile_map[cat(tiles, np.uint16).astype(np.int64)],
                          "gas": cat(gas, np.uint16).astype(np.int64), "gas_map": gas_map,
                          "decal_id": decal_map[cat(ids, np.int32).astype(np.int64)],
                          "decal_box": cat(boxes, np.float32).reshape(-1, 4),
                          "decal_flags": cat(dflags, np.uint8)}
        self.text = text
        self.crop = crop
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.frames)

    def trained_rows(self) -> np.ndarray:
        """Text rows that are right answers somewhere in these frames."""
        visible = (self.flags & BIT["visible"]) != 0
        return np.unique(self.row[visible & (self.row >= 0)])

    def __getitem__(self, i):
        f = self.frames[i]
        pic = Image.open(self.root / f["picture"]).convert("RGB")
        w, h = pic.size
        zoom = 1.0
        rng = random.Random()
        if self.crop:
            cw = ch = self.crop
            if self.augment and rng.random() < 0.5:
                # A slightly wrong tile size, as when the live view guesses it.
                zoom = rng.uniform(0.9, 1.1)
            sw, sh = cw / zoom, ch / zoom
            if zoom == 1.0:
                x0 = rng.randint(min(0, w - cw), max(0, w - cw))
                y0 = rng.randint(min(0, h - ch), max(0, h - ch))
            else:
                x0 = rng.uniform(min(0, w - sw), max(0, w - sw))
                y0 = rng.uniform(min(0, h - sh), max(0, h - sh))
            # Past the picture's edge is black, like space.
            img = pic.transform((cw, ch), Image.EXTENT, (x0, y0, x0 + sw, y0 + sh),
                                Image.NEAREST if zoom == 1.0 else Image.BILINEAR)
        else:
            cw, ch = w // CELL * CELL, h // CELL * CELL
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            img = pic.crop((x0, y0, x0 + cw, y0 + ch))
        raw = np.asarray(img, dtype=np.float32) / 255
        a = raw
        if self.augment:
            # Brighter or dimmer lighting, and a different light curve.
            a = np.clip(raw * rng.uniform(0.7, 1.3), 0, 1) ** rng.uniform(0.8, 1.25)
        gw, gh = cw // CELL, ch // CELL
        n_cells = gh * gw

        # Square centres in the frame's own pixels; squares off the picture are padding.
        sy, sx = np.mgrid[0:gh, 0:gw]
        fx = (x0 + (sx.ravel() + 0.5) * CELL / zoom)
        fy = (y0 + (sy.ravel() + 0.5) * CELL / zoom)
        on_picture = (fx >= 0) & (fx < w) & (fy >= 0) & (fy < h)
        dark = raw.reshape(gh, CELL, gw, CELL, 3).mean(axis=(1, 3, 4)).ravel() < DARK

        r = slice(f["first"], f["first"] + f["count"])
        bx, flags, rows, groups, depth, mobs = (self.box[r], self.flags[r], self.row[r], self.group[r],
                                                self.depth[r], self.mob[r])
        cx = ((bx[:, 0] + bx[:, 2]) / 2 - x0) * zoom
        cy = ((bx[:, 1] + bx[:, 3]) / 2 - y0) * zoom
        inside = (cx >= 0) & (cx < cw) & (cy >= 0) & (cy < ch)
        cells = (np.floor(cy / CELL).astype(np.int64) * gw + np.floor(cx / CELL).astype(np.int64))
        visible = (flags & BIT["visible"]) != 0
        named = inside & (rows >= 0)
        pos = np.stack([cells[named & visible], rows[named & visible]], 1)
        ignore = np.stack([cells[named & ~visible], rows[named & ~visible]], 1)
        slot_box = np.zeros((n_cells, SLOTS, 4), np.float32)
        slot_on = np.zeros((n_cells, SLOTS), np.float32)
        slot_pos = []                                   # (cell, slot, text row)
        player = np.zeros(n_cells, np.float32)
        subject = np.zeros(n_cells, np.float32)
        mob = np.zeros(n_cells, np.int64)
        walls = np.zeros(n_cells, bool)
        # Squares holding only sprites the player can't see are neither right
        # nor wrong for "something", player and mob: dark and faint are judged
        # from pixels and can be off. Padding never counts.
        mask = on_picture.astype(np.float32)
        mask[cells[inside & ~visible]] = 0
        per_cell: dict[int, list[int]] = {}
        for k in np.nonzero(inside)[0]:
            cell = cells[k]
            if flags[k] & BIT["subject"]:
                subject[cell] = 1
            if not visible[k]:
                continue
            per_cell.setdefault(cell, []).append(k)
            if flags[k] & BIT["humanoid"]:
                player[cell] = 1
            if groups[k] == 1:
                walls[cell] = True
        for cell, ks in per_cell.items():
            mask[cell] = 1
            # Topmost first; among equals, the bigger one.
            ks.sort(key=lambda k: (-int(depth[k]), -float((bx[k, 2] - bx[k, 0]) * (bx[k, 3] - bx[k, 1]))))
            for j, k in enumerate(ks[:SLOTS]):
                slot_on[cell, j] = 1
                slot_box[cell, j] = [(cx[k] % CELL) / CELL, (cy[k] % CELL) / CELL,
                                     math.log(max((bx[k, 2] - bx[k, 0]) * zoom, 1) / CELL),
                                     math.log(max((bx[k, 3] - bx[k, 1]) * zoom, 1) / CELL)]
                if rows[k] >= 0:
                    slot_pos.append((cell, j, rows[k]))
            top_mob = next((k for k in ks if mobs[k]), None)
            if top_mob is not None:
                mob[cell] = mobs[top_mob]

        tile = np.full(n_cells, -1, np.int64)
        gas = np.zeros((n_cells, max(1, self.n_gas)), np.float32)
        floor_mask = np.zeros(n_cells, np.float32)
        decal_pos = np.zeros((0, 2), np.int64)
        decal_ignore = np.zeros((0, 2), np.int64)
        if self.floor is not None and "floor" in f:
            off, rws, cls = f["floor"]
            mi, mj = (fy / FLOOR_CELL).astype(np.int64), (fx / FLOOR_CELL).astype(np.int64)
            ok = on_picture & (mi >= 0) & (mi < rws) & (mj >= 0) & (mj < cls)
            at = off + np.clip(mi, 0, rws - 1) * cls + np.clip(mj, 0, cls - 1)
            # The floor under a dark square or a wall can't be seen.
            seen = ok & ~dark & ~walls
            tile[seen] = self.floor["tiles"][at[seen]]
            floor_mask[seen] = 1
            bits = self.floor["gas"][at]
            for b, g in enumerate(self.floor["gas_map"]):
                if g >= 0:
                    gas[:, g] = ((bits >> b) & 1) * seen
            d0, dn = f["decals"]
            if dn and self.n_decals:
                ids = self.floor["decal_id"][d0:d0 + dn]
                db = self.floor["decal_box"][d0:d0 + dn]
                dv = (self.floor["decal_flags"][d0:d0 + dn] & DECAL_BITS["visible"]) != 0
                dx = ((db[:, 0] + db[:, 2]) / 2 - x0) * zoom
                dy = ((db[:, 1] + db[:, 3]) / 2 - y0) * zoom
                din = (dx >= 0) & (dx < cw) & (dy >= 0) & (dy < ch) & (ids >= 0)
                dcell = np.floor(dy / CELL).astype(np.int64) * gw + np.floor(dx / CELL).astype(np.int64)
                decal_pos = np.stack([dcell[din & dv], ids[din & dv]], 1)
                decal_ignore = np.stack([dcell[din & ~dv], ids[din & ~dv]], 1)
            floor_mask[~ok] = 0
        floor_valid = on_picture.astype(np.float32) if self.floor is not None and "floor" in f \
            else np.zeros(n_cells, np.float32)

        t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        return {"image": torch.from_numpy(a.transpose(2, 0, 1).copy()),
                "pos": t(pos).long().reshape(-1, 2), "pos_group": t(groups[named & visible]).long(),
                "ignore": t(ignore).long().reshape(-1, 2),
                "slot_pos": torch.tensor(slot_pos, dtype=torch.long).reshape(-1, 3),
                "slot_on": t(slot_on), "slot_box": t(slot_box),
                "something": t(slot_on[:, 0].copy()), "box": t(slot_box[:, 0].copy()),
                "player": t(player), "mask": t(mask), "subject": t(subject), "mob": t(mob),
                "tile": t(tile), "gas": t(gas), "floor_mask": t(floor_mask), "floor_valid": t(floor_valid),
                "decal_pos": t(decal_pos).long().reshape(-1, 2), "decal_ignore": t(decal_ignore).long().reshape(-1, 2)}


PAIRS = ("pos", "ignore", "slot_pos", "decal_pos", "decal_ignore")     # (cell, ...) rows, offset per item


def collate(items: list[dict]) -> dict:
    cells = items[0]["something"].shape[0]
    out = {k: torch.stack([it[k] for it in items]) if k == "image" else torch.cat([it[k] for it in items])
           for k in items[0] if k not in PAIRS}
    for key in PAIRS:
        width = items[0][key].shape[1]
        shift = torch.zeros(width, dtype=torch.long)
        pairs = []
        for b, it in enumerate(items):
            shift[0] = b * cells
            pairs.append(it[key] + shift)
        out[key] = torch.cat(pairs) if pairs else torch.zeros(0, width, dtype=torch.long)
    return out


class DetectModel(nn.Module):
    def __init__(self, encoder: Encoder, text_dim: int, vocab: dict | None = None, slots: int = SLOTS):
        super().__init__()
        vocab = vocab or floor_vocab(None)
        self.vocab = {k: list(v) for k, v in vocab.items()}
        self.slots = slots
        self.encoder = encoder
        self.text_proj = nn.Linear(text_dim, EMBED_DIM)
        self.name = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, EMBED_DIM * slots))
        self.something = nn.Linear(EMBED_DIM, 1)
        self.more = nn.Linear(EMBED_DIM, slots - 1)
        self.player = nn.Linear(EMBED_DIM, 1)
        self.mob = nn.Linear(EMBED_DIM, len(MOB_CLASSES))
        self.box = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, 4 * slots))
        n_tiles, n_gas, n_decals = (len(self.vocab[k]) for k in ("tile_names", "gas_names", "decal_names"))
        self.tile = nn.Linear(EMBED_DIM, n_tiles) if n_tiles else None
        self.gas = nn.Linear(EMBED_DIM, n_gas) if n_gas else None
        self.decal = nn.Linear(EMBED_DIM, n_decals) if n_decals else None
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0))
        self.register_buffer("text_vectors", torch.zeros(0, text_dim), persistent=False)
        self.register_buffer("name_ids", torch.zeros(0, dtype=torch.long), persistent=False)

    def set_text(self, table: TextTable) -> None:
        dev = self.scale.device
        self.text_vectors = torch.as_tensor(table.vectors, dtype=torch.float32, device=dev)
        ids: dict[str, int] = {}
        self.name_ids = torch.tensor([ids.setdefault(t.split(". ", 1)[0].lower(), len(ids)) for t in table.texts],
                                     device=dev)

    def texts(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        t = self.text_vectors if rows is None else self.text_vectors[rows]
        return F.normalize(self.text_proj(t), dim=-1)

    def forward(self, x):
        tiles, _ = self.encoder(x)
        t = tiles.flatten(0, 2)                                        # (B*cells, E)
        k = self.slots
        # The name head starts as a residual on stage 1's tile vectors, the same for every slot.
        q = F.normalize(t[:, None, :] + self.name(t).view(-1, k, t.shape[1]), dim=-1)     # (B*cells, slots, E)
        b = self.box(t).view(-1, k, 4)
        boxes = torch.cat([torch.sigmoid(b[..., :2]), b[..., 2:]], -1)
        out = {"names": q, "something": self.something(t).squeeze(1), "more": self.more(t),
               "player": self.player(t).squeeze(1), "mob": self.mob(t), "slot_box": boxes, "box": boxes[:, 0],
               "grid": tiles.shape[1:3]}
        for head in ("tile", "gas", "decal"):
            if getattr(self, head) is not None:
                out[head] = getattr(self, head)(t)
        return out


def load_matching(model: nn.Module, state: dict) -> list[str]:
    """Load the weights whose names and shapes fit; returns the ones left as
    they were (heads added or resized since the weights were saved)."""
    own = model.state_dict()
    fits = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(fits, strict=False)
    return sorted(k for k in own if k not in fits)


def from_stage1(start: Path, vocab: dict | None = None) -> tuple[DetectModel, TextTable, dict]:
    ckpt = load_checkpoint(start / "latest.pt")
    if not ckpt.get("text_dim"):
        raise SystemExit(f"{start} was trained without text matching, so it can't name things; "
                         "start from a model trained with text")
    encoder = Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"})
    weights = ckpt.get("ema") or ckpt["model"]
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in weights.items() if k.startswith("encoder.")})
    table = TextTable(start / "text.npz")
    model = DetectModel(encoder, table.dim, vocab)
    model.text_proj.load_state_dict({k[len("text_proj."):]: v for k, v in weights.items()
                                     if k.startswith("text_proj.")})
    nn.init.zeros_(model.name[2].weight)
    nn.init.zeros_(model.name[2].bias)
    return model, table, ckpt


def _all_slots(cells: torch.Tensor, k: int) -> torch.Tensor:
    """Row indices of every slot of these squares, (len(cells) * k,)."""
    return (cells[:, None] * k + torch.arange(k, device=cells.device)).flatten()


def losses(model: DetectModel, out: dict, b: dict, trained: torch.Tensor | None = None,
           generator: torch.Generator | None = None) -> dict:
    """trained: which text rows are right answers somewhere in the training
    frames (None: all). Only those are pushed away as wrong answers; a name
    training never shows stays neutral, so the model isn't taught to avoid it."""
    q = out["names"]
    n_cells, k = q.shape[0], q.shape[1]
    n_text = model.text_vectors.shape[0]
    dev = q.device
    pool = trained.nonzero().flatten() if trained is not None else torch.arange(n_text, device=dev)
    ign = b["ignore"][:, 1]
    if trained is not None:
        ign = ign[trained[ign]]
    # Texts asked about this step: the batch's right answers, the texts of
    # unseen sprites in it, and random others.
    rows = torch.cat([b["pos"][:, 1], ign,
                      pool[torch.randint(0, len(pool), (NEGATIVES,), device=dev, generator=generator)]]).unique()
    col = torch.full((n_text,), -1, dtype=torch.long, device=dev)
    col[rows] = torch.arange(len(rows), device=dev)
    logits = q.reshape(n_cells * k, -1) @ model.texts(rows).T * model.scale.exp() + model.bias   # (cells*slots, R)
    target = torch.zeros_like(logits)
    weight = torch.ones_like(logits)
    # In every slot of a square: its other sprites, its unseen sprites, and
    # texts sharing a right answer's name are neither right nor wrong.
    pos_cell, pos_col = b["pos"][:, 0], col[b["pos"][:, 1]]
    weight[_all_slots(pos_cell, k), pos_col.repeat_interleave(k)] = 0
    keep = col[b["ignore"][:, 1]] >= 0
    weight[_all_slots(b["ignore"][keep, 0], k), col[b["ignore"][keep, 1]].repeat_interleave(k)] = 0
    same = model.name_ids[rows][None, :] == model.name_ids[b["pos"][:, 1]][:, None]     # (P, R)
    p_i, r_i = same.nonzero(as_tuple=True)
    weight[_all_slots(pos_cell[p_i], k), r_i.repeat_interleave(k)] = 0
    sp = b["slot_pos"]
    target[sp[:, 0] * k + sp[:, 1], col[sp[:, 2]]] = 1
    weight[sp[:, 0] * k + sp[:, 1], col[sp[:, 2]]] = 1
    names = F.binary_cross_entropy_with_logits(logits, target, weight, reduction="sum") / n_cells

    m = b["mask"]
    msum = m.sum().clamp(min=1)
    on = b["slot_on"]
    something = F.binary_cross_entropy_with_logits(out["something"], on[:, 0], m, reduction="sum") / msum
    more = F.binary_cross_entropy_with_logits(out["more"], on[:, 1:], m[:, None].expand_as(on[:, 1:]),
                                              reduction="sum") / msum
    # Players are rare squares, so their misses count more.
    player = F.binary_cross_entropy_with_logits(out["player"], b["player"], m, reduction="sum",
                                                pos_weight=torch.tensor(10.0, device=dev)) / msum
    mob = (F.cross_entropy(out["mob"], b["mob"], weight=torch.tensor(MOB_WEIGHTS, device=dev), reduction="none")
           * m * on[:, 0]).sum() / (m * on[:, 0]).sum().clamp(min=1)
    box = (F.smooth_l1_loss(out["slot_box"], b["slot_box"], reduction="none", beta=0.1).sum(-1) * on).sum() \
        / on.sum().clamp(min=1)
    parts = {"names": names, "something": something, "more": more, "player": player, "mob": mob, "box": box}

    fm = b["floor_mask"]
    if "tile" in out:
        parts["tile"] = F.cross_entropy(out["tile"], b["tile"], ignore_index=-1) if (b["tile"] >= 0).any() \
            else out["tile"].sum() * 0
    if "gas" in out:
        parts["gas"] = (F.binary_cross_entropy_with_logits(out["gas"], b["gas"][:, :out["gas"].shape[1]],
                                                           reduction="none",
                                                           pos_weight=torch.tensor(5.0, device=dev)).sum(1)
                        * fm).sum() / fm.sum().clamp(min=1)
    if "decal" in out:
        valid = b["floor_valid"]
        d_target = torch.zeros_like(out["decal"])
        d_weight = valid[:, None].expand_as(out["decal"]).clone()
        d_weight[b["decal_ignore"][:, 0], b["decal_ignore"][:, 1]] = 0
        d_target[b["decal_pos"][:, 0], b["decal_pos"][:, 1]] = 1
        d_weight[b["decal_pos"][:, 0], b["decal_pos"][:, 1]] = 1
        parts["decal"] = F.binary_cross_entropy_with_logits(out["decal"], d_target, d_weight, reduction="sum") \
            / valid.sum().clamp(min=1)
    parts["total"] = sum(parts.values())
    return parts


MACRO_MIN = 20            # names with fewer held-out sprites than this are left out of the per-name average


class _Counts:
    """Precision and recall counters."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, guess: torch.Tensor, truth: torch.Tensor) -> None:
        self.tp += int((guess & truth).sum())
        self.fp += int((guess & ~truth).sum())
        self.fn += int((~guess & truth).sum())

    def scores(self, name: str) -> dict:
        return {f"{name}_precision": self.tp / max(1, self.tp + self.fp),
                f"{name}_recall": self.tp / max(1, self.tp + self.fn), f"{name}_guesses": self.tp + self.fp}


@torch.no_grad()
def evaluate(model: DetectModel, loader, device, stop=lambda: False, trained: torch.Tensor | None = None) -> dict:
    """Held-out frames. Naming: top-1/top-5 per sprite (by name, the best of
    the square's slots), split into names training showed (seen) and names it
    never did (unseen), averaged per name (macro: every name with MACRO_MIN or
    more sprites counts the same), and per group of draw depths
    (labels.GROUPS). Name sets: per square, the names scored 0.5 or more
    against the names truly centred there. "Something", second box (a second
    sprite in the square), player and mob class, leaving out squares holding
    only sprites that can't be seen, and for players the followed player's own
    square (always the middle of the view). Box overlap where both say
    something is there. Floor tile top-1, and gas and decal precision and
    recall, where the floor can be seen."""
    was_training = model.training
    model.eval()
    all_t = model.texts()
    name_ids = model.name_ids
    n_names = int(name_ids.max()) + 1 if len(name_ids) else 1
    seen_name = torch.ones(n_names, dtype=torch.bool, device=device)
    if trained is not None:
        seen_name = torch.zeros(n_names, dtype=torch.bool, device=device)
        seen_name[name_ids[trained.to(device)]] = True
    per_name = torch.zeros(n_names, 3, device=device)            # sprites, top-1 hits, top-5 hits
    per_group = torch.zeros(len(GROUPS), 3, device=device)
    per_seen = torch.zeros(2, 3, device=device)                  # unseen, seen
    mob_conf = torch.zeros(len(MOB_CLASSES), len(MOB_CLASSES), device=device)   # truth x guess
    sets, thing, second, players, gas, decal = (_Counts() for _ in range(6))
    tile_hits = tile_top3 = tile_n = 0
    ious = []
    for b in loader:
        if stop():
            break
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["image"])
        sims = torch.einsum("nke,te->nkt", out["names"], all_t).amax(1)      # (cells, T): best slot
        if len(b["pos"]):
            truth = name_ids[b["pos"][:, 1]]
            top = sims[b["pos"][:, 0]].topk(min(5, all_t.shape[0]), dim=1).indices
            hit = name_ids[top] == truth[:, None]
            stats = torch.stack([torch.ones_like(truth, dtype=torch.float), hit[:, 0].float(),
                                 hit.any(1).float()], 1)
            per_name.index_add_(0, truth, stats)
            per_group.index_add_(0, b["pos_group"], stats)
            per_seen.index_add_(0, seen_name[truth].long(), stats)
        # Name sets per square.
        n_cells = sims.shape[0]
        sure = (torch.sigmoid(sims * model.scale.exp() + model.bias) >= 0.5).nonzero()
        guess = torch.zeros(n_cells, n_names, dtype=torch.bool, device=device)
        guess[sure[:, 0], name_ids[sure[:, 1]]] = True
        there = torch.zeros_like(guess)
        there[b["pos"][:, 0], name_ids[b["pos"][:, 1]]] = True
        hidden = torch.zeros_like(guess)
        hidden[b["ignore"][:, 0], name_ids[b["ignore"][:, 1]]] = True
        guess &= ~(hidden & ~there)          # naming a hidden sprite is neither right nor wrong
        sets.add(guess, there)

        counted = b["mask"] > 0
        g, t = (out["something"] > 0) & counted, (b["something"] > 0) & counted
        thing.add(g, t)
        second.add((out["more"][:, 0] > 0) & counted, (b["slot_on"][:, 1] > 0) & counted)
        others = counted & (b["subject"] == 0)
        players.add((out["player"] > 0) & others, (b["player"] > 0) & others)
        sprite_cells = counted & t
        mob_conf.index_put_((b["mob"][sprite_cells], out["mob"][sprite_cells].argmax(1)),
                            torch.ones(int(sprite_cells.sum()), device=device), accumulate=True)
        both = g & t
        if both.any():
            ious.append(_iou(out["box"][both], b["box"][both]))
        if "tile" in out:
            seen_floor = b["tile"] >= 0
            if seen_floor.any():
                top3 = out["tile"][seen_floor].topk(min(3, out["tile"].shape[1]), dim=1).indices
                tile_hits += int((top3[:, 0] == b["tile"][seen_floor]).sum())
                tile_top3 += int((top3 == b["tile"][seen_floor][:, None]).any(1).sum())
                tile_n += int(seen_floor.sum())
        if "gas" in out:
            fm = b["floor_mask"][:, None] > 0
            gas.add((out["gas"] > 0) & fm, (b["gas"][:, :out["gas"].shape[1]] > 0) & fm)
        if "decal" in out:
            valid = b["floor_valid"][:, None] > 0
            there_d = torch.zeros_like(out["decal"], dtype=torch.bool)
            there_d[b["decal_pos"][:, 0], b["decal_pos"][:, 1]] = True
            hidden_d = torch.zeros_like(there_d)
            hidden_d[b["decal_ignore"][:, 0], b["decal_ignore"][:, 1]] = True
            decal.add((out["decal"] > 0) & valid & ~(hidden_d & ~there_d), there_d & valid)
    model.train(was_training)
    iou = torch.cat(ious) if ious else torch.zeros(0)
    ratio = lambda a, n: float(a) / max(1.0, float(n))
    per_name, per_group, per_seen, mob_conf = per_name.cpu(), per_group.cpu(), per_seen.cpu(), mob_conf.cpu()
    big = per_name[:, 0] >= MACRO_MIN
    total = per_name.sum(0)
    scores = {
        "name_top1": ratio(total[1], total[0]), "name_top5": ratio(total[2], total[0]),
        "named_sprites": int(total[0]),
        "seen_top1": ratio(per_seen[1, 1], per_seen[1, 0]), "seen_top5": ratio(per_seen[1, 2], per_seen[1, 0]),
        "seen_sprites": int(per_seen[1, 0]),
        "unseen_top1": ratio(per_seen[0, 1], per_seen[0, 0]), "unseen_top5": ratio(per_seen[0, 2], per_seen[0, 0]),
        "unseen_sprites": int(per_seen[0, 0]),
        "macro_top5": float((per_name[big, 2] / per_name[big, 0]).mean()) if big.any() else 0.0,
        "macro_names": int(big.sum()),
        **{k.replace("sets_", "name_set_"): v for k, v in sets.scores("sets").items()},
        **thing.scores("something"), **second.scores("second_box"), **players.scores("player"),
        "box_iou": float(iou.mean()) if len(iou) else 0.0,
        "box_iou_over_half": float((iou > 0.5).float().mean()) if len(iou) else 0.0}
    for k, name in enumerate(GROUPS):
        key = name.replace(" ", "_").replace("-", "_")
        scores[f"top5_{key}"] = ratio(per_group[k, 2], per_group[k, 0])
        scores[f"sprites_{key}"] = int(per_group[k, 0])
    for c, name in enumerate(MOB_CLASSES[1:], 1):
        scores[f"mob_recall_{name}"] = ratio(mob_conf[c, c], mob_conf[c].sum())
        scores[f"mob_precision_{name}"] = ratio(mob_conf[c, c], mob_conf[:, c].sum())
        scores[f"mob_squares_{name}"] = int(mob_conf[c].sum())
    if "tile_names" in model.vocab and model.tile is not None:
        scores.update({"tile_top1": ratio(tile_hits, tile_n), "tile_top3": ratio(tile_top3, tile_n),
                       "tile_squares": tile_n})
    if model.gas is not None:
        scores.update(gas.scores("gas"))
    if model.decal is not None:
        scores.update(decal.scores("decal"))
    return scores


def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Overlap of boxes given as (centre x, centre y in the square, log w, log h)."""
    def corners(t):
        w, h = t[:, 2].exp(), t[:, 3].exp()
        return t[:, 0] - w / 2, t[:, 1] - h / 2, t[:, 0] + w / 2, t[:, 1] + h / 2
    ax0, ay0, ax1, ay1 = corners(a)
    bx0, by0, bx1, by1 = corners(b)
    iw = (torch.min(ax1, bx1) - torch.max(ax0, bx0)).clamp(min=0)
    ih = (torch.min(ay1, by1) - torch.max(ay0, by0)).clamp(min=0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return (inter / union.clamp(min=1e-6)).cpu()


def _save(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def _train_mode(model: nn.Module) -> None:
    """Training mode, but with batch-norm statistics frozen at stage 1's: eight
    crops of one or two rounds are too few and too alike to re-measure them."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def _options(opts: DetectOptions) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()}


def _summary_line(scores: dict) -> list[str]:
    lines = [
        f"naming {scores['name_top1']:.1%} / {scores['name_top5']:.1%} (top 1 / top 5) on "
        f"{scores['named_sprites']:,} held-out sprites; names training showed {scores['seen_top5']:.1%}, never "
        f"showed {scores['unseen_top5']:.1%} ({scores['unseen_sprites']:,} sprites); per name "
        f"{scores['macro_top5']:.1%} ({scores['macro_names']:,} names); name sets "
        f"{scores['name_set_precision']:.1%} precise, {scores['name_set_recall']:.1%} found "
        f"({scores['name_set_guesses']:,} names scored 0.5 or more)",
        f"something {scores['something_precision']:.1%} precise, {scores['something_recall']:.1%} found; "
        f"second box {scores['second_box_precision']:.1%} precise, {scores['second_box_recall']:.1%} found; "
        f"other players {scores['player_precision']:.1%} precise, {scores['player_recall']:.1%} found; "
        f"box overlap {scores['box_iou']:.2f}",
        "mobs found (and right when guessed): " + ", ".join(
            f"{c} {scores[f'mob_recall_{c}']:.0%} ({scores[f'mob_precision_{c}']:.0%}, "
            f"{scores[f'mob_squares_{c}']:,} squares)" for c in MOB_CLASSES[1:]),
        "top 5 by group: " + ", ".join(
            f"{g} {scores['top5_' + g.replace(' ', '_').replace('-', '_')]:.1%}"
            f" ({scores['sprites_' + g.replace(' ', '_').replace('-', '_')]:,})" for g in GROUPS)]
    floor = []
    if "tile_top1" in scores:
        floor.append(f"floor tile {scores['tile_top1']:.1%} (top 3 {scores['tile_top3']:.1%}) on "
                     f"{scores['tile_squares']:,} squares")
    if "gas_precision" in scores:
        floor.append(f"gas {scores['gas_precision']:.1%} precise, {scores['gas_recall']:.1%} found")
    if "decal_precision" in scores:
        floor.append(f"decals {scores['decal_precision']:.1%} precise, {scores['decal_recall']:.1%} found")
    if floor:
        lines.append("; ".join(floor))
    return lines


def train_detect(opts: DetectOptions, progress: Progress | None = None) -> dict:
    progress = progress or Progress(None)
    started = time.time()
    out, dataset, start = (Path(p).expanduser() for p in (opts.out, opts.dataset, opts.start))
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)

    latest = out / "latest.pt"
    source = out if latest.is_file() else start
    if not (source / "text.npz").is_file():
        raise SystemExit(f"no text.npz in {source}")
    data_hash = dataset_hash(dataset)
    floor = Floor.load(dataset)
    if source is start:
        vocab = floor_vocab(floor)
        model, table, ckpt = from_stage1(start, vocab)
        shutil.copy2(start / "text.npz", out / "text.npz")
        print(f"detect: starting from {start / 'latest.pt'} (stage 1 step {ckpt.get('step', 0):,})", flush=True)
        if opts.keep_checkpoints is None:
            opts.keep_checkpoints = 5
    else:
        ckpt = load_checkpoint(latest)
        if not opts.allow_different_start and ckpt.get("start") and \
                Path(ckpt["start"]).expanduser().resolve() != start.resolve():
            raise SystemExit(f"{out} was started from {ckpt['start']}, not {start}; pick another model name, "
                             "or pass --allow-different-start to continue it anyway")
        if not opts.allow_new_data and ckpt.get("dataset_hash") and ckpt["dataset_hash"] != data_hash:
            raise SystemExit(f"{out} was trained on another version of a dataset than {dataset} (it was rebuilt, "
                             "or it is another one); its scores wouldn't compare. Pick another model name, or "
                             "pass --allow-new-data to continue it anyway")
        if opts.keep_checkpoints is None:
            # Models from before this setting kept every checkpoint.
            opts.keep_checkpoints = ckpt.get("options", {}).get("keep_checkpoints") or 0
        table = TextTable(out / "text.npz")
        vocab = ckpt.get("vocab") or floor_vocab(None)
        model = DetectModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}), table.dim,
                            vocab)
    device = pick_device(opts.device)
    model.to(device)
    model.set_text(table)

    loaded = load(dataset)
    summary = loaded[0].get("summary", {})
    train_ds = Frames(dataset, "train", table, CROP, opts.seed, augment=True, source=loaded, floor=floor, vocab=vocab)
    test_ds = Frames(dataset, "test", table, None, source=loaded, floor=floor, vocab=vocab)
    del loaded, floor
    if not len(train_ds):
        raise SystemExit(f"{dataset} has no training frames")
    trained = torch.zeros(len(table.texts), dtype=torch.bool)
    trained[torch.from_numpy(train_ds.trained_rows())] = True
    trained = trained.to(device)
    print(f"detect: {len(train_ds):,} training frames, {len(test_ds):,} held-out frames, "
          f"{len(table.texts):,} entity texts ({int(trained.sum()):,} shown in training; the others are "
          f"never pushed away), {len(vocab['tile_names']):,} floor tiles, {len(vocab['gas_names'])} gases, "
          f"{len(vocab['decal_names']):,} decals, on {device}", flush=True)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=opts.batch, shuffle=True, drop_last=True,
                                         num_workers=opts.workers, collate_fn=collate,
                                         persistent_workers=opts.workers > 0)
    # One whole frame at a time: held-out frames needn't all be the same size.
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, collate_fn=collate, num_workers=0)

    enc = list(model.encoder.parameters())
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optim = torch.optim.AdamW([{"params": enc, "lr": opts.encoder_lr, "base": opts.encoder_lr},
                               {"params": heads, "lr": opts.lr, "base": opts.lr}],
                              weight_decay=opts.weight_decay)
    step = 0
    if source is out:
        # Models from before a head was added or resized resume with a fresh one.
        fresh = load_matching(model, ckpt["model"])
        if fresh:
            print(f"detect: new or resized since this model was saved, so starting afresh: "
                  f"{', '.join(sorted({k.split('.')[0] for k in fresh}))}", flush=True)
        try:
            if fresh:
                raise ValueError
            optim.load_state_dict(ckpt["optim"])
        except ValueError:
            print("detect: the optimiser starts afresh", flush=True)
        step = ckpt["step"]
        print(f"detect: resumed {latest} at step {step:,}", flush=True)
    average = WeightAverage(model, opts.ema)
    if source is out:
        load_matching(average.model, ckpt.get("ema") or ckpt["model"])
        average.updates = ckpt.get("ema_updates", 0)
    else:
        (out / "config.json").write_text(json.dumps(
            {**_options(opts), "version": __version__, "dataset_hash": data_hash, "dataset_summary": summary},
            indent=2))

    def save(final: bool = False) -> dict:
        progress.update("evaluate", step, opts.steps, "held-out round", force=True)
        scores = evaluate(average.model, test_loader, device, lambda: progress.stop_requested, trained) \
            if len(test_ds) else {}
        state = {"model": model.state_dict(), "optim": optim.state_dict(), "step": step,
                 "ema": average.model.state_dict(), "ema_updates": average.updates,
                 "encoder_config": model.encoder.config, "text_dim": table.dim, "start": str(start),
                 "vocab": model.vocab, "slots": model.slots,
                 "options": _options(opts), "dataset_hash": data_hash, "dataset_summary": summary,
                 "version": __version__, "kind": "detect"}
        _save(latest, state)
        _save(out / "checkpoints" / f"step-{step}.pt", {k: v for k, v in state.items() if k != "optim"})
        row = {"step": step, "time": round(time.time() - started), **{k: round(v, 4) for k, v in last.items()},
               "dataset_hash": data_hash, **scores}
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        prune_checkpoints(out / "checkpoints", opts.keep_checkpoints, "name_top5", out / "metrics.jsonl")
        if scores:
            for line in _summary_line(scores):
                print(f"detect: step {step:,}: {line}", flush=True)
        return row

    _train_mode(model)
    last: dict = {}
    sums: dict = {}
    count = 0
    t0 = time.time()
    row = {}
    if step == 0 and len(test_ds):
        # What the stage-1 model already gets right, before any replay training.
        row = save()
    while step < opts.steps and not progress.stop_requested:
        for b in loader:
            if step >= opts.steps or progress.stop_requested:
                break
            if opts.minutes and time.time() - started > opts.minutes * 60:
                progress.stop_requested = True
                break
            f = min(1.0, (step + 1) / opts.warmup)
            t = min(1.0, step / max(1, opts.steps))
            for g in optim.param_groups:
                g["lr"] = g["base"] * f * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))
            b = {k: v.to(device) for k, v in b.items()}
            out_ = model(b["image"])
            ls = losses(model, out_, b, trained)
            optim.zero_grad(set_to_none=True)
            ls["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            average.update(model)
            step += 1
            # Reading a loss waits for the GPU, so they are summed there and read every 25 steps.
            for k, v in ls.items():
                sums[k] = sums.get(k, 0) + v.detach()
            count += 1
            if step % 25 == 0:
                last = {k: float(v) / count for k, v in sums.items()}
                ms = (time.time() - t0) / count * 1000
                progress.update("train detect", step, opts.steps,
                                f"loss {last['total']:.3f} (" + ", ".join(
                                    f"{k} {v:.3f}" for k, v in last.items() if k != "total") + f"), {ms:.0f} ms/step")
                print(f"detect: step {step:,}/{opts.steps:,} loss {last['total']:.4f} {ms:.0f} ms/step", flush=True)
                sums, count, t0 = {}, 0, time.time()
            if step % opts.checkpoint_every == 0:
                row = save()
    if step and (not row or row.get("step") != step):
        row = save(final=True)
    return row
