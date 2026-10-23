"""Stage 2: teach a sprite model to find and name things in real game frames.

Starts from a stage-1 model folder (its latest.pt and text.npz) and trains on
a replay dataset (dataset.py). The encoder gives one vector per 32 px square
of the frame; for each square three heads learn:

  names      which entities stand in this square: each vector is compared to
             every entity text, and the ones whose box centre falls in the
             square should match. Sprites the player can't see (dark, behind
             a wall) are neither right nor wrong answers there.
  something  whether any visible sprite is centred in the square
  player     whether a visible player or other humanoid stands in the square
             (the agent needs this most; species names are much harder)
  box        for the topmost visible sprite centred there: where its centre
             sits in the square and how big its box is

Training crops random 480x480 windows at any pixel offset, so the model does
not learn that sprites line up with its squares. Scores come from whole
frames of rounds held out by the dataset (split "test"); see evaluate().

Only names that some training frame shows are pushed away as wrong answers.
Training also varies the lighting and, slightly, the tile size; batch-norm
statistics stay at stage 1's; a moving average of the weights is what gets
scored and saved ("ema").

Writes into --out:
  latest.pt              model, weight average, optimiser, step, and the
                         dataset's fingerprint (resume loads it, and refuses
                         another dataset or stage-1 model unless told to)
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
from .labels import BIT, GROUPS, Labels, dataset_hash, group_of, load

CELL = 32
CROP = 480
NEGATIVES = 2048          # random wrong texts per step, besides every text in the batch


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


class Frames(torch.utils.data.Dataset):
    """Frames of one split with per-square targets. Only this split's sprites
    are kept, as arrays (labels.py), so loader processes get a small copy."""

    def __init__(self, dataset: Path, split: str, text: TextTable, crop: int | None, seed: int = 0,
                 augment: bool = False, source: tuple[dict, Labels] | None = None):
        data, labels = source or load(dataset)
        self.root = dataset
        frames = [f for f in data["frames"] if f["split"] == split]
        if crop:
            # Training crops are all crop x crop, so smaller frames can't be used.
            frames = [f for f in frames if f["width"] >= crop and f["height"] >= crop]
        idx = np.concatenate([np.arange(f["first"], f["first"] + f["count"]) for f in frames]) \
            if frames else np.zeros(0, np.int64)
        self.frames, n = [], 0
        for f in frames:
            self.frames.append({"picture": f["picture"], "width": f["width"], "height": f["height"],
                                "first": n, "count": f["count"], "dataset_first": f["first"]})
            n += f["count"]
        proto_row = np.array([text.row_of_entity.get(p, -1) for p in labels.prototypes] or [-1], np.int64)
        self.row = proto_row[labels.proto[idx]] if len(idx) else np.zeros(0, np.int64)
        self.box = labels.box[idx]
        self.depth = labels.depth[idx]
        self.flags = labels.flags[idx]
        self.group = np.array([group_of(int(d), int(fl)) for d, fl in zip(self.depth, self.flags)], np.int64)
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
            cw, ch = min(self.crop, w // CELL * CELL), min(self.crop, h // CELL * CELL)
            if self.augment and rng.random() < 0.5:
                # A slightly wrong tile size, as when the live view guesses it.
                zoom = rng.uniform(0.9, 1.1)
            if zoom == 1.0:
                x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
                img = pic.crop((x0, y0, x0 + cw, y0 + ch))
            else:
                sw, sh = cw / zoom, ch / zoom
                x0 = rng.uniform(min(0, w - sw), max(0, w - sw))
                y0 = rng.uniform(min(0, h - sh), max(0, h - sh))
                # Past the picture's edge is black, like space.
                img = pic.transform((cw, ch), Image.EXTENT, (x0, y0, x0 + sw, y0 + sh), Image.BILINEAR)
        else:
            cw, ch = w // CELL * CELL, h // CELL * CELL
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            img = pic.crop((x0, y0, x0 + cw, y0 + ch))
        a = np.asarray(img, dtype=np.float32) / 255
        if self.augment:
            # Brighter or dimmer lighting, and a different light curve.
            a = np.clip(a * rng.uniform(0.7, 1.3), 0, 1) ** rng.uniform(0.8, 1.25)
        gw, gh = cw // CELL, ch // CELL
        r = slice(f["first"], f["first"] + f["count"])
        bx, flags, rows, groups, depth = self.box[r], self.flags[r], self.row[r], self.group[r], self.depth[r]
        cx = ((bx[:, 0] + bx[:, 2]) / 2 - x0) * zoom
        cy = ((bx[:, 1] + bx[:, 3]) / 2 - y0) * zoom
        inside = (cx >= 0) & (cx < cw) & (cy >= 0) & (cy < ch)
        cells = (np.floor(cy / CELL).astype(np.int64) * gw + np.floor(cx / CELL).astype(np.int64))
        visible = (flags & BIT["visible"]) != 0
        named = inside & (rows >= 0)
        pos = np.stack([cells[named & visible], rows[named & visible]], 1)
        ignore = np.stack([cells[named & ~visible], rows[named & ~visible]], 1)
        box = np.zeros((gh * gw, 4), np.float32)
        something = np.zeros(gh * gw, np.float32)
        player = np.zeros(gh * gw, np.float32)
        subject = np.zeros(gh * gw, np.float32)
        # Squares holding only sprites the player can't see are neither right
        # nor wrong for "something" and "player": dark and faint are judged
        # from pixels and can be off.
        mask = np.ones(gh * gw, np.float32)
        mask[cells[inside & ~visible]] = 0
        top_depth = np.full(gh * gw, -1e9)
        for k in np.nonzero(inside)[0]:
            cell = cells[k]
            if flags[k] & BIT["subject"]:
                subject[cell] = 1
            if not visible[k]:
                continue
            something[cell] = mask[cell] = 1
            if flags[k] & BIT["humanoid"]:
                player[cell] = 1
            if depth[k] >= top_depth[cell]:
                top_depth[cell] = depth[k]
                box[cell] = [(cx[k] % CELL) / CELL, (cy[k] % CELL) / CELL,
                             math.log(max((bx[k, 2] - bx[k, 0]) * zoom, 1) / CELL),
                             math.log(max((bx[k, 3] - bx[k, 1]) * zoom, 1) / CELL)]
        return {"image": torch.from_numpy(a.transpose(2, 0, 1).copy()),
                "pos": torch.from_numpy(pos).long().reshape(-1, 2),
                "pos_group": torch.from_numpy(groups[named & visible]).long(),
                "ignore": torch.from_numpy(ignore).long().reshape(-1, 2),
                "something": torch.from_numpy(something), "player": torch.from_numpy(player),
                "mask": torch.from_numpy(mask), "subject": torch.from_numpy(subject),
                "box": torch.from_numpy(box)}


def collate(items: list[dict]) -> dict:
    cells = items[0]["something"].shape[0]
    out = {k: torch.stack([it[k] for it in items]) if k == "image" else torch.cat([it[k] for it in items])
           for k in ("image", "something", "player", "mask", "subject", "box", "pos_group")}
    for key in ("pos", "ignore"):
        pairs = [it[key] + torch.tensor([b * cells, 0]) for b, it in enumerate(items)]
        out[key] = torch.cat(pairs) if pairs else torch.zeros(0, 2, dtype=torch.long)
    return out


class DetectModel(nn.Module):
    def __init__(self, encoder: Encoder, text_dim: int):
        super().__init__()
        self.encoder = encoder
        self.text_proj = nn.Linear(text_dim, EMBED_DIM)
        self.name = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, EMBED_DIM))
        self.something = nn.Linear(EMBED_DIM, 1)
        self.player = nn.Linear(EMBED_DIM, 1)
        self.box = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, 4))
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
        # The name head starts as a residual on stage 1's tile vectors.
        q = F.normalize(t + self.name(t), dim=-1)
        b = self.box(t)
        return {"names": q, "something": self.something(t).squeeze(1), "player": self.player(t).squeeze(1),
                "box": torch.cat([torch.sigmoid(b[:, :2]), b[:, 2:]], 1), "grid": tiles.shape[1:3]}


def from_stage1(start: Path) -> tuple[DetectModel, TextTable, dict]:
    ckpt = load_checkpoint(start / "latest.pt")
    if not ckpt.get("text_dim"):
        raise SystemExit(f"{start} was trained without text matching, so it can't name things; "
                         "start from a model trained with text")
    encoder = Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"})
    weights = ckpt.get("ema") or ckpt["model"]
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in weights.items() if k.startswith("encoder.")})
    table = TextTable(start / "text.npz")
    model = DetectModel(encoder, table.dim)
    model.text_proj.load_state_dict({k[len("text_proj."):]: v for k, v in weights.items()
                                     if k.startswith("text_proj.")})
    nn.init.zeros_(model.name[2].weight)
    nn.init.zeros_(model.name[2].bias)
    return model, table, ckpt


def losses(model: DetectModel, out: dict, b: dict, trained: torch.Tensor | None = None,
           generator: torch.Generator | None = None) -> dict:
    """trained: which text rows are right answers somewhere in the training
    frames (None: all). Only those are pushed away as wrong answers; a name
    training never shows stays neutral, so the model isn't taught to avoid it."""
    q, n_cells = out["names"], out["names"].shape[0]
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
    logits = q @ model.texts(rows).T * model.scale.exp() + model.bias        # (cells, K)
    target = torch.zeros_like(logits)
    weight = torch.ones_like(logits)
    pos_cell, pos_col = b["pos"][:, 0], col[b["pos"][:, 1]]
    # Unseen sprites in a square, and texts sharing a right answer's name,
    # are neither right nor wrong there.
    keep = col[b["ignore"][:, 1]] >= 0
    weight[b["ignore"][keep, 0], col[b["ignore"][keep, 1]]] = 0
    same = model.name_ids[rows][None, :] == model.name_ids[b["pos"][:, 1]][:, None]     # (P, K)
    p_i, k_i = same.nonzero(as_tuple=True)
    weight[pos_cell[p_i], k_i] = 0
    target[pos_cell, pos_col] = 1
    weight[pos_cell, pos_col] = 1
    names = (F.binary_cross_entropy_with_logits(logits, target, weight, reduction="sum") / n_cells)
    m = b["mask"]
    something = F.binary_cross_entropy_with_logits(out["something"], b["something"], m, reduction="sum") \
        / m.sum().clamp(min=1)
    # Players are rare squares, so their misses count more.
    player = F.binary_cross_entropy_with_logits(out["player"], b["player"], m, reduction="sum",
                                                pos_weight=torch.tensor(10.0, device=dev)) / m.sum().clamp(min=1)
    s = b["something"]
    box = (F.smooth_l1_loss(out["box"], b["box"], reduction="none", beta=0.1).sum(1) * s).sum() / s.sum().clamp(min=1)
    return {"names": names, "something": something, "player": player, "box": box,
            "total": names + something + player + box}


MACRO_MIN = 20            # names with fewer held-out sprites than this are left out of the per-name average


@torch.no_grad()
def evaluate(model: DetectModel, loader, device, stop=lambda: False, trained: torch.Tensor | None = None) -> dict:
    """Held-out frames. Naming: top-1/top-5 per sprite (by name), split into
    names training showed (seen) and names it never did (unseen), averaged per
    name (macro: every name with MACRO_MIN or more sprites counts the same),
    and per group of draw depths (labels.GROUPS). Name sets: per square, the
    names scored 0.5 or more against the names truly centred there.
    "Something" and player precision and recall at 0.5, leaving out squares
    holding only sprites that can't be seen, and for players the followed
    player's own square (always the middle of the view). Box overlap where
    both say something is there."""
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
    c = dict.fromkeys(("tp", "fp", "fn", "ptp", "pfp", "pfn", "stp", "sfp", "sfn"), 0)
    ious = []
    for b in loader:
        if stop():
            break
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["image"])
        sims = out["names"] @ all_t.T
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
        c["stp"] += int((guess & there).sum())
        c["sfp"] += int((guess & ~there).sum())
        c["sfn"] += int((~guess & there).sum())

        counted = b["mask"] > 0
        g, t = (out["something"] > 0) & counted, (b["something"] > 0) & counted
        c["tp"] += int((g & t).sum())
        c["fp"] += int((g & ~t).sum())
        c["fn"] += int((~g & t).sum())
        others = counted & (b["subject"] == 0)
        pg, pt = (out["player"] > 0) & others, (b["player"] > 0) & others
        c["ptp"] += int((pg & pt).sum())
        c["pfp"] += int((pg & ~pt).sum())
        c["pfn"] += int((~pg & pt).sum())
        both = g & t
        if both.any():
            ious.append(_iou(out["box"][both], b["box"][both]))
    model.train(was_training)
    iou = torch.cat(ious) if ious else torch.zeros(0)
    ratio = lambda a, n: float(a) / max(1.0, float(n))
    per_name, per_group, per_seen = per_name.cpu(), per_group.cpu(), per_seen.cpu()
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
        "name_set_precision": ratio(c["stp"], c["stp"] + c["sfp"]),
        "name_set_recall": ratio(c["stp"], c["stp"] + c["sfn"]),
        "something_precision": ratio(c["tp"], c["tp"] + c["fp"]),
        "something_recall": ratio(c["tp"], c["tp"] + c["fn"]),
        "player_precision": ratio(c["ptp"], c["ptp"] + c["pfp"]),
        "player_recall": ratio(c["ptp"], c["ptp"] + c["pfn"]),
        "box_iou": float(iou.mean()) if len(iou) else 0.0,
        "box_iou_over_half": float((iou > 0.5).float().mean()) if len(iou) else 0.0}
    for k, name in enumerate(GROUPS):
        key = name.replace(" ", "_").replace("-", "_")
        scores[f"top5_{key}"] = ratio(per_group[k, 2], per_group[k, 0])
        scores[f"sprites_{key}"] = int(per_group[k, 0])
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
    if source is start:
        model, table, ckpt = from_stage1(start)
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
        model = DetectModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}), table.dim)
    device = pick_device(opts.device)
    model.to(device)
    model.set_text(table)

    loaded = load(dataset)
    summary = loaded[0].get("summary", {})
    train_ds = Frames(dataset, "train", table, CROP, opts.seed, augment=True, source=loaded)
    test_ds = Frames(dataset, "test", table, None, source=loaded)
    del loaded
    if not len(train_ds):
        raise SystemExit(f"{dataset} has no training frames")
    trained = torch.zeros(len(table.texts), dtype=torch.bool)
    trained[torch.from_numpy(train_ds.trained_rows())] = True
    trained = trained.to(device)
    print(f"detect: {len(train_ds):,} training frames, {len(test_ds):,} held-out frames, "
          f"{len(table.texts):,} entity texts ({int(trained.sum()):,} shown in training; the others are "
          f"never pushed away), on {device}", flush=True)
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
        # Models from before the player head resume with a fresh one.
        model.load_state_dict(ckpt["model"], strict=False)
        try:
            optim.load_state_dict(ckpt["optim"])
        except ValueError:
            print("detect: this model has a new head since it was saved, so the optimiser starts afresh", flush=True)
        step = ckpt["step"]
        print(f"detect: resumed {latest} at step {step:,}", flush=True)
    average = WeightAverage(model, opts.ema)
    if source is out:
        average.model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=False)
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
            print(f"detect: step {step:,}: naming {scores['name_top1']:.1%} / {scores['name_top5']:.1%} "
                  f"(top 1 / top 5) on {scores['named_sprites']:,} held-out sprites; names training showed "
                  f"{scores['seen_top5']:.1%}, never showed {scores['unseen_top5']:.1%} "
                  f"({scores['unseen_sprites']:,} sprites); per name {scores['macro_top5']:.1%} "
                  f"({scores['macro_names']:,} names); name sets {scores['name_set_precision']:.1%} precise, "
                  f"{scores['name_set_recall']:.1%} found; something {scores['something_precision']:.1%} "
                  f"precise, {scores['something_recall']:.1%} found; other players "
                  f"{scores['player_precision']:.1%} precise, {scores['player_recall']:.1%} found; "
                  f"box overlap {scores['box_iou']:.2f}", flush=True)
            print("detect: top 5 by group: " + ", ".join(
                f"{g} {scores['top5_' + g.replace(' ', '_').replace('-', '_')]:.1%}"
                f" ({scores['sprites_' + g.replace(' ', '_').replace('-', '_')]:,})" for g in GROUPS), flush=True)
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
                                f"loss {last['total']:.3f} (names {last['names']:.3f}, something "
                                f"{last['something']:.3f}, box {last['box']:.3f}), {ms:.0f} ms/step")
                print(f"detect: step {step:,}/{opts.steps:,} loss {last['total']:.4f} {ms:.0f} ms/step", flush=True)
                sums, count, t0 = {}, 0, time.time()
            if step % opts.checkpoint_every == 0:
                row = save()
    if step and (not row or row.get("step") != step):
        row = save(final=True)
    return row
