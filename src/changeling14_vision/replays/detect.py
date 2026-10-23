"""Stage 2: teach a sprite model to find and name things in real game frames.

Starts from a stage-1 model folder (its latest.pt and text.npz) and trains on
a replay dataset (dataset.py). The encoder gives one vector per 32 px square
of the frame; for each square three heads learn:

  names      which entities stand in this square: each vector is compared to
             every entity text, and the ones whose box centre falls in the
             square should match. Sprites the player can't see (dark, behind
             a wall) are neither right nor wrong answers there.
  something  whether any visible sprite is centred in the square
  box        for the topmost visible sprite centred there: where its centre
             sits in the square and how big its box is

Training crops random 480x480 windows at any pixel offset, so the model does
not learn that sprites line up with its squares. Scores come from whole
frames of rounds held out by the dataset (split "test").

Writes into --out:
  latest.pt              model, optimiser, step (resume loads it)
  checkpoints/step-N.pt  weights at every checkpoint
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
from ..pretrain.train import load_checkpoint, pick_device
from ..progress import Progress

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


class Frames(torch.utils.data.Dataset):
    """Frames of one split with per-square targets."""

    def __init__(self, dataset: Path, split: str, text: TextTable, crop: int | None, seed: int = 0):
        data = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
        self.root = dataset
        self.frames = [f for f in data["frames"] if f["split"] == split]
        self.text = text
        self.crop = crop
        self.seed = seed

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        img = np.asarray(Image.open(self.root / f["picture"]).convert("RGB"), dtype=np.float32) / 255
        h, w = img.shape[:2]
        if self.crop:
            rng = random.Random()
            cw, ch = min(self.crop, w // CELL * CELL), min(self.crop, h // CELL * CELL)
            x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        else:
            cw, ch = w // CELL * CELL, h // CELL * CELL
            x0, y0 = (w - cw) // 2, (h - ch) // 2
        img = img[y0:y0 + ch, x0:x0 + cw]
        gw, gh = cw // CELL, ch // CELL
        pos, ignore = [], []                          # (cell, text row)
        box = np.zeros((gh * gw, 4), np.float32)
        something = np.zeros(gh * gw, np.float32)
        top_depth = np.full(gh * gw, -1e9)
        for s in f["sprites"]:
            bx0, by0, bx1, by1 = s["box"]
            cx, cy = (bx0 + bx1) / 2 - x0, (by0 + by1) / 2 - y0
            if not (0 <= cx < cw and 0 <= cy < ch):
                continue
            cell = int(cy // CELL) * gw + int(cx // CELL)
            row = self.text.row_of_entity.get(s["prototype"])
            if row is not None:
                (pos if s["visible"] else ignore).append((cell, row))
            if s["visible"]:
                something[cell] = 1
                if s["draw_depth"] >= top_depth[cell]:
                    top_depth[cell] = s["draw_depth"]
                    box[cell] = [(cx % CELL) / CELL, (cy % CELL) / CELL,
                                 math.log(max(bx1 - bx0, 1) / CELL), math.log(max(by1 - by0, 1) / CELL)]
        return {"image": torch.from_numpy(img.transpose(2, 0, 1).copy()),
                "pos": torch.tensor(pos, dtype=torch.long).reshape(-1, 2),
                "ignore": torch.tensor(ignore, dtype=torch.long).reshape(-1, 2),
                "something": torch.from_numpy(something), "box": torch.from_numpy(box)}


def collate(items: list[dict]) -> dict:
    cells = items[0]["something"].shape[0]
    out = {"image": torch.stack([it["image"] for it in items]),
           "something": torch.cat([it["something"] for it in items]),
           "box": torch.cat([it["box"] for it in items])}
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
        return {"names": q, "something": self.something(t).squeeze(1),
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


def losses(model: DetectModel, out: dict, b: dict, generator: torch.Generator | None = None) -> dict:
    q, n_cells = out["names"], out["names"].shape[0]
    n_text = model.text_vectors.shape[0]
    dev = q.device
    # Texts asked about this step: every text in the batch plus random others.
    rows = torch.cat([b["pos"][:, 1], b["ignore"][:, 1],
                      torch.randint(0, n_text, (NEGATIVES,), device=dev, generator=generator)]).unique()
    col = torch.full((n_text,), -1, dtype=torch.long, device=dev)
    col[rows] = torch.arange(len(rows), device=dev)
    logits = q @ model.texts(rows).T * model.scale.exp() + model.bias        # (cells, K)
    target = torch.zeros_like(logits)
    weight = torch.ones_like(logits)
    pos_cell, pos_col = b["pos"][:, 0], col[b["pos"][:, 1]]
    # Unseen sprites in a square, and texts sharing a right answer's name,
    # are neither right nor wrong there.
    weight[b["ignore"][:, 0], col[b["ignore"][:, 1]]] = 0
    same = model.name_ids[rows][None, :] == model.name_ids[b["pos"][:, 1]][:, None]     # (P, K)
    p_i, k_i = same.nonzero(as_tuple=True)
    weight[pos_cell[p_i], k_i] = 0
    target[pos_cell, pos_col] = 1
    weight[pos_cell, pos_col] = 1
    names = (F.binary_cross_entropy_with_logits(logits, target, weight, reduction="sum") / n_cells)
    something = F.binary_cross_entropy_with_logits(out["something"], b["something"])
    m = b["something"]
    box = (F.smooth_l1_loss(out["box"], b["box"], reduction="none", beta=0.1).sum(1) * m).sum() / m.sum().clamp(min=1)
    return {"names": names, "something": something, "box": box, "total": names + something + box}


@torch.no_grad()
def evaluate(model: DetectModel, loader, device, stop=lambda: False) -> dict:
    """Held-out frames: naming top-1/top-5 per sprite (by name), "something"
    precision and recall at 0.5, and box overlap where both agree."""
    model.eval()
    all_t = model.texts()
    h1 = h5 = named = tp = fp = fn = 0
    ious = []
    for b in loader:
        if stop():
            break
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["image"])
        if len(b["pos"]):
            top = (out["names"][b["pos"][:, 0]] @ all_t.T).topk(min(5, all_t.shape[0]), dim=1).indices
            hit = model.name_ids[top] == model.name_ids[b["pos"][:, 1]][:, None]
            h1 += int(hit[:, 0].sum())
            h5 += int(hit.any(1).sum())
            named += len(b["pos"])
        guess = out["something"] > 0
        truth = b["something"] > 0
        tp += int((guess & truth).sum())
        fp += int((guess & ~truth).sum())
        fn += int((~guess & truth).sum())
        both = guess & truth
        if both.any():
            ious.append(_iou(out["box"][both], b["box"][both]))
    model.train()
    iou = torch.cat(ious) if ious else torch.zeros(0)
    return {"name_top1": h1 / max(1, named), "name_top5": h5 / max(1, named), "named_sprites": named,
            "something_precision": tp / max(1, tp + fp), "something_recall": tp / max(1, tp + fn),
            "box_iou": float(iou.mean()) if len(iou) else 0.0,
            "box_iou_over_half": float((iou > 0.5).float().mean()) if len(iou) else 0.0}


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
    if source is start:
        model, table, ckpt = from_stage1(start)
        shutil.copy2(start / "text.npz", out / "text.npz")
        print(f"detect: starting from {start / 'latest.pt'} (stage 1 step {ckpt.get('step', 0):,})", flush=True)
    else:
        ckpt = load_checkpoint(latest)
        table = TextTable(out / "text.npz")
        model = DetectModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}), table.dim)
    device = pick_device(opts.device)
    model.to(device)
    model.set_text(table)

    train_ds = Frames(dataset, "train", table, CROP, opts.seed)
    test_ds = Frames(dataset, "test", table, None)
    if not len(train_ds):
        raise SystemExit(f"{dataset} has no training frames")
    print(f"detect: {len(train_ds):,} training frames, {len(test_ds):,} held-out frames, "
          f"{len(table.texts):,} entity texts, on {device}", flush=True)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=opts.batch, shuffle=True, drop_last=True,
                                         num_workers=opts.workers, collate_fn=collate,
                                         persistent_workers=opts.workers > 0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, collate_fn=collate, num_workers=0)

    enc = list(model.encoder.parameters())
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optim = torch.optim.AdamW([{"params": enc, "lr": opts.encoder_lr, "base": opts.encoder_lr},
                               {"params": heads, "lr": opts.lr, "base": opts.lr}],
                              weight_decay=opts.weight_decay)
    step = 0
    if source is out:
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"detect: resumed {latest} at step {step:,}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(
            {**{k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()},
             "version": __version__}, indent=2))

    def save(final: bool = False) -> dict:
        progress.update("evaluate", step, opts.steps, "held-out round", force=True)
        scores = evaluate(model, test_loader, device, lambda: progress.stop_requested) if len(test_ds) else {}
        state = {"model": model.state_dict(), "optim": optim.state_dict(), "step": step,
                 "encoder_config": model.encoder.config, "text_dim": table.dim, "start": str(start),
                 "options": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()},
                 "version": __version__, "kind": "detect"}
        _save(latest, state)
        _save(out / "checkpoints" / f"step-{step}.pt", {k: v for k, v in state.items() if k != "optim"})
        row = {"step": step, "time": round(time.time() - started), **{k: round(v, 4) for k, v in last.items()},
               **scores}
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if scores:
            print(f"detect: step {step:,}: naming {scores['name_top1']:.1%} / {scores['name_top5']:.1%} "
                  f"(top 1 / top 5) on {scores['named_sprites']:,} held-out sprites; something "
                  f"{scores['something_precision']:.1%} precise, {scores['something_recall']:.1%} found; "
                  f"box overlap {scores['box_iou']:.2f}", flush=True)
        return row

    model.train()
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
            ls = losses(model, out_, b)
            optim.zero_grad(set_to_none=True)
            ls["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            for k, v in ls.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            count += 1
            if step % 25 == 0:
                last = {k: v / count for k, v in sums.items()}
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
