"""Train (or resume) single-sprite pretraining.

Writes into the training folder (--out):
  index.json            labels and splits, fixed for the life of this folder
  config.json           the options this folder was started with
  latest.pt             model, heads, optimizer and step: what a resume loads
  checkpoints/step-N.pt weights at every checkpoint, all kept for rollback
  metrics.jsonl         one line per checkpoint: losses and held-out scores

and, when a run folder is set, status.json, metrics.jsonl and one watch-set
folder per checkpoint (docs/run-folder.md).
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import __version__
from ..model.encoder import EMBED_DIM, Encoder, build_encoder, count_parameters
from ..progress import Progress, _write_json_atomic
from .data import SpriteDataset, load_or_build_index
from .watch import WATCHSET_SIZE, draw_watchset, pick_watchset

EVAL_LIMIT = 2048


@dataclass
class TrainOptions:
    data: Path
    out: Path
    size: str = "tiny"
    steps: int = 20000
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup: int = 500
    checkpoint_every: int = 1000
    checkpoint_minutes: float = 15.0
    minutes: float | None = None       # stop (cleanly) after this long
    workers: int = 2
    device: str = "auto"
    seed: int = 14
    fresh: bool = False


class PretrainModel(nn.Module):
    """The encoder plus the stage-1 heads. Only the encoder is exported."""

    def __init__(self, encoder: Encoder, n_categories: int, n_states: int):
        super().__init__()
        self.encoder = encoder
        self.category = nn.Linear(EMBED_DIM, n_categories)
        self.state = nn.Linear(EMBED_DIM, n_states)
        self.direction = nn.Linear(EMBED_DIM, 4)

    def forward(self, x):
        _tiles, g = self.encoder(x)
        return self.category(g), self.state(g), self.direction(g)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(size: str, index: dict) -> PretrainModel:
    return PretrainModel(build_encoder(size), len(index["categories"]), len(index["states"]))


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(path: Path) -> tuple[PretrainModel, dict]:
    """Rebuild the model from a latest.pt or checkpoints/step-N.pt file."""
    ckpt = load_checkpoint(path)
    model = PretrainModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}),
                          ckpt["n_categories"], ckpt["n_states"])
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def lr_at(step: int, opts: TrainOptions) -> float:
    if step < opts.warmup:
        return opts.lr * (step + 1) / opts.warmup
    t = min(1.0, (step - opts.warmup) / max(1, opts.steps - opts.warmup))
    return opts.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))


def _ignore_stop_signals(_worker_id: int) -> None:
    # The app stops a job by signalling its whole process group. Workers must
    # outlive that signal so the main process can save a checkpoint first.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _loader(ds, batch: int, shuffle: bool, workers: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                       drop_last=shuffle, generator=g, worker_init_fn=_ignore_stop_signals,
                                       persistent_workers=workers > 0 and shuffle)


def _subset(ds: SpriteDataset, limit: int, seed: int) -> SpriteDataset:
    if len(ds.items) > limit:
        ds.items = random.Random(seed).sample(ds.items, limit)
    return ds


def _losses(outputs, cat, state, direction):
    cat_logits, state_logits, dir_logits = outputs
    loss_cat = F.cross_entropy(cat_logits, cat, label_smoothing=0.1)
    loss_state = F.cross_entropy(state_logits, state, ignore_index=-1, label_smoothing=0.1) \
        if (state >= 0).any() else cat_logits.sum() * 0
    loss_dir = F.cross_entropy(dir_logits, direction, ignore_index=-1) \
        if (direction >= 0).any() else cat_logits.sum() * 0
    return loss_cat + loss_state + 0.5 * loss_dir, (loss_cat, loss_state, loss_dir)


@torch.no_grad()
def evaluate(model: PretrainModel, loader, device, stop=lambda: False) -> dict:
    """Top-1/top-5 on whichever labels the split has."""
    model.eval()
    n_cat = n_state = n_dir = 0
    cat1 = cat5 = st1 = st5 = dir1 = 0
    for x, cat, state, direction in loader:
        if stop():
            break
        x, cat, state, direction = x.to(device), cat.to(device), state.to(device), direction.to(device)
        c, s, d = model(x)
        top = c.topk(min(5, c.shape[1]), dim=1).indices
        cat1 += (top[:, 0] == cat).sum().item()
        cat5 += (top == cat[:, None]).any(dim=1).sum().item()
        n_cat += len(cat)
        m = state >= 0
        if m.any():
            top = s[m].topk(min(5, s.shape[1]), dim=1).indices
            st1 += (top[:, 0] == state[m]).sum().item()
            st5 += (top == state[m][:, None]).any(dim=1).sum().item()
            n_state += int(m.sum())
        m = direction >= 0
        if m.any():
            dir1 += (d[m].argmax(1) == direction[m]).sum().item()
            n_dir += int(m.sum())
    model.train()
    out = {"n": n_cat}
    if n_cat:
        out.update(category_top1=cat1 / n_cat, category_top5=cat5 / n_cat)
    if n_state:
        out.update(state_top1=st1 / n_state, state_top5=st5 / n_state)
    if n_dir:
        out.update(direction_top1=dir1 / n_dir)
    return out


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def train(opts: TrainOptions, progress: Progress | None = None) -> dict:
    progress = progress or Progress(None)
    started = time.time()
    out = opts.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    latest = out / "latest.pt"
    if opts.fresh and latest.exists():
        latest.unlink()
    _seed_everything(opts.seed)

    progress.update("index sprites", message="reading the sprite catalogue", force=True)
    index = load_or_build_index(opts.data, out, seed=opts.seed, progress=progress)
    print(f"train: {index['counts']} images, {len(index['categories'])} categories, "
          f"{len(index['states']):,} states", flush=True)

    device = pick_device(opts.device)
    model = build_model(opts.size, index).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": opts.weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}], lr=opts.lr)
    step = 0
    if latest.exists():
        ckpt = load_checkpoint(latest, map_location=device)
        if ckpt["encoder_config"]["size"] != opts.size:
            raise SystemExit(f"{out} holds a {ckpt['encoder_config']['size']} model, not {opts.size}; "
                             "pick another --out or pass --fresh")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"train: resumed {latest} at step {step:,}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(
            {**{k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()},
             "version": __version__}, indent=2))
    params = count_parameters(model.encoder)
    print(f"train: {opts.size} encoder, {params / 1e6:.1f}M parameters, on {device}", flush=True)

    train_ds = SpriteDataset(opts.data, index, "train", train=True, seed=opts.seed)
    unseen_ds = _subset(SpriteDataset(opts.data, index, "unseen", train=False, seed=opts.seed), EVAL_LIMIT, opts.seed)
    newview_ds = _subset(SpriteDataset(opts.data, index, "newview", train=False, seed=opts.seed), EVAL_LIMIT, opts.seed)
    loader = _loader(train_ds, opts.batch, True, opts.workers, opts.seed + step)
    watch_items = pick_watchset(index, opts.seed, WATCHSET_SIZE)

    def checkpoint(reason: str) -> dict:
        # On a stop request, save straight away: the app kills the job after ~15 s.
        scores = {"unseen": {}, "newview": {}}
        if reason != "stop":
            progress.update("evaluate", step, opts.steps, "scoring held-out sprites", force=True)
            stop = lambda: progress.stop_requested
            scores = {"unseen": evaluate(model, _loader(unseen_ds, opts.batch, False, 0, 0), device, stop),
                      "newview": evaluate(model, _loader(newview_ds, opts.batch, False, 0, 0), device, stop)}
        name = f"step-{step:07d}"
        state = {"step": step, "model": model.state_dict(), "optim": optim.state_dict(),
                 "encoder_config": model.encoder.config, "n_categories": len(index["categories"]),
                 "n_states": len(index["states"]), "out": str(out), "version": __version__,
                 "options": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()}}
        _save(latest, state)
        _save(out / "checkpoints" / f"{name}.pt", {k: v for k, v in state.items() if k != "optim"})
        row = {"step": step, "time": time.time(), "reason": reason, "loss": avg_loss(),
               "unseen_category_top1": scores["unseen"].get("category_top1"),
               "unseen_category_top5": scores["unseen"].get("category_top5"),
               "newview_state_top1": scores["newview"].get("state_top1"),
               "newview_state_top5": scores["newview"].get("state_top5"),
               "newview_direction_top1": scores["newview"].get("direction_top1"),
               "images_seen": step * opts.batch}
        for target in filter(None, (out, progress.run_dir)):
            with open(target / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if progress.run_dir and reason != "stop":
            draw_watchset(model, opts.data, index, watch_items, device,
                          progress.run_dir / "watchset" / name, seed=opts.seed)
            _write_json_atomic(progress.run_dir / "watchset" / name / "meta.json",
                               {k: v for k, v in row.items() if v is not None})
        print(f"checkpoint {name}: " + ", ".join(
            f"{k} {v:.3f}" for k, v in row.items() if isinstance(v, float) and k != "time"), flush=True)
        return row

    window: list[float] = []

    def avg_loss() -> float | None:
        return sum(window) / len(window) if window else None

    model.train()
    last_ckpt_time = time.time()
    last_row = None
    t0, step0 = time.time(), step
    stopped = False
    if step == 0:
        last_row = checkpoint("start")
    while step < opts.steps and not stopped:
        for x, cat, state, direction in loader:
            if progress.stop_requested or (opts.minutes and time.time() - started > opts.minutes * 60):
                stopped = True
                break
            x, cat, state, direction = x.to(device), cat.to(device), state.to(device), direction.to(device)
            for group in optim.param_groups:
                group["lr"] = lr_at(step, opts)
            loss, _parts = _losses(model(x), cat, state, direction)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            window.append(loss.item())
            del window[:-100]
            rate = (time.time() - t0) / max(1, step - step0)
            progress.update("train", step, opts.steps, f"loss {avg_loss():.3f}, {1 / rate * opts.batch:.0f} img/s",
                            eta_seconds=rate * (opts.steps - step))
            if step % 50 == 0:
                print(f"step {step:,}/{opts.steps:,}  loss {avg_loss():.4f}  lr {lr_at(step, opts):.2e}  "
                      f"{rate * 1000:.0f} ms/step", flush=True)
            if (step % opts.checkpoint_every == 0
                    or time.time() - last_ckpt_time > opts.checkpoint_minutes * 60):
                last_row = checkpoint("interval")
                last_ckpt_time = time.time()
            if step >= opts.steps:
                break
    if not last_row or last_row["step"] != step:
        last_row = checkpoint("stop" if stopped else "end")
    done = step >= opts.steps
    if last_row["unseen_category_top1"] is None:
        message = f"stopped at step {step:,}; checkpoint saved, run again to resume"
    else:
        message = (f"unseen category {last_row['unseen_category_top1']:.1%}, "
                   f"new-view state {last_row['newview_state_top1'] or 0:.1%}")
    progress.update("done" if done else "stopped", step, opts.steps, message, eta_seconds=0, force=True)
    return {"complete": done, "step": step, "last": last_row}
