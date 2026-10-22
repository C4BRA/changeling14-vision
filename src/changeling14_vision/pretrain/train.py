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

import contextlib
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
from .text import DEFAULT_TEXT_MODEL, TextTable, build_text_table
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
    amp: bool = False                  # half precision on MPS/CUDA (bfloat16 on CPU)
    text_model: str | None = DEFAULT_TEXT_MODEL   # None: no text matching


class PretrainModel(nn.Module):
    """The encoder plus the stage-1 heads. Only the encoder is exported."""

    def __init__(self, encoder: Encoder, n_categories: int, n_states: int, n_components: int = 0,
                 text_dim: int = 0):
        super().__init__()
        self.encoder = encoder
        self.category = nn.Linear(EMBED_DIM, n_categories)
        self.state = nn.Linear(EMBED_DIM, n_states)
        self.direction = nn.Linear(EMBED_DIM, 4)
        self.components = nn.Linear(EMBED_DIM, max(1, n_components))
        self.text_dim = text_dim
        if text_dim:
            # Frozen sentence vectors come in; a learned map puts them in the encoder's space.
            self.text_proj = nn.Linear(text_dim, EMBED_DIM)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.register_buffer("text_vectors", torch.zeros(0, max(1, text_dim)), persistent=False)

    def set_text(self, vectors, texts: list[str] | None = None) -> None:
        self.text_vectors = torch.as_tensor(vectors, dtype=torch.float32, device=self.category.weight.device)
        # Entity names, for captions: "name. description" -> "name".
        self.text_names = [t.split(". ", 1)[0] for t in texts] if texts else None

    def text_embeddings(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        t = self.text_vectors if rows is None else self.text_vectors[rows]
        return F.normalize(self.text_proj(t), dim=-1)

    def forward(self, x) -> dict:
        _tiles, g = self.encoder(x)
        return {"category": self.category(g), "state": self.state(g), "direction": self.direction(g),
                "components": self.components(g), "global": g}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(size: str, index: dict, text_dim: int = 0) -> PretrainModel:
    return PretrainModel(build_encoder(size), len(index["categories"]), len(index["states"]),
                         len(index.get("components", ())), text_dim)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(path: Path) -> tuple[PretrainModel, dict]:
    """Rebuild the model from a latest.pt or checkpoints/step-N.pt file."""
    ckpt = load_checkpoint(path)
    model = PretrainModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}),
                          ckpt["n_categories"], ckpt["n_states"], ckpt.get("n_components", 0),
                          ckpt.get("text_dim", 0))
    model.load_state_dict(ckpt["model"])
    text_file = Path(ckpt.get("out", "")) / "text.npz"
    if model.text_dim and text_file.is_file():
        table = TextTable(text_file)
        model.set_text(table.vectors, table.texts)
    return model, ckpt


def _mixed_precision(device: torch.device, enabled: bool):
    """(autocast context factory, grad scaler or None). Off by default: half
    precision is usually faster on Apple and NVIDIA GPUs, but check that the
    loss still falls before relying on it."""
    if not enabled:
        return contextlib.nullcontext, None
    dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    scaler = None
    if dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler(device.type)
        except (RuntimeError, TypeError, ValueError):
            scaler = None
    return (lambda: torch.autocast(device_type=device.type, dtype=dtype)), scaler


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
    sampler = torch.utils.data.WeightedRandomSampler(ds.weights(), len(ds), generator=g) if shuffle else None
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                                       drop_last=shuffle, worker_init_fn=_ignore_stop_signals,
                                       persistent_workers=workers > 0 and shuffle)


def _subset(ds: SpriteDataset, limit: int, seed: int) -> SpriteDataset:
    if len(ds.items) > limit:
        ds.items = random.Random(seed).sample(ds.items, limit)
    return ds


def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _text_loss(model: PretrainModel, out: dict, b: dict) -> torch.Tensor:
    """SigLIP: every image-text pair in the batch is its own yes/no question."""
    m = b["text"] >= 0
    if not model.text_dim or int(m.sum()) < 2:
        return out["category"].sum() * 0
    rows = b["text"][m]
    zi = F.normalize(out["global"][m].float(), dim=-1)
    zt = model.text_embeddings(rows)
    logits = zi @ zt.T * model.logit_scale.exp() + model.logit_bias
    labels = (rows[:, None] == rows[None, :]).float() * 2 - 1
    return -F.logsigmoid(labels * logits).sum() / len(rows)


def _losses(out: dict, b: dict):
    zero = out["category"].sum() * 0
    loss_cat = F.cross_entropy(out["category"], b["category"], label_smoothing=0.1)
    loss_state = F.cross_entropy(out["state"], b["state"], ignore_index=-1, label_smoothing=0.1) \
        if (b["state"] >= 0).any() else zero
    loss_dir = F.cross_entropy(out["direction"], b["direction"], ignore_index=-1) \
        if (b["direction"] >= 0).any() else zero
    m = b["has_components"]
    loss_comp = F.binary_cross_entropy_with_logits(out["components"][m], b["components"][m]) * 10 \
        if m.any() else zero
    return loss_cat + loss_state + 0.5 * loss_dir + loss_comp, (loss_cat, loss_state, loss_dir, loss_comp)


@torch.no_grad()
def evaluate(model: PretrainModel, loader, device, stop=lambda: False) -> dict:
    """Top-1/top-5 on whichever labels the split has."""
    model.eval()
    n_cat = n_state = n_dir = n_txt = 0
    cat1 = cat5 = st1 = st5 = dir1 = tx1 = tx5 = 0
    tp = fp = fn = 0
    all_text = model.text_embeddings() if model.text_dim and len(model.text_vectors) else None
    for batch in loader:
        if stop():
            break
        b = _to(batch, device)
        cat, state, direction = b["category"], b["state"], b["direction"]
        out = model(b["x"])
        c, s, d = out["category"], out["state"], out["direction"]
        m = b["text"] >= 0
        if all_text is not None and m.any():
            sims = F.normalize(out["global"][m], dim=-1) @ all_text.T
            top = sims.topk(min(5, sims.shape[1]), dim=1).indices            # (n, 5)
            hit = (top[:, :, None] == b["text_rows"][m][:, None, :]).any(-1)  # (n, 5)
            tx1 += hit[:, 0].sum().item()
            tx5 += hit.any(1).sum().item()
            n_txt += int(m.sum())
        m = b["has_components"]
        if m.any():
            pred = out["components"][m] > 0
            truth = b["components"][m] > 0.5
            tp += (pred & truth).sum().item()
            fp += (pred & ~truth).sum().item()
            fn += (~pred & truth).sum().item()
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
    if n_txt:
        out.update(text_top1=tx1 / n_txt, text_top5=tx5 / n_txt, text_n=n_txt)
    if tp + fp + fn:
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        out.update(components_precision=precision, components_recall=recall,
                   components_f1=2 * precision * recall / max(1e-9, precision + recall))
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

    text = None
    if opts.text_model:
        try:
            progress.update("embed text", message=opts.text_model, force=True)
            text = TextTable(build_text_table(opts.data, out, model_name=opts.text_model, progress=progress))
            print(f"train: {len(text.texts):,} entity texts from {text.model}", flush=True)
        except ImportError:
            print("train: sentence-transformers is not installed, so training runs without text matching "
                  '(pip install -e ".[text]")', flush=True)
        except ValueError as exc:
            print(f"train: no text matching: {exc}", flush=True)

    device = pick_device(opts.device)
    model = build_model(opts.size, index, text.dim if text else 0).to(device)
    if text:
        model.set_text(text.vectors, text.texts)
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
        if ckpt.get("text_dim", 0) != model.text_dim:
            raise SystemExit(f"{out} was trained {'with' if ckpt.get('text_dim') else 'without'} text matching; "
                             "pass --fresh to start over, or match the earlier setting")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"train: resumed {latest} at step {step:,}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(
            {**{k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()},
             "version": __version__}, indent=2))
    autocast, scaler = _mixed_precision(device, opts.amp)
    params = count_parameters(model.encoder)
    print(f"train: {opts.size} encoder, {params / 1e6:.1f}M parameters, on {device}", flush=True)

    train_ds = SpriteDataset(opts.data, index, "train", train=True, seed=opts.seed, text=text)
    unseen_ds = _subset(SpriteDataset(opts.data, index, "unseen", train=False, seed=opts.seed, worn=False,
                                      text=text), EVAL_LIMIT, opts.seed)
    newview_ds = _subset(SpriteDataset(opts.data, index, "newview", train=False, seed=opts.seed, worn=False,
                                       text=text), EVAL_LIMIT, opts.seed)
    loader = _loader(train_ds, opts.batch, True, opts.workers, opts.seed + step)
    watch_items = pick_watchset(index, opts.seed, WATCHSET_SIZE, named=text is not None)

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
                 "n_states": len(index["states"]), "n_components": len(index.get("components", ())),
                 "text_dim": model.text_dim, "text_model": text.model if text else None,
                 "out": str(out), "version": __version__,
                 "options": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()}}
        _save(latest, state)
        _save(out / "checkpoints" / f"{name}.pt", {k: v for k, v in state.items() if k != "optim"})
        row = {"step": step, "time": time.time(), "reason": reason, "loss": avg_loss(),
               "unseen_category_top1": scores["unseen"].get("category_top1"),
               "unseen_category_top5": scores["unseen"].get("category_top5"),
               "unseen_components_f1": scores["unseen"].get("components_f1"),
               "unseen_text_top1": scores["unseen"].get("text_top1"),
               "unseen_text_top5": scores["unseen"].get("text_top5"),
               "newview_state_top1": scores["newview"].get("state_top1"),
               "newview_state_top5": scores["newview"].get("state_top5"),
               "newview_direction_top1": scores["newview"].get("direction_top1"),
               "images_seen": step * opts.batch}
        for target in filter(None, (out, progress.run_dir)):
            with open(target / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if progress.run_dir and reason != "stop":
            draw_watchset(model, opts.data, index, watch_items, device,
                          progress.run_dir / "watchset" / name, seed=opts.seed, text=text)
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
        for batch in loader:
            if progress.stop_requested or (opts.minutes and time.time() - started > opts.minutes * 60):
                stopped = True
                break
            b = _to(batch, device)
            for group in optim.param_groups:
                group["lr"] = lr_at(step, opts)
            with autocast():
                out_ = model(b["x"])
            out_ = {k: v.float() for k, v in out_.items()}
            loss, _parts = _losses(out_, b)
            loss = loss + _text_loss(model, out_, b)
            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
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
        message = (f"unseen: category {last_row['unseen_category_top1']:.1%}, "
                   f"components F1 {last_row['unseen_components_f1'] or 0:.2f}; "
                   f"new-view sprite {last_row['newview_state_top1'] or 0:.1%}")
    progress.update("done" if done else "stopped", step, opts.steps, message, eta_seconds=0, force=True)
    return {"complete": done, "step": step, "last": last_row}
