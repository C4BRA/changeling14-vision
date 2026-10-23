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
import copy
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
from .data import CELLS, PAIR_FRACTION, SpriteDataset, load_or_build_index, sprite_views
from .text import DEFAULT_TEXT_MODEL, TextTable, build_text_table
from .watch import WATCHSET_SIZE, draw_watchset, pick_watchset

EVAL_LIMIT = 2048
CAPTURE_PICTURES = 2048            # schedule-free: pictures that re-measure batch-norm statistics


@dataclass
class TrainOptions:
    data: Path
    out: Path
    size: str = "tiny"
    steps: int = 20000
    batch: int = 128
    lr: float | None = None            # None: scaled from the batch size (see base_lr)
    weight_decay: float = 0.05
    warmup: int | None = None          # steps; None: the first 64,000 pictures
    checkpoint_every: int | None = None   # steps; None: every 128,000 pictures
    checkpoint_minutes: float = 15.0
    minutes: float | None = None       # stop (cleanly) after this long
    workers: int = 2
    device: str = "auto"
    seed: int = 14
    fresh: bool = False
    amp: bool = False                  # half precision on MPS/CUDA (bfloat16 on CPU)
    text_model: str | None = DEFAULT_TEXT_MODEL   # None: no text matching
    rich_text: bool = False            # average name, suffix, folder and component phrases (text.py)
    ema: float = 0.999                 # decay of the weight average that is scored and saved
    optimizer: str = "adamw"           # adamw (cosine schedule) or schedulefree (no schedule)

    def __post_init__(self):
        # Batch 128 is the reference: bigger batches take proportionally fewer,
        # larger steps over the same pictures.
        if self.lr is None:
            self.lr = base_lr(self.optimizer) * math.sqrt(self.batch / 128)
        if self.warmup is None:
            self.warmup = max(1, round(500 * 128 / self.batch))
        if self.checkpoint_every is None:
            self.checkpoint_every = max(1, round(1000 * 128 / self.batch))


def base_lr(optimizer: str) -> float:
    return 2e-3 if optimizer == "schedulefree" else 1e-3


class PretrainModel(nn.Module):
    """The encoder plus the stage-1 heads. Only the encoder is exported."""

    def __init__(self, encoder: Encoder, n_categories: int, n_states: int, n_components: int = 0,
                 text_dim: int = 0):
        super().__init__()
        self.encoder = encoder
        self.category = nn.Linear(EMBED_DIM, n_categories)
        # Older models named each exact sprite with a 20k-way layer; newer ones
        # (n_states 0) learn "same sprite" by comparing two views instead.
        self.state = nn.Linear(EMBED_DIM, n_states) if n_states else None
        self.identity = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, 128)) \
            if not n_states else None
        self.direction = nn.Linear(EMBED_DIM, 4)
        self.components = nn.Linear(EMBED_DIM, max(1, n_components))
        self.text_dim = text_dim
        if text_dim:
            # Frozen sentence vectors come in; a learned map puts them in the encoder's space.
            self.text_proj = nn.Linear(text_dim, EMBED_DIM)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))
            # Temperature of the comparison against every entity text.
            self.name_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.register_buffer("text_vectors", torch.zeros(0, max(1, text_dim)), persistent=False)

    def set_text(self, vectors, texts: list[str] | None = None) -> None:
        self.text_vectors = torch.as_tensor(vectors, dtype=torch.float32, device=self.category.weight.device)
        # Entity names, for captions: "name. description" -> "name".
        self.text_names = [t.split(". ", 1)[0] for t in texts] if texts else None

    def text_embeddings(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        t = self.text_vectors if rows is None else self.text_vectors[rows]
        return F.normalize(self.text_proj(t), dim=-1)

    def forward(self, x) -> dict:
        tiles, g = self.encoder(x)
        out = {"category": self.category(g), "direction": self.direction(g), "components": self.components(g),
               "global": g, "tiles": tiles.flatten(1, 2)}            # tiles: (B, cells, E)
        if self.state is not None:
            out["state"] = self.state(g)
        return out


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(size: str, index: dict, text_dim: int = 0) -> PretrainModel:
    return PretrainModel(build_encoder(size), len(index["categories"]), 0,
                         len(index.get("components", ())), text_dim)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(path: Path) -> tuple[PretrainModel, dict]:
    """Rebuild the model from a latest.pt or checkpoints/step-N.pt file."""
    ckpt = load_checkpoint(path)
    model = PretrainModel(Encoder(**{k: v for k, v in ckpt["encoder_config"].items() if k != "tile"}),
                          ckpt["n_categories"], ckpt["n_states"], ckpt.get("n_components", 0),
                          ckpt.get("text_dim", 0))
    # The weight average is what training scores and what gets benchmarked and exported.
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
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
    # use() keeps each item's name rows lined up with it (they used to stay in
    # the old order, which scrambled the held-out naming scores).
    if len(ds.items) > limit:
        ds.use(random.Random(seed).sample(ds.items, limit))
    return ds


def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _text_loss(model: PretrainModel, out: dict, b: dict) -> torch.Tensor:
    """Two views of the same question, "which entity is this?":

    - SigLIP: every image-text pair in the batch is its own yes/no question.
    - Every name: a softmax over the whole text table, so each named sprite is
      pushed away from every other entity at every step, most from the ones it
      is most easily confused with. Any of the sprite's entities counts as right.
    The masks are applied without reading anything back from the GPU.
    """
    if not model.text_dim:
        return out["category"].sum() * 0
    m = (b["text"] >= 0).float()
    n = m.sum().clamp(min=1)
    rows = b["text"].clamp(min=0)
    zi = F.normalize(out["global"].float(), dim=-1)
    zt = model.text_embeddings(rows)
    logits = zi @ zt.T * model.logit_scale.exp() + model.logit_bias
    labels = (rows[:, None] == rows[None, :]).float() * 2 - 1
    pair = m[:, None] * m[None, :]
    siglip = -(F.logsigmoid(labels * logits) * pair).sum() / n

    return siglip + _every_name_loss(model, zi, b["text_rows"], model.text_embeddings())


def _every_name_loss(model: PretrainModel, z: torch.Tensor, wanted: torch.Tensor,
                     all_t: torch.Tensor) -> torch.Tensor:
    """Softmax over every entity text; z (N, E) unit vectors, wanted (N, k)
    right rows padded with -1. Rows with none add nothing."""
    m = (wanted >= 0).any(1).float()
    every = z @ all_t.T * model.name_scale.exp()                         # (N, texts)
    right = torch.zeros_like(every, dtype=torch.bool)
    right.scatter_(1, wanted.clamp(min=0), wanted >= 0)
    right[:, 0] |= ~right.any(1)          # rows with no name: any target, masked out below
    nll = torch.logsumexp(every, 1) - torch.logsumexp(every.masked_fill(~right, -1e4), 1)
    return (nll * m).sum() / m.sum().clamp(min=1)


def _tile_loss(model: PretrainModel, out: dict, b: dict) -> torch.Tensor:
    """Four-sprite canvases: each tile names (and groups) the sprite in it."""
    tiles = out["tiles"].float()
    cells = tiles.shape[1]
    if cells != CELLS:
        return tiles.sum() * 0
    loss = _masked_ce(model.category(tiles).flatten(0, 1), b["tile_category"].flatten())
    if model.text_dim:
        zt = F.normalize(tiles.flatten(0, 1), dim=-1)
        loss = loss + _every_name_loss(model, zt, b["tile_rows"].flatten(0, 1), model.text_embeddings())
    return loss


def _identity_loss(model: PretrainModel, z1_all: torch.Tensor, z2: torch.Tensor, b: dict, k: int) -> torch.Tensor:
    """SimCLR-style: the second view of sprite i must pick sprite i's first
    view out of the whole batch. Other copies of the same sprite in the batch
    also count as right. z1_all (B, E) and z2 (k, E) are encoder outputs."""
    if model.identity is None or k == 0:
        return z1_all.sum() * 0
    p1 = F.normalize(model.identity(z1_all.float()), dim=-1)
    p2 = F.normalize(model.identity(z2.float()), dim=-1)
    logits = p2 @ p1.T / 0.1                                              # (k, B)
    state = b["state"]
    same = (state[:k, None] == state[None, :]) & (state[:k, None] >= 0)
    same |= torch.eye(k, len(state), dtype=torch.bool, device=state.device)
    nll = torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~same, -1e4), 1)
    m = b["pair"][:k].float()
    return (nll * m).sum() / m.sum().clamp(min=1)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy over the rows with a label (-1 = none), with no GPU-to-CPU
    check of whether there are any: such checks stall the GPU every step."""
    m = (target >= 0).float()
    each = F.cross_entropy(logits, target.clamp(min=0), reduction="none", label_smoothing=smoothing)
    return (each * m).sum() / m.sum().clamp(min=1)


def _losses(out: dict, b: dict):
    loss_cat = _masked_ce(out["category"], b["category"], 0.1)
    loss_state = _masked_ce(out["state"], b["state"], 0.1) if "state" in out else out["category"].sum() * 0
    loss_dir = _masked_ce(out["direction"], b["direction"])
    m = b["has_components"].float()
    each = F.binary_cross_entropy_with_logits(out["components"], b["components"], reduction="none").mean(1)
    loss_comp = (each * m).sum() / m.sum().clamp(min=1) * 10
    return loss_cat + loss_state + 0.5 * loss_dir + loss_comp, (loss_cat, loss_state, loss_dir, loss_comp)


def _optimizer(opts: TrainOptions, groups: list[dict]):
    if opts.optimizer == "schedulefree":
        try:
            import schedulefree
        except ImportError as exc:
            raise SystemExit('the schedule-free optimiser needs: pip install -e ".[train]"') from exc
        optim = schedulefree.AdamWScheduleFree(groups, lr=opts.lr, warmup_steps=opts.warmup)
        optim.train()
        return optim
    if opts.optimizer != "adamw":
        raise SystemExit(f"unknown optimiser {opts.optimizer!r}; pick adamw or schedulefree")
    return torch.optim.AdamW(groups, lr=opts.lr)


class WeightAverage:
    """An exponential moving average of the weights (and batch-norm statistics),
    kept in a second copy of the model that is scored, saved and exported.
    The decay ramps up from 0, so early checkpoints are not stuck near the
    random start."""

    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        avg = [v for v in self.model.state_dict().values() if v.dtype.is_floating_point]
        new = [v for v in model.state_dict().values() if v.dtype.is_floating_point]
        torch._foreach_lerp_(avg, new, 1 - d)
        for a, v in zip(self.model.state_dict().values(), model.state_dict().values()):
            if not a.dtype.is_floating_point:
                a.copy_(v)

    @torch.no_grad()
    def capture(self, model: nn.Module, optim, loader, device, batches: int = 16) -> None:
        """Schedule-free: its evaluation weights are already an average, so copy
        them, then re-measure batch-norm statistics at those weights (they were
        gathered at the training weights)."""
        optim.eval()
        self.model.load_state_dict(model.state_dict())
        optim.train()
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum = None          # a plain average over the batches below
        with torch.no_grad():
            for n, b in enumerate(loader):
                if n >= batches:
                    break
                self.model(b["x"].to(device))
        self.model.eval()


def _memory_note(device) -> str:
    """GPU memory in use, so a slowdown can be told apart from memory growth."""
    try:
        if device.type == "mps":
            return f", GPU memory {torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
        if device.type == "cuda":
            return f", GPU memory {torch.cuda.memory_reserved(device) / 2**30:.1f} GB"
    except (AttributeError, RuntimeError):
        pass
    return ""


def _name_hits(z: torch.Tensor, wanted: torch.Tensor, all_text: torch.Tensor) -> tuple[int, int, int]:
    """(top-1 hits, top-5 hits, n) of unit vectors z against every text."""
    m = (wanted >= 0).any(1)
    if not m.any():
        return 0, 0, 0
    top = (z[m] @ all_text.T).topk(min(5, all_text.shape[0]), dim=1).indices      # (n, 5)
    hit = (top[:, :, None] == wanted[m][:, None, :]).any(-1)                      # (n, 5)
    return hit[:, 0].sum().item(), hit.any(1).sum().item(), int(m.sum())


@torch.no_grad()
def evaluate(model: PretrainModel, loader, device, stop=lambda: False) -> dict:
    """Top-1/top-5 on whichever labels the split has (whole pictures, and
    tiles of four-sprite canvases)."""
    model.eval()
    n_cat = n_state = n_dir = n_txt = n_tile = n_ttxt = 0
    cat1 = cat5 = st1 = st5 = dir1 = tx1 = tx5 = tile1 = ttx1 = ttx5 = 0
    tp = fp = fn = 0
    all_text = model.text_embeddings() if model.text_dim and len(model.text_vectors) else None
    for batch in loader:
        if stop():
            break
        b = _to(batch, device)
        cat, state, direction = b["category"], b["state"], b["direction"]
        out = model(b["x"])
        c, d = out["category"], out["direction"]
        if all_text is not None:
            h1, h5, n = _name_hits(F.normalize(out["global"], dim=-1), b["text_rows"], all_text)
            tx1, tx5, n_txt = tx1 + h1, tx5 + h5, n_txt + n
            zt = F.normalize(out["tiles"].flatten(0, 1), dim=-1)
            h1, h5, n = _name_hits(zt, b["tile_rows"].flatten(0, 1), all_text)
            ttx1, ttx5, n_ttxt = ttx1 + h1, ttx5 + h5, n_ttxt + n
        tc = b["tile_category"].flatten()
        mt = tc >= 0
        if mt.any():
            tile1 += (model.category(out["tiles"].flatten(0, 1)[mt]).argmax(1) == tc[mt]).sum().item()
            n_tile += int(mt.sum())
        m = b["has_components"]
        if m.any():
            pred = out["components"][m] > 0
            truth = b["components"][m] > 0.5
            tp += (pred & truth).sum().item()
            fp += (pred & ~truth).sum().item()
            fn += (~pred & truth).sum().item()
        m = cat >= 0
        if m.any():
            top = c[m].topk(min(5, c.shape[1]), dim=1).indices
            cat1 += (top[:, 0] == cat[m]).sum().item()
            cat5 += (top == cat[m][:, None]).any(dim=1).sum().item()
            n_cat += int(m.sum())
        m = state >= 0
        if "state" in out and m.any():
            s = out["state"]
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
    if n_tile:
        out.update(tile_category_top1=tile1 / n_tile, tile_n=n_tile)
    if n_ttxt:
        out.update(tile_text_top1=ttx1 / n_ttxt, tile_text_top5=ttx5 / n_ttxt)
    if tp + fp + fn:
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        out.update(components_precision=precision, components_recall=recall,
                   components_f1=2 * precision * recall / max(1e-9, precision + recall))
    return out


@torch.no_grad()
def embed(model: PretrainModel, ds: SpriteDataset, batch: int, device, stop=lambda: False) -> torch.Tensor:
    """Unit-length whole-picture embeddings of a dataset, in order."""
    model.eval()
    chunks = []
    for b in _loader(ds, batch, False, 0, 0):
        if stop():
            break
        chunks.append(F.normalize(model(b["x"].to(device))["global"].float(), dim=-1))
    model.train()
    return torch.cat(chunks) if chunks else torch.zeros(0, EMBED_DIM, device=device)


def newview_retrieval(model: PretrainModel, queries: SpriteDataset, gallery: SpriteDataset, batch: int,
                      device, stop=lambda: False) -> dict:
    """Same-sprite recognition without a per-sprite layer: each held-out view
    looks for its sprite among one training picture per sprite (top-1/top-5)."""
    q, g = embed(model, queries, batch, device, stop), embed(model, gallery, batch, device, stop)
    if not len(q) or not len(g):
        return {}
    position = {(it["sheet"], it["state"]): n for n, it in enumerate(gallery.items)}
    want = torch.tensor([position.get((it["sheet"], it["state"]), -1) for it in queries.items[:len(q)]],
                        device=device)
    top = (q @ g.T).topk(min(5, len(g)), dim=1).indices
    m = want >= 0
    hit = top[m] == want[m][:, None]
    n = int(m.sum())
    return {"state_top1": hit[:, 0].float().mean().item(), "state_top5": hit.any(1).float().mean().item(),
            "state_n": n} if n else {}


def retrieval_sets(index: dict, data: Path, seed: int, limit: int, text=None) -> tuple[SpriteDataset, SpriteDataset]:
    """Held-out views (up to limit) and a gallery: one training picture of each
    of their sprites plus as many other sprites as distractors."""
    queries = _subset(SpriteDataset(data, index, "newview", train=False, seed=seed, worn=False, text=text),
                      limit, seed)
    wanted = {(it["sheet"], it["state"]) for it in queries.items}
    first: dict[tuple[str, str], dict] = {}
    for it in index["items"]:
        if it["split"] == "train" and not it["worn"]:
            first.setdefault((it["sheet"], it["state"]), it)
    others = [k for k in first if k not in wanted]
    keys = sorted(wanted & first.keys()) + random.Random(seed).sample(others, min(limit, len(others)))
    gallery = SpriteDataset(data, index, "train", train=False, seed=seed, worn=False)
    return queries, gallery.use([first[k] for k in keys])


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
    if opts.fresh:
        # Labels and texts come from the catalogue, which may be another one now.
        for name in ("latest.pt", "index.json", "text.npz"):
            (out / name).unlink(missing_ok=True)
    _seed_everything(opts.seed)

    progress.update("index sprites", message="reading the sprite catalogue", force=True)
    index = load_or_build_index(opts.data, out, seed=opts.seed, progress=progress)
    print(f"train: {index['counts']} images, {len(index['categories'])} categories, "
          f"{len(index['states']):,} states", flush=True)

    text = None
    if opts.text_model:
        try:
            progress.update("embed text", message=opts.text_model, force=True)
            text = TextTable(build_text_table(opts.data, out, model_name=opts.text_model, progress=progress,
                                              rich=opts.rich_text))
            print(f"train: {len(text.texts):,} entity texts from {text.model}"
                  f"{' with folder and component phrases' if text.rich else ''}", flush=True)
            if text.rich != opts.rich_text:
                print("train: this model folder keeps the text it started with; --fresh rebuilds it", flush=True)
        except ImportError:
            print("train: sentence-transformers is not installed, so training runs without text matching "
                  '(pip install -e ".[text]")', flush=True)
        except ValueError as exc:
            print(f"train: no text matching: {exc}", flush=True)

    device = pick_device(opts.device)
    model = build_model(opts.size, index, text.dim if text else 0).to(device)
    if text:
        model.set_text(text.vectors, text.texts)
    average = WeightAverage(model, opts.ema)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": opts.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    optim = _optimizer(opts, groups)
    step = 0
    if latest.exists():
        ckpt = load_checkpoint(latest, map_location=device)
        if ckpt["encoder_config"]["size"] != opts.size:
            raise SystemExit(f"{out} holds a {ckpt['encoder_config']['size']} model, not {opts.size}; "
                             "pick another --out or pass --fresh")
        if ckpt.get("text_dim", 0) != model.text_dim:
            raise SystemExit(f"{out} was trained {'with' if ckpt.get('text_dim') else 'without'} text matching; "
                             "pass --fresh to start over, or match the earlier setting")
        if ckpt.get("options", {}).get("optimizer", "adamw") != opts.optimizer:
            raise SystemExit(f"{out} was trained with the {ckpt['options'].get('optimizer', 'adamw')} optimiser; "
                             "keep it, or start a new model")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        average.model.load_state_dict(ckpt.get("ema") or ckpt["model"])
        average.updates = ckpt.get("ema_updates", ckpt["step"])
        step = ckpt["step"]
        print(f"train: resumed {latest} at step {step:,}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(
            {**{k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()},
             "version": __version__}, indent=2))
    autocast, scaler = _mixed_precision(device, opts.amp)
    params = count_parameters(model.encoder)
    print(f"train: {opts.size} encoder, {params / 1e6:.1f}M parameters, on {device}", flush=True)

    train_items = [it for it in index["items"] if it["split"] == "train"]
    train_ds = SpriteDataset(opts.data, index, "train", train=True, seed=opts.seed, text=text,
                             views=sprite_views(train_items))
    unseen_ds = _subset(SpriteDataset(opts.data, index, "unseen", train=False, seed=opts.seed, worn=False,
                                      text=text), EVAL_LIMIT, opts.seed)
    # The same unseen sprites, four to a canvas, to score the per-tile output.
    unseen_tiles = _subset(SpriteDataset(opts.data, index, "unseen", train=False, seed=opts.seed, worn=False,
                                         text=text, mosaic=1.0), EVAL_LIMIT // 2, opts.seed)
    newview_ds = _subset(SpriteDataset(opts.data, index, "newview", train=False, seed=opts.seed, worn=False,
                                       text=text), EVAL_LIMIT, opts.seed)
    if model.state is None:
        queries, gallery = retrieval_sets(index, opts.data, opts.seed, EVAL_LIMIT // 2, text)
    pairs = max(1, int(opts.batch * PAIR_FRACTION)) if model.identity is not None else 0
    loader = _loader(train_ds, opts.batch, True, opts.workers, opts.seed + step)
    watch_items = pick_watchset(index, opts.seed, WATCHSET_SIZE, named=text is not None)

    def checkpoint(reason: str) -> dict:
        # On a stop request, save straight away: the app kills the job after ~15 s.
        scores = {"unseen": {}, "newview": {}, "tiles": {}}
        if opts.optimizer == "schedulefree":
            # About 2,048 pictures (512 on a stop) whatever the batch size.
            pictures = 512 if reason == "stop" else CAPTURE_PICTURES
            average.capture(model, optim, _loader(train_ds, opts.batch, True, 0, step), device,
                            batches=max(1, round(pictures / opts.batch)))
        if reason != "stop":
            progress.update("evaluate", step, opts.steps, "scoring held-out sprites", force=True)
            stop = lambda: progress.stop_requested
            scores = {"unseen": evaluate(average.model, _loader(unseen_ds, opts.batch, False, 0, 0), device, stop),
                      "newview": evaluate(average.model, _loader(newview_ds, opts.batch, False, 0, 0), device, stop),
                      "tiles": evaluate(average.model, _loader(unseen_tiles, opts.batch, False, 0, 0), device, stop)}
            if model.state is None:
                scores["newview"].update(newview_retrieval(average.model, queries, gallery, opts.batch, device, stop))
        name = f"step-{step:07d}"
        state = {"step": step, "model": model.state_dict(), "optim": optim.state_dict(),
                 "ema": average.model.state_dict(), "ema_updates": average.updates,
                 "encoder_config": model.encoder.config, "n_categories": len(index["categories"]),
                 "n_states": model.state.out_features if model.state is not None else 0, "n_components": len(index.get("components", ())),
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
               "unseen_tile_text_top1": scores["tiles"].get("tile_text_top1"),
               "unseen_tile_text_top5": scores["tiles"].get("tile_text_top5"),
               "unseen_tile_category_top1": scores["tiles"].get("tile_category_top1"),
               "images_seen": step * opts.batch}
        for target in filter(None, (out, progress.run_dir)):
            with open(target / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if progress.run_dir and reason != "stop":
            draw_watchset(average.model, opts.data, index, watch_items, device,
                          progress.run_dir / "watchset" / name, seed=opts.seed, text=text)
            _write_json_atomic(progress.run_dir / "watchset" / name / "meta.json",
                               {k: v for k, v in row.items() if v is not None})
        print(f"checkpoint {name}: " + ", ".join(
            f"{k} {v:.3f}" for k, v in row.items() if isinstance(v, float) and k != "time"), flush=True)
        return row

    window: list[float] = []
    pending: list[torch.Tensor] = []

    def avg_loss() -> float | None:
        return sum(window) / len(window) if window else None

    model.train()
    last_ckpt_time = time.time()
    last_row = None
    t0, step0 = time.time(), step
    stopped = False
    if step == 0:
        last_row = checkpoint("start")
        last_ckpt_time = t0 = time.time()
    while step < opts.steps and not stopped:
        for batch in loader:
            if progress.stop_requested or (opts.minutes and time.time() - started > opts.minutes * 60):
                stopped = True
                break
            b = _to(batch, device)
            if opts.optimizer == "adamw":
                for group in optim.param_groups:
                    group["lr"] = lr_at(step, opts)
            with autocast():
                # The first `pairs` samples also get their second view in the same pass.
                x = torch.cat([b["x"], b["x2"][:pairs]]) if pairs else b["x"]
                out_ = model(x)
            out_ = {k: v.float() for k, v in out_.items()}
            n = len(b["x"])
            second = out_["global"][n:]
            out_ = {k: v[:n] for k, v in out_.items()}
            loss, _parts = _losses(out_, b)
            loss = loss + _text_loss(model, out_, b) + _tile_loss(model, out_, b)
            loss = loss + _identity_loss(model, out_["global"], second, b, pairs)
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
            if opts.optimizer == "adamw":
                average.update(model)
            step += 1
            # Reading the loss waits for the GPU, so only every 10 steps.
            pending.append(loss.detach())
            if len(pending) >= 10:
                window.extend(torch.stack(pending).tolist())
                pending.clear()
                del window[:-100]
            # Step time since the last checkpoint, so checkpoints don't inflate it.
            rate = (time.time() - t0) / max(1, step - step0)
            speed = f"{1 / rate * opts.batch:.0f} img/s"
            progress.update("train", step, opts.steps, f"loss {avg_loss():.3f}, {speed}" if window else speed,
                            eta_seconds=rate * (opts.steps - step))
            if step % 50 == 0:
                print(f"step {step:,}/{opts.steps:,}  loss {avg_loss():.4f}  lr {lr_at(step, opts):.2e}  "
                      f"{rate * 1000:.0f} ms/step", flush=True)
            if (step % opts.checkpoint_every == 0
                    or time.time() - last_ckpt_time > opts.checkpoint_minutes * 60):
                began = time.time()
                last_row = checkpoint("interval")
                last_ckpt_time = t0 = time.time()
                step0 = step
                print(f"checkpoint took {last_ckpt_time - began:.0f} s{_memory_note(device)}", flush=True)
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
