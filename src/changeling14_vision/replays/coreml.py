"""Core ML export of a replay-trained (detect) model, to run on the Neural
Engine (macOS).

The export is an ML Program at one fixed input size, the size the live
overlay feeds it (the Neural Engine wants fixed shapes), cached in the model
folder as coreml/detect_<H>x<W>.mlpackage. Naming stays outside: the export
returns each square's name vectors and the overlay compares only the squares
it keeps with the entity texts, which is a small matrix product.

    changeling14-vision replays coreml <model> [--size 288x320] [--check]
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..model.encoder import EMBED_DIM

from .detect import CELL, DetectModel

OUTPUTS = ("names", "something", "more", "player", "mob", "slot_box")
FLOOR_OUTPUTS = ("tile", "gas", "decal")


class _Export(nn.Module):
    """DetectModel.forward with fixed shapes and tensor-only outputs in a fixed
    order (traced shape arithmetic doesn't convert to Core ML)."""

    def __init__(self, model: DetectModel, height: int, width: int):
        super().__init__()
        self.model = model
        self.cells = (height // CELL) * (width // CELL)
        self.floor = tuple(h for h in FLOOR_OUTPUTS if getattr(model, h) is not None)
        # Attention reads its sizes from x.shape at run time; give it fixed ones.
        from ..model.encoder import Attention
        for mod in model.encoder.modules():
            if isinstance(mod, Attention):
                mod.forward = types.MethodType(_fixed_attention(self.cells, mod.qkv.in_features, mod.heads), mod)

    def forward(self, x):
        m, k, e, n = self.model, self.model.slots, EMBED_DIM, self.cells
        tiles, _ = m.encoder(x)
        t = tiles.reshape(n, e)
        q = F.normalize(t.reshape(n, 1, e) + m.name(t).reshape(n, k, e), dim=-1)
        b = m.box(t).reshape(n, k, 4)
        boxes = torch.cat([torch.sigmoid(b[:, :, :2]), b[:, :, 2:]], -1)
        out = [q, m.something(t).reshape(n), m.more(t), m.player(t).reshape(n), m.mob(t), boxes]
        out += [getattr(m, h)(t) for h in self.floor]
        return tuple(out)


def _fixed_attention(n: int, c: int, heads: int):
    d = c // heads

    def forward(self, x):
        qkv = self.qkv(x).reshape(1, n, 3, heads, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(1, n, c))
    return forward


def package_path(folder: Path, height: int, width: int) -> Path:
    return Path(folder) / "coreml" / f"detect_{height}x{width}.mlpackage"


def packages(folder: Path) -> list[tuple[int, int, Path]]:
    """Exported sizes in a model folder: (height, width, path)."""
    found = []
    for p in sorted((Path(folder) / "coreml").glob("detect_*x*.mlpackage")):
        try:
            h, w = (int(v) for v in p.stem.split("_", 1)[1].split("x"))
        except ValueError:
            continue
        found.append((h, w, p))
    return found


def export(folder: Path, height: int, width: int, log=print) -> Path:
    """Export the model in folder at height x width (multiples of 32) and
    return the .mlpackage path; reuses an existing export of that size."""
    import coremltools as ct

    from .predict import load_detect_model

    if height % CELL or width % CELL:
        raise SystemExit(f"the input size must be multiples of {CELL}, not {height}x{width}")
    folder = Path(folder).expanduser()
    out = package_path(folder, height, width)
    stamp = out / "c14_source.json"
    latest = folder / "latest.pt"
    if stamp.is_file() and json.loads(stamp.read_text()).get("latest_mtime") == latest.stat().st_mtime:
        return out
    model, _table, _dev = load_detect_model(folder, "cpu")   # a fresh copy: the export patches its attention
    wrapper = _Export(model, height, width).eval()
    x = torch.rand(1, 3, height, width)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, x, strict=False)
    names = list(OUTPUTS + wrapper.floor)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=(1, 3, height, width), dtype=np.float32)],
                    outputs=[ct.TensorType(name=n) for n in names], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.macOS13)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(".tmp-" + out.name)
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    ml.save(str(tmp))
    (tmp / "c14_source.json").write_text(json.dumps({"latest_mtime": latest.stat().st_mtime, "outputs": names,
                                                     "height": height, "width": width}))
    if out.exists():
        import shutil
        shutil.rmtree(out)
    tmp.rename(out)
    log(f"coreml: exported {folder.name} at {height}x{width} in {time.time() - t0:.0f} s -> {out}")
    return out


def placement(path: Path) -> dict:
    """How many operations Core ML plans for the Neural Engine, GPU and CPU
    (CPU_AND_NE allows only the Neural Engine and CPU), and the CPU ones."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    compiled = ct.utils.compile_model(str(path))
    plan = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts: dict[str, int] = {}
    cpu_ops: dict[str, int] = {}
    program = plan.model_structure.program
    for fn in program.functions.values():
        for op in fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            dev = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
            counts[dev] = counts.get(dev, 0) + 1
            if dev != "NeuralEngine":
                cpu_ops[op.operator_name] = cpu_ops.get(op.operator_name, 0) + 1
    return {"devices": counts, "not_on_neural_engine": cpu_ops}


class CoreMLEngine:
    """Runs an exported detect model; returns the torch model's output dict
    (tensors on the CPU)."""

    def __init__(self, path: Path, height: int, width: int):
        import coremltools as ct

        self.height, self.width = height, width
        self.ml = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        meta = json.loads((Path(path) / "c14_source.json").read_text())
        self.names = meta["outputs"]

    def __call__(self, x: np.ndarray) -> dict:
        res = self.ml.predict({"image": np.ascontiguousarray(x, dtype=np.float32)})
        out = {k: torch.from_numpy(np.asarray(res[k], dtype=np.float32)) for k in self.names}
        out["grid"] = (self.height // CELL, self.width // CELL)
        return out


def check(folder: Path, path: Path, frames: list[Path], threshold: float = 0.5) -> dict:
    """Compare the export with the torch model on saved frames (cropped to the
    export's size): first-slot "something" and player agreement per square,
    the name top-1 agreement and box overlap for squares both keep."""
    import cv2

    from .predict import _names, _tensor, guesses_fast, load_detect_model

    model, table, _ = load_detect_model(folder, "cpu")
    words = _names(table)
    all_t = model.texts()
    h, w = (int(v) for v in Path(path).stem.split("_", 1)[1].split("x"))
    engine = CoreMLEngine(path, h, w)
    agree_s = agree_p = squares = same_name = both = 0
    ious = []
    ms_ml, ms_torch = [], []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None or img.shape[0] < h or img.shape[1] < w:
            continue
        x = _tensor(img[:h, :w], "cpu")
        t0 = time.time()
        with torch.no_grad():
            ref = model(x)
        t1 = time.time()
        got = engine(x.numpy())
        ms_torch.append(t1 - t0)
        ms_ml.append(time.time() - t1)
        s_ref = torch.sigmoid(ref["something"]) > threshold
        s_got = torch.sigmoid(got["something"].reshape(-1)) > threshold
        p_ref = torch.sigmoid(ref["player"]) > threshold
        p_got = torch.sigmoid(got["player"].reshape(-1)) > threshold
        agree_s += int((s_ref == s_got).sum())
        agree_p += int((p_ref == p_got).sum())
        squares += s_ref.numel()
        a = {(round(g.box[0] / CELL), round(g.box[1] / CELL), g.slot): g
             for g in guesses_fast(model, words, ref, threshold, all_t=all_t)}
        b = {(round(g.box[0] / CELL), round(g.box[1] / CELL), g.slot): g
             for g in guesses_fast(model, words, got, threshold, all_t=all_t)}
        for key in a.keys() & b.keys():
            both += 1
            same_name += a[key].name == b[key].name
            ax0, ay0, ax1, ay1 = a[key].box
            bx0, by0, bx1, by1 = b[key].box
            iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
            union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - iw * ih
            ious.append(iw * ih / union if union > 0 else 0.0)
    med = lambda v: round(float(np.median(v)) * 1000, 1) if v else None
    return {"frames": len(ms_ml), "size": [h, w], "something_agree": agree_s / max(1, squares),
            "player_agree": agree_p / max(1, squares), "boxes_compared": both,
            "name_top1_agree": same_name / max(1, both), "box_iou_mean": float(np.mean(ious)) if ious else None,
            "ms_coreml": med(ms_ml[1:]), "ms_torch_cpu": med(ms_torch[1:])}
