"""Export the encoder (without the pretraining heads) to standard formats.

  safetensors  PyTorch weights + config.json; load with changeling14_vision.model
  onnx         any batch and any frame size (multiples of 32); checked with onnxruntime
  mlx          batch norms folded, NHWC layout; runs with model/mlx_encoder.py alone
  coreml       fixed frame size (--height/--width); needs macOS and coremltools

Each format is checked against PyTorch on a random frame where its runtime is
installed, and the differences go into export.json. A format whose tools are
missing is skipped with a note, not an error.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .model.encoder import EMBED_DIM, TILE, Encoder, count_parameters
from .progress import Progress

FORMATS = ("safetensors", "onnx", "mlx", "coreml")


def _spec(encoder: Encoder, step: int, extra: dict | None = None) -> dict:
    return {**encoder.config, "step": step, "version": __version__, "parameters": count_parameters(encoder),
            "input": "RGB float in [0, 1], height and width multiples of 32; 1 input pixel = 1 game pixel "
                     "at default zoom; resize with nearest-neighbour only",
            "outputs": {"tiles": f"(B, H/{TILE}, W/{TILE}, {EMBED_DIM}) one feature per 32 px square",
                        "global": f"(B, {EMBED_DIM}) one feature for the whole frame"},
            **(extra or {})}


def _fold_bn_after(w, b, bn):
    s = bn["weight"] / np.sqrt(bn["running_var"] + 1e-5)
    w = w * s[:, None, None, None]
    b = ((b if b is not None else 0) - bn["running_mean"]) * s + bn["bias"]
    return w, b


def mlx_weights(encoder: Encoder) -> dict[str, np.ndarray]:
    """Fold every batch norm and move conv weights to MLX's (out, kh, kw, in) layout."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in encoder.state_dict().items()}

    def bn(prefix):
        return {k: sd[f"{prefix}.{k}"] for k in ("weight", "bias", "running_mean", "running_var")}

    out: dict[str, np.ndarray] = {}

    def put_conv(name, w, b):
        out[f"{name}.weight"] = np.ascontiguousarray(w.transpose(0, 2, 3, 1))
        if b is not None:
            out[f"{name}.bias"] = b.astype(np.float32)

    put_conv("stem", *_fold_bn_after(sd["stem.0.weight"], None, bn("stem.1")))
    cfg = encoder.config
    for i, n in enumerate(cfg["conv_depths"]):
        put_conv(f"stages.{i}.0", *_fold_bn_after(sd[f"stages.{i}.0.conv.weight"], None, bn(f"stages.{i}.0.bn")))
        for j in range(1, n + 1):
            p = f"stages.{i}.{j}"
            put_conv(f"{p}.dw", *_fold_bn_after(sd[f"{p}.dw.weight"], None, bn(f"{p}.dw_bn")))
            pre = bn(f"{p}.bn")
            s = pre["weight"] / np.sqrt(pre["running_var"] + 1e-5)
            t = pre["bias"] - pre["running_mean"] * s
            w1 = sd[f"{p}.fc1.weight"]
            put_conv(f"{p}.fc1", w1 * s[None, :, None, None], sd[f"{p}.fc1.bias"] + w1[:, :, 0, 0] @ t)
            put_conv(f"{p}.fc2", sd[f"{p}.fc2.weight"], sd[f"{p}.fc2.bias"])
    for name in ("tokenize", "pos"):
        put_conv(name, sd[f"{name}.weight"], sd[f"{name}.bias"])
    for k, v in sd.items():
        if k.startswith(("blocks.", "norm.", "tile_head.", "global_head.")):
            out[k] = v
    return out


def _max_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def export(model_dir: Path, formats: list[str], progress: Progress | None = None, checkpoint: str | None = None,
           dest: Path | None = None, height: int = 480, width: int = 480) -> dict:
    from safetensors.numpy import save_file as save_numpy
    from safetensors.torch import save_file

    from .pretrain.train import model_from_checkpoint

    progress = progress or Progress(None)
    if height % TILE or width % TILE:
        raise SystemExit(f"--height and --width must be multiples of {TILE}")
    ckpt_path = Path(checkpoint).expanduser() if checkpoint else model_dir / "latest.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"no checkpoint at {ckpt_path}; train first")
    model, ckpt = model_from_checkpoint(ckpt_path)
    encoder = model.encoder.eval()
    step = ckpt["step"]
    size = encoder.config["size"]
    dest = dest or model_dir / "export" / f"step-{step:07d}"
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"changeling14-vision-{size}-p32"
    print(f"export: {ckpt_path} (step {step:,}) to {dest}", flush=True)

    torch.manual_seed(0)
    probe = torch.rand(1, 3, height, width)
    with torch.no_grad():
        ref_tiles, ref_global = (t.numpy() for t in encoder(probe))
    result: dict = {"checkpoint": str(ckpt_path), "step": step, "size": size, "time": time.time(),
                    "frame": [height, width], "formats": {}, "failed": []}

    for n, fmt in enumerate(formats):
        progress.update("export", n, len(formats), fmt, force=True)
        entry: dict = {}
        try:
            if fmt == "safetensors":
                path = dest / f"{stem}.safetensors"
                save_file({k: v.contiguous() for k, v in encoder.state_dict().items()}, str(path))
                (dest / f"{stem}.json").write_text(json.dumps(_spec(encoder, step), indent=2))
                entry = {"path": path.name}
            elif fmt == "onnx":
                path = dest / f"{stem}.onnx"
                torch.onnx.export(encoder, (probe,), str(path), input_names=["image"],
                                  output_names=["tiles", "global"], opset_version=17, dynamo=False,
                                  dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                                "tiles": {0: "batch", 1: "rows", 2: "cols"},
                                                "global": {0: "batch"}})
                entry = {"path": path.name}
                try:
                    import onnxruntime as ort
                except ImportError:
                    entry["checked"] = "onnxruntime not installed"
                else:
                    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    tiles, glob = sess.run(None, {"image": probe.numpy()})
                    entry["max_diff"] = max(_max_diff(tiles, ref_tiles), _max_diff(glob, ref_global))
            elif fmt == "mlx":
                folder = dest / f"{stem}-mlx"
                folder.mkdir(exist_ok=True)
                save_numpy(mlx_weights(encoder), str(folder / "weights.safetensors"))
                (folder / "config.json").write_text(json.dumps(_spec(encoder, step, {"layout": "NHWC"}), indent=2))
                shutil.copyfile(Path(__file__).parent / "model" / "mlx_encoder.py", folder / "mlx_encoder.py")
                entry = {"path": folder.name}
                try:
                    import mlx.core as mx

                    from .model.mlx_encoder import MLXEncoder
                except ImportError:
                    entry["checked"] = "mlx not installed"
                else:
                    tiles, glob = MLXEncoder.load(folder)(mx.array(probe.numpy().transpose(0, 2, 3, 1)))
                    entry["max_diff"] = max(_max_diff(tiles, ref_tiles), _max_diff(glob, ref_global))
            elif fmt == "coreml":
                try:
                    import coremltools as ct
                except ImportError:
                    entry = {"skipped": "needs coremltools (pip install coremltools, macOS)"}
                else:
                    path = dest / f"{stem}-{width}x{height}.mlpackage"
                    traced = torch.jit.trace(encoder, probe)
                    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=probe.shape)],
                                         outputs=[ct.TensorType(name="tiles"), ct.TensorType(name="global")],
                                         convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13)
                    mlmodel.save(str(path))
                    entry = {"path": path.name}
                    try:
                        pred = mlmodel.predict({"image": probe.numpy()})
                        entry["max_diff"] = max(_max_diff(pred["tiles"], ref_tiles),
                                                _max_diff(pred["global"], ref_global))
                    except Exception as exc:     # prediction only works on macOS
                        entry["checked"] = f"not checked: {exc}"
            else:
                raise ValueError(f"unknown format {fmt}")
        except Exception as exc:
            entry = {"error": f"{type(exc).__name__}: {exc}"}
            result["failed"].append(fmt)
        result["formats"][fmt] = entry
        print(f"  {fmt}: {json.dumps(entry)}", flush=True)
        if progress.stop_requested:
            break

    (dest / "export.json").write_text(json.dumps(result, indent=2))
    if progress.run_dir:
        (progress.run_dir / "export.json").write_text(json.dumps(result, indent=2))
    made = [f for f, e in result["formats"].items() if "path" in e]
    progress.update("done", len(formats), len(formats), f"wrote {', '.join(made) or 'nothing'} to {dest}",
                    eta_seconds=0, force=True)
    return result
