"""Command line for the Changeling14 Vision pipeline.

    changeling14-vision sprites --repo ~/Desktop/SS13-14/wizden-ss14 \\
                                --out  ~/Desktop/SS13-14/ss14-sprite-dataset
    changeling14-vision train     --size tiny     (stage 1: single-sprite pretraining)
    changeling14-vision benchmark                 (score the latest checkpoint)
    changeling14-vision export                    (safetensors, ONNX, MLX, Core ML)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .progress import Progress

DEFAULT_REPO = "~/Desktop/SS13-14/wizden-ss14"
DEFAULT_OUT = "~/Desktop/SS13-14/ss14-sprite-dataset"
DEFAULT_MODELS = "~/Desktop/SS13-14/changeling14-models"
TRAIN_DEPS = 'the training steps need PyTorch: pip install -e ".[train]"'


def _need_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(f"error: {TRAIN_DEPS}", file=sys.stderr)
        return False
    return True


def _model_dir(args) -> Path:
    return Path(args.out or Path(DEFAULT_MODELS) / f"sprite-pretrain-{args.size or 'tiny'}").expanduser()


def _folder_config(model_dir: Path) -> dict:
    """The options a model folder was started with (train writes config.json once)."""
    try:
        return json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def _resolve_model_args(args, resuming: bool = True) -> str | None:
    """Fill --data and --size from the model folder, so `--out <model>` alone is
    enough. Returns an error message when --data names another catalogue than
    the one the folder was trained on (its labels would not match)."""
    model_dir = _model_dir(args)
    cfg = _folder_config(model_dir) if resuming else {}
    if hasattr(args, "data"):
        recorded = cfg.get("data")
        if args.data is None:
            args.data = recorded or DEFAULT_OUT
        elif recorded and resuming and (model_dir / "latest.pt").exists() and \
                Path(args.data).expanduser().resolve() != Path(recorded).expanduser().resolve():
            return (f"{model_dir} was trained on the catalogue in {recorded}, not {args.data}. "
                    "Its labels come from that catalogue, so pick another model folder for this one, "
                    "or pass --fresh to start this folder over.")
    if args.size is None:
        args.size = cfg.get("size") or "tiny"
    return None


def cmd_sprites(args) -> int:
    from .sprites.build import BuildError, Options, build

    opts = Options(repo=Path(args.repo).expanduser(), out=Path(args.out).expanduser(),
                   images=not args.no_images, force=args.force, seed=args.seed)
    progress = Progress(args.run_dir)
    progress.catch_stop()
    print(f"sprites: reading {opts.repo}, writing {opts.out}", flush=True)
    try:
        report = build(opts, progress)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print((opts.out / "report.md").read_text(), flush=True)
    return 0 if report["complete"] else 143


def cmd_train(args) -> int:
    if not _need_torch():
        return 2
    from .pretrain.data import DataError
    from .pretrain.train import TrainOptions, train

    problem = _resolve_model_args(args, resuming=not args.fresh)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    opts = TrainOptions(data=Path(args.data).expanduser().resolve(), out=_model_dir(args), size=args.size,
                        steps=args.steps, batch=args.batch, lr=args.lr, workers=args.workers,
                        checkpoint_every=args.checkpoint_every, checkpoint_minutes=args.checkpoint_minutes,
                        minutes=args.minutes, device=args.device, seed=args.seed, fresh=args.fresh,
                        amp=args.amp, text_model=None if args.no_text else args.text_model)
    progress = Progress(args.run_dir)
    progress.catch_stop()
    print(f"train: sprites from {opts.data}, model folder {opts.out}", flush=True)
    try:
        result = train(opts, progress)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if result["complete"] or not progress.stop_requested else 143


def cmd_benchmark(args) -> int:
    if not _need_torch():
        return 2
    from .pretrain.benchmark import benchmark

    _resolve_model_args(args)
    progress = Progress(args.run_dir)
    progress.catch_stop()
    report = benchmark(_model_dir(args), Path(args.data).expanduser(), progress,
                       checkpoint=args.checkpoint, device=args.device, speed_only=args.speed_only)
    return 0 if report["complete"] else 143


def cmd_export(args) -> int:
    if not _need_torch():
        return 2
    from .export import FORMATS, export

    _resolve_model_args(args)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        print(f"error: unknown format {', '.join(unknown)}; pick from {', '.join(FORMATS)}", file=sys.stderr)
        return 2
    progress = Progress(args.run_dir)
    result = export(_model_dir(args), formats, progress, checkpoint=args.checkpoint,
                    dest=Path(args.dest).expanduser() if args.dest else None,
                    height=args.height, width=args.width)
    return 0 if not result["failed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="changeling14-vision")
    parser.add_argument("--run-dir", help="report progress here (default: $C14_RUN_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sprites", help="M0: build the sprite catalogue from an SS14 checkout")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"SS14 checkout (default {DEFAULT_REPO})")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    p.add_argument("--no-images", action="store_true", help="catalogue only, don't write the cut images")
    p.add_argument("--force", action="store_true", help="write into a non-empty folder from elsewhere")
    p.add_argument("--seed", type=int, default=14)
    p.set_defaults(func=cmd_sprites)

    def model_args(p, size_default="tiny"):
        p.add_argument("--size", choices=["nano", "tiny", "small", "base"],
                       help=f"encoder size (default: the model folder's, else {size_default})")
        p.add_argument("--out", help=f"model folder (default {DEFAULT_MODELS}/sprite-pretrain-<size>)")
        p.add_argument("--device", default="auto", help="auto (MPS, then CUDA, then CPU), mps, cuda or cpu")

    p = sub.add_parser("train", help="stage 1: single-sprite pretraining on the M0 catalogue (resumes)")
    p.add_argument("--data", help=f"M0 sprite catalogue (default: the one the model folder was trained on, "
                                  f"else {DEFAULT_OUT}); a model folder keeps its catalogue for life")
    model_args(p)
    p.add_argument("--steps", type=int, default=20000, help="total optimizer steps (default 20000)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2, help="data loading processes (default 2)")
    p.add_argument("--checkpoint-every", type=int, default=1000, help="steps between checkpoints")
    p.add_argument("--checkpoint-minutes", type=float, default=15.0,
                   help="also checkpoint at least this often (default 15)")
    p.add_argument("--minutes", type=float, help="stop cleanly after this many minutes; run again to resume")
    p.add_argument("--fresh", action="store_true", help="start over instead of resuming latest.pt")
    p.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence model for matching sprites to entity names (downloaded once)")
    p.add_argument("--no-text", action="store_true", help="train without text matching")
    p.add_argument("--amp", action="store_true",
                   help="half precision on the GPU: often faster on Apple chips; check the loss still falls")
    p.add_argument("--seed", type=int, default=14)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("benchmark", help="score a checkpoint on held-out sprites, and time it")
    p.add_argument("--data", help=f"M0 sprite catalogue (default: the one the model folder was trained on, "
                                  f"else {DEFAULT_OUT})")
    model_args(p)
    p.add_argument("--checkpoint", help="a .pt file (default: the model folder's latest.pt)")
    p.add_argument("--speed-only", action="store_true", help="only time the encoder; no data needed")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("export", help="export the encoder to safetensors, ONNX, MLX and Core ML")
    model_args(p)
    p.add_argument("--checkpoint", help="a .pt file (default: the model folder's latest.pt)")
    p.add_argument("--formats", default="safetensors,onnx,mlx,coreml",
                   help="comma-separated (default safetensors,onnx,mlx,coreml; unavailable ones are skipped)")
    p.add_argument("--dest", help="output folder (default <model folder>/export/step-N)")
    p.add_argument("--height", type=int, default=480, help="frame height for fixed-size formats (default 480)")
    p.add_argument("--width", type=int, default=480, help="frame width for fixed-size formats (default 480)")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
