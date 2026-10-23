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
DEFAULT_INBOX = "~/Desktop/SS13-14/replay-inbox"
DEFAULT_LIBRARY = "~/Desktop/SS13-14/replay-library"
DEFAULT_CODEBASE = "~/Desktop/SS13-14/wizden-ss14"
DEFAULT_CLIENTS = "~/Desktop/SS13-14/replay-clients"
DEFAULT_RECORDINGS = "~/Desktop/SS13-14/replay-recordings"
DEFAULT_REPLAY_DATASET = "~/Desktop/SS13-14/replay-dataset"
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


# Training settings a model folder keeps: when continuing, any not given on
# the command line come from its config.json; for a new folder, these defaults.
KEPT = {"batch": 128, "optimizer": "adamw", "lr": None, "checkpoint_every": None,
        "rich_text": False, "amp": False, "dedup": True}


def _resolve_train_settings(args, resuming: bool) -> None:
    cfg = _folder_config(_model_dir(args)) if resuming and (_model_dir(args) / "latest.pt").exists() else {}
    for name, default in KEPT.items():
        if getattr(args, name) is None:
            setattr(args, name, cfg.get(name, default))
    if args.keep_checkpoints is None:
        # Folders from before this setting kept every checkpoint, and still do.
        args.keep_checkpoints = cfg.get("keep_checkpoints", 0) if cfg else 5


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
    _resolve_train_settings(args, resuming=not args.fresh)
    opts = TrainOptions(data=Path(args.data).expanduser().resolve(), out=_model_dir(args), size=args.size,
                        steps=args.steps, batch=args.batch, lr=args.lr, workers=args.workers,
                        checkpoint_every=args.checkpoint_every, checkpoint_minutes=args.checkpoint_minutes,
                        optimizer=args.optimizer, minutes=args.minutes, device=args.device, seed=args.seed, fresh=args.fresh,
                        amp=args.amp, text_model=None if args.no_text else args.text_model,
                        rich_text=args.rich_text, dedup=args.dedup, keep_checkpoints=args.keep_checkpoints)
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


def cmd_replays_import(args) -> int:
    from .replays.importer import import_replays

    progress = Progress(args.run_dir)
    progress.catch_stop()
    result = import_replays(Path(args.inbox), Path(args.library), args.remove_from_inbox, progress)
    for row in result.imported:
        length = f", length {row['duration']}" if row.get("duration") else ""
        print(f"imported {row['original']}: build {row['build']}{length} -> {row['file']}", flush=True)
    for name, reason in result.skipped:
        print(f"skipped {name}: {reason}", flush=True)
    print(f"replays: {len(result.imported)} imported, {len(result.skipped)} skipped, "
          f"library {Path(args.library).expanduser()}", flush=True)
    return 0


def cmd_replays_list(args) -> int:
    from collections import Counter

    from .replays.importer import load_library

    rows = load_library(Path(args.library).expanduser())
    for build, n in sorted(Counter(r["build"] for r in rows).items()):
        print(f"{build}: {n} replay(s)")
    print(f"{len(rows)} replay(s) in {Path(args.library).expanduser()}")
    return 0


def cmd_replays_record(args) -> int:
    from .replays.importer import load_library
    from .replays.record import RecordError, prepare_client, record

    library = Path(args.library).expanduser()
    rows = load_library(library)
    wanted = Path(args.replay).name
    row = next((r for r in rows if Path(r["file"]).name == wanted or r["original"] == wanted), None)
    if row is None:
        print(f"{wanted} is not in the library {library}; import it first", flush=True)
        return 1
    progress = Progress(args.run_dir)
    progress.catch_stop()
    try:
        progress.update("build client", 0, 1, f"build {row['build']}", force=True)
        client = prepare_client(row["buildForkVersion"], row["build"], Path(args.source),
                                Path(args.clients), log=lambda m: print(m, flush=True))
        out = Path(args.out).expanduser() / Path(row["file"]).stem
        result = record(library / row["file"], client, out, interval_s=args.interval,
                        players_per_moment=args.players, max_pictures=args.max_pictures,
                        start_s=args.start, end_s=args.end, seed=args.seed, progress=progress,
                        log=lambda m: print(m, flush=True))
    except RecordError as exc:
        print(f"record failed: {exc}", flush=True)
        return 1
    print(f"recorded {result['pictures']} picture(s) from {result['replay']} in {result['seconds']} s "
          f"-> {out}", flush=True)
    if result["pictures"]:
        from .replays.report import write_report
        print(f"boxes drawn on the pictures: {write_report(out)}", flush=True)
    if result["error"]:
        print(f"the recorder reported an error (see {out / 'error.txt'})", flush=True)
    return 0 if result["finished"] and not result["error"] else 1


def cmd_replays_dataset(args) -> int:
    from .replays.dataset import build_dataset
    from .replays.importer import load_library

    rows = {Path(r["file"]).stem: r for r in load_library(Path(args.library).expanduser())}
    base = Path(args.recordings).expanduser()
    if args.recording:
        recordings = [base / name for name in args.recording]
    else:
        # Only folders named after a library replay, so renamed copies are left out.
        recordings = sorted(p for p in base.iterdir() if p.name in rows and (p / "frames").is_dir()) \
            if base.is_dir() else []
    if not recordings:
        print(f"no recordings found in {base}", flush=True)
        return 1
    progress = Progress(args.run_dir)
    progress.catch_stop()
    holdout = args.holdout.split(",") if args.holdout is not None else None
    s = build_dataset(recordings, Path(args.out), rows, holdout, progress)
    if s.get("stopped"):
        print("dataset: stopped; the dataset that was there is unchanged", flush=True)
        return 143
    print(f"dataset: {s['frames']} frames ({s['train_frames']} train, {s['test_frames']} test, "
          f"test rounds {', '.join(s['test_rounds']) or 'none'}), dropped {s['repeats_dropped']} near-repeats, "
          f"{s['black_dropped']} black and {s.get('other_view_dropped', 0)} zoomed pictures, "
          f"{s['visible_sprites']} of {s['sprites']} sprites visible -> {Path(args.out).expanduser()}", flush=True)
    return 0


def cmd_replays_train(args) -> int:
    try:
        from .replays.detect import DetectOptions, train_detect
    except ImportError as exc:
        print(f"{TRAIN_DEPS} ({exc})", flush=True)
        return 1
    models = Path(args.models).expanduser()
    start = Path(args.start).expanduser()
    if not start.is_absolute() and not start.exists():
        start = models / args.start
    opts = DetectOptions(dataset=Path(args.dataset).expanduser(), start=start, out=models / args.name,
                         steps=args.steps, batch=args.batch, workers=args.workers, device=args.device,
                         minutes=args.minutes, keep_checkpoints=args.keep_checkpoints,
                         allow_new_data=args.allow_new_data, allow_different_start=args.allow_different_start)
    progress = Progress(args.run_dir)
    progress.catch_stop()
    row = train_detect(opts, progress)
    print(f"detect: done at step {row.get('step', 0):,} -> {opts.out}", flush=True)
    return 0


def _replay_model(args) -> Path:
    p = Path(args.model).expanduser()
    return p if p.exists() else Path(args.models).expanduser() / args.model


def cmd_replays_predict(args) -> int:
    from .replays.predict import predict_report

    progress = Progress(args.run_dir)
    progress.catch_stop()
    out = predict_report(_replay_model(args), Path(args.dataset), args.limit, args.threshold, args.device, progress)
    print(f"predictions: {out}", flush=True)
    return 0


def cmd_replays_live(args) -> int:
    from .replays.predict import live

    region = tuple(int(v) for v in args.region.split(",")) if args.region else None
    if region is not None and len(region) != 4:
        print("--region is left,top,width,height", flush=True)
        return 1
    live(_replay_model(args), region, args.tile_px, args.threshold, args.device, args.fps, not args.no_labels,
         args.name_threshold)
    return 0


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
    p.add_argument("--batch", type=int,
                   help="pictures per step (default: the model folder's, else 128); 256 or 512 run faster per picture on a GPU. The learning "
                        "rate, warm-up and checkpoint spacing follow it; lower --steps to train on the same pictures")
    p.add_argument("--lr", type=float, help="learning rate (default: the model folder's, else 1e-3 x "
                                            "sqrt(batch/128); 2e-3 schedule-free)")
    p.add_argument("--optimizer", choices=["adamw", "schedulefree"],
                   help="adamw with a cosine schedule (default), or schedule-free AdamW (no schedule: stop any "
                        "time); a model folder keeps its own")
    p.add_argument("--workers", type=int, default=2, help="data loading processes (default 2)")
    p.add_argument("--checkpoint-every", type=int,
                   help="steps between checkpoints (default: the model folder's, else every 128,000 pictures, "
                        "1,000 steps at batch 128)")
    p.add_argument("--checkpoint-minutes", type=float, default=15.0,
                   help="also checkpoint at least this often (default 15)")
    p.add_argument("--minutes", type=float, help="stop cleanly after this many minutes; run again to resume")
    p.add_argument("--fresh", action="store_true", help="start over instead of resuming latest.pt")
    p.add_argument("--keep-checkpoints", type=int,
                   help="newest checkpoints to keep, plus the best one; 0 keeps all (default: 5 for a new "
                        "model folder, the folder's own setting when resuming)")
    p.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence model for matching sprites to entity names (downloaded once)")
    p.add_argument("--no-text", action="store_true", help="train without text matching")
    p.add_argument("--no-dedup", dest="dedup", action="store_false", default=None,
                   help="count identical pictures separately (new models only; by default each group of "
                        "identical pictures counts once and stays on one side of the test)")
    p.add_argument("--rich-text", action="store_true", default=None,
                   help="also describe each item by its suffix, folders and components (new models only)")
    p.add_argument("--amp", action="store_true", default=None,
                   help="half precision on the GPU: often faster on Apple chips; check the loss still falls "
                        "(default: the model folder's)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="full precision, even if the folder used --amp")
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

    p = sub.add_parser("replays", help="replays: import into the library, list it")
    rsub = p.add_subparsers(dest="replays_command", required=True)
    r = rsub.add_parser("import", help="copy replay .zip files from the inbox into the library, by build")
    r.add_argument("--inbox", default=DEFAULT_INBOX, help=f"folder of replay .zip files (default {DEFAULT_INBOX})")
    r.add_argument("--library", default=DEFAULT_LIBRARY, help=f"replay library (default {DEFAULT_LIBRARY})")
    r.add_argument("--remove-from-inbox", action="store_true",
                   help="delete each replay from the inbox once it is safely copied (default: keep it)")
    r.set_defaults(func=cmd_replays_import)
    r = rsub.add_parser("list", help="replays in the library, grouped by the build that plays them")
    r.add_argument("--library", default=DEFAULT_LIBRARY, help=f"replay library (default {DEFAULT_LIBRARY})")
    r.set_defaults(func=cmd_replays_list)
    r = rsub.add_parser("record", help="play a library replay in a replay client and save pictures with answer keys")
    r.add_argument("replay", help="the replay's file name in the library")
    r.add_argument("--library", default=DEFAULT_LIBRARY, help=f"replay library (default {DEFAULT_LIBRARY})")
    r.add_argument("--source", default=DEFAULT_CODEBASE,
                   help=f"git checkout of the codebase the replay came from (default {DEFAULT_CODEBASE})")
    r.add_argument("--clients", default=DEFAULT_CLIENTS,
                   help=f"where replay clients are built, one per build (default {DEFAULT_CLIENTS})")
    r.add_argument("--out", default=DEFAULT_RECORDINGS, help=f"output folder (default {DEFAULT_RECORDINGS})")
    r.add_argument("--interval", type=float, default=30, help="seconds of replay between moments (default 30)")
    r.add_argument("--players", type=int, default=2, help="players followed at each moment (default 2)")
    r.add_argument("--max-pictures", type=int, help="stop after this many pictures")
    r.add_argument("--start", type=float, help="first second of the replay to record")
    r.add_argument("--end", type=float, help="last second of the replay to record")
    r.add_argument("--seed", type=int, default=1, help="picks which players are followed (default 1)")
    r.set_defaults(func=cmd_replays_record)
    r = rsub.add_parser("dataset", help="turn recordings into a training dataset (frames, sprites, held-out rounds)")
    r.add_argument("recording", nargs="*", help="recording folder names (default: every recorded library replay)")
    r.add_argument("--recordings", default=DEFAULT_RECORDINGS, help=f"recordings folder (default {DEFAULT_RECORDINGS})")
    r.add_argument("--library", default=DEFAULT_LIBRARY, help=f"replay library (default {DEFAULT_LIBRARY})")
    r.add_argument("--out", default=DEFAULT_REPLAY_DATASET, help=f"output folder (default {DEFAULT_REPLAY_DATASET})")
    r.add_argument("--holdout", help="comma-separated round ids kept for testing (default: about 1 in 5 rounds)")
    r.set_defaults(func=cmd_replays_dataset)
    r = rsub.add_parser("train", help="train a sprite model to find and name things in replay frames")
    r.add_argument("name", help="new model folder name (inside --models); resumes if it exists")
    r.add_argument("--from", dest="start", required=True,
                   help="the sprite model folder to start from (a name inside --models, or a path)")
    r.add_argument("--dataset", default=DEFAULT_REPLAY_DATASET, help=f"replay dataset (default {DEFAULT_REPLAY_DATASET})")
    r.add_argument("--models", default=DEFAULT_MODELS, help=f"models folder (default {DEFAULT_MODELS})")
    r.add_argument("--steps", type=int, default=6000, help="training steps (default 6000)")
    r.add_argument("--batch", type=int, default=8, help="480x480 crops per step (default 8)")
    r.add_argument("--minutes", type=float, help="stop cleanly after this long")
    r.add_argument("--workers", type=int, default=2, help="picture loading processes (default 2)")
    r.add_argument("--device", default="auto", help="auto, mps, cuda or cpu (default auto)")
    r.add_argument("--keep-checkpoints", type=int,
                   help="newest checkpoints to keep, plus the best one; 0 keeps all (default: 5 for a new "
                        "model, the model's own setting when resuming)")
    r.add_argument("--allow-new-data", action="store_true",
                   help="resume although the dataset was rebuilt or is another one")
    r.add_argument("--allow-different-start", action="store_true",
                   help="resume although --from names another stage-1 model than the one it started from")
    r.set_defaults(func=cmd_replays_train)
    r = rsub.add_parser("predict", help="draw a replay model's guesses next to the answers on held-out frames")
    r.add_argument("model", help="replay model folder name (inside --models) or path")
    r.add_argument("--dataset", default=DEFAULT_REPLAY_DATASET, help=f"replay dataset (default {DEFAULT_REPLAY_DATASET})")
    r.add_argument("--models", default=DEFAULT_MODELS, help=f"models folder (default {DEFAULT_MODELS})")
    r.add_argument("--limit", type=int, default=60, help="held-out frames to draw (default 60)")
    r.add_argument("--threshold", type=float, default=0.5, help="how sure a square must be to count (default 0.5)")
    r.add_argument("--device", default="auto", help="auto, mps, cuda or cpu (default auto)")
    r.set_defaults(func=cmd_replays_predict)
    r = rsub.add_parser("live", help="watch the game on screen and show a replay model's boxes and names")
    r.add_argument("model", help="replay model folder name (inside --models) or path")
    r.add_argument("--models", default=DEFAULT_MODELS, help=f"models folder (default {DEFAULT_MODELS})")
    r.add_argument("--region", help="left,top,width,height of the game view in screen points (default: drag a box)")
    r.add_argument("--tile-px", type=float, help="captured pixels per game tile (default: worked out)")
    r.add_argument("--threshold", type=float, default=0.5, help="how sure a square must be to count (default 0.5)")
    r.add_argument("--fps", type=float, default=10, help="pictures per second (default 10)")
    r.add_argument("--name-threshold", type=float, default=0.6,
                   help='show "?" instead of a name the model is less sure of (default 0.6)')
    r.add_argument("--no-labels", action="store_true", help="boxes only")
    r.add_argument("--device", default="auto", help="auto, mps, cuda or cpu (default auto)")
    r.set_defaults(func=cmd_replays_live)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
