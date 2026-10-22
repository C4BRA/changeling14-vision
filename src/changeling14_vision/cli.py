"""Command line for the Changeling14 Vision pipeline.

    changeling14-vision sprites --repo ~/Desktop/SS13-14/wizden-ss14 \\
                                --out  ~/Desktop/SS13-14/ss14-sprite-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .progress import Progress

DEFAULT_REPO = "~/Desktop/SS13-14/wizden-ss14"
DEFAULT_OUT = "~/Desktop/SS13-14/ss14-sprite-dataset"


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
