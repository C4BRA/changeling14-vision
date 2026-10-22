"""Report progress in the run-folder format (docs/run-folder.md).

Every pipeline step uses this. With no run folder it does nothing, so steps
behave the same with or without the training app watching.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import time
from pathlib import Path

MIN_INTERVAL_S = 0.5


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


class Progress:
    def __init__(self, run_dir: str | Path | None = None):
        if run_dir is None:
            run_dir = os.environ.get("C14_RUN_DIR") or None
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._last_write = 0.0
        self._stage = ""
        self.stop_requested = False

    def catch_stop(self) -> None:
        """Turn SIGTERM/SIGINT into a flag the step checks between units of work."""

        def handler(_signum, _frame):
            self.stop_requested = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def update(self, stage: str, step: int | None = None, total: int | None = None,
               message: str = "", eta_seconds: float | None = None, force: bool = False) -> None:
        if not self.run_dir:
            return
        now = time.time()
        if not force and stage == self._stage and now - self._last_write < MIN_INTERVAL_S:
            return
        self._stage = stage
        self._last_write = now
        data = {"stage": stage, "message": message, "updated_at": now}
        if step is not None:
            data["step"] = step
        if total is not None:
            data["total"] = total
        if eta_seconds is not None:
            data["eta_seconds"] = eta_seconds
        _write_json_atomic(self.run_dir / "status.json", data)

    def watchset(self, checkpoint: str, images: list[Path], meta: dict | None = None) -> None:
        """Copy images into watchset/<checkpoint>/ for the app's gallery."""
        if not self.run_dir:
            return
        out = self.run_dir / "watchset" / checkpoint
        out.mkdir(parents=True, exist_ok=True)
        for n, src in enumerate(images):
            shutil.copyfile(src, out / f"{n:03d}-{src.parent.name}-{src.name}")
        if meta is not None:
            _write_json_atomic(out / "meta.json", meta)
