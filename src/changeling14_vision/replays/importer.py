"""Import replays: read each replay's metadata and file it in a library.

A replay .zip from an SS14 server holds _replay/replay.yml (written when
recording starts: engine version, codebase and its commit, start time) and
_replay/replay_final.yml (written when it ends: duration, end tick, file
count). Import reads only those two files, never plays the replay, and copies
it into the library, grouped by the exact build that recorded it, because a
replay only plays back on that build:

  <library>/<fork>-<commit[:12]>/<name>.zip
  <library>/replays.jsonl      one row per replay (metadata, fingerprint, path)

The inbox is left as it is unless remove_from_inbox is set. A replay that is
incomplete (no replay_final.yml) or unreadable is reported and skipped.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

META = "_replay/replay.yml"
META_FINAL = "_replay/replay_final.yml"
# Metadata kept in the library (Robust's ReplayConstants, plus content's round,
# map and server). Player lists in replay_final.yml are left out.
KEEP = ("name", "time", "engineVersion", "buildForkId", "buildForkVersion", "startTick", "startTime",
        "isClientRecording", "duration", "endTick", "endTime", "fileCount", "size",
        "uncompressedSize", "roundId", "map", "gamemode", "server_id", "server_name")


class ReplayError(Exception):
    pass


def _find(names: list[str], wanted: str) -> str | None:
    """The zip entry for a metadata file, wherever the _replay folder sits."""
    if wanted in names:
        return wanted
    tail = "/" + wanted
    return next((n for n in names if n.endswith(tail)), None)


def read_metadata(path: Path) -> dict:
    """The replay's merged replay.yml and replay_final.yml (only kept keys)."""
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            meta_name, final_name = _find(names, META), _find(names, META_FINAL)
            if meta_name is None:
                raise ReplayError("no _replay/replay.yml: not an SS14 replay")
            if final_name is None:
                raise ReplayError("no _replay/replay_final.yml: the recording never finished")
            meta = yaml.safe_load(z.read(meta_name)) or {}
            final = yaml.safe_load(z.read(final_name)) or {}
    except zipfile.BadZipFile as exc:
        raise ReplayError(f"not a readable zip ({exc})") from exc
    merged = {**meta, **final}
    out = {k: merged[k] for k in KEEP if k in merged and not isinstance(merged[k], (dict, list))}
    for key in ("buildForkId", "buildForkVersion", "engineVersion"):
        if not out.get(key):
            raise ReplayError(f"replay.yml has no {key}, so the build that can play it is unknown")
    return out


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_key(meta: dict) -> str:
    """Folder name for replays that play on the same build."""
    fork = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(meta["buildForkId"]))
    return f"{fork}-{str(meta['buildForkVersion'])[:12]}"


def load_library(library: Path) -> list[dict]:
    path = library / "replays.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@dataclass
class ImportResult:
    imported: list[dict]
    skipped: list[tuple[str, str]]       # (file, reason)


def import_replays(inbox: Path, library: Path, remove_from_inbox: bool = False, progress=None) -> ImportResult:
    inbox, library = Path(inbox).expanduser(), Path(library).expanduser()
    library.mkdir(parents=True, exist_ok=True)
    rows = load_library(library)
    known = {r["sha256"] for r in rows}
    files = sorted(p for p in inbox.glob("*.zip") if p.is_file()) if inbox.is_dir() else []
    imported, skipped = [], []
    for n, path in enumerate(files):
        if progress:
            progress.update("import replays", n, len(files), path.name)
            if progress.stop_requested:
                break
        try:
            meta = read_metadata(path)
        except ReplayError as exc:
            skipped.append((path.name, str(exc)))
            continue
        digest = fingerprint(path)
        if digest in known:
            skipped.append((path.name, "already in the library"))
            continue
        target = library / build_key(meta) / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{target.stem}-{digest[:8]}{target.suffix}")
        tmp = target.with_name(target.name + ".part")
        shutil.copy2(path, tmp)
        tmp.replace(target)
        row = {"file": str(target.relative_to(library)), "original": path.name, "sha256": digest,
               "bytes": target.stat().st_size, "build": build_key(meta), **meta}
        rows.append(row)
        known.add(digest)
        imported.append(row)
        if remove_from_inbox:
            path.unlink()
    tmp = library / "replays.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(library / "replays.jsonl")
    if progress:
        progress.update("import replays", len(files), len(files), f"{len(imported)} imported", force=True)
    return ImportResult(imported, skipped)
