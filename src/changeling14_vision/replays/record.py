"""Record pictures and answer keys from a replay, by playing it in a real client.

A replay only plays back on the exact build that recorded it, so each build in
the library gets its own replay client:

  <clients>/<build>/            a git worktree of the codebase at the replay's commit,
                                with recorder/C14RecorderSystem.cs added to
                                Content.Client and Content.Replay built

The recorder is only ever added to these worktrees. The checkout you play or
develop with is not changed.

Recording copies the replay into the client's user data folder
(<user data>/c14-recorder/replay.zip), writes job.txt next to it, and starts the
client. The recorder walks the replay, saves <moment>-<player>.png and .json
pairs into c14-recorder/out, writes done.txt (or error.txt) and closes the
client. Those files are then moved into the output folder, and the copies made
in the user data folder are removed. Nothing in the library is changed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

RECORDER_DIR = Path(__file__).resolve().parents[3] / "recorder"
ADDON_DIR = Path("Content.Client/Replay/C14")
STAMP = ".c14-recorder"


class RecordError(Exception):
    pass


def user_data_dir() -> Path:
    """Where Robust keeps user data for "Space Station 14" (the replay client's folder)."""
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        return home / "Library/Application Support/Space Station 14/data"
    if system == "Windows":
        return Path(os.environ["APPDATA"]) / "Space Station 14/data"
    base = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share"))
    return base / "Space Station 14/data"


def _run(cmd: list[str], cwd: Path | None = None, log=None) -> None:
    if log:
        log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-40:]
        raise RecordError(f"{' '.join(cmd)} failed ({proc.returncode}):\n" + "\n".join(tail))


def _recorder_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(RECORDER_DIR.glob("*.cs")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def prepare_client(commit: str, build: str, source: Path, clients: Path, log=print) -> Path:
    """A built replay client with the recorder for one build. Reuses it when up to date."""
    source, clients = Path(source).expanduser(), Path(clients).expanduser()
    if not (source / ".git").exists():
        raise RecordError(f"{source} is not a git checkout of the codebase")
    dest = clients / build
    stamp = dest / STAMP
    wanted = f"{commit}\n{_recorder_hash()}\n"
    if stamp.is_file() and stamp.read_text() == wanted:
        return dest

    clients.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        # The checkout may be shallow, so ask for the commit itself first.
        subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", commit],
                       capture_output=True, text=True)
        _run(["git", "-C", str(source), "worktree", "add", "--detach", str(dest), commit], log=log)
    try:
        _run(["git", "-C", str(dest), "submodule", "update", "--init", "--recursive", "--depth", "1",
              "RobustToolbox"], log=log)
    except RecordError:
        _run(["git", "-C", str(dest), "submodule", "update", "--init", "--recursive", "RobustToolbox"], log=log)

    addon = dest / ADDON_DIR
    addon.mkdir(parents=True, exist_ok=True)
    for f in RECORDER_DIR.glob("*.cs"):
        shutil.copy2(f, addon / f.name)
    _run(["dotnet", "build", str(dest / "Content.Replay/Content.Replay.csproj"), "-c", "Release"],
         cwd=dest, log=log)
    stamp.write_text(wanted)
    return dest


def _client_command(client: Path) -> list[str]:
    dll = client / "bin/Content.Replay/Content.Replay.dll"
    if not dll.is_file():
        raise RecordError(f"no built replay client at {dll}")
    return ["dotnet", str(dll), "--cvar", "replay.ignore_errors=true", "--cvar", "audio.mastervolume=0"]


def _first_line(path: Path) -> str | None:
    """The file's first line, or None while it is missing or being rewritten."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    return lines[0] if lines else None


def write_job(path: Path, **settings) -> None:
    lines = [f"{k}={v}" for k, v in settings.items() if v is not None]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def record(replay: Path, client: Path, out: Path, interval_s: float = 30, players_per_moment: int = 2,
           settle_frames: int = 12, max_pictures: int | None = None, start_s: float | None = None,
           end_s: float | None = None, seed: int = 1, timeout_s: float = 6 * 3600,
           load_timeout_s: float = 600, progress=None, log=print) -> dict:
    """Play one replay in the client and collect its pictures into out/frames.
    The client is stopped if the replay hasn't loaded after load_timeout_s (a
    client stuck on its loading screen otherwise holds the job for hours)."""
    replay, client, out = Path(replay).expanduser(), Path(client), Path(out).expanduser()
    root = user_data_dir() / "c14-recorder"
    if (root / "job.txt").exists():
        raise RecordError(f"{root / 'job.txt'} already exists: another recording may be running "
                          "(delete it if not)")
    if (root / "out").is_dir() and any((root / "out").iterdir()):
        # Pictures from a recording that was never collected: keep them.
        raise RecordError(f"{root / 'out'} still holds pictures from an earlier recording; move them out first")
    for name in ("done.txt", "error.txt", "progress.txt"):
        (root / name).unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(replay, root / "replay.zip")
    write_job(root / "job.txt", replay="/c14-recorder/replay.zip", interval_s=interval_s,
              players_per_moment=players_per_moment, settle_frames=settle_frames,
              max_pictures=max_pictures, start_s=start_s, end_s=end_s, seed=seed)

    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    started = time.time()
    proc = None
    try:
        with open(out / "client.log", "w", encoding="utf-8") as client_log:
            proc = subprocess.Popen(_client_command(client), cwd=client, stdout=client_log,
                                    stderr=subprocess.STDOUT)
            note = "starting the client"
            loaded = False
            while proc.poll() is None:
                time.sleep(2)
                note = _first_line(root / "progress.txt") or note
                loaded = loaded or not note.startswith(("starting", "job read", "loading"))
                if not loaded and time.time() - started > load_timeout_s:
                    proc.terminate()
                    raise RecordError(f"the replay hadn't loaded after {load_timeout_s:.0f} s (last: {note}); "
                                      f"see {out / 'client.log'}")
                saved = len(list((root / "out").glob("*.png"))) if (root / "out").is_dir() else 0
                if progress:
                    progress.update("record", saved, max_pictures or 0, f"{replay.name}: {note}")
                    if progress.stop_requested:
                        proc.terminate()
                if time.time() - started > timeout_s:
                    proc.terminate()
                    raise RecordError(f"recording took longer than {timeout_s:.0f} s")
        if (root / "out").is_dir():
            for f in (root / "out").iterdir():
                shutil.move(str(f), frames / f.name)
        error = (root / "error.txt").read_text(errors="replace") if (root / "error.txt").is_file() else None
        finished = (root / "done.txt").is_file()
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        # Only remove the copies made here; the library is never touched.
        (root / "job.txt").unlink(missing_ok=True)
        (root / "replay.zip").unlink(missing_ok=True)
    pictures = len(list(frames.glob("*.png")))
    if error:
        (out / "error.txt").write_text(error, encoding="utf-8")
    return {"replay": replay.name, "pictures": pictures, "finished": finished, "error": error,
            "seconds": round(time.time() - started, 1)}
