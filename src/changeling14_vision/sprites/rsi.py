"""Read RSI sprite sheets (*.rsi folders with a meta.json and one PNG per state).

In a state PNG, frames are laid out left to right, top to bottom, in
size.x by size.y cells: every frame of direction 0, then direction 1, and so on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class State:
    name: str
    directions: int
    frames: int
    delays: tuple[tuple[float, ...], ...] = ()

    def frames_in(self, direction: int) -> int:
        """Directions can have different frame counts; each has its own delays row."""
        if direction < len(self.delays):
            return max(1, len(self.delays[direction]))
        return self.frames


@dataclass
class Sheet:
    path: str                     # relative to Textures/, e.g. "Objects/Tools/wrench.rsi"
    root: Path                    # absolute folder
    license: object
    copyright: object
    width: int
    height: int
    states: dict[str, State] = field(default_factory=dict)
    error: str | None = None


def read_sheet(root: Path, textures_dir: Path) -> Sheet:
    rel = root.relative_to(textures_dir).as_posix()
    try:
        # Some meta.json files start with a UTF-8 byte order mark.
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return Sheet(rel, root, None, None, 0, 0, error=f"unreadable meta.json: {exc}")
    size = meta.get("size") or {}
    sheet = Sheet(rel, root, meta.get("license"), meta.get("copyright"),
                  int(size.get("x", 32)), int(size.get("y", 32)))
    for raw in meta.get("states") or []:
        name = raw.get("name")
        if not isinstance(name, str):
            continue
        directions = int(raw.get("directions", 1) or 1)
        delays = tuple(tuple(float(d) for d in row) for row in (raw.get("delays") or []))
        frames = max((len(row) for row in delays), default=1) or 1
        sheet.states[name] = State(name, directions, frames, delays)
    return sheet


def find_sheets(textures_dir: Path) -> list[Path]:
    return sorted(p for p in textures_dir.rglob("*.rsi") if p.is_dir() and (p / "meta.json").is_file())


def cut_state(sheet: Sheet, state: State):
    """Yield (direction, frame, image) for every cell of a state, at native size."""
    from PIL import Image

    png = sheet.root / f"{state.name}.png"
    with Image.open(png) as img:
        img = img.convert("RGBA")
        columns = max(1, img.width // sheet.width)
        index = -1
        for direction in range(state.directions):
            for frame in range(state.frames_in(direction)):
                index += 1
                row, col = divmod(index, columns)
                box = (col * sheet.width, row * sheet.height, (col + 1) * sheet.width, (row + 1) * sheet.height)
                if box[2] > img.width or box[3] > img.height:
                    raise ValueError(f"{png.name}: cell {index} outside a {img.width}x{img.height} image")
                yield direction, frame, img.crop(box)
