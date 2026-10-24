"""The sprites of a replay dataset, kept as compact arrays.

dataset.json (version 2) lists the frames; each frame's sprites are the rows
first..first+count of sprites.npz:

  prototypes, names   the distinct prototype ids and in-game names
  proto, name         per sprite: index into those lists (int32)
  box                 x0, y0, x1, y1 in frame pixels (float32)
  origin              where the entity stands (float32)
  depth               draw depth (int16)
  flags               bits, see FLAGS (uint8)
  mob                 MOB_CLASSES index (uint8; 0 for anything that isn't a mob)

That is about 40 bytes a sprite instead of about 300 in JSON, and the frame
list stays small enough to read quickly. Version 1 datasets (sprites inside
dataset.json) are still read, by turning them into the same arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FLAGS = ("visible", "dark", "blocked", "faint", "humanoid", "alive", "subject")
BIT = {name: 1 << i for i, name in enumerate(FLAGS)}
MOB_CLASSES = ("none", "humanoid", "borg", "animal", "hostile")

# Groups that scores are split by, from the draw depths in
# Content.Shared/DrawDepth/DrawDepth.cs (Objects = 0).
GROUPS = ("floor", "walls", "objects", "wall-mounted", "items", "mobs", "doors", "other", "followed player")


def group_of(depth: int, flags: int) -> int:
    if flags & BIT["subject"]:
        return 8
    if flags & BIT["humanoid"]:
        return 5
    if depth <= -11:            # LowFloors .. HighFloorObjects: pipes, cables, decals, puddles
        return 0
    if depth <= -9:             # DeadMobs, SmallMobs
        return 5
    if depth <= -1:             # Walls .. WallTops
        return 1
    if depth in (0, 1, 3):      # Objects, SmallObjects, LargeObjects
        return 2
    if depth == 2:              # WallMountedItems
        return 3
    if depth == 4:              # Items
        return 4
    if depth <= 7:              # BelowMobs, Mobs, OverMobs
        return 5
    if depth <= 10:             # Doors, BlastDoors, Overdoors
        return 6
    return 7                    # gasses, effects, ghosts, overlays


class Labels:
    """All sprites of a dataset, with each frame's rows."""

    def __init__(self, prototypes, names, proto, name, box, origin, depth, flags, mob=None):
        self.prototypes = [str(p) for p in prototypes]
        self.names = [str(n) for n in names]
        self.proto = np.asarray(proto, np.int32)
        self.name = np.asarray(name, np.int32)
        self.box = np.asarray(box, np.float32).reshape(-1, 4)
        self.origin = np.asarray(origin, np.float32).reshape(-1, 2)
        self.depth = np.asarray(depth, np.int16)
        self.flags = np.asarray(flags, np.uint8)
        # Datasets from before mob classes only know humanoids.
        self.mob = np.asarray(mob, np.uint8) if mob is not None else \
            np.where((self.flags & BIT["humanoid"]) != 0, 1, 0).astype(np.uint8)

    def __len__(self):
        return len(self.proto)

    def has(self, flag: str) -> np.ndarray:
        return (self.flags & BIT[flag]) != 0

    def rows(self, frame: dict) -> slice:
        return slice(frame["first"], frame["first"] + frame["count"])

    def dicts(self, frame: dict) -> list[dict]:
        """One frame's sprites as dicts (for reports)."""
        out = []
        for i in range(*self.rows(frame).indices(len(self))):
            box = [round(float(v), 1) for v in self.box[i]]
            d = {"prototype": self.prototypes[self.proto[i]], "name": self.names[self.name[i]], "box": box,
                 "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                 "origin": [round(float(v), 1) for v in self.origin[i]], "draw_depth": int(self.depth[i])}
            d.update({f: bool(self.flags[i] & BIT[f]) for f in FLAGS})
            d["mob_class"] = MOB_CLASSES[self.mob[i]]
            out.append(d)
        return out

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, prototypes=np.array(self.prototypes), names=np.array(self.names), proto=self.proto,
                 name=self.name, box=self.box, origin=self.origin, depth=self.depth, flags=self.flags, mob=self.mob)
        tmp.replace(path)

    @classmethod
    def from_sprites(cls, frames_sprites: list[list[dict]]) -> tuple["Labels", list[tuple[int, int]]]:
        """Arrays from per-frame sprite dicts, and each frame's (first, count)."""
        w = LabelWriter()
        ranges = [w.add(sprites) for sprites in frames_sprites]
        return w.labels(), ranges


class LabelWriter:
    """Collects sprites frame by frame, as arrays, while a dataset is built."""

    def __init__(self):
        self.protos: dict[str, int] = {}
        self.names: dict[str, int] = {}
        self.parts: list[tuple] = []
        self.n = 0

    def add(self, sprites: list[dict]) -> tuple[int, int]:
        first = self.n
        self.n += len(sprites)
        self.parts.append((
            np.array([self.protos.setdefault(s["prototype"], len(self.protos)) for s in sprites], np.int32),
            np.array([self.names.setdefault(s.get("name", ""), len(self.names)) for s in sprites], np.int32),
            np.array([s["box"] for s in sprites], np.float32).reshape(-1, 4),
            np.array([s.get("origin") or [(s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2]
                      for s in sprites], np.float32).reshape(-1, 2),
            np.array([s["draw_depth"] for s in sprites], np.int16),
            np.array([sum(BIT[f] for f in FLAGS if s.get(f)) for s in sprites], np.uint8),
            np.array([MOB_CLASSES.index(s["mob_class"]) if s.get("mob_class") else int(bool(s.get("humanoid")))
                      for s in sprites], np.uint8)))
        return first, len(sprites)

    def labels(self) -> Labels:
        cols = [np.concatenate([p[i] for p in self.parts]) if self.parts else np.zeros((0,)) for i in range(7)]
        return Labels(list(self.protos), list(self.names), *cols)


def load(dataset: Path) -> tuple[dict, Labels]:
    """dataset.json (with each frame's first/count) and its sprites."""
    dataset = Path(dataset)
    data = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if data.get("version", 1) >= 2:
        z = np.load(dataset / "sprites.npz")
        labels = Labels(z["prototypes"], z["names"], z["proto"], z["name"], z["box"], z["origin"], z["depth"],
                        z["flags"], z["mob"] if "mob" in z else None)
        return data, labels
    labels, ranges = Labels.from_sprites([f.pop("sprites") for f in data["frames"]])
    for f, (first, count) in zip(data["frames"], ranges):
        f["first"], f["count"] = first, count
    return data, labels


def dataset_hash(dataset: Path) -> str:
    """Fingerprint of a dataset's frame list and sprites, saved with each model."""
    h = hashlib.sha256()
    for name in ("dataset.json", "sprites.npz", "floor.npz"):
        p = Path(dataset) / name
        if p.is_file():
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]
