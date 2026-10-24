"""What is on the floor of each frame: floor tiles, visible gas and decals.

The recorder writes, per picture, every grid in view (a matrix from picture
pixels to the grid's own coordinates, and each tile's floor type and gases)
and every decal in view with its box. The dataset keeps them in floor.npz:

  tile_names            floor types; 0 is "space" (no tile)
  gas_names             visible gases (and "Fire"); bit i of a gas value is gas i
  decal_names           decal ids
  tiles, gas            per frame, one value per CELL x CELL pixels of the frame
                        (the floor type and gas bits under that point), flattened;
                        the frame's "floor" entry is [offset, rows, columns]
  decal_id, decal_box, decal_flags
                        per decal (box in frame pixels, flags as DECAL_BITS); the
                        frame's "decals" entry is [first, count]

Frames from recordings made before the recorder wrote floors have neither entry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CELL = 8                  # floor maps hold one value per 8 x 8 frame pixels
GAS_MIN = 20              # overlay opacity (of 255) from which a gas counts as visible
DECAL_BITS = {"visible": 1, "dark": 2, "blocked": 4, "faint": 8}


class FloorWriter:
    def __init__(self):
        self.tile_index = {"space": 0}
        self.gas_index: dict[str, int] = {}
        self.decal_index: dict[str, int] = {}
        self.tiles: list[np.ndarray] = []
        self.gas: list[np.ndarray] = []
        self.offset = 0
        self.decals: list[tuple] = []
        self.n_decals = 0

    def add_maps(self, key: dict, k: float, w: int, h: int) -> list[int]:
        """Floor and gas maps of a frame w x h model pixels whose picture has k
        picture pixels per model pixel. Returns the frame's "floor" entry."""
        rows, cols = -(-h // CELL), -(-w // CELL)
        tiles = np.zeros((rows, cols), np.uint16)
        gas = np.zeros((rows, cols), np.uint16)
        filled = np.zeros((rows, cols), bool)
        py, px = np.mgrid[0:rows, 0:cols].astype(np.float64)
        px, py = (px + 0.5) * CELL * k, (py + 0.5) * CELL * k
        names = key.get("tile_names", [])
        gas_names = key.get("gas_names", [])
        for g in key.get("grids", []):
            if not g.get("tiles"):
                continue
            m11, m12, m21, m22, m31, m32 = g["m"]
            ts = g.get("tile_size", 1) or 1
            lx = np.floor((px * m11 + py * m21 + m31) / ts).astype(np.int64)
            ly = np.floor((px * m12 + py * m22 + m32) / ts).astype(np.int64)
            t = np.asarray(g["tiles"], np.int64).reshape(-1, 3)
            x0, y0 = t[:, 0].min(), t[:, 1].min()
            dense = np.zeros((t[:, 1].max() - y0 + 1, t[:, 0].max() - x0 + 1), np.int64)
            dense[t[:, 1] - y0, t[:, 0] - x0] = [self.tile_index.setdefault(names[i], len(self.tile_index))
                                                 for i in t[:, 2]]
            inside = (lx >= x0) & (ly >= y0) & (lx - x0 < dense.shape[1]) & (ly - y0 < dense.shape[0])
            val = np.zeros_like(lx)
            val[inside] = dense[ly[inside] - y0, lx[inside] - x0]
            take = (val > 0) & ~filled
            tiles[take] = val[take]
            filled |= take
            gd = np.asarray(g.get("gas") or [], np.int64).reshape(-1, 4)
            gd = gd[gd[:, 3] >= GAS_MIN]
            if len(gd):
                bits = np.zeros_like(dense)
                for x, y, i, _ in gd:
                    b = self.gas_index.setdefault(gas_names[i], len(self.gas_index))
                    if b < 16 and 0 <= y - y0 < dense.shape[0] and 0 <= x - x0 < dense.shape[1]:
                        bits[y - y0, x - x0] |= 1 << b
                gv = np.zeros_like(lx)
                gv[inside] = bits[ly[inside] - y0, lx[inside] - x0]
                gas[take] = gv[take]
        entry = [self.offset, rows, cols]
        self.tiles.append(tiles.ravel())
        self.gas.append(gas.ravel())
        self.offset += tiles.size
        return entry

    def add_decals(self, decals: list[dict]) -> list[int]:
        """decals: dicts with id (a name), box (frame pixels) and the DECAL_BITS flags."""
        first = self.n_decals
        if decals:
            self.decals.append((
                np.array([self.decal_index.setdefault(d["id"], len(self.decal_index)) for d in decals], np.int32),
                np.array([d["box"] for d in decals], np.float32).reshape(-1, 4),
                np.array([sum(v for f, v in DECAL_BITS.items() if d.get(f)) for d in decals], np.uint8)))
        self.n_decals += len(decals)
        return [first, len(decals)]

    def save(self, path: Path) -> None:
        cat = lambda parts, dtype, shape=(-1,): (np.concatenate(parts) if parts else np.zeros(0, dtype)).reshape(shape)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, tile_names=np.array(list(self.tile_index)), gas_names=np.array(list(self.gas_index) or [""]),
                 decal_names=np.array(list(self.decal_index) or [""]),
                 tiles=cat(self.tiles, np.uint16), gas=cat(self.gas, np.uint16),
                 decal_id=cat([d[0] for d in self.decals], np.int32),
                 decal_box=cat([d[1] for d in self.decals], np.float32, (-1, 4)),
                 decal_flags=cat([d[2] for d in self.decals], np.uint8),
                 has_gas=np.array(bool(self.gas_index)), has_decals=np.array(bool(self.decal_index)))
        tmp.replace(path)


class Floor:
    def __init__(self, path: Path):
        z = np.load(path)
        self.tile_names = [str(t) for t in z["tile_names"]]
        self.gas_names = [str(t) for t in z["gas_names"]] if bool(z["has_gas"]) else []
        self.decal_names = [str(t) for t in z["decal_names"]] if bool(z["has_decals"]) else []
        self.tiles, self.gas = z["tiles"], z["gas"]
        self.decal_id, self.decal_box, self.decal_flags = z["decal_id"], z["decal_box"], z["decal_flags"]

    @staticmethod
    def load(dataset: Path) -> "Floor | None":
        p = Path(dataset) / "floor.npz"
        return Floor(p) if p.is_file() else None

    def maps(self, frame: dict) -> tuple[np.ndarray, np.ndarray] | None:
        if "floor" not in frame:
            return None
        off, rows, cols = frame["floor"]
        n = rows * cols
        return self.tiles[off:off + n].reshape(rows, cols), self.gas[off:off + n].reshape(rows, cols)

    def decals(self, frame: dict) -> slice:
        first, count = frame.get("decals", (0, 0))
        return slice(first, first + count)
