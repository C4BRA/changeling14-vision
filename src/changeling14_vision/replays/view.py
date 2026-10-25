"""Where the game world is in the SS14 client window, worked out instead of
dragged by hand (the live overlay's "auto" game view).

Three sources, each checking the others:

  settings   the client's own client_config.toml (read-only): HUD layout
             (Default or Separated), viewport stretch / fixed scale / width,
             UI scale and the chat box size. From these the view is computed
             the way the client does it (Content.Client MainViewport and
             ScalingViewport): 21 x 15 tiles of 32 px times the scale,
             centred in the viewport control, clipped or letterboxed.
  picture    the floor grid gives the tile size (which also shows a zoomed
             eye) and, every frame, where the tile lines are, so the model's
             squares sit on the game's tiles. Pixels that stay put while the
             world scrolls are menus.
  player     the eye follows the player, so the player is near the view's
             centre; a player found far off-centre lowers the confidence.

The HUD (menu buttons, actions, chat, alerts, hands and inventory) floats over
the world in the Default layout, so it isn't cut out of the view: detections
under it are dropped instead. Its boxes come from the layout (the chat box
exactly, from the settings) plus the menus seen standing still.

Pure numpy apart from the optional settings file; the overlay feeds it
captures of the window's client area (one pixel per point).
"""

from __future__ import annotations

import math
import os
import time
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

TITLE_BAR = 28              # points of a standard macOS title bar
PX_PER_TILE = 32            # EyeManager.PixelsPerMeter
VIEW_H = 15                 # tiles; the width is the viewport.width setting (21 by default)
PANEL_BGR = (0x3B, 0x2C, 0x2B)  # the Separated layout's chat panel, #2B2C3B
# Eye zooms the game allows (SharedContentEyeSystem: steps of 1.5 down to 1.5^-3, up to a mob's
# MaxZoom: 1.2 revenant, 1.44 ghost; the few mobs that go to 2 and admin ghosts aren't covered).
ZOOMS = (1.5 ** -3, 1.5 ** -2, 1.5 ** -1, 1.0, 1.2, 1.44)


# The client's settings ------------------------------------------------------------

@dataclass(frozen=True)
class ClientSettings:
    """What the overlay needs from client_config.toml; the defaults are the
    upstream client's (Content.Shared CCVars), used for anything not in the file."""
    path: str | None = None
    layout: str = "default"             # "default" (HUD over the world) or "separated" (chat panel beside it)
    stretch: bool = True
    fixed_scale: int = 2
    scale_render: bool = True
    width_tiles: int = 21
    vertical_fit: bool = True
    snap_margin: int = 64
    snap_clip: int = 32
    ui_scale: float = 0.0               # 0: the OS scale (2 on a Retina screen)
    chat: tuple[float, float] | None = None     # Default layout: chat box (bottom, left) margins, UI units
    split: float = 0.6                  # Separated layout: share of the width the game gets
    found: tuple[str, ...] = ()         # settings that were in the file


def config_paths(home: Path | None = None) -> list[Path]:
    """client_config.toml files of installed SS14 clients, the plain "Space
    Station 14" one first, then the most recently written."""
    home = Path(home or Path.home())
    roots = [home / "Library/Application Support", home / ".local/share"]
    if os.environ.get("APPDATA"):
        roots.append(Path(os.environ["APPDATA"]))
    found = []
    for root in roots:
        try:
            for d in root.iterdir():
                p = d / "data" / "client_config.toml"
                if "space station" in d.name.lower() and p.is_file():
                    found.append(p)
        except OSError:
            continue
    return sorted(found, key=lambda p: (p.parent.parent.name != "Space Station 14", -p.stat().st_mtime))


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _pair(v) -> tuple[float, float] | None:
    try:
        a, b = (float(x) for x in str(v).split(","))
        return a, b
    except ValueError:
        return None


def read_settings(path: Path | None) -> ClientSettings:
    """The settings in path (missing or unreadable: the defaults). Forks keep
    some settings under their own tables (e.g. nyanotrasen.ui.layout), so a
    setting is also looked for at the end of longer names."""
    if path is None:
        return ClientSettings()
    try:
        flat = _flat(tomllib.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return ClientSettings()
    found = []

    def get(name, kind):
        keys = [name] + sorted(k for k in flat if k.endswith("." + name))
        for k in keys:
            if k in flat:
                try:
                    v = kind(flat[k])
                except (TypeError, ValueError):
                    continue
                found.append(k)
                return v
        return None

    s = ClientSettings(path=str(path))
    layout = get("ui.layout", str)
    changes = {
        "layout": "separated" if layout and layout.lower().startswith("separ") else "default",
        "stretch": get("viewport.stretch", bool), "fixed_scale": get("viewport.fixed_scale_factor", int),
        "scale_render": get("viewport.scale_render", bool), "width_tiles": get("viewport.width", int),
        "vertical_fit": get("viewport.vertical_fit", bool),
        "snap_margin": get("viewport.snap_tolerance_margin", int),
        "snap_clip": get("viewport.snap_tolerance_clip", int), "ui_scale": get("display.uiScale", float),
    }
    chat = _pair(get("ui.default_chat_size", str) or "")
    split = _pair(get("ui.separated_chat_size", str) or "")
    if chat:
        changes["chat"] = chat
    if split and 0.2 <= split[0] <= 0.95:
        changes["split"] = split[0]
    return replace(s, found=tuple(found), **{k: v for k, v in changes.items() if v is not None})


# Geometry -------------------------------------------------------------------------

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def scaled(self, k: float) -> "Rect":
        return Rect(self.x * k, self.y * k, self.w * k, self.h * k)

    def moved(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def intersect(self, o: "Rect") -> "Rect":
        x0, y0 = max(self.x, o.x), max(self.y, o.y)
        x1, y1 = min(self.right, o.right), min(self.bottom, o.bottom)
        return Rect(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def error(self, o: "Rect") -> float:
        """Largest edge difference, in the rects' pixels."""
        return max(abs(self.x - o.x), abs(self.y - o.y), abs(self.right - o.right), abs(self.bottom - o.bottom))

    def ints(self) -> tuple[int, int, int, int]:
        return int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h))


def title_bar(win_x: float, win_y: float, win_w: float, win_h: float,
              screens: list[tuple[float, float, float, float]]) -> float:
    """Points of title bar at the top of the window: none for a full-screen
    window (exactly a screen's frame), else a standard title bar."""
    for sx, sy, sw, sh in screens:
        if abs(win_w - sw) < 1 and abs(win_h - sh) < 1 and abs(win_x - sx) < 1:
            return 0.0
    return float(TITLE_BAR)


def view_width_tiles(s: ClientSettings) -> int:
    # ViewportUIController: vertical fit (with stretch) takes the server's maximum width,
    # which is 21 unless the server changes it; other widths outside 15..21 fall back to 21.
    if s.vertical_fit and s.stretch:
        return 21
    return s.width_tiles if 15 <= s.width_tiles <= 21 else 21


def control_rect(client_w: float, client_h: float, s: ClientSettings, ui_scale: float) -> Rect:
    """The viewport control within the client area (device pixels): the whole
    area in the Default layout, the part left of the chat panel in the
    Separated one (the panel is at least 300 UI units wide)."""
    if s.layout != "separated":
        return Rect(0, 0, client_w, client_h)
    w = min(client_w * s.split, client_w - 300 * ui_scale)
    return Rect(0, 0, max(1.0, w), client_h)


def draw_box(ctrl_w: float, ctrl_h: float, s: ClientSettings) -> tuple[Rect, int | None]:
    """Where ScalingViewport draws the world in its control (device pixels,
    relative to the control; may reach outside it) and the fixed scale it
    uses (None while it stretches)."""
    vw, vh = PX_PER_TILE * view_width_tiles(s), PX_PER_TILE * VIEW_H
    factor = s.fixed_scale if not s.stretch else None
    if s.stretch:
        # MainViewport.CalcSnappingFactor: an integer scale close enough to the control size wins.
        for i in range(1, 11):
            margin, clip = i * s.snap_margin, i * s.snap_clip
            dx, dy = ctrl_w - vw * i, ctrl_h - vh * i
            fits = lambda a: -clip <= a <= margin
            larger = lambda a: a > margin
            if ((fits(dx) or s.vertical_fit) and fits(dy)) or (not s.vertical_fit and fits(dx) and larger(dy)) \
                    or (larger(dx) and fits(dy)):
                factor = i
                break
    if factor is None:
        ratio = ctrl_h / vh if s.vertical_fit else min(ctrl_w / vw, ctrl_h / vh)
        w, h = vw * ratio, vh * ratio
    else:
        w, h = vw * factor, vh * factor
    return Rect((ctrl_w - w) / 2, (ctrl_h - h) / 2, w, h), factor


# HUD boxes in UI units, measured on the upstream (Wizard's Den) client: (anchor, x, y, w, h); x/y are
# from the anchor's corner, inward. The chat box comes from the settings when they have it.
HUD_LAYOUT = {
    "default": [("top-left", 0, 0, 390, 85),          # menu buttons
                ("top-left", 0, 85, 110, 315),        # actions bar
                ("bottom-left", 0, 0, 85, 78),        # inventory button
                ("bottom-centre", 0, 0, 820, 78)],    # hands and pockets
    "separated": [("top-left", 0, 0, 110, 315),
                  ("bottom-left", 0, 0, 85, 78),
                  ("bottom-centre", 0, 0, 820, 78)],
}
ALERTS = (110, 330)          # alerts column at the right edge, below the chat box (Default) or the top
CHAT_DEFAULT = (305.0, -475.0)


def hud_rects(area: Rect, s: ClientSettings, ui_scale: float) -> list[Rect]:
    """The HUD's boxes in device pixels, inside area (the client area in the
    Default layout, the viewport control in the Separated one)."""
    u = ui_scale
    out = []
    for anchor, x, y, w, h in HUD_LAYOUT["separated" if s.layout == "separated" else "default"]:
        w, h = w * u, h * u
        if anchor == "top-left":
            out.append(Rect(area.x + x * u, area.y + y * u, w, h))
        elif anchor == "bottom-left":
            out.append(Rect(area.x + x * u, area.bottom - h, w, h))
        else:
            out.append(Rect(area.x + area.w / 2 - w / 2, area.bottom - h, w, h))
    top = area.y
    if s.layout != "separated":
        bottom, left = s.chat or CHAT_DEFAULT
        # ResizableChatBox: anchored top-right with a 10-unit margin; the setting holds its
        # bottom margin and its (negative) left margin from the right edge.
        chat = Rect(area.right + left * u, area.y + 10 * u, -left * u - 10 * u, (bottom - 10) * u)
        out.append(chat)
        top = chat.bottom
    out.append(Rect(area.right - ALERTS[0] * u, top, ALERTS[0] * u, ALERTS[1] * u))
    return out


# The floor grid -------------------------------------------------------------------

def edge_profiles(grey: np.ndarray, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Mean edge strength per column boundary and per row boundary (entry i
    is between pixels i and i + 1), from every step-th row/column."""
    g = grey.astype(np.float32)
    cols = np.abs(np.diff(g[::step], axis=1)).mean(axis=0)
    rows = np.abs(np.diff(g[:, ::step], axis=0)).mean(axis=1)
    return cols, rows


def phase_scores(profile: np.ndarray, tile: float) -> np.ndarray:
    """Mean edge strength on lines tile apart, for each whole-pixel phase:
    entry p is lines at p, p + tile, ... (between pixels, so p - 0.5 ...)."""
    n = len(profile)
    t = int(round(tile))
    out = np.zeros(max(1, t), np.float32)
    for p in range(t):
        idx = np.round(np.arange(p, n, tile)).astype(int) - 1
        idx = idx[(idx >= 0) & (idx < n)]
        out[p] = profile[idx].mean() if len(idx) else 0.0
    return out


def repeat_strength(profile: np.ndarray, period: float, n: int = 8192) -> float:
    """How sharply an edge profile repeats every period pixels: the spectrum
    at 1 / period against the spectrum around it (about 1: no repeat).
    Broad changes (lighting, a dark half of the view) are taken out first."""
    if len(profile) < 3 * period:
        return 0.0
    d = profile - np.convolve(profile, np.ones(31) / 31, "same")
    spec = np.abs(np.fft.rfft(d * np.hanning(len(d)), n))
    f = n / period
    lo, hi = max(1, int(f * 0.7)), int(f * 1.3) + 1
    around = np.median(spec[lo:hi])
    # The strongest repeat nearby must be at this period itself, not a neighbour's (a seam
    # half a bigger tile apart can leak into a close candidate).
    near0, near1 = max(1, int(f * 0.88)), int(f * 1.12) + 1
    top = near0 + int(np.argmax(spec[near0:near1]))
    if around <= 0 or abs(top / f - 1) > 0.03:
        return 0.0
    return float(spec[top] / around)


def tile_from_candidates(grey: np.ndarray, candidates: list[float], min_strength: float = 1.8,
                         ) -> tuple[float | None, dict]:
    """Which candidate tile size the floor grid repeats at most sharply
    (both axes), or None when none does clearly (a dark view). Many floors
    also have a seam across each tile's middle, so half a tile can repeat
    more strongly still: half tiles are not candidates."""
    cols, rows = edge_profiles(grey)
    scores = {}
    for c in candidates:
        a, b = repeat_strength(cols, c), repeat_strength(rows, c)
        scores[c] = (a + b) / 2 if a and b else 0.8 * max(a, b)     # one axis may be all wall or all dark
    best = max(scores, key=scores.get, default=None)
    if best is None or scores[best] < min_strength:
        return None, scores
    # A big tile's harmonics fall on smaller sizes (270 px tiles repeat every 67.5): a larger
    # candidate that is a whole multiple of the best and repeats at all is the tile.
    for c in sorted(candidates, reverse=True):
        k = c / best
        if c > best and round(k) >= 2 and abs(k / round(k) - 1) <= 0.03 and scores[c] >= 0.7:
            return c, scores
    return best, scores


def _circ(a: float, b: float, period: float) -> float:
    d = (a - b) % period
    return min(d, period - d)


def grid_phase(profile: np.ndarray, tile: float, prefer: float | None = None,
               min_contrast: float = 1.6) -> float | None:
    """Where the tile lines are: the phase (0 <= phase < tile, in the
    profile's pixels) of the first line. Many floors also have a seam across
    the middle of each tile, so the best and the half-tile-off phases can be
    close: prefer (e.g. from where detected things are centred) picks
    between strong candidates. None when no line stands out (a dark view)."""
    s = phase_scores(profile, tile)
    if not len(s) or s.max() <= 0 or s.max() < min_contrast * float(np.median(s)):
        return None
    best = int(np.argmax(s))
    if prefer is None:
        return float(best)
    half = int(round((best + tile / 2) % tile)) % len(s)
    # The half-tile candidate may be a pixel or two off the exact half.
    near = [(half + d) % len(s) for d in (-2, -1, 0, 1, 2)]
    alt = max(near, key=lambda i: s[i])
    if s[alt] >= 0.6 * s[best] and _circ(alt, prefer, tile) < _circ(best, prefer, tile):
        return float(alt)
    return float(best)


def phase_from_boxes(boxes: list[tuple[float, float, float, float, float]], tile: float,
                     ) -> tuple[float | None, float | None]:
    """The tile-line phase (x, y) suggested by detected things, which mostly
    stand centred in their tiles: boxes are (x0, y0, x1, y1, weight) in
    capture pixels. None per axis when they don't agree."""
    keep = [(x0, y0, x1, y1, w) for x0, y0, x1, y1, w in boxes
            if 0.5 * tile <= x1 - x0 <= 1.5 * tile and 0.5 * tile <= y1 - y0 <= 1.5 * tile]
    if len(keep) < 5:
        return None, None
    out = []
    for axis in (0, 1):
        ang = np.array([((b[axis] + b[axis + 2]) / 2 - tile / 2) % tile / tile * 2 * math.pi for b in keep])
        wt = np.array([b[4] for b in keep])
        c, s = (wt * np.cos(ang)).sum() / wt.sum(), (wt * np.sin(ang)).sum() / wt.sum()
        out.append(math.atan2(s, c) % (2 * math.pi) / (2 * math.pi) * tile if math.hypot(c, s) > 0.35 else None)
    return out[0], out[1]


def tile_crop(world: Rect, tile: float, phase_x: float, phase_y: float) -> tuple[int, int, int, int, int, int]:
    """The part of the capture the model gets: whole tiles starting on a tile
    line, from the one cut by the world's left/top edge to past its
    right/bottom edge, always the same number of them so the model's input
    size doesn't change as the lines move. Returns x, y (may be left of or
    above the world), w, h in capture pixels and the columns and rows."""
    cols, rows = int(math.ceil(world.w / tile)) + 1, int(math.ceil(world.h / tile)) + 1
    dx = (phase_x - world.x) % tile
    dy = (phase_y - world.y) % tile
    x = world.x + dx - (tile if dx > 0.5 else 0)
    y = world.y + dy - (tile if dy > 0.5 else 0)
    return int(round(x)), int(round(y)), int(round(cols * tile)), int(round(rows * tile)), cols, rows


def cut(bgr: np.ndarray, world: Rect, x: int, y: int, w: int, h: int) -> np.ndarray:
    """bgr[y:y + h, x:x + w], black outside the world (the parts of edge
    tiles past it, letterbox bars, the Separated layout's chat panel)."""
    out = np.zeros((h, w, 3), bgr.dtype)
    wx, wy, ww, wh = world.ints()
    x0, y0 = max(x, wx, 0), max(y, wy, 0)
    x1, y1 = min(x + w, wx + ww, bgr.shape[1]), min(y + h, wy + wh, bgr.shape[0])
    if x1 > x0 and y1 > y0:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = bgr[y0:y1, x0:x1]
    return out


# Menus standing still ---------------------------------------------------------------

class StillPixels:
    """Pixels that stay the same while the world scrolls: menus. Compares
    small copies of the capture half a second apart; only comparisons where
    most of the picture changed (the camera moved) count."""

    def __init__(self, shrink: int = 4, gap: float = 0.5, moved: float = 0.35, rate: float = 0.12):
        self.shrink, self.gap, self.moved, self.rate = shrink, gap, moved, rate
        self.ref: tuple[float, np.ndarray] | None = None
        self.still: np.ndarray | None = None
        self.events = 0

    def reset(self):
        self.ref, self.still, self.events = None, None, 0

    def observe(self, bgr: np.ndarray, now: float, keep_out: Rect | None = None) -> bool:
        """keep_out (capture pixels): the player's tiles, which follow the
        camera. True when this frame counted as the world moving."""
        k = self.shrink
        small = bgr[::k, ::k].astype(np.int16).sum(axis=2)
        if self.ref is None or self.ref[1].shape != small.shape:
            self.ref, self.still, self.events = (now, small), np.zeros(small.shape, np.float32), 0
            return False
        if now - self.ref[0] < self.gap:
            return False
        diff = np.abs(small - self.ref[1])
        self.ref = (now, small)
        changed = diff > 24
        lit = small > 3 * 14                     # leave out the dark: black stays black when scrolled
        if lit.mean() < 0.2 or changed[lit].mean() < self.moved:
            return False
        still = ((diff < 9) & lit).astype(np.float32)
        if keep_out is not None:
            x, y, w, h = (int(v / k) for v in keep_out.ints())
            still[max(0, y):y + h, max(0, x):x + w] = 0
        self.still += self.rate * (still - self.still)
        self.events += 1
        return True

    def boxes(self, min_events: int = 8, level: float = 0.75, min_area: int = 6) -> list[Rect]:
        """Boxes (capture pixels) around still areas, once enough movement has been seen."""
        if self.still is None or self.events < min_events:
            return []
        import cv2

        m = (self.still > level).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
        k = self.shrink
        return [Rect(x * k, y * k, w * k, h * k) for x, y, w, h, a in stats[1:] if a >= min_area]


# The tracker ------------------------------------------------------------------------

@dataclass
class ViewEstimate:
    world: Rect                     # the visible game world, capture pixels (client area, 1 px per point)
    tile: float                     # capture pixels per tile
    hud: list[Rect]                 # menus over the world, capture pixels
    source: str                     # e.g. "settings+grid+player"
    confidence: float
    note: str = ""
    zoom: float = 1.0               # tiles shown / the unzoomed view's
    checks: dict = field(default_factory=dict)


class ViewTracker:
    """Works out the game view from the settings, the picture and the
    player, re-checks it every few seconds and keeps the last good answer
    while a check can't be made."""

    def __init__(self, settings: ClientSettings, backing: float = 2.0, every: float = 3.0):
        self.s, self.backing, self.every = settings, backing, every
        self.ui_scale = settings.ui_scale or backing
        self.current: ViewEstimate | None = None
        self.pending: ViewEstimate | None = None
        self.last_time = 0.0
        self.size: tuple[int, int] | None = None
        self.still = StillPixels()
        self.last_zoom: float | None = None

    def due(self, size: tuple[int, int], now: float) -> bool:
        return self.current is None or size != self.size or now - self.last_time >= self.every

    def geometry(self, cap_w: int, cap_h: int) -> tuple[Rect, Rect, float, int | None]:
        """From the settings alone: control and world (capture pixels), the
        tile size and the fixed scale."""
        b = self.backing
        ctrl = control_rect(cap_w * b, cap_h * b, self.s, self.ui_scale)
        box, factor = draw_box(ctrl.w, ctrl.h, self.s)
        world = box.moved(ctrl.x, ctrl.y).intersect(ctrl)
        return ctrl.scaled(1 / b), world.scaled(1 / b), box.h / VIEW_H / b, factor

    def estimate(self, bgr: np.ndarray, players: list[tuple[float, float]], now: float | None = None,
                 tile_hint: float | None = None) -> ViewEstimate:
        """bgr: the client area (capture pixels); players: centres of
        detected players (capture pixels); tile_hint: the floor grid's tile
        size if the caller measured it."""
        now = time.time() if now is None else now
        h, w = bgr.shape[:2]
        size_changed = self.size != (w, h)
        if size_changed:
            self.still.reset()
        self.size, self.last_time = (w, h), now
        ctrl, world, tile_cfg, factor = self.geometry(w, h)
        source = ["settings" if self.s.path else "defaults"]
        conf = 0.55 if self.s.path else 0.35
        notes, checks = [], {}

        grey = bgr.mean(axis=2)
        # Separated layout: find the chat panel's edge near where the setting puts it.
        if self.s.layout == "separated":
            edge = find_panel_edge(bgr, ctrl.right)
            checks["panel_edge"] = edge
            if edge is not None:
                ctrl2 = Rect(0, 0, edge, h)
                b = self.backing
                box, factor = draw_box(ctrl2.w * b, ctrl2.h * b, self.s)
                ctrl, world = ctrl2, box.scaled(1 / b).intersect(ctrl2)
                tile_cfg = box.h / VIEW_H / b
                source.append("panel")
                conf += 0.1
        # Letterbox bars must be black.
        bars = [r for r in (Rect(ctrl.x, ctrl.y, world.x - ctrl.x, ctrl.h),
                            Rect(world.right, ctrl.y, ctrl.right - world.right, ctrl.h),
                            Rect(ctrl.x, ctrl.y, ctrl.w, world.y - ctrl.y),
                            Rect(ctrl.x, world.bottom, ctrl.w, ctrl.bottom - world.bottom)) if r.w >= 2 and r.h >= 2]
        if bars:
            dark = all(grey[int(r.y):int(r.bottom), int(r.x):int(r.right)].mean() < 10 for r in bars)
            checks["letterbox_black"] = dark
            if dark:
                conf += 0.1
            else:
                conf -= 0.25
                notes.append("letterbox bars aren't black (settings out of date?)")
        # Menus: the layout's boxes plus anything seen standing still while the world moved.
        hud_area = Rect(0, 0, w, h).scaled(self.backing) if self.s.layout != "separated" else \
            ctrl.scaled(self.backing)
        menus = [r.scaled(1 / self.backing) for r in hud_rects(hud_area, self.s, self.ui_scale)]
        moving = self.still.boxes()
        if moving:
            source.append("still-menus")
        menus += moving
        # The floor grid: confirms the tile size, or shows the eye is zoomed. Menus are blanked
        # first: rows of buttons repeat too.
        x, y, ww, hh = world.ints()
        floor = grey[y:y + hh, x:x + ww].copy()
        for r in menus:
            rx, ry, rw, rh = r.moved(-x, -y).ints()
            floor[max(0, ry):max(0, ry + rh), max(0, rx):max(0, rx + rw)] = 0
        # Only the sizes the game can show are tried: the settings' tile at each eye zoom.
        if tile_hint:
            grid = tile_hint
        else:
            grid, scores = tile_from_candidates(floor, [tile_cfg / z for z in ZOOMS])
            # Another zoom only when its repeat is clearly sharper than the settings' tile's.
            if grid and abs(grid - tile_cfg) > 0.5 and scores[grid] < 1.3 * scores.get(tile_cfg, 0.0):
                grid = None
        checks["tile_settings"], checks["tile_grid"] = round(tile_cfg, 2), grid and round(grid, 2)
        # A zoom is used only once two checks in a row agree on it (the first look included).
        seen_zoom = round(tile_cfg / grid, 3) if grid else None
        level_seen = min(ZOOMS, key=lambda z: abs(seen_zoom / z - 1)) if seen_zoom else None
        self.last_zoom, previous_zoom = level_seen, self.last_zoom
        tile, zoom = tile_cfg, 1.0
        if grid:
            ratio = tile_cfg / grid
            level = min(ZOOMS, key=lambda z: abs(ratio / z - 1))
            if abs(ratio - 1) <= 0.05:
                source.append("grid")
                conf += 0.2
            elif abs(ratio / level - 1) <= 0.05 and previous_zoom == level:
                tile, zoom = tile_cfg / level, level
                source.append(f"grid (zoom {level:g})")
                conf += 0.1
            elif abs(ratio / level - 1) <= 0.05:
                notes.append(f"floor grid looks like zoom {level:g}; checking again before using it")
            else:
                notes.append(f"floor grid says {grid:.0f} px a tile, not a zoom the game has; kept {tile_cfg:.0f}")
        else:
            notes.append("no floor grid to check the tile size (dark view)")
        # The eye follows the player.
        cx, cy = world.centre
        near = min((math.hypot(px - cx, py - cy) for px, py in players), default=None)
        checks["player_offset_tiles"] = None if near is None else round(near / tile, 2)
        if near is not None and near <= 1.5 * tile:
            source.append("player")
            conf += 0.15
        elif near is not None and near > 3 * tile:
            conf -= 0.2
            notes.append("player far from the centre (camera or console view?)")
        elif near is None and grid:
            notes.append("no player found")
        elif near is None:
            conf -= 0.15
            notes = [n for n in notes if not n.startswith("no floor grid")]
            notes.append("no game world recognised: no floor grid, no player (lobby, a menu or a dark view)")
        # A quarter tile of margin: a thing whose box centre is just off a menu is usually the menu's edge.
        pad = 0.25 * tile
        hud = [Rect(r.x - pad, r.y - pad, r.w + 2 * pad, r.h + 2 * pad).intersect(world) for r in menus]
        hud = [r for r in hud if r.w > 0 and r.h > 0]
        est = ViewEstimate(world, tile, hud, "+".join(source), round(max(0.0, min(1.0, conf)), 2),
                           "; ".join(notes), round(zoom, 3), checks)
        return self._accept(est, size_changed)

    def _accept(self, est: ViewEstimate, size_changed: bool) -> ViewEstimate:
        """Hysteresis: a changed view must be seen twice (unless the window
        changed size); a weak estimate keeps the last good one."""
        cur = self.current
        if cur is None or size_changed:
            self.current, self.pending = est, None
            return est
        if est.confidence < 0.45:
            held = replace(cur, note=f"holding last good ({est.note or 'low confidence'})", hud=est.hud or cur.hud)
            self.current = held
            return held
        same = est.world.error(cur.world) <= 2 and abs(est.tile - cur.tile) <= 0.02 * cur.tile
        if same or (self.pending is not None and est.world.error(self.pending.world) <= 2
                    and abs(est.tile - self.pending.tile) <= 0.02 * est.tile):
            self.current, self.pending = (replace(est, world=cur.world, tile=cur.tile) if same else est), None
            return self.current
        self.pending = est
        return replace(cur, hud=est.hud, note="view changed; confirming")


def find_panel_edge(bgr: np.ndarray, guess: float, colour=PANEL_BGR, tol: int = 10) -> float | None:
    """The Separated layout's chat panel: the column, near guess, from which
    the rows are mostly the panel's flat colour."""
    h, w = bgr.shape[:2]
    is_panel = (np.abs(bgr.astype(np.int16) - np.array(colour, np.int16)).max(axis=2) <= tol).mean(axis=0)
    lo, hi = int(max(1, guess - 0.15 * w)), int(min(w - 2, guess + 0.15 * w))
    best = None
    for x in range(lo, hi):
        right = is_panel[x:min(w, x + 12)].mean()
        left = is_panel[max(0, x - 12):x].mean()
        score = right - left
        if right >= 0.35 and score >= 0.25 and (best is None or score > best[1]):
            best = (float(x), score)
    return best[0] if best else None


def keep_detection(cx: float, cy: float, hud: list[Rect]) -> bool:
    return not any(r.contains(cx, cy) for r in hud)
