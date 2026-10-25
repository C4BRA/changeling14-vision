"""Live overlay: a replay model's boxes and names drawn on top of the SS14
client window (macOS).

The overlay finds the client window itself (CGWindowListCopyWindowInfo),
captures only that window by its id, so covering windows and the overlay
never get in the picture, runs the model on the game-view part of it and
draws in a transparent, click-through, always-on-top window placed exactly
over the game view. It follows the window when it moves or resizes and hides
while it is minimised or on another space. A small control panel pauses,
toggles labels and floor tiles, sets the threshold, re-picks the game-view
area and quits.

Read-only: nothing is sent to the game and the client is not modified, so it
works with the normal launcher on any server. macOS asks for Screen Recording
permission for the app that runs it (Terminal, usually).

The game-view area is stored as fractions of the window in
~/.changeling14/overlay.json, so it survives moves and resizes.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STORE = Path("~/.changeling14/overlay.json")
# Owner or window names of SS14 clients: the launcher-started client is owned
# by "Space Station 14"; a client built from source runs as Content.Client.
DEFAULT_NAMES = ("Space Station 14", "Content.Client", "Robust.Client", "SS14.Loader")
EXCLUDE = ("launcher",)
MIN_SIZE = 200              # points; smaller windows are menus, tooltips and the like
TITLE_BAR = 28             # points of a standard macOS title bar, included in a window's bounds
VIEW_TILES = (21, 15)      # SS14's game view at the default zoom: 21x15 tiles, scaled to fit the window
_on_start = None            # tests: called with the app, windows and state once they exist


# Pure parts (tested without macOS) ---------------------------------------------

@dataclass(frozen=True)
class Window:
    wid: int
    owner: str
    name: str
    x: float                # points, top-left of the primary display, y down
    y: float
    w: float
    h: float
    onscreen: bool
    layer: int = 0


def windows_from_info(info: list[dict]) -> list[Window]:
    out = []
    for w in info:
        b = w.get("kCGWindowBounds") or {}
        out.append(Window(int(w.get("kCGWindowNumber", 0)), str(w.get("kCGWindowOwnerName") or ""),
                          str(w.get("kCGWindowName") or ""), float(b.get("X", 0)), float(b.get("Y", 0)),
                          float(b.get("Width", 0)), float(b.get("Height", 0)), bool(w.get("kCGWindowIsOnscreen")),
                          int(w.get("kCGWindowLayer", 0))))
    return out


def match_window(windows: list[Window], names: tuple[str, ...] = DEFAULT_NAMES) -> Window | None:
    """The largest normal-layer window whose owner or title contains one of
    names (case-insensitive), leaving out the launcher."""
    want = [n.lower() for n in names]
    found = [w for w in windows
             if w.layer == 0 and w.w >= MIN_SIZE and w.h >= MIN_SIZE
             and any(n in w.owner.lower() or n in w.name.lower() for n in want)
             and not any(e in (w.owner + " " + w.name).lower() for e in EXCLUDE)]
    return max(found, key=lambda w: w.w * w.h) if found else None


@dataclass(frozen=True)
class Area:
    """The game view as fractions of the window: left, top, width, height."""
    fx: float = 0.0
    fy: float = 0.0
    fw: float = 1.0
    fh: float = 1.0

    def clamp(self) -> "Area":
        fx, fy = min(max(self.fx, 0.0), 0.99), min(max(self.fy, 0.0), 0.99)
        return Area(fx, fy, min(max(self.fw, 0.01), 1 - fx), min(max(self.fh, 0.01), 1 - fy))

    def points(self, win: Window) -> tuple[float, float, float, float]:
        """The game view in screen points (top-left origin)."""
        return win.x + self.fx * win.w, win.y + self.fy * win.h, self.fw * win.w, self.fh * win.h

    def pixels(self, width_px: int, height_px: int) -> tuple[int, int, int, int]:
        """The game view in a capture of the window that is width_px x height_px."""
        x0, y0 = int(round(self.fx * width_px)), int(round(self.fy * height_px))
        x1, y1 = int(round((self.fx + self.fw) * width_px)), int(round((self.fy + self.fh) * height_px))
        return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)

    @staticmethod
    def from_drag(win_w: float, win_h: float, x0: float, y0: float, x1: float, y1: float) -> "Area":
        """A box dragged in window points (any corner order)."""
        lx, rx, ty, by = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        return Area(lx / win_w, ty / win_h, (rx - lx) / win_w, (by - ty) / win_h).clamp()


def default_area(win: Window) -> Area:
    """The whole window below its title bar."""
    return Area(0.0, min(0.5, TITLE_BAR / max(win.h, 1.0)), 1.0, 1.0 - min(0.5, TITLE_BAR / max(win.h, 1.0)))


def tile_from_view(view_w_px: int, view_h_px: int) -> float:
    """Pixels per tile at the default zoom: SS14 scales its 21x15-tile view to
    fit the window, so the tighter side decides (a wide window shows more
    tiles across, a tall one more down)."""
    return min(view_w_px / VIEW_TILES[0], view_h_px / VIEW_TILES[1])


def tile_from_grid(grey: np.ndarray, lo: int = 24, hi: int = 220) -> float | None:
    """Pixels per tile from the floor's grid: the strongest repeat of edge
    strength across the view (columns, then rows). None if nothing repeats
    clearly, e.g. a mostly dark view."""
    if grey.mean() < 12:                  # mostly dark: no floor grid to measure
        return None
    best = None
    for axis in (1, 0):
        d = np.abs(np.diff(grey.astype(np.float32), axis=axis)).mean(axis=1 - axis)
        d = d - d.mean()
        if len(d) <= hi or not d.any():
            continue
        ac = np.correlate(d, d, "full")[len(d) - 1:]
        ac = ac / ac[0]
        k = lo + int(np.argmax(ac[lo:hi]))
        # Many floor tiles have a seam across their middle, so half a tile can
        # repeat as strongly as a whole one: step up while the double is nearly as strong.
        while 2 * k < min(hi, len(ac)) and ac[2 * k] >= 0.85 * ac[k]:
            k *= 2
        if k > lo and ac[k] >= 0.3 and (best is None or ac[k] > best[1]):
            best = (float(k), float(ac[k]))
    return best[0] if best else None


def load_tile_points(store: Path = STORE) -> float | None:
    """The last tile size measured from the floor grid, in screen points."""
    try:
        v = float(json.loads(Path(store).expanduser().read_text()).get("tile_pt"))
        return v if v > 0 else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def save_tile_points(tile_pt: float, store: Path = STORE) -> None:
    path = Path(store).expanduser()
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        d = {}
    d["tile_pt"] = round(float(tile_pt), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.replace(tmp, path)


def load_area(store: Path = STORE) -> Area | None:
    try:
        d = json.loads(Path(store).expanduser().read_text())
        return Area(float(d["fx"]), float(d["fy"]), float(d["fw"]), float(d["fh"])).clamp()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_area(area: Area, store: Path = STORE) -> None:
    path = Path(store).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        d = {}
    d.update({"fx": area.fx, "fy": area.fy, "fw": area.fw, "fh": area.fh})
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.replace(tmp, path)


def model_to_view(box: tuple[float, float, float, float], tile_px: float, px_per_point: float,
                  cell: int = 32) -> tuple[float, float, float, float]:
    """A box in model pixels (cell per tile) -> points from the game view's
    top-left: model -> captured pixels (tile_px per tile) -> points."""
    k = tile_px / cell / px_per_point
    return tuple(v * k for v in box)


def cocoa_frame(x: float, y: float, w: float, h: float, primary_height: float) -> tuple[float, float, float, float]:
    """Top-left-origin screen points (Quartz windows) -> Cocoa's bottom-left origin."""
    return x, primary_height - y - h, w, h


def rgb(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    """predict.MOB_COLOURS are BGR for OpenCV; Cocoa wants RGB in 0..1."""
    b, g, r = bgr
    return r / 255, g / 255, b / 255


# macOS parts -------------------------------------------------------------------

PERMISSION_HELP = (
    "The overlay needs Screen Recording permission to see the game window.\n"
    "  System Settings -> Privacy & Security -> Screen Recording -> turn on the app you ran this from\n"
    "  (Terminal, iTerm, or the Claude app), then quit and reopen that app and run this again.")


@dataclass
class Result:
    found: list = field(default_factory=list)       # predict.Guess, boxes in model pixels
    floor: object = None                            # predict.FloorGuess or None
    tile_px: float = 0.0
    px_per_point: float = 2.0
    ms: float = 0.0
    fps: float = 0.0
    note: str = ""
    view: object = None                             # the captured game view this result belongs to


def capture_window(Quartz, wid: int, rect: tuple[float, float, float, float] | None = None,
                   nominal: bool = False) -> np.ndarray | None:
    """The window's own pixels as BGR (H, W, 3), or None. rect (left, top,
    width, height in screen points) captures only that part of it; nominal
    captures one pixel per point instead of the Retina two."""
    bounds = Quartz.CGRectNull if rect is None else Quartz.CGRectMake(*rect)
    opts = Quartz.kCGWindowImageBoundsIgnoreFraming | (Quartz.kCGWindowImageNominalResolution if nominal else 0)
    img = Quartz.CGWindowListCreateImage(bounds, Quartz.kCGWindowListOptionIncludingWindow, wid, opts)
    if img is None:
        return None
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    if w == 0 or h == 0:
        return None
    row = Quartz.CGImageGetBytesPerRow(img)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    a = np.frombuffer(data, dtype=np.uint8).reshape(h, row // 4, 4)[:, :w]
    return np.ascontiguousarray(a[:, :, :3])        # BGRA (little-endian, alpha first) -> BGR


def status_line(d: dict) -> str:
    """One line for whoever runs the overlay (the app reads these)."""
    return "STATUS " + json.dumps(d, separators=(",", ":"))


def run(folder: Path, names: tuple[str, ...] | None = None, tile_px: float | None = None, threshold: float = 0.5,
        name_threshold: float = 0.6, labels: bool = True, show_floor: bool = False, pick: bool = False,
        fps: float = 30, device: str = "auto", store: Path = STORE, backend: str = "auto") -> None:
    """backend: "coreml" runs the model on the Neural Engine (exported for the
    game view's size the first time, then cached), "torch" on the GPU/CPU,
    "auto" Core ML when coremltools is installed."""
    try:
        import AppKit
        import cv2
        import Quartz
        from Foundation import NSObject, NSTimer
    except ImportError as exc:
        raise SystemExit('the overlay needs macOS and: pip install -e ".[live]"') from exc
    import torch

    from .detect import CELL
    from .predict import MOB_COLOURS, _names, _tensor, _to_model, floor_guesses, guesses_fast, load_detect_model

    if not Quartz.CGPreflightScreenCaptureAccess():
        Quartz.CGRequestScreenCaptureAccess()
        raise SystemExit(PERMISSION_HELP)
    names = tuple(names) if names else DEFAULT_NAMES

    def find() -> Window | None:
        # On-screen windows only: a minimised client, or one on another space, isn't followed.
        info = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID) or []
        return match_window(windows_from_info(list(info)), names)

    win = find()
    if win is None:
        raise SystemExit(f"no SS14 client window found (looked for {', '.join(names)}); open the game first, "
                         "or pass --window-name")
    if backend == "auto":
        try:
            import coremltools  # noqa: F401
            backend = "coreml"
        except ImportError:
            backend = "torch"
    # With Core ML the torch model only supplies the entity texts and naming scale, on the CPU.
    model, table, dev = load_detect_model(folder, "cpu" if backend == "coreml" else device)
    all_t = model.texts()
    scale, bias = model.scale.exp().item(), model.bias.item()
    word = _names(table)
    engine_name = "Neural Engine (Core ML)" if backend == "coreml" else f"PyTorch on {dev}"
    print(f"overlay: following window {win.wid} ({win.owner!r}, {win.w:.0f}x{win.h:.0f} points); "
          f"model {folder.name}, {engine_name}. Ctrl-C or Quit in the panel stops it.", flush=True)

    state = {"win": win, "area": load_area(store) or default_area(win), "paused": False, "labels": labels,
             "floor": show_floor, "threshold": threshold, "tile_px": tile_px, "fixed_tile": tile_px is not None,
             "cap_size": None, "result": Result(note="starting"), "stop": False, "picking": pick}
    lock = threading.Lock()

    def worker():
        try:
            work()
        except Exception as exc:            # keep the UI alive; say what went wrong
            import traceback
            traceback.print_exc()
            with lock:
                state["result"] = Result(note=f"model error: {exc}")

    shared = {"shot": None, "seq": 0, "cap_ms": 0.0}
    ready = threading.Condition()

    def capturer():
        """Grabs the game view while the model works on the previous picture."""
        last = 0.0
        while not state["stop"]:
            wait = 1 / fps - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            w = state["win"]
            if state["paused"] or state["picking"] or w is None or not w.onscreen:
                continue
            t0 = time.time()
            # Only the game view, one pixel per point: the model works at 32 px a tile anyway.
            shot = capture_window(Quartz, w.wid, state["area"].points(w), nominal=True)
            if shot is None:
                continue
            with ready:
                shared["shot"], shared["win"] = shot, w
                shared["seq"] += 1
                shared["cap_ms"] = (time.time() - t0) * 1000
                ready.notify()

    engines: dict = {}

    def run_model(frame: np.ndarray) -> dict:
        if backend != "coreml":
            with torch.no_grad():
                return model(_tensor(frame, dev))
        size = frame.shape[:2]
        if size not in engines:
            from . import coreml
            with lock:
                state["result"] = Result(note=f"converting for the Neural Engine at {size[0]}x{size[1]} (once)")
            path = coreml.export(folder, *size, log=lambda m: print(m, flush=True))
            engines[size] = coreml.CoreMLEngine(path, *size)
        x = frame[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255
        return engines[size](x)

    def work():
        seen = 0
        frames, t_rate, rate, t_status = 0, time.time(), 0.0, 0.0
        while not state["stop"]:
            with ready:
                ready.wait_for(lambda: shared["seq"] != seen or state["stop"], timeout=0.5)
                if shared["seq"] == seen:
                    continue
                seen, shot, w, cap_ms = shared["seq"], shared["shot"], shared["win"], shared["cap_ms"]
            view = shot
            if view.shape[0] < 2 * CELL or view.shape[1] < 2 * CELL:
                continue
            px_per_point = view.shape[1] / max(1.0, state["area"].fw * w.w)
            retry = state.get("tile_source") in ("saved", "default") and time.time() - state.get("tile_tried", 0) > 3
            if not state["fixed_tile"] and (state["tile_px"] is None or state["cap_size"] != view.shape[:2] or retry):
                # The floor's grid gives the tile size at any zoom. While it can't be measured (a dark
                # view), use the last measured size, else the default zoom (21x15 tiles), and try again.
                state["tile_tried"] = time.time()
                grid = tile_from_grid(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY))
                saved = load_tile_points(store)
                if grid:
                    tile, source = grid, "grid"
                    save_tile_points(grid / px_per_point, store)
                elif saved:
                    tile, source = saved * px_per_point, "saved"
                else:
                    tile, source = tile_from_view(view.shape[1], view.shape[0]), "default"
                changed = state["tile_px"] != tile or state["cap_size"] != view.shape[:2]
                state["tile_px"], state["tile_source"], state["cap_size"] = tile, source, view.shape[:2]
                if changed:
                    print(f"overlay: game view {view.shape[1]}x{view.shape[0]} px, one tile = {tile:.0f} px ({source}; "
                          "Tile size in the panel, or --tile-px, to change it)", flush=True)
                    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(lambda: sync_tile_slider())
            t0 = time.time()
            frame = _to_model(view, state["tile_px"], cv2)
            t1 = time.time()
            out = run_model(frame)
            t2 = time.time()
            found = guesses_fast(model, word, out, state["threshold"], all_t=all_t, scale=scale, bias=bias)
            floor = floor_guesses(model, out) if state["floor"] else None
            t3 = time.time()
            frames += 1
            now = time.time()
            if now - t_rate >= 1:
                rate, frames, t_rate = frames / (now - t_rate), 0, now
            ms = {"capture": round(cap_ms, 1), "resize": round((t1 - t0) * 1000, 1),
                  "model": round((t2 - t1) * 1000, 1), "names": round((t3 - t2) * 1000, 1)}
            with lock:
                state["result"] = Result(found, floor[0] if floor else None, state["tile_px"], px_per_point,
                                         (t3 - t0) * 1000, rate, view=view)
                state["ms"] = ms
            if now - t_status >= 1:
                t_status = now
                print(status_line({"fps": round(rate, 1), "ms": ms, "engine": engine_name, "window": True,
                                   "tile_px": state["tile_px"], "tile_source": state.get("tile_source", "manual"),
                                   "input": list(frame.shape[:2]),
                                   "things": len(found)}), flush=True)

    primary_h = AppKit.NSScreen.screens()[0].frame().size.height

    class OverlayView(AppKit.NSView):
        def isFlipped(self):
            return True

        def acceptsFirstMouse_(self, event):
            return True

        def drawRect_(self, rect):
            if state["picking"]:
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0.4, 1, 0.15).set()
                AppKit.NSRectFillUsingOperation(self.bounds(), AppKit.NSCompositingOperationSourceOver)
                _text("Drag a box around the game view (not chat or the HUD)", 12, 12, (1, 1, 1), 15)
                d = getattr(self, "drag", None)
                if d:
                    (ax, ay), (bx, by) = d
                    r = AppKit.NSMakeRect(min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay))
                    AppKit.NSColor.yellowColor().set()
                    p = AppKit.NSBezierPath.bezierPathWithRect_(r)
                    p.setLineWidth_(2)
                    p.stroke()
                return
            with lock:
                res = state["result"]
            if state["paused"]:
                _text("paused", 6, 6, (1, 1, 0), 12)
                return
            if not res.tile_px:
                _text(res.note, 6, 6, (0, 1, 0), 12)
                return
            k = lambda b: model_to_view(b, res.tile_px, res.px_per_point, CELL)
            if res.floor is not None:
                cell_pt = res.tile_px / res.px_per_point
                for (r, c), _gases in res.floor.gas.items():
                    AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.8, 0, 0.8, 0.25).set()
                    AppKit.NSRectFillUsingOperation(AppKit.NSMakeRect(c * cell_pt, r * cell_pt, cell_pt, cell_pt),
                                                    AppKit.NSCompositingOperationSourceOver)
                for r, row in enumerate(res.floor.tiles):
                    for c, tile in enumerate(row):
                        _text(tile[:10], c * cell_pt + 1, (r + 1) * cell_pt - 10, (0.75, 0.75, 0.75), 7)
            for g in res.found:
                x0, y0, x1, y1 = k(g.box)
                player = g.player >= 0.5
                bgr = MOB_COLOURS.get(g.mob) or ((255, 0, 255) if player else (255, 200, 0))
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgb(bgr), 0.95).set()
                p = AppKit.NSBezierPath.bezierPathWithRect_(AppKit.NSMakeRect(x0, y0, x1 - x0, y1 - y0))
                if g.slot:                      # second and third boxes in a square: thinner, dashed
                    p.setLineWidth_(1)
                    p.setLineDash_count_phase_([3.0, 2.0], 2, 0)
                else:
                    p.setLineWidth_(2 if g.mob != "none" or player else 1.2)
                p.stroke()
                if state["labels"]:
                    label = g.name[:18] if g.names[0][1] >= name_threshold else "?"
                    if g.mob not in ("none", "humanoid"):
                        label = f"{g.mob}: {label}"
                    elif player:
                        label = f"player: {label}"
                    _text(label, x0, max(0, y0 - 12), (1, 1, 1), 9, shadow=True)
            _text(f"{len(res.found)} things, {res.fps:.0f} fps, {res.ms:.0f} ms ({engine_name})", 6, 6, (0, 1, 0), 11,
                  shadow=True)

        def mouseDown_(self, e):
            if state["picking"]:
                p = self.convertPoint_fromView_(e.locationInWindow(), None)
                self.drag = [(p.x, p.y), (p.x, p.y)]
                self.setNeedsDisplay_(True)

        def mouseDragged_(self, e):
            if state["picking"] and getattr(self, "drag", None):
                p = self.convertPoint_fromView_(e.locationInWindow(), None)
                self.drag[1] = (p.x, p.y)
                self.setNeedsDisplay_(True)

        def mouseUp_(self, e):
            if not (state["picking"] and getattr(self, "drag", None)):
                return
            (ax, ay), (bx, by) = self.drag
            self.drag = None
            w = state["win"]
            if abs(bx - ax) > 20 and abs(by - ay) > 20 and w is not None:
                area = Area.from_drag(w.w, w.h, ax, ay, bx, by)
                state["area"] = area
                save_area(area, store)
                print(f"overlay: game view saved as {area} in {Path(store).expanduser()}", flush=True)
            state["picking"] = False
            state["cap_size"] = None            # work out the tile size again
            overlay.setIgnoresMouseEvents_(True)

    def _text(s, x, y, colour, size, shadow=False):
        attrs = {AppKit.NSFontAttributeName: AppKit.NSFont.boldSystemFontOfSize_(size),
                 AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                     *colour, 1.0)}
        if shadow:
            sh = AppKit.NSShadow.alloc().init()
            sh.setShadowColor_(AppKit.NSColor.blackColor())
            sh.setShadowBlurRadius_(2)
            sh.setShadowOffset_(AppKit.NSMakeSize(0, 0))
            attrs[AppKit.NSShadowAttributeName] = sh
        AppKit.NSString.stringWithString_(s).drawAtPoint_withAttributes_(AppKit.NSMakePoint(x, y), attrs)

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    behaviour = (AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces | AppKit.NSWindowCollectionBehaviorStationary
                 | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
                 | AppKit.NSWindowCollectionBehaviorIgnoresCycle)
    overlay = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(0, 0, 100, 100), AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
    overlay.setOpaque_(False)
    overlay.setBackgroundColor_(AppKit.NSColor.clearColor())
    overlay.setHasShadow_(False)
    overlay.setIgnoresMouseEvents_(not pick)
    overlay.setLevel_(AppKit.NSStatusWindowLevel)
    overlay.setCollectionBehavior_(behaviour)
    overlay.setSharingType_(AppKit.NSWindowSharingNone)        # never in any screen capture
    view = OverlayView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 100, 100))
    overlay.setContentView_(view)

    # Control panel -------------------------------------------------------------

    class Controls(NSObject):
        def pause_(self, sender):
            state["paused"] = not state["paused"]
            sender.setTitle_("Resume" if state["paused"] else "Pause")

        def labels_(self, sender):
            state["labels"] = bool(sender.state())

        def floor_(self, sender):
            state["floor"] = bool(sender.state())

        def threshold_(self, sender):
            state["threshold"] = float(sender.doubleValue())
            thr_label.setStringValue_(f"Threshold {state['threshold']:.2f}")

        def tile_(self, sender):
            state["tile_px"] = float(sender.doubleValue())
            state["fixed_tile"] = True
            state["tile_source"] = "manual"
            tile_label.setStringValue_(f"Tile size {state['tile_px']:.0f} px")

        def pick_(self, sender):
            state["picking"] = True
            overlay.setIgnoresMouseEvents_(False)
            view.setNeedsDisplay_(True)

        def quit_(self, sender):
            state["stop"] = True
            with ready:
                ready.notify_all()
            thread.join(timeout=2)          # let a running Core ML prediction finish before exiting
            app.terminate_(None)

        def tick_(self, timer):
            w = find()
            state["win"] = w
            if w is None or not w.onscreen or w.w < MIN_SIZE:
                if overlay.isVisible():
                    overlay.orderOut_(None)
                now = time.time()
                if now - state.get("missing_said", 0) >= 1:
                    state["missing_said"] = now
                    print(status_line({"fps": 0, "engine": engine_name, "window": False}), flush=True)
                return
            area = Area() if state["picking"] else state["area"]
            x, y, ww, hh = area.points(w)
            fx, fy, fw, fh = cocoa_frame(x, y, ww, hh, primary_h)
            frame = AppKit.NSMakeRect(fx, fy, fw, fh)
            if overlay.frame() != frame:
                overlay.setFrame_display_(frame, True)
            if not overlay.isVisible():
                overlay.orderFrontRegardless()
            view.setNeedsDisplay_(True)

    controls = Controls.alloc().init()
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(40, 80, 230, 236),
        AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskUtilityWindow | AppKit.NSWindowStyleMaskNonactivatingPanel,
        AppKit.NSBackingStoreBuffered, False)
    panel.setTitle_("changeling14 overlay")
    panel.setLevel_(AppKit.NSFloatingWindowLevel)
    panel.setCollectionBehavior_(behaviour)
    panel.setHidesOnDeactivate_(False)
    box = panel.contentView()

    def button(title, action, y, kind=None, on=False):
        b = AppKit.NSButton.alloc().initWithFrame_(AppKit.NSMakeRect(12, y, 206, 24))
        b.setTitle_(title)
        if kind is not None:
            b.setButtonType_(kind)
            b.setState_(1 if on else 0)
        else:
            b.setBezelStyle_(AppKit.NSBezelStyleRounded)
        b.setTarget_(controls)
        b.setAction_(action)
        box.addSubview_(b)
        return b

    button("Pause", "pause:", 202)
    button("Labels", "labels:", 176, AppKit.NSButtonTypeSwitch, labels)
    button("Floor tiles and gas", "floor:", 152, AppKit.NSButtonTypeSwitch, show_floor)
    tile_label = AppKit.NSTextField.labelWithString_("Tile size: working it out")
    tile_label.setFrame_(AppKit.NSMakeRect(14, 130, 200, 16))
    box.addSubview_(tile_label)
    tile_slider = AppKit.NSSlider.sliderWithValue_minValue_maxValue_target_action_(tile_px or 64, 24, 200, controls,
                                                                                   "tile:")
    tile_slider.setFrame_(AppKit.NSMakeRect(12, 108, 206, 20))
    box.addSubview_(tile_slider)

    def sync_tile_slider():
        if state["tile_px"]:
            tile_slider.setDoubleValue_(state["tile_px"])
            tile_label.setStringValue_(f"Tile size {state['tile_px']:.0f} px")
    thr_label = AppKit.NSTextField.labelWithString_(f"Threshold {threshold:.2f}")
    thr_label.setFrame_(AppKit.NSMakeRect(14, 84, 200, 16))
    box.addSubview_(thr_label)
    slider = AppKit.NSSlider.sliderWithValue_minValue_maxValue_target_action_(threshold, 0.1, 0.95, controls,
                                                                              "threshold:")
    slider.setFrame_(AppKit.NSMakeRect(12, 62, 206, 20))
    box.addSubview_(slider)
    button("Re-pick game view", "pick:", 34)
    button("Quit", "quit:", 6)
    panel.orderFrontRegardless()

    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1 / 30, controls, "tick:",
                                                                                     None, True)
    AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, AppKit.NSRunLoopCommonModes)
    signal.signal(signal.SIGINT, lambda *_: controls.quit_(None))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    threading.Thread(target=capturer, daemon=True).start()
    if _on_start is not None:
        _on_start({"app": app, "overlay": overlay, "view": view, "panel": panel, "state": state, "lock": lock})
    try:
        app.run()
    finally:
        state["stop"] = True
