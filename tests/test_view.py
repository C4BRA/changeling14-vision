import numpy as np

from changeling14_vision.replays import view as V

CONFIG = """
[display]
uiScale = 1.5

[ui]
default_chat_size = "305.2135,-475"
separated_chat_size = "0.60156256,0"

[viewport]
stretch = false
fixed_scale_factor = 5
scale_render = false

[nyanotrasen.ui]
layout = "Default"
"""


def _world(h, w, tile, phase=(0, 0), seed=0):
    """A floor of tile x tile squares (lines on phase), each with its own shade and some texture."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w]
    ty, tx = (ys - phase[1]) // tile, (xs - phase[0]) // tile
    shade = (rng.integers(70, 150, (h // tile + 3, w // tile + 3)))[ty + 1, tx + 1]
    g = shade + rng.integers(-6, 7, (h, w))
    g[(xs - phase[0]) % tile == 0] = 25
    g[(ys - phase[1]) % tile == 0] = 25
    g[(xs - phase[0]) % tile == tile // 2] -= 25         # a fainter seam across each tile's middle
    return np.repeat(np.clip(g, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def test_settings_from_the_client_file_with_fork_tables(tmp_path):
    p = tmp_path / "Space Station 14" / "data" / "client_config.toml"
    p.parent.mkdir(parents=True)
    p.write_text(CONFIG)
    s = V.read_settings(p)
    assert (s.layout, s.stretch, s.fixed_scale, s.scale_render, s.ui_scale) == ("default", False, 5, False, 1.5)
    assert s.chat == (305.2135, -475.0) and abs(s.split - 0.6016) < 1e-3
    assert "nyanotrasen.ui.layout" in s.found
    assert V.config_paths(tmp_path) == []                      # only under Library/Application Support etc.
    lib = tmp_path / "Library/Application Support/Space Station 14/data"
    lib.mkdir(parents=True)
    (lib / "client_config.toml").write_text('[ui]\nlayout = "Separated"\n')
    found = V.config_paths(tmp_path)
    assert found == [lib / "client_config.toml"]
    sep = V.read_settings(found[0])
    assert sep.layout == "separated" and sep.stretch is True and sep.fixed_scale == 2   # upstream defaults
    assert V.read_settings(None) == V.ClientSettings()
    bad = tmp_path / "bad.toml"
    bad.write_text("not [valid")
    assert V.read_settings(bad) == V.ClientSettings()


def test_view_box_follows_the_clients_scaling_rules():
    fixed = V.ClientSettings(stretch=False, fixed_scale=5)
    box, f = V.draw_box(2940, 1626, fixed)                     # his window: the 3360x2400 view is clipped
    assert f == 5 and (box.x, box.y, box.w, box.h) == (-210, -387, 3360, 2400)
    box, f = V.draw_box(4000, 2600, V.ClientSettings(stretch=False, fixed_scale=2))   # letterboxed
    assert (box.x, box.y, box.w, box.h) == (1328, 820, 1344, 960)
    box, f = V.draw_box(1344 + 40, 960 + 10, V.ClientSettings())                       # close to 2x: snaps
    assert f == 2 and box.w == 1344
    assert V.draw_box(3000, 1000, V.ClientSettings())[1] == 2                          # 40 px over 2x: snaps
    box, f = V.draw_box(3000, 1100, V.ClientSettings())                                # stretched, vertical fit
    assert f is None and abs(box.h - 1100) < 1e-6 and abs(box.w - 1540) < 1e-6
    box, f = V.draw_box(3000, 1000, V.ClientSettings(vertical_fit=False, width_tiles=17, snap_margin=0,
                                                     snap_clip=0))
    assert f is None and abs(box.w - 17 * 32 * 1000 / 480) < 1e-6
    assert V.view_width_tiles(V.ClientSettings(stretch=False, width_tiles=30)) == 21


def test_hud_boxes_and_title_bar():
    s = V.read_settings(None)
    s = V.ClientSettings(chat=(305.2135, -475.0))
    hud = V.hud_rects(V.Rect(0, 0, 2940, 1626), s, 1.5)
    chat = [r.scaled(0.5) for r in hud][4]
    # Measured on his screen: the chat box's left edge at 1115 and bottom at 227.5 points.
    assert abs(chat.x - 1115) < 2 and abs(chat.bottom - 227.5) < 2 and abs(chat.right - 1462.5) < 1
    assert len(V.hud_rects(V.Rect(0, 0, 1000, 800), V.ClientSettings(layout="separated"), 1.0)) == 4
    screens = [(0, 0, 1470, 956)]
    assert V.title_bar(0, 33, 1470, 841, screens) == 28
    assert V.title_bar(0, 0, 1470, 956, screens) == 0


def test_grid_lines_seams_and_fixed_crops():
    tile = 40
    img = _world(400, 900, tile, phase=(13, 27))
    cols, rows = V.edge_profiles(img.mean(axis=2))
    assert V.grid_phase(cols, tile) == 13 and V.grid_phase(rows, tile) == 27
    # A seam as strong as the tile lines: the boxes' centres decide.
    img2 = img.copy()
    img2[:, 33::tile] = 25
    cols2, _ = V.edge_profiles(img2.mean(axis=2))
    boxes = [(13 + c * tile + 3, 27 + r * tile + 4, 13 + (c + 1) * tile - 3, 27 + (r + 1) * tile - 2, 0.9)
             for c in range(3, 9) for r in range(2, 5)]
    px, py = V.phase_from_boxes(boxes, tile)
    assert abs(px - 13) < 1 and abs(py - 28) < 1.5
    assert V.grid_phase(cols2, tile, prefer=px) == 13
    assert V.grid_phase(np.zeros(500), tile) is None
    world = V.Rect(0, 0, 900, 400)
    sizes = set()
    for ph in (0, 7, 13, 39.5):
        x, y, w, h, c, r = V.tile_crop(world, tile, ph, ph)
        assert x <= 0 + 1e-6 and y <= 0 and x + w >= 900 and y + h >= 400
        assert abs((x - ph + tile / 2) % tile - tile / 2) <= 0.5
        sizes.add((w, h, c, r))
    assert len(sizes) == 1                                      # same model input whatever the phase
    out = V.cut(img, V.Rect(100, 50, 300, 200), 80, 40, 120, 120)
    assert out.shape == (120, 120, 3) and not out[:10].any() and not out[:, :20].any()
    assert (out[10:, 20:] == img[50:160, 100:200]).all()


def _hud(img, rects, colour=(60, 80, 110)):
    for r in rects:
        x, y, w, h = r.ints()
        img[max(0, y):y + h, max(0, x):x + w] = colour
    return img


def test_tracker_default_layout_with_the_hud_over_the_world():
    s = V.ClientSettings(path="x", stretch=False, fixed_scale=5, ui_scale=1.5, chat=(305.2135, -475.0))
    tr = V.ViewTracker(s, backing=2.0)
    img = _world(813, 1470, 80, phase=(35, 59))
    hud = [r.scaled(0.5) for r in V.hud_rects(V.Rect(0, 0, 2940, 1626), s, 1.5)]
    _hud(img, hud)
    est = tr.estimate(img, [(745, 410)], now=0)
    assert est.world.ints() == (0, 0, 1470, 813) and est.tile == 80
    assert "grid" in est.source and "player" in est.source and est.confidence >= 0.85
    assert not V.keep_detection(1300, 100, est.hud)             # chat
    assert not V.keep_detection(700, 790, est.hud)              # hands
    assert V.keep_detection(735, 406, est.hud)                  # the middle of the world
    # Zoomed in one step: the grid shows 120 px tiles; the view box stays the same.
    zoomed = _world(813, 1470, 120, phase=(5, 5))
    est = tr.estimate(zoomed, [(735, 406)], now=10)
    est = tr.estimate(zoomed, [(735, 406)], now=20)             # a change is confirmed by a second look
    assert abs(est.tile - 120) < 0.5 and abs(est.zoom - 1 / 1.5) < 1e-3 and "zoom" in est.source
    # Every zoom the game allows (zoomed in: bigger tiles; a ghost zoomed out: smaller).
    import cv2
    for z in V.ZOOMS:
        t = 80 / z
        k = 80 / t
        img_z = cv2.resize(_world(int(813 * k), int(1470 * k), 80, phase=(7, 11)), (1470, 813),
                           interpolation=cv2.INTER_AREA) if t < 80 else _world(813, 1470, int(round(t)), (7, 11))
        got, _ = V.tile_from_candidates(img_z.mean(axis=2), [80 / zz for zz in V.ZOOMS])
        assert got is not None and abs(got - t) < 1, (z, got)
    # A grid that matches no zoom the game has is not believed; a dark view can't be checked.
    est2 = V.ViewTracker(s, 2.0).estimate(_world(813, 1470, 25), [], now=0)
    assert est2.tile == 80 and est2.checks["tile_grid"] is None
    est3 = V.ViewTracker(s, 2.0).estimate(np.zeros((813, 1470, 3), np.uint8), [], now=0)
    assert est3.tile == 80 and "no game world" in est3.note and est3.confidence < 0.45


def test_tracker_separated_layout_letterbox_resize_and_holding():
    s = V.ClientSettings(path="x", layout="separated", stretch=False, fixed_scale=1, ui_scale=1.0, split=0.6)
    tr = V.ViewTracker(s, backing=1.0)
    img = np.zeros((600, 1200, 3), np.uint8)
    # The chat panel from 760 (not the 0.6 the setting says): the game gets 760 x 600.
    img[:, 760:] = V.PANEL_BGR
    box, _ = V.draw_box(760, 600, s)                          # 672 x 480 at scale 1: letterboxed
    x, y, w, h = box.ints()
    img[y:y + h, x:x + w] = _world(h, w, 32, phase=(3, 3))
    est = tr.estimate(img, [(x + w / 2, y + h / 2)], now=0)
    assert est.world.ints() == (x, y, w, h) and est.checks["panel_edge"] == 760
    assert est.checks["letterbox_black"] is True and "panel" in est.source
    # The window is resized: taken at once, no confirming.
    img2 = np.zeros((700, 1400, 3), np.uint8)
    img2[:, 840:] = V.PANEL_BGR
    box2, _ = V.draw_box(840, 700, s)
    x2, y2, w2, h2 = box2.ints()
    img2[y2:y2 + h2, x2:x2 + w2] = _world(h2, w2, 32)
    est = tr.estimate(img2, [(x2 + w2 / 2, y2 + h2 / 2)], now=1)
    assert est.world.ints() == (x2, y2, w2, h2)
    # Bars that aren't black and a player far off: too unsure, keep the last good view.
    noisy = img2.copy()
    noisy[:y2] = 200
    noisy[:, :x2] = 200
    held = tr.estimate(noisy, [(10, 10)], now=5)
    assert held.world == est.world and held.note.startswith("holding last good")


def test_still_pixels_find_menus_only_while_the_world_moves():
    sp = V.StillPixels(shrink=4, gap=0.5)
    big = _world(1200, 2000, 40, seed=3)
    menu = V.Rect(20, 20, 200, 60)
    t = 0.0
    for i in range(12):                                         # standing still: nothing counts
        frame = _hud(big[:400, :800].copy(), [menu])
        assert not sp.observe(frame, t)
        t += 0.6
    assert sp.boxes() == []
    for i in range(14):                                         # walking: the world scrolls under the menu
        frame = _hud(big[i * 17:i * 17 + 400, i * 23:i * 23 + 800].copy(), [menu])
        sp.observe(frame, t, keep_out=V.Rect(340, 140, 120, 120))
        t += 0.6
    boxes = sp.boxes()
    assert len(boxes) == 1
    b = boxes[0]
    assert abs(b.x - 20) <= 4 and abs(b.y - 20) <= 4 and abs(b.w - 200) <= 8 and abs(b.h - 60) <= 8
