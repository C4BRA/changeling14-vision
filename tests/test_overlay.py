import numpy as np

from changeling14_vision.replays import overlay as ov


def _w(wid, owner, name, w, h, layer=0, x=0.0, y=0.0):
    return {"kCGWindowNumber": wid, "kCGWindowOwnerName": owner, "kCGWindowName": name, "kCGWindowLayer": layer,
            "kCGWindowIsOnscreen": True, "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}}


def test_window_matching_picks_the_largest_client_not_the_launcher():
    info = [_w(1, "Space Station 14 Launcher", "", 800, 528),
            _w(2, "Space Station 14", "", 1470, 33),                 # menu-bar strip
            _w(3, "Space Station 14", "[EN] Wizard's Den - Space Station 14", 734, 841, x=736, y=33),
            _w(4, "Space Station 14", "", 1470, 838, layer=3),      # not a normal window
            _w(5, "Terminal", "Space Station 14 notes", 300, 300),  # title match, smaller
            _w(6, "dotnet", "Content.Client", 1200, 800)]
    wins = ov.windows_from_info(info)
    assert ov.match_window(wins).wid == 6                            # largest normal-layer match
    assert ov.match_window(wins, ("Space Station 14",)).wid == 3
    assert ov.match_window(wins, ("nothing",)) is None


def test_area_fractions_round_trip_and_follow_resizes(tmp_path):
    win = ov.Window(1, "Space Station 14", "", 100.0, 50.0, 1000.0, 800.0, True)
    a = ov.Area.from_drag(win.w, win.h, 900, 700, 100, 100)          # any corner order
    assert (round(a.fx, 3), round(a.fy, 3), round(a.fw, 3), round(a.fh, 3)) == (0.1, 0.125, 0.8, 0.75)
    store = tmp_path / "overlay.json"
    ov.save_area(a, store)
    assert ov.load_area(store) == a
    assert ov.load_area(tmp_path / "missing.json") is None
    assert a.points(win) == (200.0, 150.0, 800.0, 600.0)
    bigger = ov.Window(1, "", "", 0.0, 0.0, 2000.0, 1600.0, True)    # moved and resized: same fractions
    assert a.points(bigger) == (200.0, 200.0, 1600.0, 1200.0)
    assert a.pixels(2000, 1600) == (200, 200, 1800, 1400)            # Retina capture of the 1000x800 window
    assert ov.Area(-1, 2, 5, 5).clamp().fx == 0.0
    assert ov.default_area(win).fy == ov.TITLE_BAR / 800.0


def test_model_boxes_map_to_points_and_cocoa_frames():
    # 32 px model tiles, 96 captured px per tile, 2 px per point: one tile = 48 points.
    assert ov.model_to_view((32, 64, 64, 96), 96, 2.0) == (48.0, 96.0, 96.0, 144.0)
    assert ov.cocoa_frame(10, 20, 300, 200, 1000) == (10, 780, 300, 200)
    assert ov.rgb((255, 255, 0)) == (0.0, 1.0, 1.0)                  # OpenCV BGR cyan -> RGB


def test_tile_size_from_the_floor_grid():
    assert ov.tile_from_view(2016, 1440) == 96 and ov.tile_from_view(2940, 1620) == 108
    # A floor of 80 px tiles with a fainter seam across each tile's middle.
    g = np.full((600, 1200), 100, np.uint8)
    g[:, ::80] = 30
    g[:, 40::80] = 70
    g[::80, :] = 30
    assert ov.tile_from_grid(g) == 80
    assert ov.tile_from_grid(np.zeros((600, 1200), np.uint8)) is None


def test_tile_store_keeps_the_area_and_dark_views_give_no_grid(tmp_path):
    store = tmp_path / "overlay.json"
    ov.save_area(ov.Area(0.1, 0.2, 0.5, 0.6), store)
    ov.save_tile_points(80, store)
    assert ov.load_area(store) == ov.Area(0.1, 0.2, 0.5, 0.6) and ov.load_tile_points(store) == 80
    ov.save_area(ov.Area(0, 0, 1, 1), store)                  # saving the area keeps the tile size
    assert ov.load_tile_points(store) == 80
    dark = np.random.default_rng(0).integers(0, 8, (600, 900)).astype(np.uint8)
    assert ov.tile_from_grid(dark) is None
    assert ov.status_line({"fps": 12.5}).startswith('STATUS {"fps":12.5')
