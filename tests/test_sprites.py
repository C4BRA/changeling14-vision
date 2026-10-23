import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from changeling14_vision.progress import Progress
from changeling14_vision.sprites import licenses
from changeling14_vision.sprites.build import BuildError, Options, build

PROTOTYPES = """
- type: entity
  id: BaseItem
  abstract: true
  name: item
  components:
  - type: Item
  - type: Sprite
    state: icon

- type: entity
  parent: BaseItem
  id: Wrench
  name: wrench
  description: A wrench.
  components:
  - type: Sprite
    sprite: Objects/Tools/wrench.rsi

- type: entity
  parent: BaseItem
  id: FancyWrench
  components:
  - type: Sprite
    sprite: Objects/Tools/wrench.rsi
    layers:
    - state: on
    - map: [ "enum.SomeVisuals.Layer" ]

- type: entity
  parent: BaseItem
  id: DarkWrench
  name: dark wrench
  components:
  - type: Sprite
    sprite: Objects/Tools/wrench.rsi
    layers:
    - state: icon
      color: "#808080"
    - state: spin
      visible: false

- type: entity
  id: NcThing
  name: nc thing
  components:
  - type: Sprite
    sprite: Objects/Fun/nc.rsi
    state: icon

- type: entity
  parent: !type:CreateVariants
    values: [ Wrench, Wrench ]
  id: !type:CreateVariants
    values: [ PipeA, PipeB ]
  components:
  - type: Sprite
    layers:
    - sprite: !type:CreateVariants
        values: [ Objects/Tools/wrench.rsi, Objects/Tools/missing.rsi ]
      state: icon
"""


def make_sheet(root: Path, rel: str, license_id: str, states: list[dict], size=32):
    sheet = root / "Resources" / "Textures" / rel
    sheet.mkdir(parents=True)
    (sheet / "meta.json").write_text("﻿" + json.dumps(
        {"version": 1, "license": license_id, "copyright": "by test", "size": {"x": size, "y": size}, "states": states}))
    for st in states:
        cells = sum(len(row) for row in st.get("delays", [])) or st.get("directions", 1)
        img = Image.new("RGBA", (size * min(cells, 4), size * ((cells + 3) // 4)), (0, 0, 0, 0))
        for n in range(cells):
            r, c = divmod(n, 4)
            img.paste((10 * n, 100, 200, 255), (c * size, r * size, (c + 1) * size, (r + 1) * size))
        img.save(sheet / f"{st['name']}.png")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "ss14"
    make_sheet(root, "Objects/Tools/wrench.rsi", "CC-BY-SA-3.0", [
        {"name": "icon"},
        {"name": "on", "directions": 4},
        # Directions with different frame counts.
        {"name": "spin", "directions": 4, "delays": [[0.1, 0.1, 0.1], [0.1], [0.1], [0.1]]},
    ])
    make_sheet(root, "Objects/Fun/nc.rsi", "CC-BY-NC-SA-3.0", [{"name": "icon"}])
    protos = root / "Resources" / "Prototypes" / "Entities"
    protos.mkdir(parents=True)
    (protos / "test.yml").write_text(PROTOTYPES)
    return root


def entities_by_id(out: Path) -> dict:
    rows = [json.loads(line) for line in (out / "catalogue" / "entities.jsonl").read_text().splitlines()]
    return {row["id"]: row for row in rows}


def test_build_catalogue(repo, tmp_path):
    out = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    report = build(Options(repo=repo, out=out), Progress(run_dir))

    assert report["complete"]
    assert report["sheets"] == {**report["sheets"], "total": 2, "included": 1, "excluded": 1}
    # icon 1 + on 4 + spin 3+1+1+1
    assert report["images"]["cut"] == 11
    assert (out / "images/Objects/Tools/wrench/spin/d0_f2.png").exists()
    assert (out / "images/Objects/Tools/wrench/spin/d3_f0.png").exists()
    assert not (out / "images/Objects/Fun").exists()

    ents = entities_by_id(out)
    assert ents["Wrench"]["name"] == "wrench"
    assert ents["Wrench"]["chain"] == ["BaseItem"]
    sprites = {(s["role"], s.get("state"), s["status"]) for s in ents["Wrench"]["sprites"]}
    assert ("sprite", "icon", "clean") in sprites
    # Item has no sprite of its own: falls back to the Sprite sheet.
    assert ("item", None, "clean") in sprites

    # `on` stays a state name, not a boolean; the inherited `state: icon` is ignored with layers.
    fancy = [(s["role"], s.get("state"), s["status"]) for s in ents["FancyWrench"]["sprites"]]
    assert ("layer", "on", "clean") in fancy
    assert ("layer", None, "runtime_state") in fancy
    assert not any(role == "sprite" for role, _, _ in fancy)

    assert ents["NcThing"]["sprites"][0]["status"] == "excluded_license"
    assert ents["PipeA"]["sprites"][0]["status"] == "clean"
    assert ents["PipeB"]["sprites"][0]["status"] == "missing_sheet"

    links = [json.loads(line) for line in (out / "catalogue" / "sprite_links.jsonl").read_text().splitlines()]
    icon = next(link for link in links if link["state"] == "icon")
    assert {"entity": "Wrench", "role": "sprite"} in icon["entities"]

    credits = list(csv.reader(open(out / "credits.csv")))
    assert credits[1][:2] == ["Objects/Tools/wrench.rsi", "CC-BY-SA-3.0"]
    excluded = list(csv.reader(open(out / "excluded.csv")))
    assert excluded[1][2] == "non-commercial license"

    # Whole-item looks: one untinted layer reuses the cut image; tinted layers are stacked.
    looks = [json.loads(line) for line in (out / "catalogue" / "entity_images.jsonl").read_text().splitlines()]
    plain = next(look for look in looks if "Wrench" in look["entities"])
    assert not plain["composite"] and plain["path"] == "images/Objects/Tools/wrench/icon/d0_f0.png"
    assert "PipeA" in plain["entities"]
    dark = next(look for look in looks if look["entities"] == ["DarkWrench"])
    assert dark["composite"] and dark["tinted"] and dark["layers"] == ["Objects/Tools/wrench.rsi|icon"]
    # The hidden spin layer is left out, and the tint multiplies the icon's (0, 100, 200).
    assert Image.open(out / dark["path"]).getpixel((5, 5)) == (0, 50, 100, 255)
    assert not any("NcThing" in look["entities"] for look in looks)

    manifest = json.loads((out / "manifest.json").read_text())
    assert "catalogue/entities.jsonl" in manifest["files"]

    status = json.loads((run_dir / "status.json").read_text())
    assert status["stage"] == "done"
    assert list((run_dir / "watchset" / "sample").glob("*.png"))


def test_rebuild_replaces_own_outputs(repo, tmp_path):
    out = tmp_path / "dataset"
    build(Options(repo=repo, out=out))
    (out / "images" / "stale.png").write_bytes(b"x")
    build(Options(repo=repo, out=out))
    assert not (out / "images" / "stale.png").exists()


def test_refuses_unsafe_output(repo, tmp_path):
    with pytest.raises(BuildError, match="inside the source"):
        build(Options(repo=repo, out=repo / "out"))
    other = tmp_path / "busy"
    other.mkdir()
    (other / "keep.txt").write_text("mine")
    with pytest.raises(BuildError, match="not empty"):
        build(Options(repo=repo, out=other))
    assert (other / "keep.txt").exists()


def test_stop_request_leaves_partial_report(repo, tmp_path):
    progress = Progress(None)
    progress.stop_requested = True
    report = build(Options(repo=repo, out=tmp_path / "d"), progress)
    assert not report["complete"]


@pytest.mark.parametrize("value,reason", [
    ("CC-BY-SA-3.0", None),
    ("CC0-1.0", None),
    ("cc0", None),
    ("CC-BY-NC-SA-3.0", "non-commercial license"),
    ("Custom", "custom license"),
    (None, "missing license"),
    ("GPL", "unrecognised license 'GPL'"),
])
def test_license_policy(value, reason):
    assert licenses.exclusion_reason(value) == reason


def test_replay_import_files_by_build_and_keeps_the_inbox(tmp_path):
    import zipfile
    from changeling14_vision.cli import main
    from changeling14_vision.replays.importer import load_library
    inbox, library = tmp_path / "inbox", tmp_path / "library"
    inbox.mkdir()

    def replay(name, final=True, **meta):
        with zipfile.ZipFile(inbox / name, "w") as z:
            z.writestr("_replay/replay.yml", "\n".join(f"{k}: {v}" for k, v in meta.items()))
            if final:
                z.writestr("_replay/replay_final.yml", "duration: 3600.5\nendTick: 108000\n")
            z.writestr("_replay/data_0.dat", b"x" * 10)

    meta = dict(engineVersion="267.1.0", buildForkId="wizards", buildForkVersion="c1384c6f0123456789abcdef")
    replay("round1.zip", **meta)
    replay("round2.zip", **{**meta, "buildForkVersion": "deadbeef0123456789"})
    replay("broken.zip", final=False, **meta)
    (inbox / "notes.zip").write_bytes(b"not a zip")
    assert main(["replays", "import", "--inbox", str(inbox), "--library", str(library)]) == 0
    rows = load_library(library)
    assert sorted(r["build"] for r in rows) == ["wizards-c1384c6f0123", "wizards-deadbeef0123"]
    assert (library / rows[0]["file"]).is_file() and rows[0]["endTick"] == 108000
    assert sorted(p.name for p in inbox.iterdir()) == ["broken.zip", "notes.zip", "round1.zip", "round2.zip"]
    # Importing again adds nothing.
    assert main(["replays", "import", "--inbox", str(inbox), "--library", str(library)]) == 0
    assert len(load_library(library)) == 2


def test_replay_dataset_shrinks_marks_dark_and_holds_out_rounds(tmp_path):
    import numpy as np
    from PIL import Image

    from changeling14_vision.replays.dataset import build_dataset

    recs = []
    for rec in ("roundA", "roundB"):
        frames = tmp_path / rec / "frames"
        frames.mkdir(parents=True)
        for i, shade in enumerate((200, 200, 240)):        # the second picture repeats the first
            a = np.zeros((960, 1344, 3), np.uint8)
            a[:, :672] = shade
            Image.fromarray(a).save(frames / f"{i:05d}-5.png")
            ents = [{"uid": k, "prototype": "WallSolid", "name": "wall", "center": [32 + 700 * k, 32],
                     "box": [700 * k, 0, 700 * k + 64, 64], "draw_depth": -2, "rotation": 0, "blocked": False,
                     "alive": False, "humanoid": False, "subject": False} for k in range(2)]
            (frames / f"{i:05d}-5.json").write_text(json.dumps(
                {"picture": f"{i:05d}-5.png", "tick": i, "time": i * 30.0, "subject": 5, "eye_rotation": 0,
                 "zoom": 1, "width": 1344, "height": 960, "entities": ents}))
        recs.append(tmp_path / rec)
    from changeling14_vision.replays.labels import load

    s = build_dataset(recs, tmp_path / "out", holdout=["roundB"])
    assert (s["frames"], s["repeats_dropped"], s["test_frames"]) == (4, 2, 2)
    data, labels = load(tmp_path / "out")
    assert data["version"] == 2 and "sprites" not in data["frames"][0]
    f = data["frames"][0]
    assert (f["width"], f["height"]) == (672, 480)
    lit, dark = labels.dicts(f)
    assert lit["box"] == [0.0, 0.0, 32.0, 32.0] and lit["visible"] and lit["prototype"] == "WallSolid"
    assert dark["dark"] and not dark["visible"]
    assert not lit["faint"]
    assert (tmp_path / "out" / "report.html").is_file()

    # A rebuild is made aside and swapped in; the old one is kept once.
    build_dataset(recs[:1], tmp_path / "out", holdout=[])
    assert load(tmp_path / "out")[0]["summary"]["frames"] == 2
    assert load(tmp_path / ".out.previous")[0]["summary"]["frames"] == 4
    assert not (tmp_path / ".out.building").exists()

    class Stop:
        stop_requested = False

        def update(self, *a, **k):
            self.stop_requested = True
    assert build_dataset(recs, tmp_path / "out", holdout=[], progress=Stop())["stopped"]
    assert load(tmp_path / "out")[0]["summary"]["frames"] == 2           # left as it was


def test_faint_floor_sprites():
    import numpy as np

    from changeling14_vision.replays.dataset import _looks_faint

    floor = np.full((96, 96, 3), 0.4, np.float32)
    assert _looks_faint(floor, 32, 32, 64, 64)                     # nothing there
    pipe = floor.copy()
    pipe[47:49, 32:64] = 0.9                                       # a thin pipe: low average, sharp line
    assert not _looks_faint(pipe, 32, 32, 64, 64)
    splat = floor.copy()
    splat[40:56, 40:56] = 0.9
    assert not _looks_faint(splat, 32, 32, 64, 64)
    # Carpet in a carpeted room: the ring is all carpet, so it can't be judged faint.
    carpets = [(x, y, x + 32, y + 32) for x in (0, 32, 64) for y in (0, 32, 64)]
    assert not _looks_faint(floor, 32, 32, 64, 64, carpets)


def test_detect_trains_from_a_stage1_model(tmp_path):
    import numpy as np
    import torch
    from PIL import Image

    from changeling14_vision.pretrain.train import build_model
    from changeling14_vision.replays.detect import DetectOptions, train_detect

    start = tmp_path / "stage1"
    start.mkdir()
    texts = ["wall. a wall", "light. a lamp", "airlock"]
    vec = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    np.savez(start / "text.npz", texts=np.array(texts), vectors=vec.astype(np.float16),
             entities=np.array(["WallSolid", "Poweredlight", "Airlock"]),
             entity_text=np.array([0, 1, 2], dtype=np.int32), model=np.array("test"), rich=np.array(False))
    m = build_model("nano", {"categories": ["a", "b"]}, text_dim=8)
    torch.save({"model": m.state_dict(), "encoder_config": m.encoder.config, "text_dim": 8, "step": 5},
               start / "latest.pt")

    ds = tmp_path / "ds"
    (ds / "frames").mkdir(parents=True)
    frames = []
    for i, split in enumerate(("train", "train", "test")):
        size = (480, 480) if split == "train" else (96, 128)       # held-out frames may differ in size
        Image.fromarray(np.full((*size, 3), 40 * i + 50, np.uint8)).save(ds / "frames" / f"{i}.png")
        sprites = [{"prototype": "WallSolid", "name": "wall", "box": [0, 0, 32, 32], "draw_depth": 0,
                    "blocked": False, "dark": False, "visible": True, "humanoid": False, "alive": False,
                    "subject": False},
                   {"prototype": "Poweredlight", "name": "light", "box": [40, 40, 72, 72], "draw_depth": 2,
                    "blocked": True, "dark": False, "visible": False, "humanoid": False, "alive": False,
                    "subject": False}]
        frames.append({"picture": f"frames/{i}.png", "split": split, "round": str(i), "width": size[1],
                       "height": size[0], "sprites": sprites})
    (ds / "dataset.json").write_text(json.dumps({"version": 1, "frames": frames}))

    opts = DetectOptions(dataset=ds, start=start, out=tmp_path / "det", steps=3, batch=2, workers=0,
                         device="cpu", checkpoint_every=2, warmup=1, keep_checkpoints=2)
    row = train_detect(opts)
    assert row["step"] == 3 and "name_top1" in row and row["named_sprites"] == 1
    assert row["seen_sprites"] == 1 and row["unseen_sprites"] == 0 and row["sprites_walls"] == 0
    assert "name_set_precision" in row and "top5_followed_player" in row
    # Two newest kept (step 2, 3) plus the best; step 0 goes unless it was the best.
    kept = {p.name for p in (tmp_path / "det" / "checkpoints").glob("*.pt")}
    assert {"step-2.pt", "step-3.pt"} <= kept and len(kept) <= 3
    ckpt = torch.load(tmp_path / "det" / "latest.pt", weights_only=False)
    assert "ema" in ckpt and ckpt["dataset_hash"]
    # Resuming continues from the saved step.
    opts.steps = 4
    assert train_detect(opts)["step"] == 4
    # ...but not on a rebuilt dataset, unless asked to.
    (ds / "dataset.json").write_text(json.dumps({"version": 1, "frames": frames, "note": "rebuilt"}))
    opts.steps = 5
    with pytest.raises(SystemExit, match="another version"):
        train_detect(opts)
    opts.allow_new_data = True
    assert train_detect(opts)["step"] == 5


def test_detect_frames_targets(tmp_path):
    import numpy as np
    from PIL import Image

    from changeling14_vision.replays.detect import Frames

    class Table:
        row_of_entity = {"WallSolid": 0, "Poweredlight": 1, "Airlock": 2}
        texts = ["wall", "light", "airlock"]

    (tmp_path / "frames").mkdir()
    Image.fromarray(np.full((480, 672, 3), 120, np.uint8)).save(tmp_path / "frames" / "0.png")
    base = {"draw_depth": 0, "blocked": False, "dark": False, "humanoid": False, "alive": False, "subject": False}
    sprites = [{**base, "prototype": "WallSolid", "box": [0, 0, 32, 32], "draw_depth": -8, "visible": True},
               {**base, "prototype": "Poweredlight", "box": [64, 64, 96, 96], "draw_depth": 2, "visible": False,
                "dark": True},
               {**base, "prototype": "MobHuman", "box": [320, 224, 352, 256], "draw_depth": 6, "visible": True,
                "humanoid": True, "subject": True}]
    (tmp_path / "dataset.json").write_text(json.dumps({"version": 1, "frames": [
        {"picture": "frames/0.png", "split": "test", "width": 672, "height": 480, "sprites": sprites}]}))
    ds = Frames(tmp_path, "test", Table, None)
    it = ds[0]
    gw = 672 // 32
    assert it["pos"].tolist() == [[0, 0]] and it["ignore"].tolist() == [[2 * gw + 2, 1]]
    assert it["pos_group"].tolist() == [1]                       # walls
    assert it["mask"][2 * gw + 2] == 0 and it["mask"][0] == 1    # only a hidden light there
    assert it["subject"][7 * gw + 10] == 1 and it["player"][7 * gw + 10] == 1
    assert ds.trained_rows().tolist() == [0]
    # Training augments: crops stay 480 x 480 and lighting changes.
    aug = Frames(tmp_path, "test", Table, 480, augment=True)
    shapes = {tuple(aug[0]["image"].shape) for _ in range(10)}
    assert shapes == {(3, 480, 480)}


def test_predict_report_from_a_replay_model(tmp_path):
    from changeling14_vision.replays.predict import predict_report

    test_detect_trains_from_a_stage1_model(tmp_path)
    page = predict_report(tmp_path / "det", tmp_path / "ds", device="cpu", threshold=0.0)
    text = page.read_text()
    assert "held-out frames" in text and "<figure>" in text


def test_prune_checkpoints_keeps_newest_and_best(tmp_path):
    from changeling14_vision.pretrain.train import prune_checkpoints

    for step in (0, 100, 200, 300, 400):
        (tmp_path / f"step-{step:07d}.pt").write_bytes(b"x")
    rows = [{"step": s, "score": v} for s, v in ((0, 0.1), (100, 0.9), (200, 0.2), (300, 0.3), (400, None))]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    prune_checkpoints(tmp_path, 2, "score", tmp_path / "metrics.jsonl")
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["step-0000100.pt", "step-0000300.pt", "step-0000400.pt"]
    prune_checkpoints(tmp_path, 0, "score", tmp_path / "metrics.jsonl")      # 0 keeps all
    assert len(list(tmp_path.glob("*.pt"))) == 3
