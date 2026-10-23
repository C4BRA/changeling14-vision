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
            a = np.zeros((144, 192, 3), np.uint8)
            a[:, :96] = shade
            Image.fromarray(a).save(frames / f"{i:05d}-5.png")
            ents = [{"uid": k, "prototype": "WallSolid", "name": "wall", "center": [48 + 96 * k, 48],
                     "box": [96 * k, 0, 96 * k + 96, 96], "draw_depth": -10, "rotation": 0, "blocked": False,
                     "alive": False, "humanoid": False, "subject": False} for k in range(2)]
            (frames / f"{i:05d}-5.json").write_text(json.dumps(
                {"picture": f"{i:05d}-5.png", "tick": i, "time": i * 30.0, "subject": 5, "eye_rotation": 0,
                 "zoom": 1, "width": 192, "height": 144, "entities": ents}))
        recs.append(tmp_path / rec)
    s = build_dataset(recs, tmp_path / "out", holdout=["roundB"])
    assert (s["frames"], s["repeats_dropped"], s["test_frames"]) == (4, 2, 2)
    data = json.loads((tmp_path / "out" / "dataset.json").read_text())
    f = data["frames"][0]
    assert (f["width"], f["height"]) == (64, 48)
    lit, dark = f["sprites"]
    assert lit["box"] == [0.0, 0.0, 32.0, 32.0] and lit["visible"]
    assert dark["dark"] and not dark["visible"]
    assert (tmp_path / "out" / "report.html").is_file()
