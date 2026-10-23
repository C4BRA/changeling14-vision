import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from changeling14_vision.cli import main
from changeling14_vision.model.encoder import EMBED_DIM, SIZES, build_encoder, count_parameters
from changeling14_vision.pretrain.data import build_index

# Parameter budgets from the plan: nano ~2-3M, tiny ~5-10M, small ~25M, base ~80M.
BUDGET = {"nano": (1.5e6, 3.5e6), "tiny": (5e6, 10.5e6), "small": (18e6, 30e6), "base": (60e6, 90e6)}


@pytest.mark.parametrize("size", list(SIZES))
def test_sizes_share_outputs(size):
    enc = build_encoder(size).eval()
    lo, hi = BUDGET[size]
    assert lo <= count_parameters(enc) <= hi
    with torch.no_grad():
        tiles, pooled = enc(torch.rand(2, 3, 96, 160))
    assert tiles.shape == (2, 3, 5, EMBED_DIM)       # one token per 32 px square
    assert pooled.shape == (2, EMBED_DIM)


def test_tokens_follow_the_tile_grid():
    """Changing one tile's pixels changes that tile's conv features most: the
    stem and downsamples line up with the 32 px grid."""
    enc = build_encoder("nano").eval()
    x = torch.zeros(1, 3, 96, 96)
    with torch.no_grad():
        base = enc.tokenize(enc.stages(enc.stem(x)))
        x[:, :, 40:56, 72:88] = 1.0                   # inside tile (row 1, col 2)
        moved = enc.tokenize(enc.stages(enc.stem(x)))
    diff = (moved - base).abs().sum(1)[0]
    assert diff.argmax().item() == 1 * 3 + 2


def _fake_dataset(root: Path) -> Path:
    """A tiny M0-shaped catalogue: 21 sheets in 3 categories, 4 directions each."""
    rows = []
    rng = np.random.default_rng(0)
    sheets = [f"Objects/Tools/tool{i}.rsi" for i in range(10)] + \
             [f"Structures/Doors/door{i}.rsi" for i in range(10)] + ["Tiles/floor.rsi"]
    for s in sheets:
        colour = rng.integers(40, 255, size=3)
        for state in ("icon", "open"):
            for d in range(4):
                rel = Path("images") / s[:-4] / state / f"d{d}_f0.png"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                arr = np.zeros((32, 32, 4), np.uint8)
                arr[4 + d * 4:20 + d * 4, 8:24, :3] = colour
                arr[4 + d * 4:20 + d * 4, 8:24, 3] = 255
                Image.fromarray(arr, "RGBA").save(root / rel)
                rows.append({"path": rel.as_posix(), "sheet": s, "state": state, "direction": d,
                             "frame": 0, "width": 32, "height": 32})
    (root / "catalogue").mkdir(parents=True, exist_ok=True)
    with open(root / "catalogue" / "images.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    entities, links = [], []
    for s in sheets[:-1]:
        name = Path(s).stem
        comps = ["Sprite", "Item", "Tool"] if "Tools" in s else ["Sprite", "Door", "Airlock"]
        entities.append({"id": name, "components": comps, "name": name.rstrip("0123456789"),
                         "description": f"A {name} for testing."})
        links += [{"sheet": s, "state": st, "entities": [{"entity": name, "role": "sprite"}]} for st in ("icon", "open")]
    with open(root / "catalogue" / "entities.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in entities)
    with open(root / "catalogue" / "sprite_links.jsonl", "w") as f:
        f.writelines(json.dumps(link) + "\n" for link in links)
    return root


@pytest.fixture(autouse=True)
def small_component_vocab(monkeypatch):
    monkeypatch.setattr("changeling14_vision.pretrain.data.MIN_COMPONENT_STATES", 5)


def test_index_holds_out_whole_sheets(tmp_path):
    index = build_index(_fake_dataset(tmp_path / "ds"), unseen_fraction=0.3, seed=3)
    by_sheet = {}
    for it in index["items"]:
        by_sheet.setdefault(it["sheet"], set()).add(it["split"] == "unseen")
    assert all(len(v) == 1 for v in by_sheet.values())   # a sheet is wholly in or out
    assert index["counts"].get("unseen") and index["counts"].get("newview")
    assert all(it["state_id"] == -1 for it in index["items"] if it["split"] == "unseen")
    # Sprite is on every linked sprite, so it says nothing and is not a label.
    assert index["components"] == ["Airlock", "Door", "Item", "Tool"]
    floor = next(it for it in index["items"] if it["sheet"] == "Tiles/floor.rsi")
    assert floor["components"] is None
    wrench = next(it for it in index["items"] if it["sheet"] == "Objects/Tools/tool0.rsi")
    assert [index["components"][c] for c in wrench["components"]] == ["Item", "Tool"]


def test_names_go_to_whole_items(tmp_path):
    from changeling14_vision.pretrain.data import is_worn
    data = _fake_dataset(tmp_path / "ds")
    look = Path("images/_entities/tool0.png")
    (data / look).parent.mkdir(parents=True)
    Image.new("RGBA", (32, 32), (60, 60, 60, 255)).save(data / look)
    with open(data / "catalogue" / "entity_images.jsonl", "w") as f:
        f.write(json.dumps({"path": look.as_posix(), "composite": True, "sheet": "Objects/Tools/tool0.rsi",
                            "state": "icon", "width": 32, "height": 32, "entities": ["tool0"]}) + "\n")
        f.write(json.dumps({"path": "images/Structures/Doors/door0/icon/d0_f0.png", "composite": False,
                            "sheet": "Structures/Doors/door0.rsi", "state": "icon", "entities": ["door0"]}) + "\n")
    index = build_index(data)
    named = {(it["sheet"], it["state"]): it["entities"] for it in index["items"]}
    # The stacked look carries the name; tool0's lone layers no longer do.
    assert named[("Objects/Tools/tool0.rsi", "@tool0")] == ["tool0"]
    assert named[("Objects/Tools/tool0.rsi", "icon")] == []
    # A one-layer item keeps its name on every direction of that state, not on its other states.
    assert named[("Structures/Doors/door0.rsi", "icon")] == ["door0"]
    assert named[("Structures/Doors/door0.rsi", "open")] == []
    assert is_worn("soles-equipped-FEET") and is_worn("equipped-HELMET") and not is_worn("icon-unshaded")


def test_train_resume_benchmark_export(tmp_path):
    data = _fake_dataset(tmp_path / "ds")
    out, run = tmp_path / "model", tmp_path / "run"
    args = ["--run-dir", str(run), "train", "--data", str(data), "--out", str(out), "--size", "nano",
            "--batch", "8", "--workers", "0", "--checkpoint-every", "4", "--device", "cpu"]
    assert main(args + ["--steps", "6"]) == 0
    status = json.loads((run / "status.json").read_text())
    assert status["stage"] == "done" and status["step"] == 6
    marks = sorted(p.name for p in (run / "watchset").iterdir())
    assert marks == ["step-0000000", "step-0000004", "step-0000006"]
    assert json.loads((run / "watchset" / "step-0000004" / "meta.json").read_text())["step"] == 4
    assert list((run / "watchset" / "step-0000004").glob("*.png"))
    preds = json.loads((run / "watchset" / "step-0000004" / "predictions.json").read_text())
    for row in preds["items"]:
        # Sprites with no entity name fall back to the folder group.
        assert row["kind"] == "name" or row["name"] == row["group"]
        assert (run / "watchset" / "step-0000004" / row["image"]).is_file()
        assert row["rank"] in (None, 1, 2, 3, 4, 5) and 1 <= len(row["guesses"]) <= 5
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 3

    # Resume carries on from latest.pt.
    assert main(args + ["--steps", "8"]) == 0
    assert torch.load(out / "latest.pt", weights_only=False)["step"] == 8
    assert sorted(p.name for p in (out / "checkpoints").iterdir())[-1] == "step-0000008.pt"

    run2 = tmp_path / "run2"
    assert main(["--run-dir", str(run2), "benchmark", "--data", str(data), "--out", str(out),
                 "--size", "nano", "--device", "cpu"]) == 0
    report = json.loads((run2 / "benchmark.json").read_text())
    assert report["step"] == 8 and report["speed"]["ms_median"] > 0
    assert "category_top1" in report["unseen"] and (out / "benchmarks" / "step-0000008.md").exists()
    assert "components_f1" in report["unseen"]

    run3 = tmp_path / "run3"
    assert main(["--run-dir", str(run3), "export", "--out", str(out), "--size", "nano",
                 "--formats", "safetensors,onnx,mlx", "--height", "64", "--width", "96"]) == 0
    result = json.loads((run3 / "export.json").read_text())
    assert result["failed"] == []
    onnxruntime = pytest.importorskip("onnxruntime")
    onnx_file = next((out / "export").glob("*/changeling14-vision-nano-p32.onnx"))
    session = onnxruntime.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    tiles, pooled = session.run(None, {"image": np.random.rand(2, 3, 160, 128).astype(np.float32)})
    assert tiles.shape == (2, 5, 4, EMBED_DIM) and pooled.shape == (2, EMBED_DIM)   # any frame size
    for fmt in ("onnx", "mlx"):
        entry = result["formats"][fmt]
        if "max_diff" in entry:
            assert entry["max_diff"] < 1e-3, (fmt, entry)


def test_model_folder_keeps_its_catalogue(tmp_path, capsys):
    data = _fake_dataset(tmp_path / "ds")
    out = tmp_path / "model"
    common = ["--batch", "8", "--workers", "0", "--checkpoint-every", "100", "--device", "cpu", "--no-text"]
    assert main(["train", "--data", str(data), "--out", str(out), "--size", "nano", "--steps", "2"] + common) == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["data"] == str(data.resolve()) and cfg["size"] == "nano"

    # --out alone continues the model on its own catalogue and size.
    assert main(["train", "--out", str(out), "--steps", "3"] + common) == 0
    assert torch.load(out / "latest.pt", weights_only=False)["step"] == 3

    # Another catalogue would scramble the labels: refused unless the folder starts over.
    other = _fake_dataset(tmp_path / "other")
    capsys.readouterr()
    assert main(["train", "--data", str(other), "--out", str(out), "--steps", "4"] + common) == 2
    assert "trained on the catalogue" in capsys.readouterr().err
    assert main(["train", "--data", str(other), "--out", str(out), "--size", "nano", "--steps", "1",
                 "--fresh"] + common) == 0
    assert json.loads((out / "config.json").read_text())["data"] == str(other.resolve())

    run = tmp_path / "bench"
    assert main(["--run-dir", str(run), "benchmark", "--out", str(out), "--device", "cpu"]) == 0
    assert json.loads((run / "benchmark.json").read_text())["size"] == "nano"


def test_mlx_matches_torch_with_trained_batch_norms():
    mx = pytest.importorskip("mlx.core")
    from changeling14_vision.export import mlx_weights
    from changeling14_vision.model.mlx_encoder import MLXEncoder

    torch.manual_seed(1)
    enc = build_encoder("nano")
    enc.train()
    with torch.no_grad():                   # give the batch norms non-trivial statistics
        for _ in range(3):
            enc(torch.rand(4, 3, 64, 64) * 0.7 + 0.2)
        for m in enc.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.2, 0.2)
    enc.eval()
    x = torch.rand(2, 3, 64, 96)
    with torch.no_grad():
        ref_tiles, ref_global = enc(x)
    mlx_enc = MLXEncoder({k: mx.array(v) for k, v in mlx_weights(enc).items()}, enc.config)
    tiles, pooled = mlx_enc(mx.array(x.numpy().transpose(0, 2, 3, 1)))
    assert np.abs(np.array(tiles) - ref_tiles.numpy()).max() < 1e-3
    assert np.abs(np.array(pooled) - ref_global.numpy()).max() < 1e-3


def test_stop_signal_to_process_group_saves_checkpoint(tmp_path):
    """The app sends SIGTERM to the whole process group, data workers included."""
    import os
    import signal
    import subprocess
    import sys
    import time

    data = _fake_dataset(tmp_path / "ds")
    out, run = tmp_path / "model", tmp_path / "run"
    proc = subprocess.Popen(
        [sys.executable, "-m", "changeling14_vision", "--run-dir", str(run), "train", "--data", str(data),
         "--out", str(out), "--size", "nano", "--batch", "8", "--workers", "2", "--steps", "100000",
         "--checkpoint-every", "100000", "--device", "cpu"],
        start_new_session=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            if json.loads((run / "status.json").read_text()).get("step", 0) >= 3:
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(timeout=30) == 143
    assert json.loads((run / "status.json").read_text())["stage"] == "stopped"
    assert torch.load(out / "latest.pt", weights_only=False)["step"] >= 3


def test_text_matching_trains_and_scores(tmp_path, monkeypatch):
    """With a stand-in sentence model: the text table is built once, the SigLIP
    loss runs, and unseen sprites are scored by naming them from all texts."""
    from changeling14_vision.pretrain import text as text_mod
    from changeling14_vision.pretrain.train import TrainOptions, train
    from changeling14_vision.progress import Progress

    def fake_embedder(model_name, device=None):
        def embed(texts, batch=256):
            rng = [np.random.default_rng(abs(hash(t)) % 2**32).normal(size=16) for t in texts]
            return np.stack(rng)
        return embed

    monkeypatch.setattr(text_mod, "sentence_embedder", fake_embedder)
    data = _fake_dataset(tmp_path / "ds")
    out, run = tmp_path / "model", tmp_path / "run"
    opts = TrainOptions(data=data, out=out, size="nano", steps=4, batch=8, workers=0, device="cpu",
                        checkpoint_every=4, warmup=1, text_model="fake")
    result = train(opts, Progress(run))
    table = text_mod.TextTable(out / "text.npz")
    assert table.dim == 16 and "tool. A tool0 for testing." in table.texts
    assert result["last"]["unseen_text_top5"] is not None
    preds = json.loads((run / "watchset" / "step-0000004" / "predictions.json").read_text())
    named = [r for r in preds["items"] if r["kind"] == "name"]
    assert named and all(r["name"] in r["names"] and ". " not in r["name"] for r in named)
    for r in named:
        guessed = [g["name"].lower() for g in r["guesses"]]
        assert len(guessed) == len(set(guessed))          # one entry per name
        right = {n.lower() for n in r["names"]}
        assert r["rank"] == next((k + 1 for k, g in enumerate(guessed) if g in right), None)
    ckpt = torch.load(out / "latest.pt", weights_only=False)
    assert ckpt["text_dim"] == 16 and ckpt["text_model"] == "fake"
    assert ckpt["ema"].keys() == ckpt["model"].keys() and ckpt["ema_updates"] == ckpt["step"]
    # Resuming without text matching is refused rather than silently mixing setups.
    with pytest.raises(SystemExit):
        train(TrainOptions(data=data, out=out, size="nano", steps=6, batch=8, workers=0, device="cpu",
                           text_model=None), Progress(None))
    from changeling14_vision.pretrain.benchmark import benchmark
    report = benchmark(out, data, Progress(tmp_path / "bench"), device="cpu")
    assert "text_top1" in report["unseen"]


def test_watchset_prefers_things_players_see():
    from changeling14_vision.pretrain.watch import pick_watchset, preferred
    assert preferred("Clothing/Eyes") and preferred("Objects/Tools") and preferred("Mobs/Pets")
    assert not preferred("Mobs/Customization") and not preferred("Structures/Walls")
    assert not preferred("Objects/Tiles") and not preferred("Effects")

    def item(cat, n):
        return {"split": "unseen", "direction": -1, "path": f"{cat}{n}_f0.png", "worn": False,
                "category": cat, "entities": ["e"], "sheet": "s", "state": "x"}
    cats = ["Clothing/Eyes", "Mobs/Customization", "Objects/Tools"]
    index = {"counts": {"unseen": 1}, "categories": cats,
             "items": [item(c, n) for c in range(3) for n in range(5)]}
    picked = pick_watchset(index, 1, n=6, named=True)
    assert {cats[it["category"]] for it in picked} == {"Clothing/Eyes", "Objects/Tools"}
    picked = pick_watchset(index, 1, n=12, named=True)       # runs short: markings fill in
    assert sum(cats[it["category"]] == "Mobs/Customization" for it in picked) == 2


def test_name_loss_counts_every_entity_and_skips_unnamed():
    from changeling14_vision.pretrain.train import PretrainModel, _text_loss
    from changeling14_vision.model.encoder import build_encoder
    torch.manual_seed(0)
    model = PretrainModel(build_encoder("nano"), 3, 4, 2, text_dim=8)
    model.set_text(torch.randn(5, 8), [f"t{i}. x" for i in range(5)])
    g = torch.randn(3, EMBED_DIM, requires_grad=True)
    out = {"global": g, "category": torch.zeros(3, 3)}
    rows = torch.tensor([[0, 3, -1], [1, -1, -1], [-1, -1, -1]])
    b = {"text": torch.tensor([0, 1, -1]), "text_rows": rows}
    loss = _text_loss(model, out, b)
    loss.backward()
    assert torch.isfinite(loss) and g.grad[0].abs().sum() > 0
    assert g.grad[2].abs().sum() == 0            # no name: no pull either way


def test_schedulefree_trains_resumes_and_benchmarks(tmp_path):
    pytest.importorskip("schedulefree")
    data = _fake_dataset(tmp_path / "ds")
    out = tmp_path / "model"
    args = ["train", "--data", str(data), "--out", str(out), "--size", "nano", "--batch", "8",
            "--workers", "0", "--checkpoint-every", "3", "--device", "cpu", "--no-text",
            "--optimizer", "schedulefree"]
    assert main(args + ["--steps", "3"]) == 0
    assert main(args + ["--steps", "5"]) == 0
    ckpt = torch.load(out / "latest.pt", weights_only=False)
    assert ckpt["step"] == 5 and ckpt["n_states"] == 0
    row = json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])
    assert "unseen_tile_category_top1" in row
    run = tmp_path / "run"
    assert main(["--run-dir", str(run), "benchmark", "--data", str(data), "--out", str(out),
                 "--size", "nano", "--device", "cpu"]) == 0
    # A different optimiser can't continue this model.
    with pytest.raises(SystemExit, match="schedulefree"):
        main(args[:-1] + ["adamw", "--steps", "6"])


def test_rich_text_uses_only_catalogue_facts(tmp_path):
    from changeling14_vision.pretrain import text as text_mod
    bucket = {"id": "Bucket", "name": "bucket", "description": "It's a boring old bucket.", "suffix": "Full",
              "source": "Entities/Objects/Tools/bucket.yml",
              "components": ["Item", "Clothing", "Edible", "Storage", "SomethingNew"]}
    assert text_mod.entity_phrases(bucket) == [
        "bucket. It's a boring old bucket.", "bucket, full", "bucket, objects, tools",
        "bucket: holds items inside, an item you can pick up"]
    assert text_mod.entity_phrases({"id": "X", "name": "", "components": ["Item"]}) == []
    assert text_mod._words("OuterClothing") == "outer clothing"

    root = tmp_path / "ds" / "catalogue"
    root.mkdir(parents=True)
    plain = dict(bucket, id="Bucket2", suffix=None, components=[])
    (root / "entities.jsonl").write_text(json.dumps(bucket) + "\n" + json.dumps(plain) + "\n")
    seen = []

    def embed(texts, batch=256):
        seen.extend(texts)
        return np.stack([np.random.default_rng(len(t)).normal(size=8) for t in texts])

    path = text_mod.build_text_table(tmp_path / "ds", tmp_path / "m", embed=embed, model_name="fake", rich=True)
    table = text_mod.TextTable(path)
    # Both share the shown text, so they share one row; every phrase is embedded once.
    assert table.texts == ["bucket. It's a boring old bucket."] and table.rich
    assert table.rows_for(["Bucket", "Bucket2"]) == [0] and len(seen) == len(set(seen)) == 4
    assert abs(np.linalg.norm(table.vectors[0]) - 1) < 1e-2


def test_continued_model_keeps_its_training_settings(tmp_path, monkeypatch):
    """Continuing with only --out (or the app's --steps) keeps the folder's batch,
    optimiser, learning rate and text settings; anything given explicitly wins,
    and a new folder gets the defaults."""
    import changeling14_vision.pretrain.train as train_mod
    seen = {}

    def fake_train(opts, progress=None):
        seen["opts"] = opts
        return {"complete": True}

    monkeypatch.setattr(train_mod, "train", fake_train)
    data = _fake_dataset(tmp_path / "ds")
    model = tmp_path / "model"
    model.mkdir()
    (model / "latest.pt").write_bytes(b"")
    (model / "config.json").write_text(json.dumps({
        "data": str(data.resolve()), "size": "nano", "batch": 512, "optimizer": "schedulefree", "lr": 0.004,
        "checkpoint_every": 250, "rich_text": True, "amp": True}))

    assert main(["train", "--out", str(model), "--steps", "100"]) == 0
    o = seen["opts"]
    assert (o.batch, o.optimizer, o.lr, o.checkpoint_every, o.rich_text, o.amp, o.size) == (
        512, "schedulefree", 0.004, 250, True, True, "nano")

    assert main(["train", "--out", str(model), "--batch", "256", "--no-amp"]) == 0
    assert (seen["opts"].batch, seen["opts"].amp, seen["opts"].optimizer) == (256, False, "schedulefree")

    assert main(["train", "--out", str(tmp_path / "new"), "--data", str(data), "--size", "nano"]) == 0
    o = seen["opts"]
    assert (o.batch, o.optimizer, o.lr, o.rich_text, o.amp) == (128, "adamw", 1e-3, False, False)


def test_dedup_counts_identical_pictures_once(tmp_path):
    from changeling14_vision.pretrain.data import SpriteDataset
    from changeling14_vision.pretrain.duplicates import write_report
    data = _fake_dataset(tmp_path / "ds")
    plain = build_index(data, unseen_fraction=0.3, seed=3)
    # Copy a training door's picture into an unseen tool sheet, so a group spans both.
    unseen = next(it for it in plain["items"] if it["split"] == "unseen" and "Tools" in it["sheet"])
    trained = next(it for it in plain["items"] if it["split"] == "train" and "Doors" in it["sheet"])
    (data / unseen["path"]).write_bytes((data / trained["path"]).read_bytes())
    index = build_index(data, unseen_fraction=0.3, seed=3, dedup=True)
    assert index["dedup"] and not plain.get("dedup")
    groups = {}
    for it in index["items"]:
        groups.setdefault(it["pixels"], []).append(it)
    for g in groups.values():
        # One label set and one picture's weight per group, per animation.
        assert len({it["category"] for it in g}) == 1 and len({tuple(it["entities"]) for it in g}) == 1
        assert abs(sum(it["weight"] for it in g) - 1) < 1e-9
        assert all(it["split"] != "newview" for it in g) or len(g) == 1
        assert len({it["split"] == "unseen" for it in g}) == 1      # never on both sides of the test
    moved = next(it for it in index["items"] if it["path"] == unseen["path"])
    assert moved["split"] == "train" and moved.get("moved")
    # The door's copy in the tool sheet: the folder most copies share, and both names.
    assert index["categories"][moved["category"]] == "Structures/Doors"
    assert set(moved["entities"]) == set(trained["entities"]) | set(unseen["entities"])
    ds = SpriteDataset(data, index, "train", train=True)
    assert len(ds.weights()) == len(ds) and min(ds.weights()) > 0
    write_report(index, data, tmp_path / "duplicates.html")
    page = (tmp_path / "duplicates.html").read_text()
    assert "Biggest groups" in page and "moved into training" in page and "data:image/png" in page
