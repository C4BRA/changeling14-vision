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
    return root


def test_index_holds_out_whole_sheets(tmp_path):
    index = build_index(_fake_dataset(tmp_path / "ds"), unseen_fraction=0.3, seed=3)
    by_sheet = {}
    for it in index["items"]:
        by_sheet.setdefault(it["sheet"], set()).add(it["split"] == "unseen")
    assert all(len(v) == 1 for v in by_sheet.values())   # a sheet is wholly in or out
    assert index["counts"].get("unseen") and index["counts"].get("newview")
    assert all(it["state_id"] == -1 for it in index["items"] if it["split"] == "unseen")


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
