"""Time the parts of a replay-training step on a real dataset.

  python tools/time_detect_step.py <dataset> <stage-1 model folder> [steps]

Prints the average milliseconds of each part (the device is synchronised
between parts, so each number is that part's own time).
"""

import sys
import time
from pathlib import Path

import torch

from changeling14_vision.pretrain.text import TextTable
from changeling14_vision.pretrain.train import WeightAverage, pick_device
from changeling14_vision.replays import detect
from changeling14_vision.replays.floor import Floor


def main():
    dataset, start = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    dev = pick_device("auto")
    floor = Floor.load(dataset)
    model, table, _ = detect.from_stage1(start, detect.floor_vocab(floor))
    model.to(dev)
    model.set_text(table)
    ds = detect.Frames(dataset, "train", table, detect.CROP, augment=True, floor=floor, vocab=model.vocab)
    trained = torch.zeros(len(table.texts), dtype=torch.bool)
    trained[torch.from_numpy(ds.trained_rows())] = True
    trained = trained.to(dev)
    loader = torch.utils.data.DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, drop_last=True, num_workers=0,
                                         collate_fn=detect.collate)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    average = WeightAverage(model, 0.998)
    detect._train_mode(model)
    sync = (torch.mps.synchronize if dev.type == "mps" else torch.cuda.synchronize if dev.type == "cuda"
            else (lambda: None))
    times: dict[str, float] = {}

    def mark(name, t0):
        sync()
        t = time.time()
        times[name] = times.get(name, 0.0) + (t - t0) * 1000
        return t

    n = 0
    it = iter(loader)
    while n < steps + 3:
        t = time.time()
        b = next(it, None)
        if b is None:
            it = iter(loader)
            continue
        t = mark("load", t)
        b = {k: v.to(dev) for k, v in b.items()}
        t = mark("to device", t)
        out = model(b["image"])
        t = mark("forward", t)
        ls = detect.losses(model, out, b, trained)
        t = mark("losses", t)
        optim.zero_grad(set_to_none=True)
        ls["total"].backward()
        t = mark("backward", t)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        t = mark("optimiser", t)
        average.update(model)
        t = mark("weight average", t)
        n += 1
        if n == 3:                  # warm-up steps don't count
            times.clear()
    total = sum(times.values())
    for k, v in times.items():
        print(f"{k:15s} {v / steps:8.1f} ms")
    print(f"{'total':15s} {total / steps:8.1f} ms on {dev}")

    # The losses on their own, part by part.
    with torch.no_grad():
        b = {k: v.to(dev) for k, v in next(iter(loader)).items()}
    out = model(b["image"])
    for _ in range(2):
        sync()
        t0 = time.time()
        detect.losses(model, out, b, trained)
        sync()
    print(f"losses alone (forward only) {(time.time() - t0) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
