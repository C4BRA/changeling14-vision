"""Where does a training step spend its time on this machine?

    python scripts/profile_training.py --data ~/Desktop/SS13-14/ss14-sprite-dataset

Prints one line per measurement; nothing is written. Takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from changeling14_vision.pretrain.data import SpriteDataset, build_index
from changeling14_vision.pretrain.train import _loader, build_model, pick_device


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev, runs=10, warm=3):
    for _ in range(warm):
        fn()
    sync(dev)
    t = time.perf_counter()
    for _ in range(runs):
        fn()
    sync(dev)
    return (time.perf_counter() - t) / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", default="tiny")
    args = ap.parse_args()
    dev = pick_device()
    print(f"machine {platform.platform()} {platform.processor()} torch {torch.__version__} device {dev}")
    data = Path(args.data).expanduser()
    index = build_index(data)
    n_states = len(index["states"])
    print(f"index: {index['counts']}, {n_states:,} states")

    def step_fn(model, opt, x, y, amp=False, heads=True):
        def run():
            ctx = torch.autocast(dev.type, dtype=torch.float16) if amp else torch.autocast(dev.type, enabled=False)
            with ctx:
                out = model(x) if heads else {"global": model.encoder(x)[1]}
            loss = (F.cross_entropy(out["category"].float(), y) + F.cross_entropy(out["state"].float(), y)
                    if heads else out["global"].float().pow(2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return run

    for batch in (128, 256, 512):
        model = build_model(args.size, index).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.rand(batch, 3, 64, 64, device=dev)
        y = torch.randint(0, len(index["categories"]), (batch,), device=dev)
        for label, kw in (("fp32", {}), ("amp fp16", {"amp": True}), ("encoder only", {"heads": False})):
            try:
                s = timed(step_fn(model, opt, x, y, **kw), dev)
                print(f"step  b{batch:<4} {label:13} {s * 1000:7.1f} ms  {batch / s:7.0f} img/s")
            except Exception as exc:  # noqa: BLE001
                print(f"step  b{batch:<4} {label:13} failed: {exc}")
        if batch == 128:
            xc = x.contiguous(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
            s = timed(step_fn(model, opt, xc, y), dev)
            print(f"step  b{batch:<4} {'channels_last':13} {s * 1000:7.1f} ms  {batch / s:7.0f} img/s")

    # Parts of the encoder, forward + backward, batch 256.
    model = build_model(args.size, index).to(dev)
    enc = model.encoder
    x = torch.rand(256, 3, 64, 64, device=dev, requires_grad=False)

    def part(fn):
        def run():
            out = fn()
            out.float().pow(2).mean().backward()
        return run

    with torch.no_grad():
        conv_out = enc.stages(enc.stem(x))
    print(f"part  stem+conv stages    {timed(part(lambda: enc.stages(enc.stem(x))), dev) * 1000:7.1f} ms")
    tok = conv_out.detach().requires_grad_()

    def tail():
        t = enc.tokenize(tok)
        t = t + enc.pos(t)
        t = t.flatten(2).transpose(1, 2)
        for blk in enc.blocks:
            t = blk(t)
        return enc.norm(t)
    print(f"part  transformer         {timed(part(tail), dev) * 1000:7.1f} ms")
    g = torch.rand(256, 256, device=dev, requires_grad=True)
    print(f"part  20k-way state head  {timed(part(lambda: model.state(g)), dev) * 1000:7.1f} ms")
    # Depthwise vs dense 3x3 at the stem's resolution.
    c = enc.config["conv"][0]
    feat = torch.rand(256, c, 64, 64, device=dev)
    dw = torch.nn.Conv2d(c, c, 3, padding=1, groups=c).to(dev)
    dense = torch.nn.Conv2d(c, c, 3, padding=1).to(dev)
    print(f"op    depthwise 3x3 c{c}   {timed(part(lambda: dw(feat)), dev) * 1000:7.1f} ms")
    print(f"op    dense 3x3 c{c}       {timed(part(lambda: dense(feat)), dev) * 1000:7.1f} ms")
    bn = torch.nn.BatchNorm2d(c).to(dev)
    print(f"op    batchnorm c{c}       {timed(part(lambda: bn(feat)), dev) * 1000:7.1f} ms")
    s = timed(lambda: torch.ones(1, device=dev).item(), dev, runs=50)
    print(f"op    one .item() sync    {s * 1000:7.2f} ms")

    # Data loading alone.
    ds = SpriteDataset(data, index, "train", train=True)
    for workers in (0, 2, 4, 6):
        loader = _loader(ds, 256, True, workers, 0)
        it = iter(loader)
        next(it)
        t = time.perf_counter()
        n = 0
        for _ in range(8 if workers else 3):
            n += len(next(it)["x"])
        dt = time.perf_counter() - t
        print(f"data  workers {workers}           {n / dt:7.0f} img/s")
        del it, loader


if __name__ == "__main__":
    main()
