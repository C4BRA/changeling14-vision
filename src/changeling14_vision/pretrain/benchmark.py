"""Score a pretraining checkpoint on every held-out sprite, and time the encoder.

Writes benchmarks/step-N.json and .md next to the checkpoint, and
benchmark.json, benchmark.md, per_category.csv plus a "mistakes" watch-set
folder (the most confident wrong answers) into the run folder.
"""

from __future__ import annotations

import csv
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

from ..model.encoder import count_parameters
from ..progress import Progress, _write_json_atomic
from .data import SpriteDataset
from .text import TextTable
from .train import _loader, model_from_checkpoint, newview_retrieval, pick_device, retrieval_sets
from .watch import draw_watchset

SPEED_SIZE = (480, 480)     # 15 x 15 tiles, the plan's reference frame


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_encoder(encoder, device, size=SPEED_SIZE, runs: int = 30) -> dict:
    encoder.eval()
    x = torch.rand(1, 3, *size, device=device)
    for _ in range(5):
        encoder(x)
    _sync(device)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        encoder(x)
        _sync(device)
        times.append((time.perf_counter() - t) * 1000)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3   # bytes on macOS, KB on Linux
    return {"frame": list(size), "device": str(device), "ms_median": statistics.median(times),
            "ms_p90": sorted(times)[int(len(times) * 0.9) - 1], "process_peak_rss_mb": round(peak_mb),
            "machine": f"{platform.system()} {platform.machine()}"}


@torch.no_grad()
def _predict(model, ds, device, batch, progress, stage, known: set[int] | None = None):
    """known: text rows some training picture carries, to split naming into
    items training showed and items it never did."""
    loader = _loader(ds, batch, False, 0, 0)
    rows = []
    done = 0
    all_text = model.text_embeddings() if model.text_dim and len(model.text_vectors) else None
    name_of = model.text_name_ids.cpu().tolist() if len(model.text_name_ids) else None
    for batch in loader:
        if progress.stop_requested:
            break
        out = model(batch["x"].to(device))
        c, d = out["category"], out["direction"]
        cat, state, direction = batch["category"], batch["state"], batch["direction"]
        cp = c.softmax(1).cpu()
        top5 = c.topk(min(5, c.shape[1]), 1).indices.cpu()
        s = out.get("state")
        stop5 = s.topk(min(5, s.shape[1]), 1).indices.cpu() if s is not None else None
        dirs = d.argmax(1).cpu()
        comp_pred = (out["components"] > 0).cpu()
        comp_true = batch["components"] > 0.5
        for i in range(len(cat)):
            row = {"cat": int(cat[i]), "cat_top5": top5[i].tolist(), "cat_conf": float(cp[i].max()),
                   "state": int(state[i]) if stop5 is not None else -1,
                   "state_top5": stop5[i].tolist() if stop5 is not None else [],
                   "dir": int(direction[i]), "dir_pred": int(dirs[i])}
            if all_text is not None and batch["text"][i] >= 0:
                g = torch.nn.functional.normalize(out["global"][i:i + 1], dim=-1)
                top = (g @ all_text.T).topk(min(5, all_text.shape[0]), dim=1).indices[0].cpu().tolist()
                rows_wanted = {int(r) for r in batch["text_rows"][i] if r >= 0}
                wanted = {name_of[r] for r in rows_wanted} if name_of else rows_wanted
                top = [name_of[t] for t in top] if name_of else top        # the right name counts
                row["text_hit"] = (top[0] in wanted, any(t in wanted for t in top))
                if known is not None:
                    row["known_item"] = bool(rows_wanted & known)
            if batch["has_components"][i]:
                p, t = comp_pred[i], comp_true[i]
                row["comp"] = (int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()))
            rows.append(row)
        done += len(cat)
        progress.update(stage, done, len(ds))
    return rows


def _scores(rows) -> dict:
    out = {"n": len(rows)}
    if rows:
        out["category_top1"] = sum(r["cat_top5"][0] == r["cat"] for r in rows) / len(rows)
        out["category_top5"] = sum(r["cat"] in r["cat_top5"] for r in rows) / len(rows)
    st = [r for r in rows if r["state"] >= 0]
    if st:
        out["state_top1"] = sum(r["state_top5"][0] == r["state"] for r in st) / len(st)
        out["state_top5"] = sum(r["state"] in r["state_top5"] for r in st) / len(st)
    th = [r["text_hit"] for r in rows if "text_hit" in r]
    if th:
        out.update(text_n=len(th), text_top1=sum(a for a, _ in th) / len(th), text_top5=sum(b for _, b in th) / len(th))
    for key, flag in (("known", True), ("new", False)):
        th = [r["text_hit"] for r in rows if "text_hit" in r and r.get("known_item") is flag]
        if th:
            out.update({f"text_{key}_n": len(th), f"text_{key}_top1": sum(a for a, _ in th) / len(th),
                        f"text_{key}_top5": sum(b for _, b in th) / len(th)})
    cr = [r["comp"] for r in rows if "comp" in r]
    if cr:
        tp, fp, fn = (sum(x[k] for x in cr) for k in range(3))
        p, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
        out.update(components_n=len(cr), components_precision=p, components_recall=r_,
                   components_f1=2 * p * r_ / max(1e-9, p + r_))
    dr = [r for r in rows if r["dir"] >= 0]
    if dr:
        out["direction_top1"] = sum(r["dir_pred"] == r["dir"] for r in dr) / len(dr)
    return out


def _render_md(r: dict) -> str:
    lines = [f"# Benchmark: {r['size']} encoder, step {r['step']:,}", "",
             f"Checkpoint `{r['checkpoint']}`, {r['parameters'] / 1e6:.1f}M parameters.", ""]
    s = r["speed"]
    lines += ["## Speed", f"- {s['ms_median']:.1f} ms median ({s['ms_p90']:.1f} ms p90) per "
              f"{s['frame'][1]}x{s['frame'][0]} frame on {s['device']} ({s['machine']}).",
              f"- Process peak memory {s['process_peak_rss_mb']:,} MB.", ""]
    if "unseen" in r:
        u, n = r["unseen"], r["newview"]
        lines += ["## Held-out sprites",
                  "| Test | Images | Top-1 | Top-5 |", "|---|---|---|---|",
                  f"| Category of sprites from unseen sheets | {u['n']:,} | {u.get('category_top1', 0):.1%} | "
                  f"{u.get('category_top5', 0):.1%} |",
                  f"| Exact sprite, new direction or frame | {n['n']:,} | {n.get('state_top1', 0):.1%} | "
                  f"{n.get('state_top5', 0):.1%} |",
                  f"| Facing direction (four-way sprites) | | {n.get('direction_top1', 0):.1%} | |", "",
                  *([f"Naming unseen-sheet sprites from all entity texts ({u['text_n']:,} linked sprites): "
                     f"top-1 {u['text_top1']:.1%}, top-5 {u['text_top5']:.1%}.", ""] if "text_n" in u else []),
                  *([f"Of those, items no training picture shows ({u['text_new_n']:,}): top-5 {u['text_new_top5']:.1%}; "
                     f"items also shown in other training sheets ({u.get('text_known_n', 0):,}): "
                     f"top-5 {u.get('text_known_top5', 0):.1%}.", ""] if "text_new_n" in u else []),
                  *([f"Naming new views of training sprites ({n['text_n']:,}): top-1 {n['text_top1']:.1%}, "
                     f"top-5 {n['text_top5']:.1%}.", ""] if "text_n" in n else []),
                  f"Entity components on unseen sheets ({u.get('components_n', 0):,} linked sprites): "
                  f"F1 {u.get('components_f1', 0):.2f}, precision {u.get('components_precision', 0):.2f}, "
                  f"recall {u.get('components_recall', 0):.2f}.", "",
                  "## Unseen sheets by category", "| Category | Images | Top-1 |", "|---|---|---|"]
        lines += [f"| {c['category']} | {c['n']:,} | {c['top1']:.1%} |" for c in r["per_category"]]
        lines.append("")
    if not r["complete"]:
        lines += ["**Stopped early; scores are partial.**", ""]
    return "\n".join(lines)


def benchmark(model_dir: Path, data: Path, progress: Progress | None = None, checkpoint: str | None = None,
              device: str = "auto", speed_only: bool = False, batch: int = 128) -> dict:
    progress = progress or Progress(None)
    ckpt_path = Path(checkpoint).expanduser() if checkpoint else model_dir / "latest.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"no checkpoint at {ckpt_path}; train first")
    progress.update("load", message=str(ckpt_path), force=True)
    model, ckpt = model_from_checkpoint(ckpt_path)
    dev = pick_device(device)
    model.to(dev).eval()
    step = ckpt["step"]
    report = {"checkpoint": str(ckpt_path), "step": step, "size": ckpt["encoder_config"]["size"],
              "parameters": count_parameters(model.encoder), "time": time.time(), "complete": True}

    progress.update("time encoder", message=f"{SPEED_SIZE[1]}x{SPEED_SIZE[0]} frames on {dev}", force=True)
    report["speed"] = time_encoder(model.encoder, dev)

    if not speed_only:
        index = json.loads((Path(ckpt.get("out", model_dir)) / "index.json").read_text())
        text_file = Path(ckpt.get("out", model_dir)) / "text.npz"
        text = TextTable(text_file) if model.text_dim and text_file.is_file() else None
        known = ({r for it in index["items"] if it["split"] == "train" for r in text.rows_for(it.get("entities", []))}
                 if text else None)
        unseen = _predict(model, SpriteDataset(data, index, "unseen", train=False, worn=False, text=text), dev, batch, progress,
                          "score unseen sheets", known)
        newview = _predict(model, SpriteDataset(data, index, "newview", train=False, worn=False, text=text), dev, batch, progress,
                           "score new views", known)
        report["complete"] = not progress.stop_requested
        report["unseen"], report["newview"] = _scores(unseen), _scores(newview)
        if model.state is None and not progress.stop_requested:
            # Newer models have no per-sprite layer: same-sprite recognition is
            # measured by looking each held-out view up among the training sprites.
            progress.update("score new views", message="looking up each view among the training sprites", force=True)
            queries, gallery = retrieval_sets(index, data, ckpt.get("options", {}).get("seed", 14), 10**9, text)
            report["newview"].update(newview_retrieval(model, queries, gallery, batch, dev,
                                                       lambda: progress.stop_requested))
        per_cat = []
        for c, name in enumerate(index["categories"]):
            rows = [r for r in unseen if r["cat"] == c]
            if rows:
                per_cat.append({"category": name, "n": len(rows),
                                "top1": sum(r["cat_top5"][0] == c for r in rows) / len(rows)})
        report["per_category"] = sorted(per_cat, key=lambda c: -c["n"])
        if progress.run_dir and unseen:
            items = SpriteDataset(data, index, "unseen", train=False, worn=False, text=text).items
            wrong = sorted((i for i, r in enumerate(unseen) if r["cat_top5"][0] != r["cat"]),
                           key=lambda i: -unseen[i]["cat_conf"])[:24]
            draw_watchset(model, data, index, [items[i] for i in wrong], dev,
                          progress.run_dir / "watchset" / f"mistakes-step-{step:07d}", text=text, captions=True)
            _write_json_atomic(progress.run_dir / "watchset" / f"mistakes-step-{step:07d}" / "meta.json",
                               {"step": step, "what": "most confident wrong answers on unseen sheets"})

    md = _render_md(report)
    targets = [model_dir / "benchmarks"] + ([progress.run_dir] if progress.run_dir else [])
    for i, target in enumerate(targets):
        target.mkdir(parents=True, exist_ok=True)
        stem = f"step-{step:07d}" if i == 0 else "benchmark"
        (target / f"{stem}.json").write_text(json.dumps(report, indent=2))
        (target / f"{stem}.md").write_text(md)
    if progress.run_dir and report.get("per_category"):
        with open(progress.run_dir / "per_category.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["category", "n", "top1"])
            w.writeheader()
            w.writerows(report["per_category"])
    print(md, flush=True)
    s = report["speed"]
    summary = f"{s['ms_median']:.1f} ms/frame"
    if "unseen" in report:
        summary += (f", unseen category {report['unseen'].get('category_top1', 0):.1%}, "
                    f"components F1 {report['unseen'].get('components_f1', 0):.2f}, "
                    f"new-view state {report['newview'].get('state_top1', 0):.1%}")
    progress.update("done" if report["complete"] else "stopped", message=summary, eta_seconds=0, force=True)
    return report
