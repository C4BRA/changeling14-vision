"""M0: build the sprite catalogue from an SS14 checkout.

Reads Resources/Textures and Resources/Prototypes, never writes to the
checkout, and writes everything into one output folder:

  images/<sheet>/<state>/d<direction>_f<frame>.png   every cell, native size
  catalogue/sheets.jsonl         one line per sheet: license, size, states
  catalogue/images.jsonl         one line per cut image
  catalogue/entities.jsonl       one line per entity: name, parent chain, sprite refs
  catalogue/sprite_links.jsonl   one line per (sheet, state): the entities that draw it
  catalogue/entity_images.jsonl  one line per distinct look of an entity on the floor: its
                                 layers stacked and tinted (images/_entities/), or the cut
                                 image itself when that is one untinted layer
  credits.md, credits.csv        every sheet used, with license and copyright
  excluded.csv                   every sheet left out, with the reason
  report.json, report.md         counts, and how many sprite links came out clean
  manifest.json                  source commit, tool version, file hashes
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import __version__
from ..progress import Progress
from . import licenses
from .compose import Composer
from .prototypes import Entity, SpriteRef, load_entities, resolve, sprite_refs
from .rsi import Sheet, cut_state, find_sheets, read_sheet

OWNED = ("images", "catalogue", "credits.md", "credits.csv", "excluded.csv",
         "report.json", "report.md", "manifest.json")
WATCHSET_SAMPLES = 36


class BuildError(Exception):
    pass


@dataclass
class Options:
    repo: Path
    out: Path
    images: bool = True
    force: bool = False
    seed: int = 14


def git_origin(repo: Path) -> str | None:
    """The checkout's GitHub-style name (owner/repo) from its origin remote, if any."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"],
                             capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    url = out.stdout.strip().removesuffix(".git").rstrip("/")
    parts = url.replace(":", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 and url else None


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _prepare_out(opts: Options) -> None:
    repo, out = opts.repo.resolve(), opts.out.resolve()
    if out == repo or repo in out.parents:
        raise BuildError(f"output {out} is inside the source checkout; pick a folder outside it")
    if out.exists() and any(out.iterdir()):
        if not (out / "manifest.json").exists() and not opts.force:
            raise BuildError(f"{out} is not empty and has no manifest.json from an earlier build; "
                             "use --force to write into it anyway")
        for name in OWNED:
            target = out / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    (out / "catalogue").mkdir(parents=True, exist_ok=True)


def _jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ("" if value is None else json.dumps(value))


def classify(ref: SpriteRef, sheets: dict[str, Sheet], included: set[str]) -> str:
    """clean, runtime_state, runtime_layer, texture, no_sheet, missing_sheet,
    missing_state, bad_sheet or excluded_license."""
    if ref.texture and not ref.rsi:
        return "texture"
    if not ref.rsi and ref.state is None:
        return "runtime_layer"   # sheet and state both filled in by game logic
    if not ref.rsi:
        return "no_sheet"
    sheet = sheets.get(ref.rsi)
    if sheet is None:
        return "missing_sheet"
    if sheet.error:
        return "bad_sheet"
    if ref.rsi not in included:
        return "excluded_license"
    if ref.role in ("item", "clothing"):
        return "clean"
    if ref.state is None:
        return "runtime_state"
    if ref.state not in sheet.states:
        return "missing_state"
    return "clean"


def _compose_entities(out: Path, entity_refs: dict[str, list[tuple[SpriteRef, str]]],
                      sheets: dict[str, Sheet], progress: Progress) -> int:
    """Write catalogue/entity_images.jsonl: one row per distinct look, with
    every entity that looks exactly like that. Returns the number of rows."""
    composer = Composer(sheets)
    simple: dict[tuple[str, str], list[str]] = defaultdict(list)
    stacked: dict[bytes, dict] = {}
    ids = sorted(entity_refs)
    for i, entity_id in enumerate(ids):
        if i % 500 == 0:
            progress.update("compose items", i, len(ids), "stacking layers and colours")
        look = composer.compose(entity_refs[entity_id])
        if look is None:
            continue
        if look.simple:
            simple[look.layers[0]].append(entity_id)
            continue
        key = hashlib.sha256(look.image.tobytes() + repr(look.image.size).encode()).digest()
        if key not in stacked:
            rel = Path("images") / "_entities" / f"{entity_id}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            look.image.save(out / rel)
            sheet, state = look.layers[0]
            stacked[key] = {"path": rel.as_posix(), "composite": True, "sheet": sheet, "state": state,
                            "layers": [f"{a}|{b}" for a, b in look.layers], "tinted": look.tinted,
                            "width": look.image.width, "height": look.image.height, "entities": []}
        stacked[key]["entities"].append(entity_id)
    rows = list(stacked.values())
    for (sheet, state), users in sorted(simple.items()):
        rel = Path("images") / sheet[:-len(".rsi")] / state / "d0_f0.png"
        if (out / rel).is_file():
            rows.append({"path": rel.as_posix(), "composite": False, "sheet": sheet, "state": state,
                         "layers": [f"{sheet}|{state}"], "tinted": False, "entities": users})
    _jsonl(out / "catalogue" / "entity_images.jsonl", rows)
    return len(rows)


def build(opts: Options, progress: Progress | None = None) -> dict:
    progress = progress or Progress(None)
    started = time.time()
    resources = opts.repo / "Resources"
    textures, prototypes = resources / "Textures", resources / "Prototypes"
    if not textures.is_dir() or not prototypes.is_dir():
        raise BuildError(f"{opts.repo} doesn't look like an SS14 checkout (no Resources/Textures and Resources/Prototypes)")
    _prepare_out(opts)
    out = opts.out
    problems: list[str] = []

    # 1. Sheets and licenses ------------------------------------------------
    progress.update("read sheets", force=True)
    sheet_dirs = find_sheets(textures)
    sheets: dict[str, Sheet] = {}
    excluded: dict[str, str] = {}
    for i, path in enumerate(sheet_dirs):
        sheet = read_sheet(path, textures)
        sheets[sheet.path] = sheet
        reason = sheet.error or licenses.exclusion_reason(sheet.license)
        if reason:
            excluded[sheet.path] = reason
        progress.update("read sheets", i + 1, len(sheet_dirs))
    included = {p for p in sheets if p not in excluded}

    # 2. Entities and sprite links ------------------------------------------
    progress.update("read prototypes", message="parsing YAML", force=True)
    entities, load_problems = load_entities(prototypes)
    problems += load_problems
    problems += resolve(entities)

    links: dict[tuple[str, str | None], list[dict]] = defaultdict(list)
    ref_status = Counter()
    entity_rows = []
    drawable = clean_entities = 0
    entity_refs: dict[str, list[tuple[SpriteRef, str]]] = {}
    for entity in sorted(entities.values(), key=lambda e: e.id):
        rows = []
        for ref in sprite_refs(entity):
            status = classify(ref, sheets, included)
            if not entity.abstract:
                entity_refs.setdefault(entity.id, []).append((ref, status))
            if not entity.abstract:
                ref_status[status] += 1
            row = {k: v for k, v in asdict(ref).items() if v not in (None, (), True) or k in ("rsi", "state")}
            row["status"] = status
            rows.append(row)
            if status == "clean" and not entity.abstract:
                states = [ref.state] if ref.state else sorted(sheets[ref.rsi].states)
                for state in states:
                    links[(ref.rsi, state)].append({"entity": entity.id, "role": ref.role})
        if not entity.abstract and rows:
            drawable += 1
            if any(r["status"] == "clean" and r["role"] in ("sprite", "layer", "icon") for r in rows):
                clean_entities += 1
        entity_rows.append({
            "id": entity.id,
            "name": entity.name,
            "description": entity.description,
            "suffix": entity.suffix,
            "abstract": entity.abstract,
            "parents": list(entity.parents),
            "chain": list(entity.chain),
            "source": entity.source,
            "components": sorted(entity.components),
            "sprites": rows,
        })
    cat = out / "catalogue"
    _jsonl(cat / "entities.jsonl", entity_rows)
    _jsonl(cat / "sprite_links.jsonl", (
        {"sheet": sheet, "state": state, "entities": users}
        for (sheet, state), users in sorted(links.items(), key=lambda kv: (kv[0][0], kv[0][1] or ""))))

    # 3. Cut images ---------------------------------------------------------
    image_rows = []
    cut_errors = 0
    stopped = False
    ordered = sorted(included)
    t0 = time.time()
    for i, path in enumerate(ordered):
        if progress.stop_requested:
            stopped = True
            break
        sheet = sheets[path]
        for state in sheet.states.values():
            base = Path("images") / path[:-len(".rsi")] / state.name
            try:
                cells = list(cut_state(sheet, state))
            except (OSError, ValueError) as exc:
                cut_errors += 1
                problems.append(f"{path}/{state.name}: {exc}")
                continue
            if opts.images:
                (out / base).mkdir(parents=True, exist_ok=True)
            for direction, frame, img in cells:
                rel = base / f"d{direction}_f{frame}.png"
                if opts.images:
                    img.save(out / rel, optimize=False)
                image_rows.append({"path": rel.as_posix(), "sheet": path, "state": state.name,
                                   "direction": direction, "frame": frame,
                                   "width": img.width, "height": img.height})
        done = i + 1
        rate = (time.time() - t0) / done
        progress.update("cut sprites", done, len(ordered), f"{len(image_rows):,} images",
                        eta_seconds=rate * (len(ordered) - done))
    _jsonl(cat / "images.jsonl", image_rows)

    # 3b. Whole entities, layers stacked and tinted ----------------------------
    looks = 0
    if opts.images and not stopped:
        looks = _compose_entities(out, entity_refs, sheets, progress)

    # 4. Sheets, credits, exclusions -----------------------------------------
    used_sheets = {sheet for (sheet, _state) in links}
    _jsonl(cat / "sheets.jsonl", (
        {"sheet": s.path, "license": s.license, "copyright": s.copyright,
         "size": [s.width, s.height], "included": s.path in included,
         "excluded_reason": excluded.get(s.path), "linked_to_entity": s.path in used_sheets,
         "states": [{"name": st.name, "directions": st.directions, "frames": st.frames} for st in s.states.values()]}
        for s in sorted(sheets.values(), key=lambda s: s.path)))
    with open(out / "credits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "license", "copyright"])
        for path in ordered:
            w.writerow([path, licenses.normalize(sheets[path].license), _text(sheets[path].copyright)])
    with open(out / "excluded.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "license", "reason"])
        for path in sorted(excluded):
            w.writerow([path, _text(sheets[path].license), excluded[path]])
    commit = git_commit(opts.repo)
    with open(out / "credits.md", "w", encoding="utf-8") as f:
        f.write("# Sprite credits\n\n")
        source = git_origin(opts.repo) or opts.repo.name
        f.write(f"Sprites from {source} (commit {commit or 'unknown'}), "
                "used under the license stated for each sheet.\n\n")
        f.write("| Sheet | License | Copyright |\n|---|---|---|\n")
        for path in ordered:
            copyright_text = _text(sheets[path].copyright).replace("|", "\\|").replace("\n", " ")
            f.write(f"| {path} | {licenses.normalize(sheets[path].license)} | {copyright_text} |\n")

    # 5. Report and manifest -------------------------------------------------
    license_counts = Counter(licenses.normalize(s.license) or "missing" for s in sheets.values())
    concrete = sum(1 for e in entities.values() if not e.abstract)
    report = {
        "complete": not stopped,
        "source": {"repo": str(opts.repo), "origin": git_origin(opts.repo), "commit": commit},
        "sheets": {"total": len(sheets), "included": len(included), "excluded": len(excluded),
                   "excluded_by_reason": dict(Counter(excluded.values()).most_common()),
                   "by_license": dict(license_counts.most_common()),
                   "included_linked_to_entity": len(used_sheets & included)},
        "images": {"cut": len(image_rows), "written": len(image_rows) if opts.images else 0,
                   "state_errors": cut_errors},
        "entities": {"total": len(entities), "concrete": concrete, "abstract": len(entities) - concrete,
                     "concrete_with_sprite_refs": drawable,
                     "concrete_with_clean_world_sprite": clean_entities},
        "sprite_refs": dict(ref_status.most_common()),
        "sprite_links": len(links),
        "entity_looks": looks,
        "problems": len(problems),
        "seconds": round(time.time() - started, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "problems.txt").write_text("\n".join(problems) + ("\n" if problems else ""))
    (out / "report.md").write_text(render_report(report))
    manifest = {
        "tool": "changeling14-vision sprites",
        "version": __version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": report["source"],
        "license_policy": {"allowed": sorted(licenses.ALLOWED),
                           "excluded": "non-commercial, custom, missing or unrecognised"},
        "files": {p.relative_to(out).as_posix(): _sha256(p)
                  for p in sorted([*cat.glob("*.jsonl"), out / "credits.csv", out / "excluded.csv"])},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if opts.images and image_rows:
        rng = random.Random(opts.seed)
        firsts = [r for r in image_rows if r["direction"] == 0 and r["frame"] == 0]
        sample = rng.sample(firsts, min(WATCHSET_SAMPLES, len(firsts)))
        progress.watchset("sample", [out / r["path"] for r in sample], {"images": len(image_rows)})
    progress.update("done" if not stopped else "stopped", len(ordered) if not stopped else None,
                    len(ordered), f"{len(image_rows):,} images, {clean_entities:,} entities with clean sprites",
                    eta_seconds=0, force=True)
    return report


def render_report(r: dict) -> str:
    s, e, refs = r["sheets"], r["entities"], r["sprite_refs"]
    total_refs = sum(refs.values()) or 1
    lines = [
        "# Sprite catalogue report",
        "",
        f"Source: `{r['source']['repo']}` at commit `{r['source']['commit'] or 'unknown'}`."
        + ("" if r["complete"] else " **Stopped early; counts are partial.**"),
        "",
        "## Sheets",
        f"- {s['total']:,} sheets, {s['included']:,} included, {s['excluded']:,} excluded.",
    ]
    lines += [f"  - {reason}: {n:,}" for reason, n in s["excluded_by_reason"].items()]
    lines += [f"- {s['included_linked_to_entity']:,} included sheets are drawn by at least one entity.",
              "- By license: " + ", ".join(f"{k} {v:,}" for k, v in s["by_license"].items()) + ".",
              "",
              "## Images",
              f"- {r['images']['cut']:,} images cut (every state, direction and animation frame).",
              f"- {r['images']['state_errors']:,} states could not be cut (see problems.txt).",
              f"- {r.get('entity_looks', 0):,} item looks: each entity's layers stacked with their colours, "
              "as it appears on the floor (catalogue/entity_images.jsonl).",
              "",
              "## Entities",
              f"- {e['total']:,} entity prototypes: {e['concrete']:,} concrete, {e['abstract']:,} abstract.",
              f"- {e['concrete_with_sprite_refs']:,} concrete entities reference a sprite.",
              f"- {e['concrete_with_clean_world_sprite']:,} have at least one clean world sprite link "
              "(sheet exists, license allowed, state exists).",
              "",
              "## Sprite links (concrete entities)",
              "| Status | Count | Share |",
              "|---|---|---|"]
    lines += [f"| {k} | {v:,} | {v / total_refs:.0%} |" for k, v in refs.items()]
    lines += ["",
              "`runtime_state` means the layer's state is picked by game logic (doors, fill levels, "
              "damage); `runtime_layer` means the whole layer is (action icons, body parts, overlays). "
              "Both are flagged, never guessed; mapping them to meanings is the next step.",
              "",
              f"{r['problems']:,} problems logged in problems.txt. Built in {r['seconds']} s.",
              ""]
    return "\n".join(lines)
