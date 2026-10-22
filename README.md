# Changeling14 Vision

An open vision encoder for Space Station 14: a small model that turns SS14
screenshots and sprites into features that know what every object is and what
state it is in. This repo holds the dataset pipeline, training code and
benchmark. It is early: the sprite catalogue (M0) and single-sprite
pretraining of the encoder (stage 1) exist; replay import comes next.

The desktop control panel lives in
[C4BRA/changeling14-training-gui](https://github.com/C4BRA/changeling14-training-gui).
Every step here is a plain command, so the pipeline runs with or without it.

## Install

Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## M0: sprite catalogue

Reads an SS14 checkout (Wizden or a fork) and writes the object catalogue that
every later step uses. It never writes into the checkout.

```sh
changeling14-vision sprites \
    --repo ~/Desktop/SS13-14/wizden-ss14 \
    --out  ~/Desktop/SS13-14/ss14-sprite-dataset
```

Those two paths are the defaults. It takes about a minute and writes:

| Path | What |
|---|---|
| `images/<sheet>/<state>/d<dir>_f<frame>.png` | every state, direction and animation frame, at native size |
| `catalogue/sheets.jsonl` | every sheet: license, copyright, size, states, whether it's included |
| `catalogue/images.jsonl` | one line per cut image |
| `catalogue/entities.jsonl` | every entity: name, description, parent chain, components, sprite refs with a status |
| `catalogue/sprite_links.jsonl` | for each (sheet, state), the entities that draw it |
| `credits.md`, `credits.csv` | every sheet used, with its license and copyright |
| `excluded.csv` | every sheet left out, and why |
| `report.md`, `report.json` | counts, and how many sprite links came out clean |
| `manifest.json` | source commit, tool version, file hashes |

Licenses: only CC-BY-SA, CC-BY and CC0 sheets are used. Non-commercial,
custom, missing and unrecognised licenses are excluded and listed.

Sprite-link statuses: `clean` (sheet and state exist, license allowed),
`runtime_state` and `runtime_layer` (picked by game logic, so flagged rather
than guessed), `excluded_license`, `missing_state`, `missing_sheet`,
`no_sheet`, `texture`.

On Wizden master (October 2026): 3,195 sheets, 76 excluded as non-commercial,
77k images, and 9.2k of 10.1k concrete entities with a clean world sprite.

Rebuilding replaces the previous output. The step refuses a non-empty folder
it didn't create unless you pass `--force`.

## Stage 1: single-sprite pretraining

The encoder is a hybrid: convolutions read the pixels inside each 32 px
square and turn it into one token, then a small transformer reasons across
the squares (the FastViT / MobileCLIP layout). It comes in four sizes with
the same outputs: nano 2.3M, tiny 9.1M, small 24M and base 76M parameters.
The input and output spec is in [docs/encoder.md](docs/encoder.md).

Training needs PyTorch. On the Air it runs on the GPU through MPS.

```sh
pip install -e ".[train,export]"
changeling14-vision train --size tiny          # reads the M0 catalogue
changeling14-vision benchmark --size tiny      # held-out scores and ms per frame
changeling14-vision export --size tiny         # safetensors, ONNX, MLX, Core ML
```

`train` draws each sprite from the M0 catalogue onto a 96×96 canvas at a
random spot, on a plain, noisy or real floor-tile background, with
whole-number zoom and nearest-neighbour rotation. It learns three things
from the catalogue's own labels: the coarse category from the sheet's folder
(`Objects/Tools`, `Structures/Doors`, 48 in all), the exact sprite (sheet and
state), and the facing direction. Screen HUD art (`Interface/`, `Actions/`,
`Markers/`, …) is left out. Held and worn overlays (`inhand-*`,
`equipped-*`) are a third of the catalogue but alone are small fragments of
a mob, so they train at a quarter of the weight and are left out of the
scores.

What it holds out, so the scores mean something:

- **Unseen sheets:** 10% of sheets are never trained on. They are scored on
  category, since their exact sprites are new.
- **New views:** one direction or animation frame of each trained sprite is
  held back. These are scored on the exact sprite and on facing.

The model folder (default `~/Desktop/SS13-14/changeling14-models/sprite-pretrain-<size>`)
holds `latest.pt`, every checkpoint under `checkpoints/`, `metrics.jsonl`,
and the label index. Running `train` again resumes from `latest.pt`;
`--fresh` starts over. `--minutes 120` stops cleanly after two hours, so a
long run can be split across nights. Stopping from the app saves a
checkpoint first.

At every checkpoint (every 1,000 steps, and at least every 15 minutes) the
run folder gets a watch set: 24 sprites from unseen sheets, drawn with the
model's category guess, its confidence and the trained sprite it thinks
they look most like, framed green or red.

`benchmark` scores every held-out sprite, breaks the unseen-sheet score down
by category (`per_category.csv`), times a 480×480 frame, and puts the 24
most confident mistakes into the watch set.

`export` writes the encoder without the training heads, and checks each
format against PyTorch where its runtime is installed.

Defaults: 20,000 steps of 128 sprites with AdamW and a cosine schedule.
Memory peaked around 3 GB at batch 128 on CPU; pass `--batch 64` if the Air
starts swapping.

## Progress for the training app

Steps report progress through a run folder; see
[docs/run-folder.md](docs/run-folder.md). Pass `--run-dir` or set
`C14_RUN_DIR`. The sprite step writes its progress there plus a sample
gallery of cut sprites.

## License

Code: MIT. Sprites keep their own licenses (see `credits.md` in each build);
none are stored in this repo.
