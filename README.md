# Changeling14 Vision

An open vision encoder for Space Station 14: a small model that turns SS14
screenshots and sprites into features that know what every object is and what
state it is in. This repo holds the dataset pipeline, training code and
benchmark. It is early: only the first step (M0, the sprite catalogue) exists.

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

## Progress for the training app

Steps report progress through a run folder; see
[docs/run-folder.md](docs/run-folder.md). Pass `--run-dir` or set
`C14_RUN_DIR`. The sprite step writes its progress there plus a sample
gallery of cut sprites.

## License

Code: MIT. Sprites keep their own licenses (see `credits.md` in each build);
none are stored in this repo.
