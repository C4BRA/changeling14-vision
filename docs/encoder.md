# Encoder spec

The Changeling14 encoder turns an SS14 frame into one feature per 32 px
square plus one feature for the whole frame. Every size has the same inputs
and output shapes, so code written for one size runs on all of them. The
features of different sizes only share one space once the later stages
(text alignment and distillation from base) train them into it; until then a
head trained on one size needs retraining for another.

## Input

- RGB, float, values in [0, 1]. PyTorch and ONNX take `(B, 3, H, W)`; MLX
  takes `(B, H, W, 3)`.
- `H` and `W` must be multiples of 32. A 15×15-tile view is 480×480.
- One input pixel is one game pixel at the default zoom. Resize only with
  nearest-neighbour, by whole numbers, never with smooth interpolation.
- Normalisation happens inside the model; feed raw [0, 1] values.

## Outputs

| Name | Shape | Meaning |
|---|---|---|
| `tiles` | `(B, H/32, W/32, 256)` | one feature per 32 px square, row-major from the top left |
| `global` | `(B, 256)` | one feature for the whole frame |

Square `(r, c)` covers pixels `[32r, 32r+32) × [32c, 32c+32)`. Its feature
also sees a little past its edges, so an object straddling a boundary is
seen whole.

## Layout

| Part | What it does |
|---|---|
| stem | 3×3 conv at full resolution, so single-pixel detail survives |
| 4 conv stages | each opens with a 2×2 stride-2 conv, so every downsample lines up with the tile grid; blocks are a 3×3 depthwise conv and a 1×1 MLP, both residual |
| tokenise | 2×2 stride-2 conv: one token per 32 px square |
| position | 3×3 depthwise conv over the token grid, so any frame size works |
| transformer | pre-norm blocks (softmax attention + MLP) across all squares |
| heads | `tiles` and `global` linear projections to 256 |

Only `Conv2d`, `BatchNorm2d`, `LayerNorm`, `Linear`, GELU and softmax
attention are used.

## Sizes

| Size | Parameters | Conv widths | Transformer width × depth | 480×480 on a 4-core x86 CPU |
|---|---|---|---|---|
| nano | 2.3M | 16 → 128 | 160 × 6 | 46 ms |
| tiny | 9.1M | 24 → 192 | 256 × 10 | 72 ms |
| small | 24.2M | 32 → 256 | 384 × 12 | 169 ms |
| base | 76.2M | 48 → 384 | 640 × 14 | 345 ms |

The CPU numbers are only a reference point; `changeling14-vision benchmark`
measures the real ones on the Air.

## Files from `export`

| Format | File | Notes |
|---|---|---|
| safetensors | `changeling14-vision-<size>-p32.safetensors` + `.json` | PyTorch state dict for `changeling14_vision.model.Encoder`, config alongside |
| ONNX | `changeling14-vision-<size>-p32.onnx` | dynamic batch, height and width |
| MLX | `changeling14-vision-<size>-p32-mlx/` | `weights.safetensors` (batch norms folded, NHWC), `config.json`, and `mlx_encoder.py` to run it with only `mlx` installed |
| Core ML | `changeling14-vision-<size>-p32-<W>x<H>.mlpackage` | fixed frame size; built on macOS with coremltools |

`export.json` records, for each format, the largest difference from PyTorch
on a random frame.
