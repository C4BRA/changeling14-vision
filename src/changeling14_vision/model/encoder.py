"""Hybrid encoder: convolutions inside each 32 px square, a transformer across squares.

Layout (FastViT / MobileCLIP style, standard layers only):

  RGB image, H and W multiples of 32
  stem      3x3 conv at full resolution (keeps single-pixel detail)
  stages    four conv stages, each opened by a 2x2 stride-2 conv, so every
            downsample lines up with the 32 px grid; 3x3 depthwise convs let
            each square see a little past its edges
  tokenise  2x2 stride-2 conv: one token per 32 px square
  position  3x3 depthwise conv over the token grid (works at any grid size)
  mixer     pre-norm transformer blocks across all squares
  outputs   tiles  (B, H/32, W/32, 256)  one feature per square
            global (B, 256)              one feature for the whole image

Every size (nano, tiny, small, base) has the same outputs; only widths and
depths change. Only Conv2d, BatchNorm2d, LayerNorm, Linear, GELU and softmax
attention are used, so the model exports to ONNX, Core ML and MLX.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

TILE = 32
EMBED_DIM = 256

# conv: widths of the stem and the four conv stages; conv_depths: blocks per stage.
SIZES: dict[str, dict] = {
    "nano": dict(conv=(16, 24, 48, 96, 128), conv_depths=(1, 1, 2, 1), dim=160, depth=6, heads=4),
    "tiny": dict(conv=(24, 32, 64, 128, 192), conv_depths=(1, 2, 2, 2), dim=256, depth=10, heads=4),
    "small": dict(conv=(32, 48, 96, 192, 256), conv_depths=(1, 2, 3, 3), dim=384, depth=12, heads=6),
    "base": dict(conv=(48, 64, 128, 256, 384), conv_depths=(2, 2, 4, 4), dim=640, depth=14, heads=10),
}


class ConvBlock(nn.Module):
    """3x3 depthwise conv for space, then a 1x1 MLP for channels; both residual."""

    def __init__(self, channels: int, expand: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.bn = nn.BatchNorm2d(channels)
        self.fc1 = nn.Conv2d(channels, channels * expand, 1)
        self.fc2 = nn.Conv2d(channels * expand, channels, 1)

    def forward(self, x):
        x = x + self.dw_bn(self.dw(x))
        return x + self.fc2(F.gelu(self.fc1(self.bn(x))))


class Downsample(nn.Module):
    """2x2 stride-2 conv: non-overlapping, so it never blurs across the tile grid.
    With stride 1 it only changes the width (a 1x1 conv)."""

    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, stride, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Encoder(nn.Module):
    def __init__(self, conv=(24, 32, 64, 128, 192), conv_depths=(1, 2, 2, 2), dim=256, depth=10,
                 heads=4, embed_dim: int = EMBED_DIM, size: str | None = None, stem_stride: int = 1):
        super().__init__()
        if stem_stride not in (1, 2, 4):
            raise ValueError("stem_stride must be 1, 2 or 4")
        self.config = dict(size=size, conv=tuple(conv), conv_depths=tuple(conv_depths), dim=dim,
                           depth=depth, heads=heads, embed_dim=embed_dim, tile=TILE, stem_stride=stem_stride)
        # stem_stride 1: 3x3 conv at full resolution. 2 or 4: a non-overlapping
        # patch conv that still sees every pixel, and the first stages skip
        # their downsample so the token grid stays one per 32 px square.
        if stem_stride == 1:
            stem = nn.Conv2d(3, conv[0], 3, padding=1, bias=False)
        else:
            stem = nn.Conv2d(3, conv[0], stem_stride, stride=stem_stride, bias=False)
        self.stem = nn.Sequential(stem, nn.BatchNorm2d(conv[0]), nn.GELU())
        skip = {1: 0, 2: 1, 4: 2}[stem_stride]
        stages = []
        for i, n in enumerate(conv_depths):
            stages.append(nn.Sequential(Downsample(conv[i], conv[i + 1], 1 if i < skip else 2),
                                        *[ConvBlock(conv[i + 1]) for _ in range(n)]))
        self.stages = nn.Sequential(*stages)
        self.tokenize = nn.Conv2d(conv[len(conv_depths)], dim, 2, stride=2)
        self.pos = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.tile_head = nn.Linear(dim, embed_dim)
        self.global_head = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, H, W) RGB in [0, 1], H and W multiples of 32.
        Returns (tiles (B, H/32, W/32, E), global (B, E))."""
        x = self.stages(self.stem((x - 0.5) * 2.0))
        x = self.tokenize(x)
        x = x + self.pos(x)
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)            # (B, h*w, C)
        for block in self.blocks:
            t = block(t)
        t = self.norm(t)
        tiles = self.tile_head(t).reshape(b, h, w, -1)
        pooled = self.global_head(t.mean(dim=1))
        return tiles, pooled


def build_encoder(size: str = "tiny") -> Encoder:
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; pick one of {', '.join(SIZES)}")
    return Encoder(**SIZES[size], size=size)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
