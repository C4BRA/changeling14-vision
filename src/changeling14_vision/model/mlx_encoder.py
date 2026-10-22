"""The encoder for MLX (Apple silicon), for inference.

Loads the weights written by `changeling14-vision export --formats mlx`:
batch norms are folded into the convolutions next to them and convolution
weights are already in MLX's (out, kh, kw, in) layout. Needs only `mlx`, not
PyTorch, so a bot can run it on its own:

    from changeling14_vision.model.mlx_encoder import MLXEncoder
    enc = MLXEncoder.load("export/step-0020000/changeling14-vision-tiny-p32-mlx")
    tiles, pooled = enc(frame)      # frame: (B, H, W, 3) float in [0, 1], NHWC

The export folder carries a copy of this file, so it also works without the
changeling14-vision package installed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx


def _gelu(x):
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2.0)))


def _layer_norm(x, w, b, eps=1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps) * w + b


class MLXEncoder:
    def __init__(self, weights: dict[str, mx.array], config: dict):
        self.w = weights
        self.config = config

    @classmethod
    def load(cls, folder: str | Path) -> "MLXEncoder":
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        return cls(mx.load(str(folder / "weights.safetensors")), config)

    def _conv(self, x, name, stride=1, padding=0, groups=1):
        y = mx.conv2d(x, self.w[f"{name}.weight"], stride=stride, padding=padding, groups=groups)
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def _linear(self, x, name):
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """x: (B, H, W, 3) RGB in [0, 1], H and W multiples of 32.
        Returns (tiles (B, H/32, W/32, E), global (B, E))."""
        c = self.config
        x = _gelu(self._conv((x - 0.5) * 2.0, "stem", padding=1))
        for i, n in enumerate(c["conv_depths"]):
            x = self._conv(x, f"stages.{i}.0", stride=2)
            ch = c["conv"][i + 1]
            for j in range(1, n + 1):
                p = f"stages.{i}.{j}"
                x = x + self._conv(x, f"{p}.dw", padding=1, groups=ch)
                x = x + self._conv(_gelu(self._conv(x, f"{p}.fc1")), f"{p}.fc2")
        x = self._conv(x, "tokenize", stride=2)
        x = x + self._conv(x, "pos", padding=1, groups=c["dim"])
        b, h, w, d = x.shape
        t = x.reshape(b, h * w, d)
        heads = c["heads"]
        for k in range(c["depth"]):
            p = f"blocks.{k}"
            y = _layer_norm(t, self.w[f"{p}.norm1.weight"], self.w[f"{p}.norm1.bias"])
            qkv = self._linear(y, f"{p}.attn.qkv").reshape(b, h * w, 3, heads, d // heads).transpose(2, 0, 3, 1, 4)
            q, kk, v = qkv[0], qkv[1], qkv[2]
            y = mx.fast.scaled_dot_product_attention(q, kk, v, scale=1.0 / math.sqrt(d // heads))
            t = t + self._linear(y.transpose(0, 2, 1, 3).reshape(b, h * w, d), f"{p}.attn.proj")
            y = _layer_norm(t, self.w[f"{p}.norm2.weight"], self.w[f"{p}.norm2.bias"])
            t = t + self._linear(_gelu(self._linear(y, f"{p}.fc1")), f"{p}.fc2")
        t = _layer_norm(t, self.w["norm.weight"], self.w["norm.bias"])
        tiles = self._linear(t, "tile_head").reshape(b, h, w, -1)
        pooled = self._linear(mx.mean(t, axis=1), "global_head")
        return tiles, pooled
