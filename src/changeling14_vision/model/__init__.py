"""The Changeling14 encoder: a convolutional front end that turns each 32 px
square into one token, then a small transformer across the squares.

See docs/encoder.md for the input and output spec.
"""

from .encoder import EMBED_DIM, SIZES, TILE, Encoder, build_encoder

__all__ = ["EMBED_DIM", "SIZES", "TILE", "Encoder", "build_encoder"]
