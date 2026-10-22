"""Draw each entity the way the game shows it on the floor.

SS14 builds most items from several layers, some of them grey and tinted
by a colour in the prototype (a black jumpsuit is a grey jumpsuit layer
tinted #3f3f3f, plus a trinkets layer on top). Cutting layers one by one
loses that: a lamp glow or a welder flame on its own is not the item.

compose() stacks an entity's visible layers in order, first direction and
first frame, centred like the game does, with the layer and Sprite colours
applied. When the result is just one untinted layer, the cut image already
is the item and no new picture is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageColor

from .prototypes import SpriteRef
from .rsi import Sheet, cut_state

DRAWN = ("sprite", "layer")


def parse_color(value: object) -> tuple[int, int, int, int] | None:
    """'#3f3f3f', '#3f3f3f80', 'red', 'LightGray' -> RGBA; None if unreadable or white."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        rgba = ImageColor.getcolor(value.strip(), "RGBA")
    except ValueError:
        return None
    return None if rgba == (255, 255, 255, 255) else rgba


def _tint(img: Image.Image, colours: list[tuple[int, int, int, int]]) -> Image.Image:
    for colour in colours:
        bands = [band.point(lambda v, c=c: v * c // 255) for band, c in zip(img.split(), colour)]
        img = Image.merge("RGBA", bands)
    return img


@dataclass
class Composite:
    image: Image.Image
    layers: list[tuple[str, str]]   # (sheet, state) drawn, bottom first
    tinted: bool

    @property
    def simple(self) -> bool:
        """One untinted layer: the cut image of that state is already the item."""
        return len(self.layers) == 1 and not self.tinted


class Composer:
    def __init__(self, sheets: dict[str, Sheet]):
        self.sheets = sheets
        self._cells: dict[tuple[str, str], Image.Image | None] = {}

    def cell(self, rsi: str, state: str) -> Image.Image | None:
        """Direction 0, frame 0 of a state."""
        key = (rsi, state)
        if key not in self._cells:
            sheet = self.sheets[rsi]
            try:
                self._cells[key] = next(cut_state(sheet, sheet.states[state]))[2]
            except (OSError, ValueError, StopIteration):
                self._cells[key] = None
        return self._cells[key]

    def compose(self, refs: list[tuple[SpriteRef, str]]) -> Composite | None:
        """refs: the entity's sprite refs with their link status (build.classify).
        None when the bottom layer can't be drawn (picked at runtime, missing,
        or from an excluded sheet): without it the picture would mislead."""
        drawn = [(ref, status) for ref, status in refs if ref.role in DRAWN]
        if not drawn:
            return None
        cells, layers, tinted = [], [], False
        for n, (ref, status) in enumerate(drawn):
            usable = ref.visible and status == "clean" and ref.state is not None
            img = self.cell(ref.rsi, ref.state) if usable else None
            if img is None:
                if n == 0:
                    return None
                continue
            colours = [c for c in (parse_color(ref.color), parse_color(ref.sprite_color)) if c]
            if colours:
                img, tinted = _tint(img, colours), True
            cells.append(img)
            layers.append((ref.rsi, ref.state))
        width, height = max(c.width for c in cells), max(c.height for c in cells)
        canvas = Image.new("RGBA", (width, height))
        for img in cells:
            canvas.alpha_composite(img, ((width - img.width) // 2, (height - img.height) // 2))
        if canvas.getchannel("A").getbbox() is None:
            return None
        return Composite(canvas, layers, tinted)
