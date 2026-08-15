"""Thin wrapper around ml.models.style_transfer.

Preloads every available style checkpoint once at startup (main.py's
lifespan handler constructs a single StyleService instance) and delegates
every actual call to apply_style()/list_styles()/load_style_model(), which
already own the [0,1]<->[0,255] normalization and the per-(style,device)
model cache — nothing here reimplements that logic, per the Phase 2 module's
own contract (see ml/models/style_transfer.py and
ml/weights/styles/README.md).
"""

import io

import torch
from PIL import Image

from ..ml.models import apply_style, list_styles, load_style_model

THUMBNAIL_SIZE = 128


class StyleService:
    def __init__(self, device: torch.device):
        self.device = device
        self.styles = list_styles()  # sourced from ml/weights/styles/*.pth — not hardcoded here
        for name in self.styles:
            load_style_model(name, device=device)  # warms ml.models' own cache at startup
        self._thumbnail_cache: dict = {}

    def available_styles(self) -> list:
        return list(self.styles)

    def apply(self, stego: torch.Tensor, style_name: str) -> torch.Tensor:
        """stego: (1,3,H,W) or (3,H,W) float [0,1]. Returns styled tensor, same convention."""
        if style_name not in self.styles:
            raise ValueError(f"unknown style {style_name!r}; available: {self.styles}")
        return apply_style(stego, style_name, device=self.device)

    def thumbnail_png_bytes(self, style_name: str) -> bytes:
        """A small cached preview PNG for style_name, so GET /api/styles's
        thumbnail_url links resolve to something real instead of a dead
        reference. Generated once per style (from a procedural placeholder
        image, not a project asset — so this doesn't depend on ml/data being
        present) and cached in memory for the life of the process."""
        if style_name not in self.styles:
            raise ValueError(f"unknown style {style_name!r}; available: {self.styles}")
        if style_name not in self._thumbnail_cache:
            base = _placeholder_image(THUMBNAIL_SIZE)
            styled = apply_style(base, style_name, device=self.device, size=THUMBNAIL_SIZE)
            buf = io.BytesIO()
            styled.save(buf, format="PNG")
            self._thumbnail_cache[style_name] = buf.getvalue()
        return self._thumbnail_cache[style_name]


def _placeholder_image(size: int) -> Image.Image:
    """A small procedurally-generated sample image (diagonal color gradient)
    used as the base for style-preview thumbnails."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (int(255 * x / size), int(255 * y / size), 180)
    return img
