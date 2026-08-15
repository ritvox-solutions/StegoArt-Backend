"""Image <-> tensor conversion and text<->tensor encoding for the backend.

Built entirely on top of the Phase 1 ml package — this module does not
reimplement resizing, [0,1] normalization, or the text/bit-tensor scheme; it
adapts ml.utils' functions to the bytes-in / base64-out shape HTTP requests
and responses need, and adds EXIF stripping on both the way in and out.
"""

import base64
import io

import torch
from PIL import Image

from ..ml.utils.image_utils import load_image_as_tensor, tensor_to_pil
from ..ml.utils.text_encoding import tensor_to_text, text_to_tensor

__all__ = [
    "strip_exif",
    "load_upload_as_tensor",
    "tensor_to_base64_png",
    "encode_text_secret",
    "decode_text_secret",
]


def strip_exif(image: Image.Image) -> Image.Image:
    """Returns a copy of `image` holding only pixel data — no EXIF or any
    other metadata dict. Applied to every image coming in (an upload) and
    going out (a response), so nothing from the original file, or from any
    intermediate PIL operation, leaks into what we send back."""
    return Image.frombytes(image.mode, image.size, image.tobytes())


def load_upload_as_tensor(data: bytes) -> torch.Tensor:
    """Raw image bytes -> (3, IMG_SIZE, IMG_SIZE) float [0,1] tensor, EXIF stripped."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = strip_exif(img)
    return load_image_as_tensor(img)


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """(3,H,W) or (1,3,H,W) float [0,1] tensor -> EXIF-free base64-encoded PNG string."""
    t = tensor[0] if tensor.dim() == 4 else tensor
    img = strip_exif(tensor_to_pil(t))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_text_secret(text: str) -> torch.Tensor:
    """Validates against MAX_TEXT_CHARS and 8-bit-ASCII encodability (via
    ml.utils.text_encoding's own checks — not duplicated here) and returns a
    (3, IMG_SIZE, IMG_SIZE) tensor. Raises ValueError with the original
    message on violation; callers translate that to an HTTP 422."""
    return text_to_tensor(text)


def decode_text_secret(tensor: torch.Tensor) -> str:
    """(3,H,W) or (1,3,H,W) recovered-secret tensor -> decoded ASCII string."""
    t = tensor[0] if tensor.dim() == 4 else tensor
    return tensor_to_text(t)
