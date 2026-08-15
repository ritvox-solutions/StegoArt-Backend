"""Preprocessing helpers for cover and secret images.

Cover images and image-type secrets share the exact same normalization
scheme (resize to IMG_SIZE, [0,1] float tensor) so a single pair of
functions covers both.
"""

from PIL import Image
import torch
import torchvision.transforms as T

from ..config import IMG_SIZE

_to_tensor = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),  # -> [0,1] float32, shape (3, IMG_SIZE, IMG_SIZE)
    ]
)
_to_pil = T.ToPILImage()


def load_image_as_tensor(path_or_pil) -> torch.Tensor:
    """Load an image (file path or PIL.Image), resize to IMG_SIZE, and
    return a (3, IMG_SIZE, IMG_SIZE) float32 tensor in [0,1]."""
    img = path_or_pil if isinstance(path_or_pil, Image.Image) else Image.open(path_or_pil)
    img = img.convert("RGB")
    return _to_tensor(img)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Inverse of load_image_as_tensor: (3,H,W) float tensor in [0,1] -> PIL.Image."""
    return _to_pil(tensor.detach().cpu().clamp(0, 1))
