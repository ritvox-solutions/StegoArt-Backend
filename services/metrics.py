"""PSNR/SSIM for the encode response.

Thin wrapper around ml.utils.metrics (scikit-image under the hood) — not a
reimplementation. Always compares cover vs. the UNSTYLED stego image, per
Mode A: that's the meaningful signal-preservation comparison. A styled image
is expected to differ heavily from the cover by design, so it's never used
here even when apply_style was requested.
"""

import torch

from ..ml.utils.metrics import psnr as _psnr
from ..ml.utils.metrics import ssim as _ssim


def cover_stego_metrics(cover: torch.Tensor, stego: torch.Tensor) -> tuple:
    """cover, stego: (1,3,H,W) or (3,H,W) float [0,1] tensors. Returns (psnr, ssim)."""
    c = cover[0] if cover.dim() == 4 else cover
    s = stego[0] if stego.dim() == 4 else stego
    return _psnr(c, s), _ssim(c, s)
