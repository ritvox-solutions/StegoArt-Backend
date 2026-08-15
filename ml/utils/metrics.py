"""PSNR / SSIM / text-accuracy metrics used during training and evaluation."""

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy_hwc(tensor: torch.Tensor):
    """(3,H,W) float tensor in [0,1] -> (H,W,3) numpy array in [0,1]."""
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def psnr(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """PSNR between two (3,H,W) float tensors in [0,1]."""
    a, b = _to_numpy_hwc(img_a), _to_numpy_hwc(img_b)
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def ssim(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """SSIM between two (3,H,W) float tensors in [0,1]."""
    a, b = _to_numpy_hwc(img_a), _to_numpy_hwc(img_b)
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=2))


def batch_psnr(batch_a: torch.Tensor, batch_b: torch.Tensor) -> float:
    """Mean PSNR over a (N,3,H,W) batch pair."""
    return sum(psnr(a, b) for a, b in zip(batch_a, batch_b)) / batch_a.shape[0]


def batch_ssim(batch_a: torch.Tensor, batch_b: torch.Tensor) -> float:
    """Mean SSIM over a (N,3,H,W) batch pair."""
    return sum(ssim(a, b) for a, b in zip(batch_a, batch_b)) / batch_a.shape[0]


def char_accuracy(original: str, recovered: str) -> float:
    """Fraction of matching characters, position-wise, against the longer
    string's length (so truncation/extra chars are penalized, not ignored)."""
    if not original and not recovered:
        return 1.0
    n = max(len(original), len(recovered))
    matches = sum(1 for i in range(min(len(original), len(recovered))) if original[i] == recovered[i])
    return matches / n
