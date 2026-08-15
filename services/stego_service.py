"""Wraps the trained steganography encoder/decoder for inference.

StegoService loads ml.models.HidingNetwork / RevealNetwork ONCE (see
main.py's lifespan handler, which constructs a single instance at startup)
and reuses those in-memory models across every request — never re-instantiate
or reload weights per request.

Also home to the two decode-confidence heuristics: there's no ground truth
available at decode time (the caller only sends a stego image + a type
hint), so "confidence" here is necessarily a self-consistency proxy computed
from the decoder's own raw output, not an accuracy measure against anything.
Both are documented as such in schemas.DecodeResponse.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from ml.config import BLOCK_SIZE, HEADER_BITS, SECRET_MAP_GRID
from ml.models import HidingNetwork, RevealNetwork
from ml.utils.text_encoding import tensor_to_text

# Empirically calibrated against this project's own trained model (measured,
# not guessed — see backend build notes): a clean recovered photo has LOW
# Laplacian variance (~0.01-0.02, matching natural-image statistics), while
# decoding noise or a *styled* stego image (a Mode A violation) gives HIGH
# variance (~0.4+, noise-like). So confidence = 1 - variance/reference,
# clamped to [0,1] — inverted from a naive "more edges = more confident"
# assumption, which measurement showed was backwards for this model.
_IMAGE_CONFIDENCE_REFERENCE_VARIANCE = 0.4


class StegoService:
    def __init__(self, weights_dir: Path, device: torch.device):
        self.device = device

        self.encoder = HidingNetwork().to(device)
        self.encoder.load_state_dict(torch.load(weights_dir / "encoder.pth", map_location=device))
        self.encoder.eval()

        self.decoder = RevealNetwork().to(device)
        self.decoder.load_state_dict(torch.load(weights_dir / "decoder.pth", map_location=device))
        self.decoder.eval()

    @torch.no_grad()
    def encode(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """cover, secret: (1,3,H,W) float [0,1]. Returns stego, same shape."""
        return self.encoder(cover.to(self.device), secret.to(self.device))

    @torch.no_grad()
    def decode(self, stego: torch.Tensor) -> torch.Tensor:
        """stego: (1,3,H,W) float [0,1]. Returns recovered secret, same shape."""
        return self.decoder(stego.to(self.device))


def text_decode_confidence(recovered: torch.Tensor) -> float:
    """Heuristic, ground-truth-free confidence for a text-secret decode: mean
    distance-from-0.5 (rescaled to [0,1]) of the decoder's raw output over the
    header + payload bit positions actually used (per the decoded length
    header). NOT an accuracy measure. 1.0 = every relevant bit was
    unambiguous; values near 0 mean the decoder output hovered near 0.5 —
    e.g. when decoding noise or a styled/non-stego image.

    Empirically (measured against this project's own trained model, n=8
    samples across all 4 styles): clean decodes score 0.997-0.999; decodes
    of a styled stego image score 0.39-0.59. A threshold around 0.8 cleanly
    separates the two in that sample — not a guarantee, but a reasonable
    default if a frontend wants to surface a low-confidence warning."""
    t = recovered[0] if recovered.dim() == 4 else recovered
    decoded_text = tensor_to_text(t)  # canonical decode, reused as-is (not reimplemented)

    mono = t.mean(dim=0)
    blocks = mono.view(SECRET_MAP_GRID, BLOCK_SIZE, SECRET_MAP_GRID, BLOCK_SIZE)
    grid = blocks.mean(dim=(1, 3)).flatten()

    used_bits = min(HEADER_BITS + len(decoded_text) * 8, grid.numel())
    if used_bits == 0:
        return 0.0
    certainty = (grid[:used_bits] - 0.5).abs() * 2
    return float(certainty.mean().clamp(0, 1))


def image_decode_confidence(recovered: torch.Tensor) -> float:
    """Heuristic, ground-truth-free confidence for an image-secret decode: a
    Laplacian-variance-based proxy, squashed to [0,1]. NOT an accuracy
    measure, and NOT simply "sharper is more confident" — measured against
    this project's own model, a clean recovered photo has LOW local
    variance (it looks like a normal photo), while decoding noise or a
    *styled* stego image produces a HIGH-variance, noise-like output. So low
    variance -> high confidence here, the opposite of a naive sharpness
    score. Known blind spot: decoding a real-but-wrong image (e.g. a
    non-stego photo) can still land at a middling-to-high score, since its
    output is photo-like even though it isn't the actual hidden secret —
    this heuristic mainly catches noise-like failures (styled/corrupted
    input), not "is this the right image."""
    t = recovered[0] if recovered.dim() == 4 else recovered
    gray = t.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=t.device).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, kernel, padding=1)
    variance = laplacian.var().item()
    return float(max(0.0, min(1.0, 1.0 - variance / _IMAGE_CONFIDENCE_REFERENCE_VARIANCE)))
