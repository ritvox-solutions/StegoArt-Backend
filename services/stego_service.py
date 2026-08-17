"""Wraps the trained steganography encoder/decoder for inference.

StegoService loads ml.models.HidingNetwork / RevealNetwork ONCE (see
main.py's lifespan handler, which constructs a single instance at startup)
and reuses those in-memory models across every request — never
re-instantiate or reload weights per request.

Uses the "v3" style-robust weights (encoder.pth/decoder.pth) — fine-tuned
with a frozen style-transfer network wired in as a differentiable
distortion layer, so styled-image text recovery is reliable for 3-4 of the
4 styles instead of collapsing to random guessing — see ml/README.md's
style-robust training experiment.

Also home to the decode-confidence heuristic: there's no ground truth
available at decode time (the caller only sends a stego image), so
"confidence" here is necessarily a self-consistency proxy computed from the
decoder's own raw output, not an accuracy measure. Documented as such in
schemas.DecodeResponse.
"""

from pathlib import Path

import torch

from ml.config import BLOCK_SIZE, HEADER_BITS, SECRET_MAP_GRID
from ml.models import HidingNetwork, RevealNetwork
from ml.utils.text_encoding import tensor_to_text


class StegoService:
    def __init__(self, weights_dir: Path, device: torch.device):
        self.device = device

        self.encoder = HidingNetwork().to(self.device)
        self.encoder.load_state_dict(torch.load(weights_dir / "encoder.pth", map_location=self.device))
        self.encoder.eval()

        self.decoder = RevealNetwork().to(self.device)
        self.decoder.load_state_dict(torch.load(weights_dir / "decoder.pth", map_location=self.device))
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
    e.g. when decoding noise or a non-stego image.

    Empirically, with the current style-robust ("v3") weights (see
    ml/README.md's style-robust training experiment): clean decodes still
    score ~0.997-0.999, and — because v3 was specifically trained to make
    styled decodes bit-certain, not just accurate — *styled* decodes now
    typically also score high, ~0.92-0.95 across all 4 styles, INCLUDING
    udnie, whose actual measured aggregate accuracy (~38% char accuracy) is
    much worse than candy/mosaic/rain_princess's (~92-96%). That's a real
    blind spot: this heuristic reads bit-certainty, and v3 apparently
    produces confident-looking-but-sometimes-wrong bits for udnie specifically,
    so a high score here does NOT reliably distinguish "styled but fine" from
    "styled and wrong" the way it did against the pre-v3 model (clean
    ~0.998 vs styled ~0.39-0.59, cleanly separated by an 0.8 threshold —
    no longer true). Treat this purely as "how sure the decoder itself
    seemed," not as a robustness or style-specific accuracy signal."""
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
