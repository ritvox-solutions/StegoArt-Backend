"""Wraps the trained steganography encoder/decoder for inference.

StegoService loads TWO ml.models.HidingNetwork / RevealNetwork pairs ONCE
(see main.py's lifespan handler, which constructs a single instance at
startup) and reuses those in-memory models across every request — never
re-instantiate or reload weights per request.

Two pairs, not one, because a single style-robust model can't be good at
both: text secrets need the "v3" weights (encoder.pth/decoder.pth — trained
on mixed text+image secrets, reliable styled-text recovery for 3-4 styles),
while image secrets need a separate model trained EXCLUSIVELY on image
secrets (encoder_image.pth/decoder_image.pth) to get meaningfully better
styled-image recovery (SSIM ~0.31 -> ~0.42, closing the udnie-specific gap)
— see ml/README.md's style-robust training experiment. The image-dedicated
model was verified to have LOST v3's styled-text robustness (8-22% char
accuracy vs v3's 92-96%), confirming these can't be merged into one model;
StegoService routes by secret_type so callers get whichever pair is
actually good at their request instead of a compromise.

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

# Originally calibrated against the pre-v3 (clean-only) weights: a clean
# recovered photo had LOW Laplacian variance (~0.01-0.02, matching natural-
# image statistics), while decoding noise or a styled stego image (a Mode A
# violation, back when styling reliably broke recovery) gave HIGH variance
# (~0.4+, noise-like). So confidence = 1 - variance/reference, clamped to
# [0,1] — inverted from a naive "more edges = more confident" assumption.
#
# STILL A BLIND SPOT with the current image-secret-dedicated model
# (encoder_image.pth/decoder_image.pth): measured clean=0.987, styled=
# 0.97-0.99 across all 4 styles — i.e. this heuristic reports high
# confidence essentially always, clean or styled, good or degraded. That's
# NOT because nothing improved (styled image-secret SSIM did go from ~0.31
# to ~0.42 with this dedicated model — genuinely more recognizable, see
# ml/README.md) — it's that this heuristic's underlying signal (local pixel
# variance) stays low/photo-like for this model's output regardless of
# actual fidelity, so it never tracked "styled vs clean" the way early,
# rougher models happened to. This is exactly why
# EncodeResponse.styled_decode_supported is unconditionally False for image
# secrets regardless of what this heuristic reports — don't rely on this
# score alone to gate that decision.
_IMAGE_CONFIDENCE_REFERENCE_VARIANCE = 0.4


class StegoService:
    def __init__(self, weights_dir: Path, device: torch.device):
        self.device = device
        self.encoder_text, self.decoder_text = self._load_pair(weights_dir, "encoder.pth", "decoder.pth")
        self.encoder_image, self.decoder_image = self._load_pair(
            weights_dir, "encoder_image.pth", "decoder_image.pth"
        )

    def _load_pair(self, weights_dir: Path, encoder_file: str, decoder_file: str):
        encoder = HidingNetwork().to(self.device)
        encoder.load_state_dict(torch.load(weights_dir / encoder_file, map_location=self.device))
        encoder.eval()

        decoder = RevealNetwork().to(self.device)
        decoder.load_state_dict(torch.load(weights_dir / decoder_file, map_location=self.device))
        decoder.eval()
        return encoder, decoder

    def _pair_for(self, secret_type: str):
        return (self.encoder_image, self.decoder_image) if secret_type == "image" else (self.encoder_text, self.decoder_text)

    @torch.no_grad()
    def encode(self, cover: torch.Tensor, secret: torch.Tensor, secret_type: str) -> torch.Tensor:
        """cover, secret: (1,3,H,W) float [0,1]. secret_type: "text" or "image",
        selects which trained pair does the embedding. Returns stego, same shape."""
        encoder, _ = self._pair_for(secret_type)
        return encoder(cover.to(self.device), secret.to(self.device))

    @torch.no_grad()
    def decode(self, stego: torch.Tensor, secret_type: str) -> torch.Tensor:
        """stego: (1,3,H,W) float [0,1]. secret_type: "text" or "image", MUST
        match whatever was passed to encode() for this stego image — using the
        wrong pair's decoder on the other type's stego is unsupported (the two
        pairs' encoders embed differently). Returns recovered secret, same shape."""
        _, decoder = self._pair_for(secret_type)
        return decoder(stego.to(self.device))


def text_decode_confidence(recovered: torch.Tensor) -> float:
    """Heuristic, ground-truth-free confidence for a text-secret decode: mean
    distance-from-0.5 (rescaled to [0,1]) of the decoder's raw output over the
    header + payload bit positions actually used (per the decoded length
    header). NOT an accuracy measure. 1.0 = every relevant bit was
    unambiguous; values near 0 mean the decoder output hovered near 0.5 —
    e.g. when decoding noise or a styled/non-stego image.

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


def image_decode_confidence(recovered: torch.Tensor) -> float:
    """Heuristic, ground-truth-free confidence for an image-secret decode: a
    Laplacian-variance-based proxy, squashed to [0,1]. NOT an accuracy
    measure, and NOT simply "sharper is more confident": a clean recovered
    photo has LOW local variance (it looks like a normal photo), so low
    variance -> high confidence here, the opposite of a naive sharpness
    score.

    Known blind spots: (1) decoding a real-but-wrong image (e.g. a
    non-stego photo) can still land at a middling-to-high score, since its
    output is photo-like even though it isn't the actual hidden secret.
    (2) With the current image-secret-dedicated model, this ALSO scores a
    styled-image decode as high-confidence (~0.97-0.99 measured, barely
    below clean's ~0.99) even though styled recovery, while now more
    recognizable than before (SSIM ~0.42 — see _IMAGE_CONFIDENCE_REFERENCE_
    VARIANCE's comment and ml/README.md), still isn't reliable. This
    heuristic mainly catches noise-like failures, not "is this the right
    image" or "was this actually styled" — don't treat a high score here as
    proof a styled image-secret decode is trustworthy."""
    t = recovered[0] if recovered.dim() == 4 else recovered
    gray = t.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=t.device).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, kernel, padding=1)
    variance = laplacian.var().item()
    return float(max(0.0, min(1.0, 1.0 - variance / _IMAGE_CONFIDENCE_REFERENCE_VARIANCE)))
