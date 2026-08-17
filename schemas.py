"""Pydantic request/response models for the StegoArt API.

Only response shapes live here as full models — the two multipart endpoints
(/api/encode, /api/decode) take File(...)/Form(...) parameters directly in
their router functions rather than a single body model, since FastAPI can't
bind a unified Pydantic model to multipart/form-data containing files.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class EncodeResponse(BaseModel):
    stego_image_base64: str = Field(description="Base64-encoded PNG of the plain (unstyled) stego image.")
    styled_image_base64: Optional[str] = Field(
        default=None, description="Base64-encoded PNG of the styled stego image, present only if apply_style was true."
    )
    psnr: float = Field(description="PSNR (dB) between cover and the UNSTYLED stego image.")
    ssim: float = Field(description="SSIM between cover and the UNSTYLED stego image.")
    styled_decode_supported: bool = Field(
        description=(
            "Whether styled_image_base64 (when present) can reasonably be sent to /api/decode. "
            "True whenever no style was applied (styled_image_base64 is None, so the question is "
            "moot). As of the style-robust ('v3') model (see ml/README.md's style-robust training "
            "experiment), also true under most conditions — styled-image text recovery is now "
            "usually accurate (measured >=97% char accuracy through ~200 characters for candy/"
            "mosaic/rain_princess/udnie alike). One carve-out: with style_name='udnie' specifically, "
            "accuracy collapses for longer text (79% at 250 chars, down to ~6% near the 510-char "
            "max) while the other 3 styles stay strong (75-100%) even at max length — so this is "
            "false for udnie + text longer than ~200 characters. Check DecodeResponse.confidence "
            "for a per-decode signal either way. The frontend should surface this rather than "
            "hardcoding the same assumption."
        )
    )


class DecodeResponse(BaseModel):
    recovered_text: str = Field(description="The decoded secret message.")
    confidence: float = Field(
        description=(
            "Heuristic, UNCALIBRATED [0,1] signal — there is no ground truth at decode time, so "
            "this is not an accuracy measure. Mean bit-certainty of the decoded header+payload bits."
        )
    )


class StyleInfo(BaseModel):
    name: str
    thumbnail_url: str


class StylesResponse(BaseModel):
    styles: List[StyleInfo]
