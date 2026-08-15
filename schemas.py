"""Pydantic request/response models for the StegoArt API.

Only response shapes live here as full models — the two multipart endpoints
(/api/encode, /api/decode) take File(...)/Form(...) parameters directly in
their router functions rather than a single body model, since FastAPI can't
bind a unified Pydantic model to multipart/form-data containing files.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SecretType(str, Enum):
    text = "text"
    image = "image"


class EncodeResponse(BaseModel):
    stego_image_base64: str = Field(description="Base64-encoded PNG of the plain (unstyled) stego image.")
    styled_image_base64: Optional[str] = Field(
        default=None, description="Base64-encoded PNG of the styled stego image, present only if apply_style was true."
    )
    psnr: float = Field(description="PSNR (dB) between cover and the UNSTYLED stego image.")
    ssim: float = Field(description="SSIM between cover and the UNSTYLED stego image.")
    styled_decode_supported: bool = Field(
        description=(
            "False whenever a styled image was returned. Style transfer destroys the hidden "
            "signal (Mode A, confirmed empirically in ml/test_style_transfer.py) — only "
            "stego_image_base64 should ever be sent to /api/decode, never styled_image_base64. "
            "The frontend should surface this rather than hardcoding the same assumption."
        )
    )


class DecodeResponse(BaseModel):
    recovered_text: Optional[str] = Field(default=None, description="Present when secret_type_hint was 'text'.")
    recovered_image_base64: Optional[str] = Field(
        default=None, description="Base64-encoded PNG, present when secret_type_hint was 'image'."
    )
    confidence: float = Field(
        description=(
            "Heuristic, UNCALIBRATED [0,1] signal — there is no ground truth at decode time, so "
            "this is not an accuracy measure. Text: mean bit-certainty of the decoded header+payload "
            "bits. Image: a Laplacian-variance sharpness/plausibility proxy."
        )
    )


class StyleInfo(BaseModel):
    name: str
    thumbnail_url: str


class StylesResponse(BaseModel):
    styles: List[StyleInfo]
