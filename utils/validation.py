"""Upload validation: real content-based file-type checking (not just the
upload's declared filename/Content-Type) and a size limit. Raises
fastapi.HTTPException with a clean message on any violation, so router code
never needs its own try/except for these checks — and FastAPI's default
HTTPException handling means the client always gets a clean 4xx + message,
never a stack trace.
"""

import io

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


def validate_image_upload(data: bytes, field_name: str) -> None:
    """Raises HTTPException(4xx) unless `data` is a real, fully-decodable
    image of an allowed format under the size limit. Format is determined by
    Pillow actually parsing the file's content (magic bytes + structure),
    never by trusting the upload's filename extension or declared
    Content-Type.

    Deliberately uses .load() (a full pixel decode) rather than Pillow's
    .verify(), which its own docs describe as a quick structural check only
    — a file can pass .verify() and still throw partway through a real
    decode (e.g. truncated/corrupted pixel data past a valid-looking
    header). This function is the one place that risk gets caught and
    turned into a clean 400; every other image load in this app (see
    utils/preprocessing.py) can then assume the bytes it's given are good."""
    if not data:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} is {len(data) / 1e6:.1f}MB, exceeds the 10MB limit",
        )

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        # Deliberately broad: PIL's decode-failure exception types are
        # inconsistent across formats/versions (UnidentifiedImageError,
        # OSError, SyntaxError for bad PNG chunks, zlib/struct errors for
        # corrupt compressed data, ...). This is a validation boundary for
        # untrusted input whose entire job is "turn any decode failure into
        # a clean 400" — enumerating PIL's exception types here would be
        # fragile and this is exactly the place a catch-all is correct.
        raise HTTPException(status_code=400, detail=f"{field_name} is not a valid image file")

    if img.format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be one of {sorted(ALLOWED_IMAGE_FORMATS)} (detected: {img.format})",
        )


def validate_style_name(style_name, available_styles: list) -> None:
    if style_name not in available_styles:
        raise HTTPException(
            status_code=400,
            detail=f"unknown style {style_name!r}; available: {available_styles}",
        )
