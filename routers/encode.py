"""POST /api/encode — hide a text or image secret inside a cover image,
optionally rendering an additional styled copy.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from schemas import EncodeResponse, SecretType
from services.metrics import cover_stego_metrics
from services.stego_service import StegoService
from services.style_service import StyleService
from utils.preprocessing import encode_text_secret, load_upload_as_tensor, tensor_to_base64_png
from utils.validation import validate_image_upload, validate_style_name

router = APIRouter()

# Measured (see ml/README.md's style-robust training experiment, "udnie
# length sensitivity" note): with the v3 weights, udnie-styled text recovery
# stays strong (>=97% char accuracy) through ~200 chars, then falls off a
# cliff — 79% at 250, down to 6% by 480 (near MAX_TEXT_CHARS). The other 3
# styles (candy/mosaic/rain_princess) stay strong (75-100%) even near max
# length, so this carve-out is udnie-specific, not a general length limit.
_UDNIE_SAFE_TEXT_CHARS = 200


def get_stego_service(request: Request) -> StegoService:
    return request.app.state.stego_service


def get_style_service(request: Request) -> StyleService:
    return request.app.state.style_service


@router.post("/encode", response_model=EncodeResponse)
async def encode(
    cover_image: UploadFile = File(..., description="Cover image (JPEG/PNG/WEBP, max 10MB)."),
    secret_type: SecretType = Form(...),
    secret_text: Optional[str] = Form(None, description="Required when secret_type is 'text'."),
    secret_image: Optional[UploadFile] = File(None, description="Required when secret_type is 'image'."),
    apply_style: bool = Form(False),
    style_name: Optional[str] = Form(
        None, description="One of: candy, mosaic, rain_princess, udnie. Required when apply_style is true."
    ),
    stego_service: StegoService = Depends(get_stego_service),
    style_service: StyleService = Depends(get_style_service),
):
    try:
        cover_bytes = await cover_image.read()
        validate_image_upload(cover_bytes, "cover_image")
        cover_tensor = load_upload_as_tensor(cover_bytes).unsqueeze(0)

        if secret_type == SecretType.text:
            if not secret_text:
                raise HTTPException(status_code=422, detail="secret_text is required when secret_type is 'text'")
            try:
                secret_tensor = encode_text_secret(secret_text).unsqueeze(0)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
        else:
            if secret_image is None:
                raise HTTPException(status_code=422, detail="secret_image is required when secret_type is 'image'")
            secret_bytes = await secret_image.read()
            validate_image_upload(secret_bytes, "secret_image")
            secret_tensor = load_upload_as_tensor(secret_bytes).unsqueeze(0)

        if apply_style:
            if not style_name:
                raise HTTPException(status_code=422, detail="style_name is required when apply_style is true")
            validate_style_name(style_name, style_service.available_styles())

        stego_tensor = stego_service.encode(cover_tensor, secret_tensor)
        psnr, ssim = cover_stego_metrics(cover_tensor, stego_tensor)

        styled_b64 = None
        if apply_style:
            styled_tensor = style_service.apply(stego_tensor, style_name)
            styled_b64 = tensor_to_base64_png(styled_tensor)

        # v3 (style-robust) weights make styled-image text recovery usually accurate,
        # but styled-image *image*-secret recovery is still unreliable, and udnie
        # specifically collapses on long text — see EncodeResponse.styled_decode_
        # supported's Field description and _UDNIE_SAFE_TEXT_CHARS above.
        styled_decode_supported = (not apply_style) or (
            secret_type == SecretType.text
            and not (style_name == "udnie" and len(secret_text) > _UDNIE_SAFE_TEXT_CHARS)
        )

        return EncodeResponse(
            stego_image_base64=tensor_to_base64_png(stego_tensor),
            styled_image_base64=styled_b64,
            psnr=psnr,
            ssim=ssim,
            styled_decode_supported=styled_decode_supported,
        )
    finally:
        await cover_image.close()
        if secret_image is not None:
            await secret_image.close()
