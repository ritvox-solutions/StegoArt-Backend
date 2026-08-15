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

        return EncodeResponse(
            stego_image_base64=tensor_to_base64_png(stego_tensor),
            styled_image_base64=styled_b64,
            psnr=psnr,
            ssim=ssim,
            styled_decode_supported=not apply_style,
        )
    finally:
        await cover_image.close()
        if secret_image is not None:
            await secret_image.close()
