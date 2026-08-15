"""POST /api/decode — extract a hidden secret from a (plain, unstyled) stego
image. See EncodeResponse.styled_decode_supported / Mode A: a styled image
should never be sent here — the Reveal Network can't recover a usable secret
from one (measured in ml/test_style_transfer.py).
"""

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from ..schemas import DecodeResponse, SecretType
from ..services.stego_service import StegoService, image_decode_confidence, text_decode_confidence
from ..utils.preprocessing import decode_text_secret, load_upload_as_tensor, tensor_to_base64_png
from ..utils.validation import validate_image_upload

router = APIRouter()


def get_stego_service(request: Request) -> StegoService:
    return request.app.state.stego_service


@router.post("/decode", response_model=DecodeResponse)
async def decode(
    stego_image: UploadFile = File(..., description="A plain (unstyled) stego image (JPEG/PNG/WEBP, max 10MB)."),
    secret_type_hint: SecretType = Form(...),
    stego_service: StegoService = Depends(get_stego_service),
):
    try:
        data = await stego_image.read()
        validate_image_upload(data, "stego_image")
        stego_tensor = load_upload_as_tensor(data).unsqueeze(0)

        recovered = stego_service.decode(stego_tensor)

        if secret_type_hint == SecretType.text:
            return DecodeResponse(
                recovered_text=decode_text_secret(recovered),
                recovered_image_base64=None,
                confidence=text_decode_confidence(recovered),
            )
        return DecodeResponse(
            recovered_text=None,
            recovered_image_base64=tensor_to_base64_png(recovered),
            confidence=image_decode_confidence(recovered),
        )
    finally:
        await stego_image.close()
