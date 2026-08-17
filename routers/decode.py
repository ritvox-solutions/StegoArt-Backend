"""POST /api/decode — extract a hidden text secret from a stego image. Works
on plain (unstyled) images always; also works on styled images under most
conditions (see EncodeResponse.styled_decode_supported and ml/README.md's
style-robust training experiment) — check that field rather than assuming
either way, since it varies by style and (for udnie specifically) text
length.
"""

from fastapi import APIRouter, Depends, File, Request, UploadFile

from schemas import DecodeResponse
from services.stego_service import StegoService, text_decode_confidence
from utils.preprocessing import decode_text_secret, load_upload_as_tensor
from utils.validation import validate_image_upload

router = APIRouter()


def get_stego_service(request: Request) -> StegoService:
    return request.app.state.stego_service


@router.post("/decode", response_model=DecodeResponse)
async def decode(
    stego_image: UploadFile = File(..., description="A plain (unstyled) stego image (JPEG/PNG/WEBP, max 10MB)."),
    stego_service: StegoService = Depends(get_stego_service),
):
    try:
        data = await stego_image.read()
        validate_image_upload(data, "stego_image")
        stego_tensor = load_upload_as_tensor(data).unsqueeze(0)

        recovered = stego_service.decode(stego_tensor)

        return DecodeResponse(
            recovered_text=decode_text_secret(recovered),
            confidence=text_decode_confidence(recovered),
        )
    finally:
        await stego_image.close()
