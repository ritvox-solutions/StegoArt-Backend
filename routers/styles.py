"""GET /api/styles, GET /api/styles/{style_name}/thumbnail.

The style list is sourced from StyleService.available_styles(), which in
turn comes from ml.models.list_styles() scanning ml/weights/styles/*.pth —
never a hardcoded list, so this always reflects whatever checkpoints are
actually present on disk.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ..schemas import StyleInfo, StylesResponse
from ..services.style_service import StyleService

router = APIRouter()


def get_style_service(request: Request) -> StyleService:
    return request.app.state.style_service


@router.get("/styles", response_model=StylesResponse)
async def get_styles(style_service: StyleService = Depends(get_style_service)):
    styles = [
        StyleInfo(name=name, thumbnail_url=f"/api/styles/{name}/thumbnail")
        for name in style_service.available_styles()
    ]
    return StylesResponse(styles=styles)


@router.get("/styles/{style_name}/thumbnail")
async def get_style_thumbnail(style_name: str, style_service: StyleService = Depends(get_style_service)):
    if style_name not in style_service.available_styles():
        raise HTTPException(status_code=404, detail=f"unknown style {style_name!r}")
    png_bytes = style_service.thumbnail_png_bytes(style_name)
    return Response(content=png_bytes, media_type="image/png")
