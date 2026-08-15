"""StegoArt backend — FastAPI app entrypoint.

Loads the trained steganography encoder/decoder and every available style
checkpoint ONCE, in the lifespan handler below, then serves /api/encode,
/api/decode, /api/styles for the rest of the process's life. Nothing in this
app re-loads weights per request — see services/stego_service.py and
services/style_service.py.

Self-contained: model code and weights live under ./ml (an embedded copy of
the standalone `ml` repo's inference-relevant modules, kept in sync
manually — see ml/README.md in this repo). No sibling `ml` checkout needed.

Run (from the folder CONTAINING this repo's clone, so `backend` resolves as
a package):
    python -m backend.main
or, for autoreload during development:
    uvicorn backend.main:app --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import decode, encode, styles
from .services.stego_service import StegoService
from .services.style_service import StyleService

BACKEND_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = BACKEND_ROOT / "ml" / "weights"

# Comma-separated list of allowed browser origins, e.g.
#   CORS_ALLOWED_ORIGINS="https://stegoart.vercel.app,https://stegoart.com"
# Defaults to the local Vite dev server so `npm run dev` keeps working
# out of the box. Set this env var in production instead of editing code.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] loading models on device={device} ...")

    app.state.device = device
    app.state.stego_service = StegoService(WEIGHTS_DIR, device)
    app.state.style_service = StyleService(device)

    print(f"[startup] ready — styles available: {app.state.style_service.available_styles()}")
    yield
    # nothing to explicitly release — process exit frees the in-memory models


app = FastAPI(
    title="StegoArt API",
    description="CNN-autoencoder image steganography, with optional feed-forward neural style transfer.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Last-resort safety net so an unexpected bug never leaks a raw traceback
    # to the client. Explicit HTTPException calls throughout the app (4xx,
    # with a clean message) are handled by FastAPI's own default machinery
    # and never reach this handler.
    print(f"[unhandled] {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


app.include_router(encode.router, prefix="/api", tags=["encode"])
app.include_router(decode.router, prefix="/api", tags=["decode"])
app.include_router(styles.router, prefix="/api", tags=["styles"])


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1 by default: in production this process sits behind a reverse
    # proxy (Caddy/nginx) on the same machine, which is the thing actually
    # exposed to the internet — the backend itself never needs to bind
    # 0.0.0.0. Override via env vars if your setup differs.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)
