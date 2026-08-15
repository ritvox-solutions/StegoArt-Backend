# StegoArt — Backend

> Part of the StegoArt project (3 separate repos: `frontend`, `backend`,
> `ml`). For clone/setup instructions covering all three, see the
> [root README](../README.md).

FastAPI service that wraps the trained steganography and style-transfer
models from `../ml` and exposes them as an HTTP API for the frontend.

## Clone

```bash
git clone <backend-repo-url> backend
```

Clone this alongside the `ml` repo (and `frontend`, if you want the UI too)
under one parent folder — **not** as a subfolder of `ml` or vice versa:

```
stegoart/
├── frontend/
├── backend/   <- this repo
└── ml/
```

This repo imports `ml` as a sibling Python package at runtime (see
"Important: directory layout" below), so it won't run standalone without
the `ml` repo cloned next to it. Full multi-repo setup:
[root README](../README.md).

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/encode` | Hide a text or image secret in a cover image; optionally render a styled copy |
| POST | `/api/decode` | Extract a secret from a plain (unstyled) stego image |
| GET | `/api/styles` | List available style-transfer presets + thumbnail URLs |
| GET | `/api/styles/{name}/thumbnail` | PNG thumbnail for one style |
| GET | `/health` | Liveness check |

Interactive docs (Swagger UI) at `/docs` once the server is running.

## Important: directory layout

This service loads model weights from `../ml/weights` and imports Python
modules from the `ml` package directly (`ml.models`, `ml.utils`, ...). It
**must be run with the project root (the parent of both `backend/` and
`ml/`) as the working directory / on `sys.path`**, not from inside
`backend/` itself — both `backend` and `ml` are imported as top-level
packages.

```bash
# from the steganography/ project root:
pip install -r backend/requirements.txt
python -m backend.main
# or, with autoreload:
uvicorn backend.main:app --reload
```

Server listens on `http://127.0.0.1:8000`. CORS is restricted to
`http://localhost:5173` / `http://127.0.0.1:5173` (the frontend dev server).

## Tests

```bash
# from the project root:
pytest backend/tests
```

15 tests covering encode/decode/styles, validation, and error handling.
Model weights load once per test session (see `tests/conftest.py`), not
per-test.

## Models loaded at startup

Both the steganography encoder/decoder and every style checkpoint under
`ml/weights/styles/*.pth` are loaded once in the FastAPI `lifespan` handler
(`main.py`) and kept in `app.state` — no per-request reloading. Runs on GPU
automatically if `torch.cuda.is_available()`, otherwise CPU.

## Mode A: styling breaks decoding

`EncodeResponse.styled_decode_supported` is `False` whenever a styled image
is returned — style transfer overwrites the same pixel-level pattern the
Reveal Network depends on. Only ever send `stego_image_base64` to
`/api/decode`, never `styled_image_base64`. See `ml/test_style_transfer.py`
for the measured degradation.
