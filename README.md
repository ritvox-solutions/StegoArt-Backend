# StegoArt — Backend

> Part of the StegoArt project (3 repos: `frontend`, `backend`, `ml`). This
> repo is self-contained and runs standalone — see [Clone](#clone) below.
> The [root README](../README.md) covers running the full app (backend +
> frontend) together; the `ml` repo is only needed if you want to retrain
> or experiment with the models, not to run this API.

FastAPI service that wraps the trained steganography and style-transfer
models and exposes them as an HTTP API for the frontend.

## Clone

```bash
git clone <backend-repo-url> backend
cd backend
```

This repo is self-contained and flat: this repo's own root is the app
root (no outer package wrapper), and it embeds its own copy of the model
code and trained weights under `ml/` (see [ml/README.md](ml/README.md) for
what that is) — no sibling `ml` checkout is required to run it. Every
import inside this repo is absolute, resolved against its own root, so it
runs correctly from **inside its own clone** — the same way any standard
PaaS/Docker Python app does (`cd backend && uvicorn main:app`), with no
special parent-folder requirement.

```bash
cd backend
pip install -r requirements.txt
python main.py
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/encode` | Hide a text secret in a cover image; optionally render a styled copy |
| POST | `/api/decode` | Extract a secret from a plain (unstyled) stego image |
| GET | `/api/styles` | List available style-transfer presets + thumbnail URLs |
| GET | `/api/styles/{name}/thumbnail` | PNG thumbnail for one style |
| GET | `/health` | Liveness check |

Interactive docs (Swagger UI) at `/docs` once the server is running.

## Run

```bash
# from inside this repo's own root:
pip install -r requirements.txt
python main.py
# or, with autoreload:
uvicorn main:app --reload
```

Server listens on `http://127.0.0.1:8000`. CORS defaults to
`http://localhost:5173` / `http://127.0.0.1:5173` (the frontend dev
server) — override with the `CORS_ALLOWED_ORIGINS` env var (or a `.env`
file, see `.env.example`) in production (see Deploying below). `HOST`,
`PORT`, and `RELOAD` env vars are also read by `python main.py` (defaults:
`127.0.0.1`, `8000`, `false`).

## Deploying

Any standard Python/ASGI hosting approach works, since this repo runs like
a normal flat FastAPI app (`uvicorn main:app`, run from its own root) —
Railway, Render, Fly.io, Docker, a bare VPS, etc. Config templates for a
bare VPS + your own reverse proxy are in `deploy/`:

1. **DNS**: point a subdomain (e.g. `api.yourdomain.com`, or use whatever
   your reverse proxy/tunnel — Caddy, nginx, Netbird, Cloudflare Tunnel —
   gives you) at your backend.
2. **Install the app** on the server:
   ```bash
   cd /opt && sudo mkdir stegoart && cd stegoart
   git clone <backend-repo-url> backend
   python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
   ```
3. **Config**: `cp backend/.env.example backend/.env` and edit
   `CORS_ALLOWED_ORIGINS` to your frontend's real origin. `main.py` loads
   this file automatically — no shell exports or systemd `Environment=`
   needed.
4. **HTTPS**: if you're fronting this with your own reverse proxy rather
   than a platform that handles TLS for you, `deploy/Caddyfile` is a
   minimal example (auto-provisions/renews a free Let's Encrypt cert and
   proxies to `127.0.0.1:8000`). Skip this if your proxy/tunnel already
   terminates HTTPS for you.
5. **Run the backend as a service**: copy `deploy/stegoart-backend.service`
   to `/etc/systemd/system/`, edit `WorkingDirectory` and `ExecStart` to
   match your paths, then:
   ```bash
   sudo systemctl enable --now stegoart-backend
   ```
6. **Verify**: `curl https://api.yourdomain.com/health` → `{"status":"ok"}`.
7. **Point the frontend at it**: on Vercel, set the `VITE_API_BASE_URL`
   project environment variable to your backend's public URL and redeploy
   (see `../frontend/README.md`). It must be `https://`, not `http://` — a
   Vercel page is served over HTTPS and browsers block it from calling a
   plain-HTTP API (mixed content).

**Vercel preview deployments** get random URLs
(`project-git-branch-user.vercel.app`) that won't match a fixed
`CORS_ALLOWED_ORIGINS` entry. Either add your Vercel *production* domain
only (previews just won't be able to call the API), or set
`allow_origin_regex=r"https://.*\.vercel\.app"` alongside/instead of
`allow_origins` in `main.py` if you want previews working too.

## Tests

```bash
# from inside this repo's own root:
pytest tests
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
`/api/decode`, never `styled_image_base64`. See the standalone `ml` repo's
`test_style_transfer.py` for the measured degradation.
