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
```

This repo is self-contained: it embeds its own copy of the model code and
trained weights under `ml/` (see [ml/README.md](ml/README.md) for what
that is and how it relates to the standalone `ml` repo) — no sibling `ml`
checkout is required to run it.

The one layout requirement is that `backend/` needs to be importable as a
Python package, so run it from **the folder containing this clone**, not
from inside the clone itself:

```
somewhere/
└── backend/   <- this repo clones here
```

```bash
cd somewhere/
pip install -r backend/requirements.txt
python -m backend.main
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/encode` | Hide a text or image secret in a cover image; optionally render a styled copy |
| POST | `/api/decode` | Extract a secret from a plain (unstyled) stego image |
| GET | `/api/styles` | List available style-transfer presets + thumbnail URLs |
| GET | `/api/styles/{name}/thumbnail` | PNG thumbnail for one style |
| GET | `/health` | Liveness check |

Interactive docs (Swagger UI) at `/docs` once the server is running.

## Run

```bash
# from the folder containing this clone:
pip install -r backend/requirements.txt
python -m backend.main
# or, with autoreload:
uvicorn backend.main:app --reload
```

Server listens on `http://127.0.0.1:8000`. CORS defaults to
`http://localhost:5173` / `http://127.0.0.1:5173` (the frontend dev
server) — override with the `CORS_ALLOWED_ORIGINS` env var in production
(see Deploying below). `HOST`, `PORT`, and `RELOAD` env vars are also
read by `python -m backend.main` (defaults: `127.0.0.1`, `8000`, `false`).

## Deploying (VPS + domain, frontend on Vercel)

Config templates for this are in `deploy/`. Summary:

1. **DNS**: point a subdomain (e.g. `api.yourdomain.com`) at your server's
   public IP with an A/AAAA record.
2. **Install the app** on the server:
   ```bash
   cd /opt && sudo mkdir stegoart && cd stegoart
   git clone <backend-repo-url> backend
   python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
   ```
3. **HTTPS via Caddy** (auto-provisions/renews a free Let's Encrypt cert —
   no certbot cron jobs to manage): install Caddy, copy `deploy/Caddyfile`
   to `/etc/caddy/Caddyfile`, edit the domain, `sudo systemctl reload caddy`.
   Caddy terminates HTTPS on 80/443 and proxies to the backend on
   `127.0.0.1:8000` — the backend process itself is never exposed directly.
4. **Run the backend as a service**: copy `deploy/stegoart-backend.service`
   to `/etc/systemd/system/`, edit `WorkingDirectory`, `ExecStart`, and
   `CORS_ALLOWED_ORIGINS` (your Vercel URL + any custom domain), then:
   ```bash
   sudo systemctl enable --now stegoart-backend
   ```
5. **Verify**: `curl https://api.yourdomain.com/health` → `{"status":"ok"}`.
6. **Point the frontend at it**: on Vercel, set the `VITE_API_BASE_URL`
   project environment variable to `https://api.yourdomain.com` and
   redeploy (see `../frontend/README.md`). It must be `https://`, not
   `http://` — a Vercel page is served over HTTPS and browsers block it
   from calling a plain-HTTP API (mixed content).

**Vercel preview deployments** get random URLs
(`project-git-branch-user.vercel.app`) that won't match a fixed
`CORS_ALLOWED_ORIGINS` entry. Either add your Vercel *production* domain
only (previews just won't be able to call the API), or set
`allow_origin_regex=r"https://.*\.vercel\.app"` alongside/instead of
`allow_origins` in `main.py` if you want previews working too.

## Tests

```bash
# from the folder containing this clone:
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
`/api/decode`, never `styled_image_base64`. See the standalone `ml` repo's
`test_style_transfer.py` for the measured degradation.
