# Embedded copy of `ml` (inference subset)

This is a private, embedded copy of the parts of the standalone `ml` repo
needed to *run* the trained models — not a separate dependency. It exists so
the `backend` repo can be cloned and run entirely on its own, with no
sibling `ml` checkout required.

## What's here vs. what's not

Included: `config.py`, `models/` (network + style-transfer architectures),
`utils/` (text encoding, image conversion, PSNR/SSIM), and `weights/`
(`encoder.pth`, `decoder.pth`, `weights/styles/*.pth`).

Not included: `data/` (dataset download/loading), `train.py`, `evaluate.py`,
`export.py`, `notebooks/`, `test_style_transfer.py` — anything only needed
for training or offline evaluation, not for serving `/api/encode` and
`/api/decode`.

## Keeping this in sync

These files are plain copies from the `ml` repo, at whatever commit backend
last synced against. If the `ml` repo's model code, `config.py`, or
`weights/` change (e.g. retraining produces new weights, or the network
architecture changes), re-copy the same files from `ml/` into this folder —
this directory's internal relative imports (`from ..config import ...`
etc.) mirror `ml`'s own structure exactly, so a straight file copy is all
that's needed; nothing here should be hand-edited independently of `ml`.

## Current weights: style-robust v3 (not the clean-only original)

`encoder.pth`/`decoder.pth` here are **not** `ml/weights/`'s original
clean-trained weights — they're the "v3" style-robust variant from
`ml/README.md`'s "Style-robust training experiment" section
(`ml/weights_style_robust_v3/`), deliberately trading clean-path fidelity
for text-secret recovery from a *styled* stego image. candy/mosaic/
rain_princess stay strong (75-100% char accuracy) even at long text; udnie
is just as strong up to ~200 characters but collapses past that (79% at
250 chars, ~6% near the 510-char max) — see `ml/README.md`'s "udnie length
sensitivity" table. `routers/encode.py`'s `_UDNIE_SAFE_TEXT_CHARS`
threshold bakes this into `EncodeResponse.styled_decode_supported`, so
callers don't need to know this nuance themselves. Image-secret recovery
from a styled image is still unreliable regardless of style or length
(SSIM ~0.31).

Consequences for anything reading this backend's `EncodeResponse`:
cover/stego PSNR and SSIM will read noticeably lower than the ~30dB/~0.93
this project's docs elsewhere describe as "the" numbers — that's expected
with these weights, not a regression. See `ml/README.md` for the full
before/after comparison table and how to reproduce or revert to the
original (`ml/weights/`) weights.
