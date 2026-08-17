# Embedded copy of `ml` (inference subset)

This is a private, embedded copy of the parts of the standalone `ml` repo
needed to *run* the trained models — not a separate dependency. It exists so
the `backend` repo can be cloned and run entirely on its own, with no
sibling `ml` checkout required.

## What's here vs. what's not

Included: `config.py`, `models/` (network + style-transfer architectures),
`utils/` (text encoding, image conversion, PSNR/SSIM), and `weights/`
(`encoder.pth`/`decoder.pth` — see "Model weights" below — plus
`weights/styles/*.pth`).

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

## Model weights: the "v3" style-robust text model

`StegoService` (`../services/stego_service.py`) loads a single trained
encoder/decoder pair — `encoder.pth`/`decoder.pth`, the "v3" style-robust
variant (from `ml/weights_style_robust_v3/`), not `ml/weights/`'s original
clean-trained weights. This project only hides text secrets, so v3 (which
was tuned specifically for styled-**text** recovery) is the one pair the
backend needs. candy/mosaic/rain_princess stay strong (75-100% char
accuracy) even at long text; udnie is just as strong up to ~200 characters
but collapses past that (79% at 250 chars, ~6% near the 510-char max) —
see `ml/README.md`'s "udnie length sensitivity" table. `routers/encode.py`'s
`_UDNIE_SAFE_TEXT_CHARS` threshold bakes this into
`EncodeResponse.styled_decode_supported`.

An earlier iteration of this project also supported image secrets, with a
second, image-dedicated model pair (`encoder_image.pth`/`decoder_image.pth`,
from `ml/weights_style_robust_img/`) loaded alongside v3 and routed by
`secret_type`. That path has been removed — the backend, API, and frontend
are text-only now — so those weights are no longer shipped here (still kept
in the standalone `ml` repo's `weights_style_robust_img/` for reference).

Consequence for anything reading this backend's `EncodeResponse`:
cover/stego PSNR and SSIM (computed against v3's output, per
`routers/encode.py`) will read noticeably lower than the ~30dB/~0.93 this
project's docs elsewhere describe as "the" numbers — that's expected with
these weights, not a regression. See `ml/README.md` for the full
before/after comparison tables and how to reproduce or revert to the
original (`ml/weights/`) weights.
