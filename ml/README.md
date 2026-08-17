# Embedded copy of `ml` (inference subset)

This is a private, embedded copy of the parts of the standalone `ml` repo
needed to *run* the trained models — not a separate dependency. It exists so
the `backend` repo can be cloned and run entirely on its own, with no
sibling `ml` checkout required.

## What's here vs. what's not

Included: `config.py`, `models/` (network + style-transfer architectures),
`utils/` (text encoding, image conversion, PSNR/SSIM), and `weights/`
(`encoder.pth`/`decoder.pth` + `encoder_image.pth`/`decoder_image.pth` —
two separate trained pairs, see "Two model pairs" below — plus
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

## Two model pairs, not one (not the clean-only original either)

`StegoService` (`../services/stego_service.py`) loads **two** trained
encoder/decoder pairs and routes by `secret_type` — a single style-robust
model can't be good at both text and image secrets (verified: training one
model on both, or reusing one for the other, measurably breaks whichever
wasn't its focus). Neither pair is `ml/weights/`'s original clean-trained
weights.

- **`encoder.pth`/`decoder.pth`** — the "v3" style-robust variant (from
  `ml/weights_style_robust_v3/`), used for **text** secrets. candy/mosaic/
  rain_princess stay strong (75-100% char accuracy) even at long text;
  udnie is just as strong up to ~200 characters but collapses past that
  (79% at 250 chars, ~6% near the 510-char max) — see `ml/README.md`'s
  "udnie length sensitivity" table. `routers/encode.py`'s
  `_UDNIE_SAFE_TEXT_CHARS` threshold bakes this into
  `EncodeResponse.styled_decode_supported`.
- **`encoder_image.pth`/`decoder_image.pth`** — the image-secret-dedicated
  variant (from `ml/weights_style_robust_img/`), used for **image**
  secrets. Styled-image recovery is meaningfully better than v3's would be
  (SSIM ~0.40-0.44 across all 4 styles, no longer a udnie-specific outlier)
  but still not reliable enough to flag `styled_decode_supported: true` —
  it's "recognizable but degraded," not "trustworthy." This pair would NOT
  handle text secrets well if you tried (verified: 8-22% char accuracy on
  styled text) — never use it for `secret_type=text`.

Consequences for anything reading this backend's `EncodeResponse`:
cover/stego PSNR and SSIM (always computed against the text/v3 pair's
output, per `routers/encode.py`) will read noticeably lower than the
~30dB/~0.93 this project's docs elsewhere describe as "the" numbers —
that's expected with these weights, not a regression. See `ml/README.md`
for the full before/after comparison tables and how to reproduce or revert
to the original (`ml/weights/`) weights.
