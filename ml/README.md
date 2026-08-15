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
