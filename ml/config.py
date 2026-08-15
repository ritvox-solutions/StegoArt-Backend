"""Shared constants for the StegoArt ML prototype.

Import this module everywhere instead of hardcoding sizes — the FastAPI
backend (built later, see /02_Technical_Requirements_Document.md) will need
these exact same constants to stay in sync with whatever weights get
exported to /ml/weights.
"""

IMG_SIZE = 256                  # cover / stego / secret spatial size (H = W)
COVER_CHANNELS = 3
SECRET_CHANNELS = 3             # unified secret tensor shape: (3, IMG_SIZE, IMG_SIZE)
                                 # both text and image secrets are encoded into this
                                 # same shape, so one Encoder/Decoder pair handles both

# --- Text <-> tensor bit-map scheme (see utils/text_encoding.py) ---
SECRET_MAP_GRID = 64            # bits are packed into a SECRET_MAP_GRID x SECRET_MAP_GRID grid
BLOCK_SIZE = IMG_SIZE // SECRET_MAP_GRID   # each bit is tiled into a BLOCK_SIZE x BLOCK_SIZE pixel block
HEADER_BITS = 16                # 16-bit big-endian length header, prepended to the payload
BIT_CAPACITY = SECRET_MAP_GRID * SECRET_MAP_GRID
MAX_TEXT_CHARS = (BIT_CAPACITY - HEADER_BITS) // 8   # = 510 chars (4096-bit grid, 8-bit ASCII)

assert IMG_SIZE % SECRET_MAP_GRID == 0, "SECRET_MAP_GRID must evenly divide IMG_SIZE"

# --- Training defaults (TRD §3.1: L_total = alpha * L_cover + beta * L_secret) ---
ALPHA_COVER_LOSS = 1.0
BETA_SECRET_LOSS = 0.75

# --- Normalization ---
# Every tensor (cover, secret, stego, recovered secret) lives in [0, 1].
# Both networks end in a Sigmoid, so there's no [-1,1]/ImageNet-mean
# normalization to reverse — the FastAPI inference side just needs a
# divide-by-255 on the way in and a multiply-by-255 on the way out.
