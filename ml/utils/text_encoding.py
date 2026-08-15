"""Text <-> tensor bit-map encoding scheme.

Scheme: a 16-bit big-endian length header + 8-bit-ASCII payload bits are
packed into a SECRET_MAP_GRID x SECRET_MAP_GRID grid (zero-padded past the
payload), then each bit is tiled into a BLOCK_SIZE x BLOCK_SIZE pixel block
and replicated across all SECRET_CHANNELS channels. This produces a
(SECRET_CHANNELS, IMG_SIZE, IMG_SIZE) tensor — the same shape used for image
secrets — so the Encoder/Decoder networks don't need to know which kind of
secret they're handling.

At SECRET_MAP_GRID=64 this gives a 4096-bit capacity, i.e.
MAX_TEXT_CHARS = (4096 - 16) // 8 = 510 characters. See config.py.
"""

import torch

from ..config import (
    BLOCK_SIZE,
    HEADER_BITS,
    IMG_SIZE,
    MAX_TEXT_CHARS,
    SECRET_CHANNELS,
    SECRET_MAP_GRID,
)


def _text_to_bits(text: str) -> list:
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text too long: {len(text)} chars, max is {MAX_TEXT_CHARS}")
    if not all(ord(c) < 256 for c in text):
        raise ValueError("text must be encodable as 8-bit ASCII/Latin-1")

    length_bits = [int(b) for b in format(len(text), f"0{HEADER_BITS}b")]
    payload_bits = []
    for ch in text:
        payload_bits.extend(int(b) for b in format(ord(ch), "08b"))
    return length_bits + payload_bits


def text_to_tensor(text: str) -> torch.Tensor:
    """Encode `text` into a (SECRET_CHANNELS, IMG_SIZE, IMG_SIZE) float32
    tensor with values in {0.0, 1.0}."""
    bits = _text_to_bits(text)
    bits = bits + [0] * (SECRET_MAP_GRID * SECRET_MAP_GRID - len(bits))

    grid = torch.tensor(bits, dtype=torch.float32).view(SECRET_MAP_GRID, SECRET_MAP_GRID)
    tile = grid.repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1)
    return tile.unsqueeze(0).repeat(SECRET_CHANNELS, 1, 1)


def tensor_to_text(tensor: torch.Tensor) -> str:
    """Inverse of text_to_tensor. Averages channels, averages each
    BLOCK_SIZE x BLOCK_SIZE block back down to one bit, thresholds at 0.5,
    reads the 16-bit length header, then decodes that many ASCII characters.

    Works on raw decoder output (values don't need to be exactly 0/1) since
    both the channel-average and block-average smooth out per-pixel noise
    before the threshold is applied.
    """
    if tensor.shape[-3:] != (SECRET_CHANNELS, IMG_SIZE, IMG_SIZE):
        raise ValueError(f"expected shape (..., {SECRET_CHANNELS}, {IMG_SIZE}, {IMG_SIZE}), got {tuple(tensor.shape)}")

    mono = tensor.mean(dim=-3)  # (IMG_SIZE, IMG_SIZE)
    blocks = mono.view(SECRET_MAP_GRID, BLOCK_SIZE, SECRET_MAP_GRID, BLOCK_SIZE)
    grid = blocks.mean(dim=(1, 3))  # (SECRET_MAP_GRID, SECRET_MAP_GRID)
    bits = (grid.flatten() > 0.5).int().tolist()

    length_bits = bits[:HEADER_BITS]
    length = int("".join(map(str, length_bits)), 2)
    length = min(length, MAX_TEXT_CHARS)  # guard against a corrupted/garbage header

    payload_bits = bits[HEADER_BITS : HEADER_BITS + length * 8]
    chars = []
    for i in range(0, len(payload_bits) - 7, 8):
        byte = payload_bits[i : i + 8]
        chars.append(chr(int("".join(map(str, byte)), 2)))
    return "".join(chars)
