"""Feed-forward Neural Style Transfer (Johnson et al.-style transformer net).

Wraps pretrained TransformerNet checkpoints for single-pass artistic
restyling — this is inference-only code. No training happens here or
anywhere in this module; see ml/data/download_style_weights.py for how the
checkpoints in ml/weights/styles/ were obtained (a direct download, not a
training run).

Architecture and checkpoints come from pytorch/examples' fast_neural_style
(https://github.com/pytorch/examples/tree/main/fast_neural_style), which
implements "Perceptual Losses for Real-Time Style Transfer and
Super-Resolution" (Johnson, Alahi, Fei-Fei, https://arxiv.org/abs/1603.08155)
with Instance Normalization. TransformerNet below is a verbatim copy of that
repo's transformer_net.py so the pretrained state_dicts load without any
conversion.

--- IMPORTANT: normalization differs from the steganography model ---
StegoAutoencoder (ml/models/networks.py) operates entirely in [0, 1] float,
end to end (see ml/config.py). TransformerNet instead expects and returns
[0, 255] float — a plain `ToTensor()` scaled by 255, NOT divided down to
[0, 1], and NOT ImageNet mean/std normalized (that normalization only
appears in the original training script, applied to a copy of the tensor
fed through VGG for the perceptual loss — it never touches the
transformer's own input/output). Mixing these two conventions up (e.g.
feeding a stego image's raw [0, 1] tensor straight into TransformerNet, or
feeding TransformerNet's [0, 255] output straight into RevealNetwork) is a
real, silent-failure-shaped bug: both tensors have the same shape and dtype,
so nothing crashes — the output is just wrong (near-black for [0,1]-into-
TransformerNet, blown-out/clipped for [0,255]-into-RevealNetwork).

apply_style() is the one function meant to be called from outside this
module specifically so this conversion happens in exactly one place. Its
tensor in/out contract matches the rest of this codebase ([0, 1] float) —
the internal *255 / /255 conversion is entirely hidden inside it.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from ..config import IMG_SIZE

STYLES_DIR = Path(__file__).resolve().parent.parent / "weights" / "styles"
STYLE_CHECKPOINTS = {
    "candy": "candy.pth",
    "mosaic": "mosaic.pth",
    "rain_princess": "rain_princess.pth",
    "udnie": "udnie.pth",
}

_model_cache: dict = {}  # style_name -> (device_str, TransformerNet) — avoids re-loading weights every call


# --- TransformerNet: verbatim architecture from pytorch/examples/fast_neural_style ---
# (kept in this file rather than imported so this module has no dependency on
# that repo at runtime — only its checkpoint files, fetched once via
# ml/data/download_style_weights.py)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        return self.deconv3(y)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv2d(self.reflection_pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode="nearest", scale_factor=self.upsample)
        return self.conv2d(self.reflection_pad(x_in))


# --- loading + inference ---

def list_styles() -> list:
    """Style names with a checkpoint actually present on disk right now."""
    return sorted(name for name, fname in STYLE_CHECKPOINTS.items() if (STYLES_DIR / fname).exists())


def load_style_model(style_name: str, device="cpu") -> TransformerNet:
    """Load (and cache) the TransformerNet for `style_name`. Cached per
    (style_name, device) so repeated apply_style() calls in a backend don't
    pay checkpoint-loading cost every request."""
    if style_name not in STYLE_CHECKPOINTS:
        raise ValueError(f"unknown style {style_name!r}; available: {sorted(STYLE_CHECKPOINTS)}")

    device = torch.device(device)
    cache_key = (style_name, str(device))
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    ckpt_path = STYLES_DIR / STYLE_CHECKPOINTS[style_name]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found — run `python -m ml.data.download_style_weights` first"
        )

    state_dict = torch.load(ckpt_path, map_location=device)
    # Older PyTorch serialized InstanceNorm2d's (unused, since affine-only)
    # running_mean/running_var buffers; a fresh InstanceNorm2d(affine=True)
    # doesn't have them, so load_state_dict(strict=True) would fail on an
    # "unexpected key" without this — same workaround the source repo uses.
    for key in list(state_dict.keys()):
        if key.endswith(".running_mean") or key.endswith(".running_var"):
            del state_dict[key]

    model = TransformerNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model_cache[cache_key] = model
    return model


def apply_style(image, style_name: str, device="cpu", size: int = IMG_SIZE):
    """Run `image` through the pretrained `style_name` TransformerNet.

    `image` may be a PIL.Image (any mode/size — converted to RGB and resized
    to `size`x`size`) or a float tensor in [0, 1] with shape (3,H,W) or
    (1,3,H,W). The return type mirrors the input type: PIL in -> PIL out,
    tensor in -> tensor out (same shape convention, values in [0, 1]).

    Internally converts to/from the [0, 255] range TransformerNet expects —
    see the module docstring. Callers never need to think about that range;
    just don't bypass this function and call TransformerNet directly.
    """
    model = load_style_model(style_name, device=device)
    device = torch.device(device)

    input_was_pil = isinstance(image, Image.Image)
    if input_was_pil:
        tensor = T.Compose([T.Resize((size, size)), T.ToTensor()])(image.convert("RGB"))
        tensor = tensor.unsqueeze(0)
    else:
        tensor = image
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != size or tensor.shape[-2] != size:
            tensor = nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)

    tensor = (tensor * 255.0).to(device)
    with torch.no_grad():
        styled = model(tensor)
    styled = (styled.clamp(0, 255) / 255.0).cpu()

    if input_was_pil:
        return T.ToPILImage()(styled[0])
    return styled if image.dim() == 4 else styled[0]


def _timing_selftest(device="cpu", n_repeats=5):
    """Quick, no-training timing check: apply_style at IMG_SIZE, CPU, per
    style. Prints per-call seconds. Run: python -m ml.models.style_transfer"""
    dummy = torch.rand(3, IMG_SIZE, IMG_SIZE)
    for style in list_styles():
        load_style_model(style, device=device)  # warm the cache before timing
        t0 = time.time()
        for _ in range(n_repeats):
            apply_style(dummy, style, device=device)
        dt = (time.time() - t0) / n_repeats
        print(f"{style:16s} {dt * 1000:7.1f} ms/call (device={device})")


if __name__ == "__main__":
    _timing_selftest()
