import torch
import torch.nn as nn

from ..config import COVER_CHANNELS, SECRET_CHANNELS


def _conv_block(in_ch: int, out_ch: int, final: bool = False) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
    if final:
        layers.append(nn.Sigmoid())  # bounds output to [0, 1], matches our normalization
    else:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class HidingNetwork(nn.Module):
    """Encoder ("Hiding Network").

    Concatenates cover (3,H,W) and secret (3,H,W) channel-wise -> (6,H,W),
    passes through 6 Conv2D(+BN+ReLU) blocks, outputs a stego image (3,H,W)
    the same size as the cover.
    """

    CHANNELS = [64, 128, 128, 64, 32]  # + final 32->3 layer below = 6 conv layers total

    def __init__(self):
        super().__init__()
        in_ch = COVER_CHANNELS + SECRET_CHANNELS  # 3 + 3 = 6
        blocks = []
        for out_ch in self.CHANNELS:
            blocks.append(_conv_block(in_ch, out_ch))
            in_ch = out_ch
        blocks.append(_conv_block(in_ch, COVER_CHANNELS, final=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cover, secret], dim=1)
        return self.net(x)


class RevealNetwork(nn.Module):
    """Decoder ("Reveal Network").

    Mirrors HidingNetwork's channel progression: takes the stego image
    (3,H,W) and outputs the reconstructed secret (3,H,W).
    """

    CHANNELS = [32, 64, 128, 128, 64]  # + final 64->3 layer below = 6 conv layers total

    def __init__(self):
        super().__init__()
        in_ch = COVER_CHANNELS
        blocks = []
        for out_ch in self.CHANNELS:
            blocks.append(_conv_block(in_ch, out_ch))
            in_ch = out_ch
        blocks.append(_conv_block(in_ch, SECRET_CHANNELS, final=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        return self.net(stego)


class StegoAutoencoder(nn.Module):
    """Joint encoder+decoder, trained end-to-end. Each submodule is exported
    separately to /ml/weights (encoder.pth, decoder.pth) since the FastAPI
    backend loads and runs them independently (hide vs. extract endpoints)."""

    def __init__(self):
        super().__init__()
        self.encoder = HidingNetwork()
        self.decoder = RevealNetwork()

    def forward(self, cover: torch.Tensor, secret: torch.Tensor):
        stego = self.encoder(cover, secret)
        recovered_secret = self.decoder(stego)
        return stego, recovered_secret
