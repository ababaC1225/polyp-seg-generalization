"""U-Net baseline: same padding, bilinear upsampling, no pretrained weights."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, base_channels=32, boundary_head=False):
        super().__init__()
        widths = [base_channels * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([
            DoubleConv(3 if i == 0 else widths[i - 1], width)
            for i, width in enumerate(widths)
        ])
        self.decoders = nn.ModuleList([
            DoubleConv(widths[i + 1] + widths[i], widths[i]) for i in range(3, -1, -1)
        ])
        self.head = nn.Conv2d(widths[0], 1, 1)
        # Preserve baseline initialization and subsequent CPU RNG consumption.
        with torch.random.fork_rng(devices=[]):
            self.boundary_head = nn.Conv2d(widths[0], 1, 1) if boundary_head else None

    def forward(self, x, return_boundary=False):
        skips = []
        for index, encoder in enumerate(self.encoders):
            if index:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        logits = self.head(x)
        if return_boundary:
            if self.boundary_head is None:
                raise ValueError("Boundary output requested without an auxiliary head")
            return logits, self.boundary_head(x)
        return logits


def build_model(config):
    if config.get("model", "unet") != "unet":
        raise ValueError("Unsupported model")
    return UNet(config["base_channels"], config.get("boundary_supervision", {}).get("enabled", False))
