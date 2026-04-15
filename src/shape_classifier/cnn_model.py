"""Supervised CNN model for shape classification."""

from __future__ import annotations

import torch
from torch import nn


class ShapeCNN(nn.Module):
    """A compact CPU-friendly CNN for single-shape classification."""

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 1,
        width: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.features = nn.Sequential(
            _conv_block(in_channels, width),
            nn.MaxPool2d(2),
            _conv_block(width, width * 2),
            nn.MaxPool2d(2),
            _conv_block(width * 2, width * 4),
            nn.MaxPool2d(2),
            _conv_block(width * 4, width * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
