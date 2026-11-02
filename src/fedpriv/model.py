"""Small MLP for 128-d tabular PMU features (no BatchNorm, so it is Opacus-compatible)."""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 256, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
