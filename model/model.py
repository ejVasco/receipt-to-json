"""
model.py
Multilayer ANN (mlp) for per line receipt field classification
Architexture:
    linear (15 -> 64) -> batchnorm -> relu -> dropuot
    linear (64 -> 64) -> batchnorm -> relu -> dropout
    linear (64 -> 32) -> batchnorm -> relu
    linear (32 -> NUM_CLASSES)

input: feature vector of shape (batch, FEATURE_DIM)
output: logits of shape (batch, NUM_CLASSES)
"""

import torch
import torch.nn as nn

from model.dataset import FEATURE_DIM, NUM_CLASSES


class ReceiptMLP(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.Linear(FEATURE_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # layer 2
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # output
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
