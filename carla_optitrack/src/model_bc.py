import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)