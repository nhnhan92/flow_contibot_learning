"""
model.py — PropMLP: residual MLP for flow-based tip position estimation.

Input  : (B, input_size)  — [pwm1..3, flow1..3, K1..3]  (9-dim default)
Output : (B, output_size) — [x_mm, y_mm, z_mm]           (3-dim default)

Architecture:
    proj   : Linear(input_size, hidden) → LayerNorm → ReLU
    blocks : ResBlock × num_blocks
    head   : Linear(hidden, hidden//2) → ReLU → Dropout → Linear(hidden//2, output_size)

Each ResBlock:
    skip = x
    x = Linear(H,H) → LayerNorm → ReLU → Dropout → Linear(H,H) → LayerNorm
    x = ReLU(x + skip)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PropMLP(nn.Module):
    """
    Residual MLP for proprioceptive tip-position estimation.

    Parameters
    ----------
    input_size  : feature dimensionality (default 9)
    hidden_size : width of all hidden layers (default 128)
    num_blocks  : number of residual blocks (default 3)
    dropout     : dropout inside each block (0 on last block)
    output_size : output dimensionality (default 3 → x/y/z mm)
    """

    def __init__(
        self,
        input_size:  int   = 9,
        hidden_size: int   = 128,
        num_blocks:  int   = 3,
        dropout:     float = 0.1,
        output_size: int   = 3,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            ResBlock(hidden_size, dropout=dropout if i < num_blocks - 1 else 0.0)
            for i in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, input_size)  →  (B, output_size)"""
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlainMLP(nn.Module):
    """
    Plain (non-residual) MLP for tip-position estimation.

    Architecture:
        Linear(input_size, hidden) → ReLU → [Linear(hidden, hidden) → ReLU] × (num_layers-1)
        → Dropout → Linear(hidden, output_size)

    Parameters
    ----------
    input_size  : feature dimensionality (default 9)
    hidden_size : width of all hidden layers (default 128)
    num_layers  : number of hidden layers (default 3)
    dropout     : dropout before the output layer
    output_size : output dimensionality (default 3)
    """

    def __init__(
        self,
        input_size:  int   = 9,
        hidden_size: int   = 128,
        num_layers:  int   = 3,
        dropout:     float = 0.1,
        output_size: int   = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Dropout(dropout), nn.Linear(hidden_size, output_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, input_size)  →  (B, output_size)"""
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(8, 9)
    for cls, kw, tag in [
        (PropMLP,  dict(hidden_size=128, num_blocks=3,  dropout=0.1), "PropMLP "),
        (PlainMLP, dict(hidden_size=128, num_layers=3,  dropout=0.1), "PlainMLP"),
    ]:
        m = cls(input_size=9, output_size=3, **kw)
        print(f"{tag}  params: {m.n_params():,}  input: {x.shape}  output: {m(x).shape}")
