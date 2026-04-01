"""
baseline_mlp.py  –  Residual MLP baseline for flowbot error prediction.

Two input modes (controlled by `use_history`):
  - use_history=False : only the last time-step  → input size = input_size  (6)
  - use_history=True  : full sequence flattened  → input size = seq_len × input_size

Architecture per mode:
    [no history]  x[:,-1,:]  (B,6)  → project → ResBlock × num_blocks → head → (B,3)
    [history]     x.flatten  (B,240) → project → ResBlock × num_blocks → head → (B,3)

Each ResBlock:
    skip = x
    x = Linear(H,H) → LayerNorm → ReLU → Dropout → Linear(H,H) → LayerNorm
    x = ReLU(x + skip)

This baseline lets you answer:
  "Does the temporal history actually help, or is the current position enough?"
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class ResBlock(nn.Module):
    """Pre-activation residual block with LayerNorm."""
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


class ResMLP(nn.Module):
    """
    Residual MLP baseline for error prediction.

    Parameters
    ----------
    input_size  : features per time step (default 6)
    seq_len     : sequence length — only used when use_history=True (default 40)
    hidden_size : width of all hidden layers
    num_blocks  : number of residual blocks
    dropout     : dropout rate inside each block
    output_size : output dimensionality (default 3 → err_x/y/z)
    use_history : if True, flatten full sequence; if False, use only last step
    """
    def __init__(
        self,
        input_size:  int   = 6,
        seq_len:     int   = 40,
        hidden_size: int   = 128,
        num_blocks:  int   = 3,
        dropout:     float = 0.1,
        output_size: int   = 3,
        use_history: bool  = True,
    ):
        super().__init__()
        self.use_history = use_history
        self.seq_len     = seq_len
        self.input_size  = input_size

        flat_size = (seq_len * input_size) if use_history else input_size

        # Input projection → hidden_size
        self.proj = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Stacked residual blocks (no dropout after last block)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_size, dropout=dropout if i < num_blocks - 1 else 0.0)
            for i in range(num_blocks)
        ])

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)  — same shape as ResGRU input

        Returns
        -------
        pred : (B, output_size)
        """
        if self.use_history:
            x = x.reshape(x.size(0), -1)   # (B, T*F)
        else:
            x = x[:, -1, :]                # (B, F)  — last timestep only

        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, F = 4, 40, 6
    x = torch.randn(B, T, F)

    for use_hist in [True, False]:
        tag = "with history " if use_hist else "no history  "
        m = ResMLP(input_size=F, seq_len=T, hidden_size=128,
                   num_blocks=3, dropout=0.1, use_history=use_hist)
        pred = m(x)
        print(f"ResMLP [{tag}]  params: {m.n_params():>7,}  output: {pred.shape}")
