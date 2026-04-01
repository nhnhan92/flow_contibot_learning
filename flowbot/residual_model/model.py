"""
model.py  –  Residual GRU (ResGRU) for flowbot error prediction.

Each GRU layer adds a linear skip connection from its input to its output,
so the model learns residual corrections on top of the raw sequence context.

Architecture:
    Input (B, seq_len, 6)
      → ResGRULayer × num_layers   (each: GRU + skip + LayerNorm + Dropout)
      → last time-step hidden state
      → MLP head  (hidden → hidden//2 → 3)
    Output (B, 3)  ←  [err_x, err_y, err_z] in normalised units
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ResGRULayer(nn.Module):
    """
    GRU layer with residual skip connection:
        out = LayerNorm( GRU(x) + Linear(x) )
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.gru     = nn.GRU(input_size, hidden_size, batch_first=True)
        self.skip    = nn.Linear(input_size, hidden_size, bias=False)
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                        # (B, T, input_size)
        h: Optional[torch.Tensor] = None,       # (1, B, hidden_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h_new = self.gru(x, h)             # (B, T, hidden_size)
        out = self.norm(out + self.skip(x))
        return self.dropout(out), h_new


class ResGRU(nn.Module):
    """
    Stacked Residual GRU for sequence-to-error prediction.

    Parameters
    ----------
    input_size   : number of input features per time step (default 6)
    hidden_size  : GRU hidden dimension
    num_layers   : number of stacked ResGRU layers
    dropout      : dropout rate (applied after each layer except the last)
    output_size  : prediction dimensionality (default 3 → err_x/y/z)
    """
    def __init__(
        self,
        input_size:  int   = 6,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
        output_size: int   = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Build stacked layers; only apply dropout between layers, not after the last
        sizes = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList([
            ResGRULayer(
                sizes[i], sizes[i + 1],
                dropout=dropout if i < num_layers - 1 else 0.0
            )
            for i in range(num_layers)
        ])

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x:      torch.Tensor,                       # (B, T, input_size)
        hidden: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns
        -------
        pred   : (B, output_size)  — prediction from last time-step
        hidden : list of (1, B, hidden_size) tensors, one per layer
        """
        if hidden is None:
            hidden = [None] * self.num_layers

        new_hidden: List[torch.Tensor] = []
        for layer, h in zip(self.layers, hidden):
            x, h_new = layer(x, h)
            new_hidden.append(h_new)

        pred = self.head(x[:, -1, :])      # use last time-step output
        return pred, new_hidden

    # ── warm-up (no gradient) ────────────────────────────────────────────────

    @torch.no_grad()
    def warmup(
        self,
        x_seq:  torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Feed a ground-truth sequence to prime hidden states without computing
        gradients.  Returns the updated hidden state list.
        """
        _, hidden = self.forward(x_seq, hidden)
        return hidden

    # ── parameter count ──────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, F = 4, 40, 6
    model = ResGRU(input_size=F, hidden_size=64, num_layers=2, dropout=0.1)
    print(f"ResGRU  params: {model.n_params():,}")

    x = torch.randn(B, T, F)
    pred, hidden = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {pred.shape}")
    print(f"Hidden: {[h.shape for h in hidden]}")

    # warm-up then predict on tail
    h = model.warmup(x[:, :20, :])
    pred2, _ = model(x[:, 20:, :], h)
    print(f"Warmup+tail output: {pred2.shape}")
