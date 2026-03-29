"""
model.py
A3T-GCN: Attention Temporal Graph Convolutional Network.
Architecture: Temporal (GRU) → Spatial (GCN + Attention) → Temporal (GRU) → FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer: H' = σ(D^{-1/2} A D^{-1/2} H W)"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F), adj: (B, N, N)
        # Symmetric normalization with added self-loops
        adj = adj + torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / (deg * deg.transpose(-1, -2)).sqrt()
        return F.relu(self.W(torch.bmm(adj_norm, x)))


class GraphAttention(nn.Module):
    """Attention over neighbor embeddings to weight their contribution."""

    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, gcn_out: torch.Tensor, gru_out: torch.Tensor) -> torch.Tensor:
        # gcn_out, gru_out: (B, N, H)
        combined = torch.cat([gcn_out, gru_out], dim=-1)          # (B, N, 2H)
        score = torch.sigmoid(self.attn(combined))                 # (B, N, 1)
        return score * gcn_out + (1 - score) * gru_out


class A3TGCN(nn.Module):
    """
    Temporal → Spatial → Temporal sandwich with attention gating.
    Input:  x   (B, T, N, F)
            adj (B, N, N)
    Output: (B, N, 1)
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # First temporal block
        self.gru1 = nn.GRU(in_channels, hidden_channels, batch_first=True)
        # Spatial block
        self.gcn = GCNLayer(hidden_channels, hidden_channels)
        self.attention = GraphAttention(hidden_channels)
        # Second temporal block
        self.gru2 = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        # Prediction head
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape

        # --- Temporal Block 1: per-node GRU over time ---
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, F)   # (B*N, T, F)
        gru1_out, _ = self.gru1(x_reshaped)                        # (B*N, T, H)
        h1 = gru1_out[:, -1, :].reshape(B, N, -1)                  # (B, N, H) — last hidden state

        # --- Spatial Block: GCN + Attention ---
        gcn_out = self.gcn(h1, adj)                                 # (B, N, H)
        spatial_out = self.attention(gcn_out, h1)                   # (B, N, H)

        # --- Temporal Block 2: re-process spatial output ---
        spatial_seq = spatial_out.unsqueeze(2).expand(B, N, T, -1) # (B, N, T, H)
        spatial_seq = spatial_seq.reshape(B * N, T, -1)            # (B*N, T, H)
        gru2_out, _ = self.gru2(spatial_seq)                       # (B*N, T, H)
        h2 = gru2_out[:, -1, :].reshape(B, N, -1)                  # (B, N, H)

        return self.fc(h2)                                          # (B, N, 1)


def build_model(cfg: dict) -> A3TGCN:
    return A3TGCN(
        in_channels=cfg["model"]["in_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        out_channels=cfg["model"]["out_channels"],
    )
