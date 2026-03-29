"""
dataset.py
PyTorch Dataset that creates sliding 21-day windows over node features
and maps each window to the nearest weekly adjacency matrix.
"""

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data_loader import load_config


class StockGraphDataset(Dataset):
    """
    Each sample:
        x   : (lookback, N, F) — node features over the window
        adj : (N, N)           — adjacency matrix for that week
        y   : (N,)             — 5-day forward log-return for all nodes
    """

    def __init__(self, feature_store: dict, adj_matrices: list, week_dates: list[pd.Timestamp], tickers: list[str], cfg: dict):
        self.lookback = cfg["features"]["lookback"]
        self.tickers = tickers
        self.feature_cols = ["log_return", "rsi", "macd_line", "macd_signal", "macd_hist"]

        # Build (dates x N x F) feature tensor and (dates,) target array
        dates = feature_store[tickers[0]].index
        F = len(self.feature_cols)
        N = len(tickers)

        X = np.zeros((len(dates), N, F), dtype=np.float32)
        Y = np.zeros((len(dates), N), dtype=np.float32)

        for i, ticker in enumerate(tickers):
            df = feature_store[ticker]
            X[:, i, :] = df[self.feature_cols].values
            Y[:, i] = df["target"].values

        # Map each date to its nearest past week-end adjacency matrix
        week_dates = pd.DatetimeIndex(week_dates)
        self.samples = []

        for t in range(self.lookback, len(dates)):
            window_end = dates[t - 1]
            # Find latest week_end <= window_end
            valid = week_dates[week_dates <= window_end]
            if len(valid) == 0:
                continue
            adj_idx = week_dates.get_loc(valid[-1])
            adj = adj_matrices[adj_idx]

            self.samples.append((
                X[t - self.lookback:t],   # (lookback, N, F)
                adj,                       # (N, N)
                Y[t],                      # (N,)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, adj, y = self.samples[idx]
        return (
            torch.tensor(x),
            torch.tensor(adj),
            torch.tensor(y),
        )


def get_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    processed_dir = cfg["data"]["processed_dir"]

    with open(f"{processed_dir}/features.pkl", "rb") as f:
        store = pickle.load(f)
    with open(f"{processed_dir}/adjacency_matrices.pkl", "rb") as f:
        graph = pickle.load(f)

    dataset = StockGraphDataset(
        store["features"], graph["adjacency"], graph["dates"], graph["tickers"], cfg
    )

    n = len(dataset)
    train_end = int(n * cfg["training"]["train_split"])
    val_end = train_end + int(n * cfg["training"]["val_split"])

    train_ds = torch.utils.data.Subset(dataset, range(0, train_end))
    val_ds   = torch.utils.data.Subset(dataset, range(train_end, val_end))
    test_ds  = torch.utils.data.Subset(dataset, range(val_end, n))

    bs = cfg["training"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=False),
        DataLoader(val_ds,   batch_size=bs, shuffle=False),
        DataLoader(test_ds,  batch_size=bs, shuffle=False),
    )


if __name__ == "__main__":
    cfg = load_config()
    train_dl, val_dl, test_dl = get_dataloaders(cfg)
    x, adj, y = next(iter(train_dl))
    print(f"x: {x.shape}, adj: {adj.shape}, y: {y.shape}")
    # Expected: x=(32, 21, N, 5), adj=(32, N, N), y=(32, N)
