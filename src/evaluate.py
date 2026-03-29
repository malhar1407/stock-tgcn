"""
evaluate.py
Runs inference on the test set and computes:
  - MAE, RMSE (on normalized returns)
  - Directional Accuracy (sign prediction %)
  - Annualized Sharpe Ratio (simulated long/short strategy)
Saves predictions to CSV for the Streamlit app.
"""

import json
import pickle
import numpy as np
import pandas as pd
import torch

from data_loader import load_config
from dataset import get_dataloaders, StockGraphDataset
from model import build_model


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    mean = returns.mean()
    std = returns.std()
    return float((mean / std) * np.sqrt(periods_per_year)) if std > 0 else 0.0


def evaluate(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_dl = get_dataloaders(cfg)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["model_checkpoint"], map_location=device))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, adj, y in test_dl:
            x, adj = x.to(device), adj.to(device)
            pred = model(x, adj).squeeze(-1).cpu().numpy()   # (B, N)
            all_preds.append(pred)
            all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (T, N)
    targets = np.concatenate(all_targets, axis=0)   # (T, N)

    # --- Metrics ---
    mae  = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    dir_acc = (np.sign(preds) == np.sign(targets)).mean()

    # Simulated long/short: go long stocks with pred > 0, short pred < 0
    strategy_returns = np.sign(preds) * targets          # (T, N)
    portfolio_returns = strategy_returns.mean(axis=1)    # (T,) — equal weight
    sharpe = sharpe_ratio(portfolio_returns)

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "directional_accuracy": round(float(dir_acc), 4),
        "sharpe_ratio": round(sharpe, 4),
    }
    print(json.dumps(metrics, indent=2))

    # Append eval metrics to existing training history
    with open(cfg["paths"]["metrics"], "r") as f:
        history = json.load(f)
    history.update(metrics)
    with open(cfg["paths"]["metrics"], "w") as f:
        json.dump(history, f)

    # --- Save predictions CSV ---
    with open(cfg["data"]["processed_dir"] + "/adjacency_matrices.pkl", "rb") as f:
        tickers = pickle.load(f)["tickers"]

    pred_df = pd.DataFrame(preds, columns=tickers)
    pred_df.insert(0, "step", range(len(pred_df)))
    pred_df.to_csv(cfg["paths"]["predictions"], index=False)

    print(f"[INFO] Predictions saved to {cfg['paths']['predictions']}")
    return metrics


if __name__ == "__main__":
    cfg = load_config()
    evaluate(cfg)
