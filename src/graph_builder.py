"""
graph_builder.py
Builds a sequence of weekly adjacency matrices from rolling 60-day
Pearson correlation of log-returns across the stock universe.
"""

import pickle
import numpy as np
import pandas as pd

from data_loader import load_config


def build_adjacency(log_returns: pd.DataFrame, window: int, threshold: float) -> np.ndarray:
    """
    Compute correlation-based adjacency matrix for a given window of log-returns.
    Returns a symmetric NxN numpy array with self-loops removed.
    """
    corr = log_returns.rolling(window).corr().iloc[-len(log_returns.columns):]
    # corr is a MultiIndex DataFrame; take the last computed correlation matrix
    corr_matrix = log_returns.tail(window).corr().values
    np.fill_diagonal(corr_matrix, 0)  # remove self-loops
    adj = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0.0)
    return adj.astype(np.float32)


def build_adjacency_sequence(feature_store: dict, cfg: dict) -> tuple[list[np.ndarray], list[pd.Timestamp], list[str]]:
    """
    Recomputes adjacency matrix weekly over the full date range.

    Returns:
        adj_matrices : list of NxN arrays, one per week
        week_dates   : corresponding week-end dates
        tickers      : ordered list of node names
    """
    tickers = sorted(feature_store.keys())
    window = cfg["graph"]["correlation_window"]
    threshold = cfg["graph"]["correlation_threshold"]

    # Build a single DataFrame of log-returns: rows=dates, cols=tickers
    lr_df = pd.DataFrame({t: feature_store[t]["log_return"] for t in tickers})

    # Resample to weekly frequency — take last date of each week
    week_ends = lr_df.resample("W").last().index

    adj_matrices, week_dates = [], []

    for week_end in week_ends:
        window_data = lr_df.loc[:week_end].tail(window)
        if len(window_data) < window:
            continue  # not enough history yet
        adj = build_adjacency(window_data, window, threshold)
        adj_matrices.append(adj)
        week_dates.append(week_end)

    # Persist
    processed_dir = cfg["data"]["processed_dir"]
    with open(f"{processed_dir}/adjacency_matrices.pkl", "wb") as f:
        pickle.dump({
            "adjacency": adj_matrices,
            "dates": week_dates,
            "tickers": tickers,
        }, f)

    print(f"[INFO] Built {len(adj_matrices)} weekly adjacency matrices for {len(tickers)} stocks.")
    return adj_matrices, week_dates, tickers


if __name__ == "__main__":
    cfg = load_config()
    with open(f"{cfg['data']['processed_dir']}/features.pkl", "rb") as f:
        feature_store = pickle.load(f)["features"]
    build_adjacency_sequence(feature_store, cfg)
