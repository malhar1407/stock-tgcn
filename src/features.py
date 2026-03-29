"""
features.py
Computes node features (log-return, RSI, MACD) and target (k-day forward log-return).
Applies Z-score normalization to all features except RSI (Min-Max).
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_loader import load_config, get_universe


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Returns a DataFrame with 5 features + target for a single stock."""
    close = df["Close"].squeeze()
    fp = cfg["features"]

    lr = log_returns(close)
    r = rsi(close, fp["rsi_period"])
    macd_line, macd_sig, macd_hist = macd(close, fp["macd_fast"], fp["macd_slow"], fp["macd_signal"])
    target = log_returns(close).shift(-fp["target_horizon"])  # k-day forward log-return

    out = pd.DataFrame({
        "log_return": lr,
        "rsi": r,
        "macd_line": macd_line,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "target": target,
    }, index=close.index)

    return out.dropna()


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalize features in-place. Returns normalized df and fitted scalers.
    RSI: Min-Max [0,1]. All others: Z-score.
    """
    scalers = {}

    rsi_scaler = MinMaxScaler()
    df["rsi"] = rsi_scaler.fit_transform(df[["rsi"]])
    scalers["rsi"] = rsi_scaler

    zscore_cols = ["log_return", "macd_line", "macd_signal", "macd_hist", "target"]
    for col in zscore_cols:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df, scalers


def build_feature_store(cfg: dict) -> tuple[dict, dict]:
    """
    Returns:
        feature_store: {ticker: normalized DataFrame}
        scaler_store:  {ticker: {feature: scaler}}
    """
    universe = get_universe(cfg)
    feature_store, scaler_store = {}, {}

    for ticker, df in universe.items():
        feat_df = compute_features(df, cfg)
        if len(feat_df) < cfg["features"]["lookback"] + cfg["features"]["target_horizon"]:
            print(f"[WARN] {ticker} has insufficient data, skipping.")
            continue
        feat_df, scalers = normalize(feat_df)
        feature_store[ticker] = feat_df
        scaler_store[ticker] = scalers

    # Align all tickers to common date index
    common_index = None
    for df in feature_store.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    feature_store = {t: df.loc[common_index] for t, df in feature_store.items()}

    # Persist
    processed_dir = cfg["data"]["processed_dir"]
    with open(f"{processed_dir}/features.pkl", "wb") as f:
        pickle.dump({"features": feature_store, "scalers": scaler_store}, f)

    print(f"[INFO] Feature store built: {len(feature_store)} stocks, {len(common_index)} trading days.")
    return feature_store, scaler_store


if __name__ == "__main__":
    cfg = load_config()
    build_feature_store(cfg)
