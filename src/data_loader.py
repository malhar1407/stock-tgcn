"""
data_loader.py
Downloads OHLCV data for the S&P 100 universe via yfinance.
Filters by liquidity (minimum average daily volume).
"""

import os
import yaml
import yfinance as yf
import pandas as pd

# S&P 100 tickers (OEX constituents)
SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "INTU", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_data(cfg: dict) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all tickers. Returns dict of {ticker: DataFrame}."""
    raw_dir = cfg["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    data = {}

    for ticker in SP100_TICKERS:
        fpath = os.path.join(raw_dir, f"{ticker}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print(f"[WARN] No data for {ticker}, skipping.")
                continue
            df.to_csv(fpath)
        data[ticker] = df

    return data


def filter_by_liquidity(data: dict[str, pd.DataFrame], min_avg_volume: float = 5e6) -> dict[str, pd.DataFrame]:
    """Keep only tickers with average daily volume above threshold."""
    filtered = {
        ticker: df for ticker, df in data.items()
        if df["Volume"].mean() >= min_avg_volume
    }
    removed = set(data) - set(filtered)
    if removed:
        print(f"[INFO] Removed {len(removed)} illiquid tickers: {removed}")
    print(f"[INFO] Universe size after liquidity filter: {len(filtered)}")
    return filtered


def get_universe(cfg: dict) -> dict[str, pd.DataFrame]:
    data = download_data(cfg)
    return filter_by_liquidity(data)


if __name__ == "__main__":
    cfg = load_config()
    universe = get_universe(cfg)
    print(f"Final universe: {list(universe.keys())}")
