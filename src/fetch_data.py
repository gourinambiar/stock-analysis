import yfinance as yf
import pandas as pd
import os

def fetch_stocks(tickers, start, end):
    os.makedirs("data/raw", exist_ok=True)
    
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(f"data/raw/{ticker}.csv")
        print(f"Saved {ticker}.csv — {len(df)} rows")

if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    fetch_stocks(tickers, "2020-01-01", "2024-01-01")