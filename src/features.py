import pandas as pd

def add_features(df):
    # Moving Averages
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    
    # Daily Returns
    df["Daily_Return"] = df["Close"].pct_change()
    
    # Volatility (20 day rolling std of returns)
    df["Volatility"] = df["Daily_Return"].rolling(20).std()
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    
    # Bollinger Bands
    df["BB_Upper"] = df["MA_20"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["MA_20"] - 2 * df["Close"].rolling(20).std()
    
    return df.dropna()