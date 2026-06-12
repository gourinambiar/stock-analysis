import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def prepare_features(df, lag_days=5):
    # Create lag features — yesterday's price, 2 days ago, etc.
    for i in range(1, lag_days + 1):
        df[f"Lag_{i}"] = df["Close"].shift(i)
    
    # Target = tomorrow's closing price
    df["Target"] = df["Close"].shift(-1)
    
    return df.dropna()

def train_model(df):
    feature_cols = [f"Lag_{i}" for i in range(1, 6)] + ["MA_20", "RSI", "Volatility"]
    X = df[feature_cols]
    y = df["Target"]
    
    # shuffle=False is critical for time series — don't mix future into past
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"R²: {r2_score(y_test, preds):.2f}")
    
    return model, X_test, y_test, preds