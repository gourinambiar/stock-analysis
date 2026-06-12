import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
sys.path.append("C:/Users/Gouri/OneDrive/Documents/stock-analysis")
from src.features import add_features
from src.model import prepare_features, train_model

# Page config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("📈 Stock Price Analysis & Prediction")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "GOOGL", "MSFT", "TSLA"])

# Load data

import yfinance as yf

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return add_features(df)

df = load_data(ticker)

# --- Section 1: Candlestick Chart ---
st.subheader(f"{ticker} Price Chart with Moving Averages")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"],
    high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"], name="MA 20", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="MA 50", line=dict(color="blue")))
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, width='stretch')

# --- Section 2: RSI ---
st.subheader("RSI (Relative Strength Index)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")))
fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig2.update_layout(height=300)
st.plotly_chart(fig2, width='stretch')

# --- Section 3: Bollinger Bands ---
st.subheader("Bollinger Bands")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="black")))
fig3.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper Band", line=dict(color="red", dash="dash")))
fig3.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower Band", line=dict(color="green", dash="dash"),
    fill="tonexty", fillcolor="rgba(128,128,128,0.1)"
))
fig3.update_layout(height=400)
st.plotly_chart(fig3, width='stretch')

# --- Section 4: Prediction ---
st.subheader("Next Day Price Prediction (Linear Regression)")
df_model = prepare_features(df.copy())
model, X_test, y_test, preds = train_model(df_model)

pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
fig4 = go.Figure()
fig4.add_trace(go.Scatter(y=pred_df["Actual"], name="Actual", line=dict(color="blue")))
fig4.add_trace(go.Scatter(y=pred_df["Predicted"], name="Predicted", line=dict(color="orange")))
fig4.update_layout(height=400)
st.plotly_chart(fig4, width='stretch')

# Metrics
from sklearn.metrics import mean_absolute_error, r2_score
col1, col2 = st.columns(2)
col1.metric("MAE", f"${mean_absolute_error(y_test, preds):.2f}")
col2.metric("R² Score", f"{r2_score(y_test, preds):.2f}")