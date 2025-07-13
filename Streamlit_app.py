# Streamlit-Based Stock Predictor App (Automated + Password Protected)
# ------------------------------------------------
# Requirements:
# pip install streamlit yfinance pandas numpy scikit-learn matplotlib seaborn

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# --- RSI Helper Function ---
def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Feature Engineering ---
def add_features(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# --- Train Model ---
def train_model(df):
    features = ['SMA_5', 'SMA_20', 'RSI', 'Return']
    X = df[features]
    y = df['Target']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, features

# --- Predict Tomorrow ---
def predict_next_day(model, df, features):
    latest_data = df.iloc[-1:][features]
    prediction = model.predict(latest_data)[0]
    proba = model.predict_proba(latest_data)[0][prediction]
    return prediction, proba

# --- Streamlit UI ---
def main():
    st.title("📊 Stockbroker Ai - Market Monitor Dashboard")
    password = st.text_input("Enter password to unlock app:", type="password")

    if password != "Shaurya@2313":
        st.warning("Access denied. Enter correct password.")
        st.stop()

    st.markdown("Welcome to your private market overview tool.")

    if st.button("🚀 GO"):
        ticker = 'AAPL'
        df = yf.download(ticker, start='2015-01-01')
        df = add_features(df)
        model, features = train_model(df)
        pred, proba = predict_next_day(model, df, features)

        result = "📈 UP" if pred == 1 else "📉 DOWN"
        st.subheader(f"Market Signal: {result}")
        st.write(f"Confidence: {proba * 100:.2f}%")

        st.line_chart(df[['Close']])
        st.success("Analysis complete.")

if __name__ == '__main__':
    main()
