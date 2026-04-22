import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Stock Market Prediction Dashboard")

# --- Mode Selection ---
mode = st.radio("Choose data source:", ["Upload Historical CSV", "Live Current Stock"])

# --- Historical CSV Mode ---
if mode == "Upload Historical CSV":
    uploaded_file = st.file_uploader("Upload OHLCV dataset (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Debug: show columns
        st.write("Columns in uploaded CSV:", df.columns.tolist())

        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Feature engineering
        df['return'] = df['close'].pct_change()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma30'] = df['close'].rolling(window=30).mean()
        df['volatility'] = df['return'].rolling(window=10).std()

        # Target variable
        df['target'] = (df['return'].shift(-1) > 0).astype(int)
        df = df.dropna()

        # Visualization
        st.subheader("Closing Price Trend (Historical CSV)")
        st.line_chart(df[['date', 'close']].set_index('date'))

        # Model training
        features = ['return','ma10','ma30','volatility']
        X = df[features]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Model Accuracy:", accuracy_score(y_test, y_pred))

        # Predictions overlay
        df.loc[X_test.index, 'predicted'] = y_pred
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['close'], label='Close Price')
        ax.scatter(df.loc[X_test.index, 'date'],
                   df.loc[X_test.index, 'close'],
                   c=df.loc[X_test.index, 'predicted'],
                   cmap='coolwarm', label='Predicted Up/Down')
        ax.legend()
        st.pyplot(fig)

# --- Live Current Stock Mode ---
elif mode == "Live Current Stock":
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, TSLA, NVDA, META)")

    if ticker:
        # Fetch live OHLCV data
        df = yf.download(ticker, period="1y", interval="1d")
        df = df.reset_index()

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([c for c in col if c]).strip().lower() for col in df.columns.values]
        else:
            df.columns = df.columns.str.strip().str.lower()

        # Remap ticker-specific columns to generic names
        for col in df.columns:
            if col.startswith("close"):
                df.rename(columns={col: "close"}, inplace=True)
            if col.startswith("open"):
                df.rename(columns={col: "open"}, inplace=True)
            if col.startswith("high"):
                df.rename(columns={col: "high"}, inplace=True)
            if col.startswith("low"):
                df.rename(columns={col: "low"}, inplace=True)
            if col.startswith("volume"):
                df.rename(columns={col: "volume"}, inplace=True)

        # Debug: show columns
        st.write("Columns in live data (after remap):", df.columns.tolist())

        # Feature engineering
        if 'close' in df.columns:
            df['return'] = df['close'].pct_change()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma30'] = df['close'].rolling(window=30).mean()
            df['volatility'] = df['return'].rolling(window=10).std()

            # Target variable
            df['target'] = (df['return'].shift(-1) > 0).astype(int)
            df = df.dropna()

            # Visualization
            st.subheader(f"{ticker} Closing Price Trend (Live Data)")
            st.line_chart(df[['date', 'close']].set_index('date'))

            # Model training
            features = ['return','ma10','ma30','volatility']
            X = df[features]
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Model Accuracy:", accuracy_score(y_test, y_pred))

            # Predict next day
            latest_features = X.iloc[-1].values.reshape(1, -1)
            next_day_pred = model.predict(latest_features)[0]
            st.subheader("Prediction for Next Day")
            st.write("UP" if next_day_pred == 1 else "DOWN")
        else:
            st.error("No 'close' column found in data. Check ticker or column names.")
