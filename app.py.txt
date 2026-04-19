import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------------------------------
# Streamlit Dashboard
# -------------------------------
st.title("📈 Stock Market Analysis & Prediction")

# Sidebar for user input
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)

st.subheader(f"Raw Data for {ticker}")
st.write(data.tail())

# Plot closing price
st.subheader("Closing Price Trend")
fig = px.line(data, x=data.index, y="Close", title=f"{ticker} Closing Price")
st.plotly_chart(fig)

# Moving averages
st.subheader("Moving Averages")
ma50 = data['Close'].rolling(50).mean()
ma200 = data['Close'].rolling(200).mean()
fig_ma = px.line(data, x=data.index, y=[data['Close'], ma50, ma200],
                 labels={'value':'Price', 'variable':'Legend'},
                 title=f"{ticker} Closing Price with MA50 & MA200")
st.plotly_chart(fig_ma)

# Returns distribution
st.subheader("Daily Returns Distribution")
data['Daily Return'] = data['Close'].pct_change()
fig_ret = px.histogram(data, x="Daily Return", nbins=50, title="Distribution of Daily Returns")
st.plotly_chart(fig_ret)

# Volume plot
st.subheader("Trading Volume")
fig_vol = px.bar(data, x=data.index, y="Volume", title=f"{ticker} Trading Volume")
st.plotly_chart(fig_vol)

# -------------------------------
# Placeholder for ML Predictions
# -------------------------------
st.subheader("📊 Model Predictions (Demo)")
st.info("This section can be extended to show ML model forecasts. For now, it’s a placeholder.")