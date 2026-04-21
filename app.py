import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Stock Market Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload OHLCV dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna()

    # Visualization
    st.line_chart(df[['Date','Close']].set_index('Date'))

    # Model
    features = ['Return','MA10','MA30','Volatility']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    import streamlit as st
import matplotlib.pyplot as plt

st.write("Model Accuracy:", accuracy_score(y_test, y_pred))

fig, ax = plt.subplots()
ax.plot(merged['Date'], merged['Close'], label='Close Price')
ax.scatter(merged.loc[X_test.index, 'Date'],
           merged.loc[X_test.index, 'Close'],
           c=merged.loc[X_test.index, 'Predicted'],
           cmap='coolwarm', label='Predicted Up/Down')
ax.legend()
st.pyplot(fig)
