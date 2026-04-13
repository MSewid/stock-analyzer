import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("Simple Stock Analyzer")

# User inputs
ticker = st.text_input("Enter Ticker", "AAPL")
period = st.selectbox("Select Duration", ["1mo", "3mo", "6mo", "1y", "2y"])

if st.button("Analyze"):
    data = yf.download(ticker, period=period)

    if data.empty:
        st.error("Invalid ticker or no data.")
    else:
        close = data["Close"]

        # Current price
        current_price = close.iloc[-1]
        start_price = close.iloc[0]
        pct_change = ((current_price - start_price) / start_price) * 100

        # Moving average
        data["MA20"] = close.rolling(20).mean()

        # Simple ARIMA forecast
        model = ARIMA(close, order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        # Display stats
        st.subheader(ticker.upper())
        st.write(f"Current Price: ${current_price:.2f}")
        st.write(f"Change over period: {pct_change:.2f}%")

        # Plot
        fig, ax = plt.subplots()
        ax.plot(close, label="Price")
        ax.plot(data["MA20"], label="MA20")

        # Forecast dates
        future_index = pd.date_range(start=close.index[-1], periods=11, freq="B")[1:]
        ax.plot(future_index, forecast, label="Forecast")

        ax.legend()
        st.pyplot(fig)