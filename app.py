import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("Stock Analyzer")

# -----------------------
# Inputs
# -----------------------
ticker = st.text_input("Enter Ticker", "AAPL")
period = st.selectbox("Select Duration", ["1mo", "3mo", "6mo", "1y", "2y"])

sma_options = st.multiselect("Select Moving Averages", [10, 50, 200], default=[20] if 20 in [10,50,200] else [10])

use_prophet = st.checkbox("Enable Forecast (Prophet)")
log_scale = st.checkbox("Log Scale Chart")

# -----------------------
# Cache data
# -----------------------
@st.cache_data
def load_data(ticker, period):
    return yf.download(ticker, period=period)

# -----------------------
# Run analysis
# -----------------------
if st.button("Analyze"):
    try:
        data = load_data(ticker, period)

        if data.empty:
            st.error("Invalid ticker or no data found.")
            st.stop()

        close = data["Close"]

        # Stats
        current_price = float(close.iloc[-1])
        start_price = float(close.iloc[0])
        pct_change = ((current_price - start_price) / start_price) * 100

        st.subheader(ticker.upper())
        st.write(f"Current Price: ${current_price:.2f}")
        st.write(f"Change over period: {pct_change:.2f}%")

        # -----------------------
        # Indicators
        # -----------------------
        for sma in sma_options:
            data[f"SMA{sma}"] = close.rolling(sma).mean()

        # -----------------------
        # Plot Price + SMA
        # -----------------------
        fig, ax = plt.subplots()

        ax.plot(close, label="Price")

        for sma in sma_options:
            ax.plot(data[f"SMA{sma}"], label=f"SMA{sma}")

        if log_scale:
            ax.set_yscale("log")

        ax.legend()
        st.pyplot(fig)

        # -----------------------
        # Volume chart
        # -----------------------
        st.subheader("Volume")
        fig2, ax2 = plt.subplots()
        ax2.bar(data.index, data["Volume"])
        st.pyplot(fig2)

        # -----------------------
        # Prophet Forecast
        # -----------------------
        if use_prophet:
            st.subheader("Forecast (Prophet)")

            df = data.reset_index()[["Date", "Close"]]
            df.columns = ["ds", "y"]

            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            fig3 = model.plot(forecast)
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Something went wrong: {e}")