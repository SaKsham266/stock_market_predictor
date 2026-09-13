# frontend/app.py

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import sys
import os

# Allow importing data.data_loader when run via `streamlit run frontend/app.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import adjust_for_splits

API_URL = "http://127.0.0.1:8000/predict"
TIME_STEP = 90

st.title("📈 Stock Price Predictor")

# Load data
df = pd.read_csv("data/RELIANCE.csv")
df.columns = df.columns.str.strip().str.lower()
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# Fix unadjusted stock splits (e.g. RELIANCE's 2017 and 2024 bonus issues)
# so the chart and the prices sent to the API don't contain a fake ~-50%
# single-day move. Must match the same adjustment applied in train.py.
df = adjust_for_splits(df, close_col="close")

st.subheader("Raw Closing Prices")
st.subheader("Closing Price Over Time")

st.line_chart(
    df.set_index("datetime")["close"]
)

# Take last 90 prices + 10 extra days of history so the API can compute
# the 10-day rolling features (MA_10 / Volatility_10) without NaNs in the
# final 90-step window it feeds to the model.
last_100 = df["close"].tail(TIME_STEP + 10).tolist()

st.subheader("Last 90 Closing Prices")
st.write(last_100[-TIME_STEP:])

# Call API
if st.button("Predict Next Price"):
    payload = {"prices": last_100}

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            predicted_return = result["predicted_return"]

            return_pct = predicted_return * 100
            direction = "🔺" if predicted_return >= 0 else "🔻"

            col1, col2 = st.columns(2)
            col1.metric("Predicted Next Price", f"₹{prediction:.2f}")
            col2.metric(
                "Predicted Return",
                f"{direction} {return_pct:+.2f}%",
            )

            # Prepare recent data
            recent_df = df.tail(120).copy()
            recent_df["datetime"] = pd.to_datetime(recent_df["datetime"])

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))

            # Plot actual prices
            ax.plot(
                recent_df["datetime"],
                recent_df["close"],
                label="Actual Price",
                color="blue"
            )

            # Plot predicted price
            ax.axhline(
                y=prediction,
                color="red",
                linestyle="--",
                label="Predicted Next Price"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title("Actual Prices vs Predicted Next Price")
            ax.legend()

            fig.autofmt_xdate()
            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        else:
            st.error(
                f"API Error {response.status_code}: {response.text}"
            )

    except Exception as e:
        st.error(f"Request failed: {e}")