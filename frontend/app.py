# frontend/app.py

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"
TIME_STEP = 90

st.title("📈 Stock Price Predictor")

# Load data
df = pd.read_csv("data/RELIANCE.csv")
df.columns = df.columns.str.strip().str.lower()
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

st.subheader("Raw Closing Prices")
st.subheader("Closing Price Over Time")

st.line_chart(
    df.set_index("datetime")["close"]
)

# Take last 90 prices
last_90 = df["close"].tail(TIME_STEP).tolist()

st.subheader("Last 90 Closing Prices")
st.write(last_90)

# Call API
if st.button("Predict Next Price"):
    payload = {"prices": last_90}

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json()["prediction"]

            st.success(f"📈 Predicted Next Price: ₹{prediction:.2f}")

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
