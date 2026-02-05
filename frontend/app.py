# frontend/app.py
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"
TIME_STEP = 90

st.title(" Stock Price Predictor ")

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

            # Plot actual prices with dates
            plt.figure(figsize=(10, 4))
            plt.plot(
                recent_df["datetime"],
                recent_df["close"],
                label="Actual Price",
                color="blue"
            )

            # Predicted price line
            plt.axhline(
                y=prediction,
                color="red",
                linestyle="--",
                label="Predicted Next Price"
            )

            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.title("Actual Prices vs Predicted Next Price")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(plt)

        else:
            st.error(
                f"API Error {response.status_code}: {response.text}"
            )

    except Exception as e:
        st.error(f"Request failed: {e}")
