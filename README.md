# Stock Market Prediction using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock closing prices using historical data.

The goal is to study how deep learning models can capture trends in financial time-series data.

---

## Dataset
- Source: Yahoo Finance
- Stock: Reliance Industries Limited (NSE)
- Time period: 2016 – 2026
- Frequency: Daily

The dataset contains historical closing prices and is used for supervised time-series prediction.

---

## Exploratory Data Analysis (EDA)

Exploratory analysis was initially performed to understand:
- price trends
- volatility patterns
- rolling statistics
- return distributions

EDA was used to guide feature selection and was removed from the final training script to keep the codebase clean and focused.

---

## Feature Engineering

The following features were derived from the closing price:

- Daily Return
- 10-day Moving Average
- 10-day Volatility
- Closing Price

These features help the model learn both short-term and long-term market behavior.

---

## Model Architecture

- Model: LSTM Neural Network
- Layers: 2 LSTM layers with Dropout
- Look-back window: 90 days
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)

The model predicts the next-day closing price based on past sequences.

---

## Model Evaluation

The model was evaluated on a held-out test set.

- Mean Absolute Error (MAE): ~₹53
- Root Mean Squared Error (RMSE): ~₹113

The results indicate good trend prediction ability, with higher errors during volatile periods.

---

## Results Visualization

Predicted prices were compared with actual prices, and the graph is saved to the `output/` directory.

---

## How to Run

```bash

