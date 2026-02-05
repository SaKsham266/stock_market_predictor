Stock Market Prediction using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock closing prices using historical time-series data.
The objective is to study how deep learning models capture temporal patterns and trends in financial markets.

Dataset

Source: National Stock Exchange of India (NSE)

Stock: Reliance Industries Limited

Time Period: 2016 – 2026

Frequency: Daily

The dataset consists of historical daily closing prices collected from the NSE website and is used for supervised time-series forecasting.

Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to understand the statistical and temporal characteristics of the stock price data, including:

Long-term and short-term price trends

Volatility patterns

Rolling statistics

Return distributions

EDA was used to guide feature selection and modeling decisions.
To keep the training pipeline clean and focused, EDA code was excluded from the final training script.

Feature Engineering

The following features were derived from the closing price:

Daily Return

10-day Moving Average

10-day Volatility

Closing Price

These features enable the model to learn both trend-following behavior and short-term market fluctuations.

Model Architecture

Model Type: Long Short-Term Memory (LSTM) Neural Network

Architecture:

1 LSTM layer

Dropout for regularization

Fully connected output layer

Look-back Window: 90 days

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

The model predicts the next-day closing price using a fixed-length historical input sequence.

Model Evaluation

The model was evaluated on a chronologically held-out test set, ensuring no future data leakage.

Mean Absolute Error (MAE): ~₹33

Root Mean Squared Error (RMSE): ~₹89

The model demonstrates reasonable trend-following capability, with higher prediction errors observed during periods of elevated market volatility

Evaluation metrics are reported in absolute price units (INR).
Results Visualization

Predicted prices were compared against actual prices to evaluate performance visually.
The comparison plot is saved in the output/ directory.

Application & Deployment

The trained model is exposed through a FastAPI backend, enabling real-time inference.
A Streamlit frontend provides interactive visualization, including:

Historical price trends

Model input window (last 90 prices)

Next-day price prediction

Comparison between recent prices and the predicted value

This separation of concerns follows a clean ML system design pattern.

How to Run

Install dependencies

pip install -r requirements.txt


Train the model

python training/train.py


Start the API

uvicorn api.main:app --reload


Launch the frontend

streamlit run frontend/app.py

Notes

The model is trained on a single stock to demonstrate time-series forecasting and deployment concepts.

The architecture is extensible to additional stocks with retraining.

This project is intended for academic and learning purposes, not financial advice.

