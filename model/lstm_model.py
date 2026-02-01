import numpy as np
import tensorflow as tf

from  tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore

def create_sequences(X, y, time_step=60):
    Xs, ys = [], []
    for i in range(time_step, len(X)):
        Xs.append(X[i - time_step : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_lstm_model(time_steps, num_features):
    model = Sequential([
        Input(shape=(time_steps, num_features)),

        # First LSTM layer (returns sequences)
        LSTM(50, return_sequences=True),
        Dropout(0.2),

        # Second LSTM layer
        LSTM(50),
        Dropout(0.2),

        # Output layer
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model