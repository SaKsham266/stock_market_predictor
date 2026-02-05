import numpy as np
import tensorflow as tf

from  tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore



def build_lstm_model(time_steps, num_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, num_features)),
        Dropout(0.2),
        LSTM(64),
        Dense(1)
    ])
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model
