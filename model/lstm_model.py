import numpy as np
import tensorflow as tf


def build_lstm_model(time_steps, num_features):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, num_features)),

        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model