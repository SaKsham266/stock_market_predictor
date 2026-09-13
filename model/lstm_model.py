import numpy as np
import tensorflow as tf


def create_sequences(X, y, time_step=90):
    """
    Turn 2D arrays of scaled features/targets into overlapping windows
    for the LSTM: each input sample is `time_step` rows of features,
    and the target is the value immediately following that window.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:i + time_step])
        y_seq.append(y[i + time_step])
    return np.array(X_seq), np.array(y_seq)


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