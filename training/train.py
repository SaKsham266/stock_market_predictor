from model.lstm_model import build_lstm_model, create_sequences  

from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import joblib

from data.data_loader import load_stock_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
TIME_STEP = 90

data = load_stock_data("data/RELIANCE.csv")



print("Rows downloaded:", len(data))
print(data.head())

data["Return"] = data["Close"].pct_change()
data["MA_10"] = data["Close"].rolling(window=10).mean()
data["Volatility_10"] = data["Close"].rolling(window=10).std()
data = data.dropna()

features = ['Close', 'Return', 'MA_10', 'Volatility_10']

target = "Close"
print("After download:", data.shape)
print("NaNs per column:\n", data.isna().sum())

X = data[features].values
y = data[[target]].values

train_size = int(len(X) * 0.8)

X_train_raw = X[:train_size]
X_test_raw = X[train_size:]

y_train_raw = y[:train_size]
y_test_raw = y[train_size:]

X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaler.fit(X_train_raw)
y_scaler.fit(y_train_raw)

X_train_scaled = X_scaler.transform(X_train_raw)
X_test_scaled = X_scaler.transform(X_test_raw)

y_train_scaled = y_scaler.transform(y_train_raw)
y_test_scaled = y_scaler.transform(y_test_raw)

X_scaled = np.concatenate((X_train_scaled, X_test_scaled), axis=0)
y_scaled = np.concatenate((y_train_scaled, y_test_scaled), axis=0)
#print(type(X_scaled), X_scaled.shape)
#print(type(y_scaled), y_scaled.shape)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_step=TIME_STEP)

#print(X_seq.shape)
#print(y_seq.shape)

seq_train_size = int(len(X_seq) * 0.7)
seq_val_size = int(len(X_seq) * 0.15)

X_train = X_seq[:seq_train_size]
X_val = X_seq[seq_train_size:seq_train_size + seq_val_size]
X_test = X_seq[seq_train_size + seq_val_size:]

y_train = y_seq[:seq_train_size]
y_val = y_seq[seq_train_size:seq_train_size + seq_val_size]
y_test = y_seq[seq_train_size + seq_val_size:]
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

num_features = X_train.shape[2]
model = build_lstm_model(time_steps=TIME_STEP, num_features=num_features)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

y_pred_scaled = model.predict(X_test)

# Inverse scaling
y_test_actual = y_scaler.inverse_transform(y_test)
y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print("RMSE:", rmse)
print("MAE:", mae)
naive_pred = y_test_actual[:-1]
naive_actual = y_test_actual[1:]
naive_mae = mean_absolute_error(naive_actual, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(naive_actual, naive_pred))
print("Naive MAE:", naive_mae)
print("Naive RMSE:", naive_rmse)

plt.figure(figsize=(12, 5))

plt.plot(y_test_actual, label="Actual Price")
plt.plot(y_pred_actual, label="Predicted Price")

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("LSTM Stock Price Prediction")
plt.legend()

plt.tight_layout()
plt.savefig("output/prediction_plot.png")
plt.close()
model.save("model/saved_model/lstm_model.keras")
joblib.dump(X_scaler, "model/saved_model/x_scaler.pkl")
joblib.dump(y_scaler, "model/saved_model/y_scaler.pkl")
