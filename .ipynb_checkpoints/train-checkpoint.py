from model.lstm_model import build_lstm_model, create_sequences
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

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

features = ['Return', 'MA_10', 'Volatility_10', 'Close']

target = "Close"
#print("After download:", data.shape)
#print("NaNs per column:\n", data.isna().sum())
TIME_STEP = 60
X = data[features].values
y = data[[target]].values

X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)
#print(type(X_scaled), X_scaled.shape)
#print(type(y_scaled), y_scaled.shape)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_step=TIME_STEP)

#print(X_seq.shape)
#print(y_seq.shape)


train_size = int(len(X_seq) * 0.8)

X_train = X_seq[:train_size]
X_test  = X_seq[train_size:]

y_train = y_seq[:train_size]
y_test  = y_seq[train_size:]
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

model = build_lstm_model(time_steps=60, num_features=4)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
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