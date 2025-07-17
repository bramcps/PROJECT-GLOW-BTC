import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from script.preprocess import preprocess

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "btc_lstm_model.h5")
EPOCHS = 50
BATCH_SIZE = 32
SEQ_LENGTH = 180

X_train, y_train, X_val, y_val, scaler = preprocess(seq_length=SEQ_LENGTH)

# build model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# training with EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# eval
y_pred = model.predict(X_val)
y_val_inv = scaler.inverse_transform(np.concatenate([y_val.reshape(-1, 1)] * 5, axis=1))[:, 0]
y_pred_inv = scaler.inverse_transform(np.concatenate([y_pred] * 5, axis=1))[:, 0]

mse = mean_squared_error(y_val_inv, y_pred_inv)
mae = mean_absolute_error(y_val_inv, y_pred_inv)
r2 = r2_score(y_val_inv, y_pred_inv)

print("\n=== Evaluation Metrics ===")
print(f"R² Score: {r2:.4f}")
print(f"MSE     : {mse:.4f}")
print(f"MAE     : {mae:.4f}")

# save model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
