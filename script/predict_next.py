import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from script.preprocess import preprocess
import os

MODEL_PATH = os.path.join("models", "btc_lstm_model.h5")

# same as train
SEQ_LENGTH = 180

model = load_model(MODEL_PATH, compile=False)
X_train, y_train, X_val, y_val, scaler = preprocess(seq_length=SEQ_LENGTH)

# last sequence to predict
latest_sequence = X_val[-1]  # shape: (SEQ_LENGTH, num_features)
latest_sequence = np.expand_dims(latest_sequence, axis=0)  # shape: (1, SEQ_LENGTH, num_features)

# predict
predicted_scaled = model.predict(latest_sequence)
predicted_price = scaler.inverse_transform(
    np.concatenate([predicted_scaled, np.zeros((1, X_val.shape[2]-1))], axis=1)
)[0, 0]

print(f"🔮 Next Close BTC Price: ${predicted_price:,.2f}")
