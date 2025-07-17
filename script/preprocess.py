# preprocess.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from script.get_data import get_mexc_data

def preprocess(seq_length=180):
    df = get_mexc_data(limit=1000)

    # add technical feature
    df['SMA_5'] = df['Close'].rolling(window=5).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Price_Change'] = df['Close'].diff()

    df.dropna(inplace=True)

    features = ['Close', 'SMA_5', 'MACD', 'RSI', 'Price_Change']
    target = 'Close'

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i-seq_length:i])
        y.append(scaled[i, 0])  # predict Close

    X, y = np.array(X), np.array(y)

    split_index = int(len(X) * 0.8)
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:], scaler
