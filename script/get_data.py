import requests
import pandas as pd
from datetime import datetime

def get_mexc_data(limit=1000):
    url = "https://api.mexc.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": str(limit)
    }

    response = requests.get(url, params=params)
    data = response.json()

    if not isinstance(data, list):
        raise ValueError(f"Gagal ambil data dari MEXC: {data}")

    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume"
    ])

    df['Datetime'] = pd.to_datetime(df['Open Time'], unit='ms')
    df = df.sort_values("Datetime")

    df = df.astype({
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": float
    })

    print(f"Data MEXC loaded: {len(df)} rows")
    return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
