# SPOT Testnet Terms of Use
from binance.client import Client
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import time

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# 🔐 API KEYS (TESTNET)
# =========================
API_KEY = "PASTE_YOUR_TESTNET_KEY"
API_SECRET = "PASTE_YOUR_TESTNET_SECRET"

client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

symbol = "BTCUSDT"
trade_qty = 0.001  # pieni määrä turvallisesti

# =========================
# 1. Data + AI training
# =========================
def load_data():
    df = yf.download("BTC-USD", period="7d", interval="5m")
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["ema"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
    return df.dropna()

df = load_data()
features = ["Close", "rsi", "ema"]
data = df[features].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

SEQ = 30
X, y = [], []
for i in range(SEQ, len(scaled)):
    X.append(scaled[i-SEQ:i])
    y.append(scaled[i][0])

X = np.array(X)
y = np.array(y)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=3, batch_size=32)

# =========================
# 2. Trading logic
# =========================
position = None

def get_latest_price():
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])

def predict_next():
    df = load_data()
    data = df[features].values
    scaled = scaler.transform(data)

    last_seq = np.expand_dims(scaled[-SEQ:], axis=0)
    pred_scaled = model.predict(last_seq)[0,0]

    dummy = np.zeros((1, len(features)))
    dummy[0,0] = pred_scaled

    return scaler.inverse_transform(dummy)[0,0]

def trade_loop():
    global position

    price = get_latest_price()
    pred = predict_next()

    print(f"Price: {price} | Prediction: {pred}")

    if pred > price * 1.002 and position is None:
        order = client.create_order(
            symbol=symbol,
            side="BUY",
            type="MARKET",
            quantity=trade_qty
        )
        print("🟢 BUY executed")
        position = price

    elif pred < price * 0.998 and position is not None:
        order = client.create_order(
            symbol=symbol,
            side="SELL",
            type="MARKET",
            quantity=trade_qty
        )
        print("🔴 SELL executed")
        position = None

The Binance Spot Testnet and Futures Testnet are subject to the [Testnet Terms of Use](https://www.binance.com/en/about-legal/terms-testnets). <br>
Please read it carefully before proceeding.
