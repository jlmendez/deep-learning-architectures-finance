"""LSTM and GRU forecasting on a synthetic delinquency index."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers


def make_series(n: int = 180, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.0025 * t
    seasonal = 0.18 * np.sin(2 * np.pi * t / 12)
    shocks = rng.normal(0, 0.06, n)
    return 2.0 + trend + seasonal + shocks


def windows(series: np.ndarray, lookback: int = 12) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(series) - lookback):
        x.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return np.asarray(x)[..., None], np.asarray(y)


def build_recurrent(kind: str, lookback: int = 12) -> keras.Model:
    recurrent = layers.LSTM(32) if kind.lower() == "lstm" else layers.GRU(32)
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        recurrent,
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    return model


def main() -> None:
    series = make_series()
    mean, std = series.mean(), series.std()
    scaled = (series - mean) / std
    x, y = windows(scaled)
    split = int(0.75 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    naive = x_test[:, -1, 0]
    print(f"Naive RMSE: {mean_squared_error(y_test, naive) ** 0.5:.4f}")

    for kind in ("lstm", "gru"):
        model = build_recurrent(kind)
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
        model.fit(
            x_train,
            y_train,
            validation_split=0.20,
            epochs=100,
            batch_size=16,
            callbacks=[stop],
            verbose=0,
        )
        pred = model.predict(x_test, verbose=0).ravel()
        rmse = mean_squared_error(y_test, pred) ** 0.5
        print(f"{kind.upper()} RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
