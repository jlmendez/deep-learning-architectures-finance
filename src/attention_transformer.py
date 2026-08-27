"""Scaled dot-product attention and a compact Transformer-style classifier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = embeddings.shape[1]
    q = embeddings @ rng.normal(size=(d, d))
    k = embeddings @ rng.normal(size=(d, d))
    scores = q @ k.T / np.sqrt(d)
    return softmax(scores, axis=1)


def make_sequences(n: int = 3000, steps: int = 10, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, steps, 4), dtype="float32")
    y = np.zeros(n, dtype=int)
    for i in range(n):
        amount = rng.lognormal(0.0, 0.8, steps)
        new_device = rng.binomial(1, 0.12, steps)
        night = rng.binomial(1, 0.20, steps)
        failed = rng.poisson(0.20, steps)
        x[i, :, 0] = amount
        x[i, :, 1] = new_device
        x[i, :, 2] = night
        x[i, :, 3] = failed
        y[i] = int(
            np.any((amount > 2.5) & (new_device == 1) & (night == 1))
            or failed.sum() >= 3
        )
    return x, y


def build_model(steps: int = 10, features: int = 4) -> keras.Model:
    inputs = keras.Input(shape=(steps, features))
    x = layers.Dense(32)(inputs)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + attn)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    tokens = ["client", "reports", "charge", "not", "recognized"]
    embeddings = np.random.default_rng(7).normal(size=(len(tokens), 8))
    weights = scaled_dot_product_attention(embeddings)
    print("Attention matrix shape:", weights.shape)
    print("Row sums:", np.round(weights.sum(axis=1), 6))

    x, y = make_sequences()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model(x.shape[1], x.shape[2])
    stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )
    model.fit(
        x_train,
        y_train,
        validation_split=0.20,
        epochs=30,
        batch_size=64,
        callbacks=[stop],
        verbose=0,
    )
    proba = model.predict(x_test, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    print(f"Transformer-style AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, pred, digits=3))


if __name__ == "__main__":
    main()
