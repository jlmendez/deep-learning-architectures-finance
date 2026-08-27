"""MLP benchmark for synthetic transaction-risk classification."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


def make_transactions(n: int = 6000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.8, 0.8, n)
    balance_ratio = rng.beta(2.0, 5.0, n)
    new_device = rng.binomial(1, 0.12, n)
    night = rng.binomial(1, 0.20, n)
    failed_attempts = rng.poisson(0.22, n)
    international = rng.binomial(1, 0.08, n)
    x = np.column_stack([amount, balance_ratio, new_device, night, failed_attempts, international])
    risk_score = (
        0.008 * amount
        + 1.8 * balance_ratio
        + 1.1 * new_device
        + 0.8 * night
        + 0.9 * failed_attempts
        + 0.6 * international
        + rng.normal(0, 0.9, n)
    )
    threshold = np.quantile(risk_score, 0.78)
    y = (risk_score >= threshold).astype(int)
    return x, y


def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    x, y = make_transactions()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    baseline = LogisticRegression(max_iter=2000).fit(x_train_s, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(x_test_s)[:, 1])

    model = build_mlp(x_train_s.shape[1])
    callback = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=6, restore_best_weights=True
    )
    model.fit(
        x_train_s,
        y_train,
        validation_split=0.20,
        epochs=60,
        batch_size=64,
        callbacks=[callback],
        verbose=0,
    )
    proba = model.predict(x_test_s, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)

    print(f"Logistic baseline AUC: {baseline_auc:.3f}")
    print(f"MLP AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, pred, digits=3))


if __name__ == "__main__":
    main()
