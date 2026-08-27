"""Feed-forward neural network for tabular financial classification."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(n_features: int, hidden=(64,32), dropout=0.20, learning_rate=1e-3):
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in hidden:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='financial_mlp')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])
    return model
