"""LSTM and GRU builders for financial sequences."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm(steps: int, features: int, units=48, dropout=.20):
    inputs = keras.Input(shape=(steps, features))
    x = layers.LSTM(units, dropout=dropout)(inputs)
    x = layers.Dense(24, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='financial_lstm')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])
    return model


def build_gru(steps: int, features: int, units=48, dropout=.20):
    inputs = keras.Input(shape=(steps, features))
    x = layers.GRU(units, dropout=dropout)(inputs)
    x = layers.Dense(24, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='financial_gru')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])
    return model
