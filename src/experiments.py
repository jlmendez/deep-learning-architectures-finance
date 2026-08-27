"""Comparable training/evaluation loop for architecture experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def fit_and_score(model, x_train, y_train, x_test, y_test, epochs=12, batch_size=64, verbose=0):
    callbacks = [
        __import__('tensorflow').keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)
    ]
    history = model.fit(x_train, y_train, validation_split=.2, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    probability = np.asarray(model.predict(x_test, verbose=0)).ravel()
    return {
        'model': model.name,
        'roc_auc': float(roc_auc_score(y_test, probability)),
        'epochs_run': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
    }
