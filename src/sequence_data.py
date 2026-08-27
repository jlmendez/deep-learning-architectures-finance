"""Synthetic tabular and sequence datasets for deep-learning demonstrations."""
from __future__ import annotations

import numpy as np


def transaction_data(n=6000, features=12, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, features))
    logit = 1.4*x[:,0] - 1.1*x[:,1] + 0.8*x[:,2]*x[:,3] + 0.5*np.sin(x[:,4]) + rng.normal(0,.7,n)
    p = 1/(1+np.exp(-logit))
    y = rng.binomial(1,p)
    return x.astype('float32'), y.astype('float32')


def financial_sequences(n=2500, steps=24, features=5, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, steps, features)).astype('float32')
    trend = x[:,:,0].mean(axis=1) + 0.6*x[:,-4:,1].mean(axis=1) - 0.4*x[:,:,2].std(axis=1)
    y = (trend + rng.normal(0,.4,n) > 0).astype('float32')
    return x, y
