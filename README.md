# Deep Learning Architectures for Finance

Practical deep-learning patterns for financial classification and sequential data, with reproducible synthetic datasets and architecture-oriented examples.

## Highlights

- MLP for tabular transaction-risk classification
- LSTM and GRU for delinquency-index forecasting
- Scaled dot-product attention from first principles
- Compact Transformer-style classifier for transaction sequences
- Baseline-first evaluation and architecture selection by data structure

## Tech stack

Python · NumPy · pandas · scikit-learn · TensorFlow/Keras

## Repository structure

- `src/tabular_mlp.py` — tabular financial classification
- `src/sequence_models.py` — LSTM/GRU forecasting workflow
- `src/attention_transformer.py` — attention mechanics and sequence classification
- `requirements.txt`

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/tabular_mlp.py
python src/sequence_models.py
python src/attention_transformer.py
```

All examples use synthetic data. The objective is to demonstrate architecture choice, evaluation discipline and reproducible implementation rather than depend on proprietary financial datasets.
