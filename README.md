# Multicommodity Prices — Indonesia

This repository is a minimal implementation for multivariate commodity price forecasting using PyTorch.

## What it does

- loads a numeric CSV file containing multiple commodity price series
- preprocesses data with forward-fill and MinMax scaling
- builds sliding-window sequences for one-step-ahead forecasting
- trains a simple LSTM regression model
- evaluates mean absolute error (MAE) on test data

## Install

```bash
pip install -r requirements.txt
```

## Prepare data

Place a numeric CSV file in `data/csv/` or provide a path with `--csv`.
The file should contain only numeric commodity columns. Non-numeric columns are dropped.

## Run training

```bash
python src/training/train.py --csv data/csv/your_data.csv --seq_len 30 --epochs 50
```

## Notes

- This version currently trains only the `SimpleLSTM` model.
- The experiment scripts in `src/experiments/` are placeholders and assume additional CLI support.
- Use `docs/reproducing.md` for a shorter reproducibility summary.
