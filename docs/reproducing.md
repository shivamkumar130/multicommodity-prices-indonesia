# Reproducing the project

This repository is a working scaffold for commodity price forecasting.

## Steps

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide a numeric CSV file with commodity price columns.
   - Put it in `data/csv/` or pass its path with `--csv`.
3. Run training:
   ```bash
   python src/training/train.py --csv data/csv/your_data.csv --seq_len 30 --epochs 50
   ```

## What to expect

- The script loads the CSV, normalizes feature values, builds time-series sequences, and trains a simple LSTM.
- Output includes training loss, validation MAE, and per-feature MAE.

## Important

- `data/csv/` must contain a valid numeric dataset.
- The experiment scripts in `src/experiments/` are not fully wired to `train.py` yet.
