
# Preprocess: simple forward-fill, then MinMax scaling per feature.
# Exports scaler object so predictions can be denormalized.

# Functions:
# - preprocess_df(df) -> df_filled
# - fit_transform(df_train) -> (np_array, scaler_dict)
# - transform_with_scaler(df, scaler_dict) -> np_array


import numpy as np
import pandas as pd
from typing import Tuple, Dict

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # fill missing values with forward-fill then back-fill
    df_f = df.ffill().bfill()
    return df_f

def fit_transform(df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Fit MinMax (0,1) on df and return normalized numpy array and scaler dict.
    scaler dict format: {col: (min, max)}
    """
    df_proc = preprocess_df(df)
    scaler = {}
    arr = df_proc.values.astype(float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    # avoid zero division
    ranges = (maxs - mins).copy()
    ranges[ranges == 0] = 1.0
    arr_norm = (arr - mins) / ranges
    for i, c in enumerate(df_proc.columns):
        scaler[c] = (float(mins[i]), float(maxs[i]))
    return arr_norm, scaler

def transform_with_scaler(df: pd.DataFrame, scaler: Dict) -> np.ndarray:
    df_proc = preprocess_df(df)
    cols = df_proc.columns.tolist()
    arr = df_proc.values.astype(float)
    mins = np.array([scaler[c][0] for c in cols], dtype=float)
    maxs = np.array([scaler[c][1] for c in cols], dtype=float)
    ranges = (maxs - mins).copy()
    ranges[ranges == 0] = 1.0
    arr_norm = (arr - mins) / ranges
    return arr_norm

def denormalize(arr_norm: np.ndarray, cols: list, scaler: Dict) -> np.ndarray:
    mins = np.array([scaler[c][0] for c in cols], dtype=float)
    maxs = np.array([scaler[c][1] for c in cols], dtype=float)
    ranges = (maxs - mins).copy()
    ranges[ranges == 0] = 1.0
    return arr_norm * ranges + mins

if __name__ == "__main__":
    import sys
    from pathlib import Path
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\shiva\Documents\multicommodity-prices-indonesia\data\csv\india_commodities_demo.csv"
    import loader
    df = loader.load_csv_file(path)
    arr, scaler = fit_transform(df)
    print("Normalized shape:", arr.shape)
    print("Scaler sample:", list(scaler.items())[:3])
