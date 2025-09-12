# src/training/evaluate.py
import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def per_feature_mae(y_true: np.ndarray, y_pred: np.ndarray):
    # returns array of MAE per feature
    return np.mean(np.abs(y_true - y_pred), axis=0)
