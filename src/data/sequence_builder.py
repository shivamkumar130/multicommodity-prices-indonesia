# # src/data/sequence_builder.py

# Build sliding-window sequences for multivariate time-series forecasting.

# We build X: (num_samples, seq_len, n_features)
# and  y: (num_samples, n_features) which is the next-step (one-step ahead) target.


import numpy as np
from typing import Tuple

def build_sequences(arr: np.ndarray, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    arr: shape (T, n_features)
    returns X (N, seq_len, n_features), y (N, n_features)
    """
    T, F = arr.shape
    if T <= seq_len:
        raise ValueError("Time length must be greater than seq_len")
    N = T - seq_len
    X = np.zeros((N, seq_len, F), dtype=float)
    y = np.zeros((N, F), dtype=float)
    for i in range(N):
        X[i] = arr[i:i+seq_len]
        y[i] = arr[i+seq_len]   # predict immediate next step
    return X, y

if __name__ == "__main__":
    import numpy as np
    A = np.arange(100).reshape(100,1).astype(float)
    X,y = build_sequences(A, seq_len=10)
    print("X", X.shape, "y", y.shape)
