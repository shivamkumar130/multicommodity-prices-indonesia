# src/models/lstm_model.py
"""
Simple Multivariate LSTM model (PyTorch).

Usage:
    model = SimpleLSTM(input_size=n_features, hidden_size=64, num_layers=1, out_size=n_features, batchnorm=True)
"""
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, out_size=None, batchnorm=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.batchnorm = nn.BatchNorm1d(hidden_size) if batchnorm else None
        self.fc = nn.Linear(hidden_size, out_size or input_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        last = out[:, -1, :]   # take last hidden state
        if self.batchnorm is not None:
            # batchnorm expects (batch, hidden) -> (batch, hidden)
            last = self.batchnorm(last)
        out = self.fc(last)    # (batch, out_size)
        return out
