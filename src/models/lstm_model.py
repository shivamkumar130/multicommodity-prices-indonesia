\"\"\"lstm_model.py - simple PyTorch LSTM stub\"\"\"
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=13, num_layers=1, out_size=None, batchnorm=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size) if batchnorm else None
        self.fc = nn.Linear(hidden_size, out_size or input_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        if self.bn is not None:
            last = self.bn(last)
        return self.fc(last)
