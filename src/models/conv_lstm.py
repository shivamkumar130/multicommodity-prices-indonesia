# src/models/conv_lstm.py
"""
CNN + LSTM (Conv-LSTM) encoder.

Architecture:
  Input: (batch, seq_len, features)
  -> permute to (batch, features, seq_len)
  -> apply a stack of Conv1d layers along time (temporal convolutions)
     (keeps sequence length via padding)
  -> permute to (batch, seq_len, conv_channels[-1]) to feed LSTM
  -> LSTM -> take last hidden -> optional batchnorm -> FC -> out_size

Parameters:
  input_size: number of input features
  conv_channels: list of channel sizes for successive Conv1d layers (e.g. [32,64])
  kernel_size: conv kernel size (default 3)
  lstm_hidden: hidden dim for LSTM
  lstm_layers: LSTM layers
  out_size: final output dim (default = input_size)
  batchnorm: apply BatchNorm1d to projected hidden before FC
  dropout: LSTM dropout

Usage:
  model = ConvLSTM(input_size=11, conv_channels=[32,64], lstm_hidden=64, out_size=11)
  y = model(x)  # x: (batch, seq_len, input_size)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        conv_channels: List[int] = (32, 64),
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        out_size: int = None,
        batchnorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.out_size = out_size or input_size

        # Build Conv1d stack
        convs = []
        in_ch = input_size
        for ch in self.conv_channels:
            # padding to keep sequence length the same
            pad = kernel_size // 2
            convs.append(nn.Conv1d(in_ch, ch, kernel_size, padding=pad))
            convs.append(nn.ReLU())
            # optional BatchNorm on channel dim
            convs.append(nn.BatchNorm1d(ch))
            in_ch = ch
        self.conv_net = nn.Sequential(*convs)

        # LSTM that consumes conv output per time-step: input_size = last conv channel
        self.lstm = nn.LSTM(in_ch, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout)

        # projection
        self.proj = nn.Linear(lstm_hidden, lstm_hidden)
        self.batchnorm = nn.BatchNorm1d(lstm_hidden) if batchnorm else None
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(lstm_hidden, self.out_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        returns: (batch, out_size)
        """
        # permute to (batch, features, seq_len) for Conv1d
        x_conv = x.permute(0, 2, 1)  # (batch, features, seq_len)

        # pass through conv stack -> (batch, last_ch, seq_len)
        conv_out = self.conv_net(x_conv)

        # permute to (batch, seq_len, last_ch) for LSTM
        conv_out = conv_out.permute(0, 2, 1)

        # LSTM -> out: (batch, seq_len, lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(conv_out)

        # take last time-step hidden
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        proj = self.proj(last_hidden)
        if self.batchnorm is not None:
            proj = self.batchnorm(proj)
        proj = self.activation(proj)

        out = self.fc_out(proj)  # (batch, out_size)
        return out


if __name__ == "__main__":
    # smoke test
    batch, seq, feat = 8, 30, 11
    x = torch.randn(batch, seq, feat)
    m = ConvLSTM(input_size=feat, conv_channels=[32, 64], kernel_size=3, lstm_hidden=64, out_size=feat, batchnorm=True)
    y = m(x)
    print("ConvLSTM output shape:", y.shape)  # expect (8, 11)
