# src/models/attention_lstm.py
"""
Attention LSTM model (sequence attention / temporal attention).

Architecture:
  Input (batch, seq_len, features)
  -> LSTM (batch, seq_len, hidden)
  -> compute attention weights between last hidden state and each time-step output
     (dot-product style)
  -> context = sum_t alpha_t * output_t
  -> concat(context, last_hidden) -> projection -> optional batchnorm -> final FC
  -> output (batch, out_size)

Parameters:
  input_size: number of input features
  hidden_size: LSTM hidden dimension
  num_layers: LSTM layers
  out_size: output dimension (default = input_size)
  batchnorm: whether to apply BatchNorm on projected vector before final FC
  dropout: LSTM dropout (between layers)
  attention_type: 'dot' (current implementation)

Usage:
  model = AttentionLSTM(input_size=11, hidden_size=64, out_size=11, batchnorm=True)
  y = model(x)  # x: (batch, seq_len, input_size)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        out_size: int = None,
        batchnorm: bool = False,
        dropout: float = 0.0,
        attention_type: str = "dot",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size or input_size
        self.attention_type = attention_type

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        # projection of concatenated [context; last_hidden] -> hidden_size
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size) if batchnorm else None
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, self.out_size)

        # initialize weights (optional small init)
        self._reset_parameters()

    def _reset_parameters(self):
        # small initialization for linear layers
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, out_size)
        """
        # LSTM outputs
        # out: (batch, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm(x)

        # last hidden state (from LSTM outputs) shape -> (batch, hidden)
        # We can use out[:, -1, :] (last time-step) or h_n[-1]
        last_hidden = out[:, -1, :]  # (batch, hidden_size)

        # compute attention scores (dot product): energy_{t} = out_t . last_hidden
        # out: (batch, seq_len, hidden), last_hidden: (batch, hidden, 1) -> bmm
        # energy: (batch, seq_len)
        # prepare last_hidden for bmm
        last_hidden_unsq = last_hidden.unsqueeze(2)  # (batch, hidden, 1)
        energy = torch.bmm(out, last_hidden_unsq).squeeze(2)  # (batch, seq_len)

        # attention weights
        attn_weights = F.softmax(energy, dim=1)  # (batch, seq_len)

        # context vector = sum_t alpha_t * out_t
        # attn_weights: (batch, seq_len) -> (batch, 1, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)  # (batch, hidden)

        # concat context and last_hidden
        cat = torch.cat([context, last_hidden], dim=1)  # (batch, 2*hidden)

        # project
        proj = self.proj(cat)  # (batch, hidden)
        if self.batchnorm is not None:
            # BatchNorm1d expects (batch, features)
            proj = self.batchnorm(proj)
        proj = self.activation(proj)

        out_final = self.fc_out(proj)  # (batch, out_size)
        return out_final


if __name__ == "__main__":
    # quick smoke test
    batch, seq, feat = 8, 30, 11
    x = torch.randn(batch, seq, feat)
    m = AttentionLSTM(input_size=feat, hidden_size=32, num_layers=1, out_size=feat, batchnorm=True)
    y = m(x)
    print("AttentionLSTM output shape:", y.shape)  # expect (8, 11)
