# src/training/train.py
# Train a simple LSTM on your multivariate commodity data.

# Example (Windows):
# python src/training/train.py --csv "C:\Users\shiva\Documents\multicommodity-prices-indonesia\data\csv\pihps.csv" --seq_len 30 --epochs 100


import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import loader, preprocess, sequence_builder  # relative import assumes package root or adjust PYTHONPATH
from src.models.lstm_model import SimpleLSTM
from src.training.dataset import SeqDataset
from src.training.evaluate import mae, per_feature_mae

# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(model, opt, loss_fn, dl):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in dl:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n

def eval_loop(model, dl):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            out = model(xb)
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    if len(preds) == 0:
        return None, None
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return trues, preds

def main(args):
    # load
    print("Loading CSV:", args.csv)
    df = loader.load_csv_file(args.csv, date_col=None)  # set date_col if CSV has a date index
    cols = df.columns.tolist()
    print("Columns:", cols)
    arr_norm, scaler = preprocess.fit_transform(df)  # normalized arr (T, F)

    # build sequences
    seq_len = args.seq_len
    X, y = sequence_builder.build_sequences(arr_norm, seq_len=seq_len)  # X (N,seq_len,F), y (N,F)

    # split by time: first 80% samples for train, last 20% for test
    N = X.shape[0]
    split = int(N * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f"Samples: total={N}, train={X_train.shape[0]}, test={X_test.shape[0]}")

    # create dataloaders
    train_ds = SeqDataset(X_train, y_train)
    test_ds = SeqDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # model
    n_features = X.shape[2]
    model = SimpleLSTM(input_size=n_features, hidden_size=args.hidden_size,
                       num_layers=args.num_layers, out_size=n_features,
                       batchnorm=args.batchnorm, dropout=args.dropout)
    model.to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss() if args.loss == "mae" else nn.MSELoss()

    best_val = float('inf')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(model, opt, loss_fn, train_dl)
        trues, preds = eval_loop(model, test_dl)
        val_mae = mae(trues, preds)
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs} TrainLoss={train_loss:.6f} ValMAE={val_mae:.6f}")
        # save best
        if val_mae < best_val:
            best_val = val_mae
            ckpt_path = os.path.join(args.checkpoint_dir, f"best_model_epoch{epoch}.pth")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'scaler': scaler, 'cols': cols}, ckpt_path)
    # final evaluation: denormalize outputs
    trues, preds = eval_loop(model, test_dl)
    print("Final Test MAE (normalized):", mae(trues, preds))
    # denormalize and compute MAE in original units
    trues_denorm = preprocess.denormalize(trues, cols, scaler)
    preds_denorm = preprocess.denormalize(preds, cols, scaler)
    print("Final Test MAE (original units):", mae(trues_denorm, preds_denorm))
    # per-feature MAE
    pf = per_feature_mae(trues_denorm, preds_denorm)
    print("Per-feature MAE (original units):")
    for c, v in zip(cols, pf):
        print(f"  {c}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=r"C:\Users\shiva\Documents\multicommodity-prices-indonesia\data\csv\india_commodities_demo.csv",
                        help="Path to multivariate CSV file")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batchnorm", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, choices=["mse", "mae"], default="mae")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    main(args)
