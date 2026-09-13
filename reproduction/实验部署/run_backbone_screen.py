#!/usr/bin/env python3
"""Development-only backbone screen on the frozen XJTU feature snapshot.

This is a pilot for choosing the backbone of the new method.  It deliberately
does not read any external validation data and reports validation metrics by
cell (macro averaging), not by windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "基线模型/参考实现/PINN4SOH/data/XJTU data"
OUT = Path(__file__).resolve().parent / "results"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_cells(files: list[Path], valid_protocol: str | None = None) -> tuple[list[Path], list[Path]]:
    """Fixed split; optionally hold out one complete protocol for stress testing."""
    train, valid = [], []
    for protocol in sorted({p.name.split("_", 1)[0] for p in files}):
        group = sorted(p for p in files if p.name.startswith(protocol + "_"))
        if valid_protocol == protocol:
            valid.extend(group)
        else:
            for i, path in enumerate(group):
                (valid if i % 5 == 4 else train).append(path)
    return train, valid


def load_files(files: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, owners = [], [], []
    for path in files:
        frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).dropna()
        # The reference snapshot has 16 engineered charge-curve features and
        # one capacity column.  Capacity is normalized by the XJTU 2 Ah rating.
        x = frame.iloc[:, :16].to_numpy(dtype=np.float32)
        y = (frame.iloc[:, 16].to_numpy(dtype=np.float32) / 2.0)
        xs.append(x)
        ys.append(y)
        owners.extend([path.name] * len(y))
    return np.concatenate(xs), np.concatenate(ys), owners


class MLP(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(32, 64, 3, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.net(x.unsqueeze(1)))


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1, dilation=1), nn.GELU(),
                                 nn.Conv1d(32, 32, 3, padding=2, dilation=2), nn.GELU(),
                                 nn.Conv1d(32, 64, 3, padding=4, dilation=4), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.net(x.unsqueeze(1)))


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(1, 64, 2, batch_first=True, dropout=.1)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1])


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        layer = nn.TransformerEncoderLayer(32, 4, 64, .1, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        z = self.encoder(self.embed(x.unsqueeze(-1)))
        return self.head(z.mean(dim=1))


MODELS = {"MLP": MLP, "CNN1D": CNN1D, "TCN": TCN, "LSTM": LSTM,
          "Transformer": Transformer}


def metrics(model, loader, owners, y_true, device):
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(x.to(device)).cpu().numpy().ravel()
                               for x, _ in loader])
    by_cell = {}
    for owner, yt, yp in zip(owners, y_true, pred):
        by_cell.setdefault(owner, ([], []))
        by_cell[owner][0].append(float(yt))
        by_cell[owner][1].append(float(yp))
    maes = [mean_absolute_error(v[0], v[1]) for v in by_cell.values()]
    rmses = [mean_squared_error(v[0], v[1]) ** .5 for v in by_cell.values()]
    return float(np.mean(maes)), float(np.mean(rmses)), pred


def run_one(name, seed, train_x, train_y, valid_x, valid_y, owners, device, epochs):
    seed_everything(seed)
    model = MODELS[name]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y[:, None])),
                              batch_size=512, shuffle=True)
    valid_loader = DataLoader(TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y[:, None])),
                              batch_size=1024, shuffle=False)
    best = float("inf")
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            pred = model(x.to(device))
            loss = loss_fn(pred, y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val_mae, val_rmse, _ = metrics(model, valid_loader, owners, valid_y, device)
        if val_mae < best:
            best, best_state, stale = val_mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    val_mae, val_rmse, _ = metrics(model, valid_loader, owners, valid_y, device)
    return {"model": name, "seed": seed, "val_macro_mae": val_mae,
            "val_macro_rmse": val_rmse, "epochs": epoch + 1,
            "parameters": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--valid-protocol", default=None,
                    help="hold out one complete XJTU protocol, e.g. 3C")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("*.csv"))
    train_files, valid_files = split_cells(files, args.valid_protocol)
    train_x, train_y, _ = load_files(train_files)
    valid_x, valid_y, owners = load_files(valid_files)
    mu, sigma = train_x.mean(0), train_x.std(0)
    sigma[sigma < 1e-7] = 1.0
    train_x = ((train_x - mu) / sigma).astype(np.float32)
    valid_x = ((valid_x - mu) / sigma).astype(np.float32)
    suffix = "" if args.valid_protocol is None else f"_{args.valid_protocol}_holdout"
    manifest = {"data_snapshot": str(DATA), "protocol": "XJTU engineered 16-feature pilot",
                "validation_protocol_holdout": args.valid_protocol,
                "train_cells": [p.name for p in train_files],
                "validation_cells": [p.name for p in valid_files],
                "normalization": "train-cell rows only", "test_used": False,
                "seeds": [int(x) for x in args.seeds.split(",")], "epochs_max": args.epochs}
    (OUT / f"split_manifest{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name in MODELS:
        for seed in manifest["seeds"]:
            result = run_one(name, seed, train_x, train_y, valid_x, valid_y, owners, device, args.epochs)
            print(result, flush=True)
            results.append(result)
    with (OUT / f"results{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    summary = []
    for name in MODELS:
        rows = [r for r in results if r["model"] == name]
        summary.append({"model": name, "mean_mae": float(np.mean([r["val_macro_mae"] for r in rows])),
                        "std_mae": float(np.std([r["val_macro_mae"] for r in rows])),
                        "mean_rmse": float(np.mean([r["val_macro_rmse"] for r in rows])),
                        "parameters": rows[0]["parameters"]})
    summary.sort(key=lambda r: (r["mean_mae"], r["std_mae"]))
    (OUT / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
