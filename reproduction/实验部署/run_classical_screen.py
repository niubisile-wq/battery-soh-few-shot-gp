#!/usr/bin/env python3
"""Classical reference screen using exactly the pilot split and normalization."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "基线模型/参考实现/PINN4SOH/data/XJTU data"
OUT = Path(__file__).resolve().parent / "results"


def split_cells(files):
    train, valid = [], []
    for protocol in sorted({p.name.split("_", 1)[0] for p in files}):
        group = sorted(p for p in files if p.name.startswith(protocol + "_"))
        for i, path in enumerate(group):
            (valid if i % 5 == 4 else train).append(path)
    return train, valid


def load_files(files):
    xs, ys, owners = [], [], []
    for path in files:
        f = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).dropna()
        xs.append(f.iloc[:, :16].to_numpy(np.float32))
        ys.append(f.iloc[:, 16].to_numpy(np.float32) / 2.0)
        owners.extend([path.name] * len(f))
    return np.concatenate(xs), np.concatenate(ys), owners


def score(y, pred, owners):
    cell = {}
    for o, a, b in zip(owners, y, pred):
        cell.setdefault(o, ([], [])); cell[o][0].append(a); cell[o][1].append(b)
    mae = [mean_absolute_error(v[0], v[1]) for v in cell.values()]
    rmse = [mean_squared_error(v[0], v[1]) ** .5 for v in cell.values()]
    return float(np.mean(mae)), float(np.mean(rmse))


def main():
    files = sorted(DATA.glob("*.csv")); train_files, valid_files = split_cells(files)
    xtr, ytr, _ = load_files(train_files); xva, yva, owners = load_files(valid_files)
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.002)),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               max_features=0.8, random_state=0, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=.05,
                                 subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
                                 objective="reg:squarederror", n_jobs=8, random_state=0),
    }
    rows = []
    for name, model in models.items():
        model.fit(xtr, ytr); pred = model.predict(xva)
        mae, rmse = score(yva, pred, owners)
        row = {"model": name, "val_macro_mae": mae, "val_macro_rmse": rmse}
        print(row, flush=True); rows.append(row)
    OUT.mkdir(exist_ok=True)
    with (OUT / "classical_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (OUT / "classical_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
