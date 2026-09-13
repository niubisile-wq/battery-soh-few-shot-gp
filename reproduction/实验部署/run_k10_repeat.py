#!/usr/bin/env python3
"""Five-seed K=10 repeatability check for the selected baseline families."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

import run_formal_protocol as formal

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "formal_protocol"


def score(y, p):
    return float(mean_absolute_error(y, p)), float(mean_squared_error(y, p) ** .5)


def main():
    seeds = [0, 1, 2, 3, 4]
    source = formal.read_cells(formal.SOURCE_ZIP)
    target = formal.read_cells(formal.TARGET_ZIP)
    src_train = [c for i, c in enumerate(source) if i % 5 != 4]
    src_val = [c for i, c in enumerate(source) if i % 5 == 4]
    train_x, train_y = np.concatenate([c[1] for c in src_train]), np.concatenate([c[2] for c in src_train])
    val_x, val_y = np.concatenate([c[1] for c in src_val]), np.concatenate([c[2] for c in src_val])
    mu, sd = train_x.mean((0, 2), keepdims=True), train_x.std((0, 2), keepdims=True)
    sd[sd < 1e-6] = 1
    norm = lambda x: ((x - mu) / sd).astype("float32")
    train_x, val_x = norm(train_x), norm(val_x)
    target = [(n, norm(x), y) for n, x, y in target]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = {"TCN": formal.NN_MODELS["TCN"], "LSTM": formal.NN_MODELS["LSTM"], "PatchTSTLite": formal.NN_MODELS["PatchTSTLite"]}
    rows = []
    for model_name, ctor in selected.items():
        for seed in seeds:
            print(f"RUN model={model_name} seed={seed}", flush=True)
            base = formal.fit_nn(ctor, train_x, train_y, val_x, val_y, device, epochs=10, value=seed)
            variants = ["full_finetune", "head_only"] if model_name != "LSTM" and model_name != "PatchTSTLite" else ["full_finetune"]
            for variant in variants:
                maes, rmses = [], []
                for _, x, y in target:
                    if len(y) <= 10:
                        continue
                    model = formal.adapt(base, x[:10], y[:10], variant, device, steps=5)
                    a, b = score(y[10:], formal.predict(model, x[10:], device)); maes.append(a); rmses.append(b)
                rows.append({"K": 10, "model": model_name, "variant": variant, "seed": seed, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "target_cells": len(maes)})
    # Target-only is trained from scratch on each target cell's support set.
    ctor = formal.NN_MODELS["TCN"]
    for seed in seeds:
        maes, rmses = [], []
        for _, x, y in target:
            if len(y) <= 10:
                continue
            model = formal.fit_nn(ctor, x[:10], y[:10], val_x, val_y, device, epochs=10, value=seed + 1000, early_stop=False)
            a, b = score(y[10:], formal.predict(model, x[10:], device)); maes.append(a); rmses.append(b)
        rows.append({"K": 10, "model": "TCN", "variant": "target_only", "seed": seed, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "target_cells": len(maes)})
    # Linear trend is deterministic and is included once.
    maes, rmses = [], []
    train_mean = float(train_y.mean())
    for _, _, y in target:
        if len(y) <= 10:
            continue
        p = formal.linear_trend(None, y[:10], len(y[10:]), train_mean)
        a, b = score(y[10:], p); maes.append(a); rmses.append(b)
    rows.append({"K": 10, "model": "linear_trend", "variant": "linear_trend", "seed": -1, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "target_cells": len(maes)})
    out = OUT / "k10_repeat_results.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    grouped = {}
    for row in rows: grouped.setdefault(f"{row['model']}/{row['variant']}", []).append(row)
    summary = []
    for setting, values in grouped.items():
        summary.append({"setting": setting, "runs": len(values), "mean_mae": float(np.mean([x["macro_mae"] for x in values])), "std_mae": float(np.std([x["macro_mae"] for x in values], ddof=1)) if len(values) > 1 else 0.0, "mean_rmse": float(np.mean([x["macro_rmse"] for x in values])), "std_rmse": float(np.std([x["macro_rmse"] for x in values], ddof=1)) if len(values) > 1 else 0.0})
    (OUT / "k10_repeat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({"device": str(device), "rows": len(rows), "results": str(out), "summary": str(OUT / 'k10_repeat_summary.json')}, indent=2))


if __name__ == "__main__":
    main()
