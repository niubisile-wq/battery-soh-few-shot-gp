#!/usr/bin/env python3
"""Fair K=10 repeatability run for every main baseline.

All source-pretrained methods use the same deterministic 1,000-window source
budget. Trend extrapolation is written to a separate track because it consumes
support SOH labels rather than V/I/time signals.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import run_formal_protocol as formal

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / os.environ.get("FAIR_OUT", "formal_protocol_all")
SEED_START = int(os.environ.get("SEED_START", "0"))
SEED_COUNT = int(os.environ.get("SEED_COUNT", "5"))
SEEDS = list(range(SEED_START, SEED_START + SEED_COUNT))
K = 10
SOURCE_BUDGET = 1000
EPOCHS = 25
ADAPT_STEPS = 30


def score(y, p):
    return float(mean_absolute_error(y, p)), float(mean_squared_error(y, p) ** .5)


def classical(name, seed):
    if name == "SVR_RBF":
        return make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=.002))
    return XGBRegressor(n_estimators=50, max_depth=6, learning_rate=.05,
                        subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
                        objective="reg:squarederror", n_jobs=4,
                        random_state=seed)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    keep = np.linspace(0, len(train_x) - 1, min(SOURCE_BUDGET, len(train_x))).astype(int)
    train_x, train_y = train_x[keep], train_y[keep]
    flat_train = train_x.reshape(len(train_x), -1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    cell_rows = []
    manifest = {
        "task": "XJTU_to_HUST", "K": K, "primary_track": "raw_V_I_time",
        "source_train_cells": [c[0] for c in src_train],
        "source_validation_cells": [c[0] for c in src_val],
        "target_cells": [c[0] for c in target],
        "cycle_order": "sorted by numeric cycle_number before validity filtering",
        "charge_selector": "positive current segment; resampled to 128 points",
        "support": "first 10 valid chronological cycles",
        "query": "all later valid chronological cycles",
        "split_before_window": True, "query_labels_used_for_selection": False,
        "source_training_window_budget": SOURCE_BUDGET,
        "epochs": EPOCHS, "adapt_steps": ADAPT_STEPS, "seeds": SEEDS,
        "raw_models": list(formal.NN_MODELS),
        "classical_models": ["SVR_RBF", "XGBoost"],
        "trend_track_separate": True,
    }
    (OUT / "fair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

    for model_name, ctor in formal.NN_MODELS.items():
        for seed in SEEDS:
            print(f"NN {model_name} seed={seed}", flush=True)
            base = formal.fit_nn(ctor, train_x, train_y, val_x, val_y, device, EPOCHS, seed)
            for variant in ("source_only", "full_finetune", "head_only", "target_only"):
                maes, rmses = [], []
                for _, x, y in target:
                    if len(y) <= K:
                        continue
                    if variant == "source_only": model = base
                    elif variant == "target_only": model = formal.fit_nn(ctor, x[:K], y[:K], val_x, val_y, device, EPOCHS, seed + 1000, early_stop=False)
                    else: model = formal.adapt(base, x[:K], y[:K], variant, device, ADAPT_STEPS)
                    a, b = score(y[K:], formal.predict(model, x[K:], device)); maes.append(a); rmses.append(b)
                    cell_rows.append({"track":"raw_V_I_time","K":K,"model":model_name,"variant":variant,"seed":seed,"cell_id":_ ,"mae":a,"rmse":b})
                rows.append({"track": "raw_V_I_time", "K": K, "model": model_name, "variant": variant, "seed": seed, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "target_cells": len(maes), "source_budget": SOURCE_BUDGET})

    for name in ("SVR_RBF", "XGBoost"):
        for seed in SEEDS:
            print(f"CLASSICAL {name} seed={seed}", flush=True)
            source_model = classical(name, seed).fit(flat_train, train_y)
            for variant in ("source_only", "target_only"):
                maes, rmses = [], []
                for _, x, y in target:
                    if len(y) <= K:
                        continue
                    if variant == "source_only": model = source_model
                    else: model = classical(name, seed).fit(x[:K].reshape(K, -1), y[:K])
                    a, b = score(y[K:], model.predict(x[K:].reshape(len(x) - K, -1))); maes.append(a); rmses.append(b)
                    cell_rows.append({"track":"raw_V_I_time","K":K,"model":name,"variant":variant,"seed":seed,"cell_id":_ ,"mae":a,"rmse":b})
                rows.append({"track": "raw_V_I_time", "K": K, "model": name, "variant": variant, "seed": seed, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "target_cells": len(maes), "source_budget": SOURCE_BUDGET})

    maes, rmses = [], []
    for _, _, y in target:
        if len(y) <= K:
            continue
        p = formal.linear_trend(None, y[:K], len(y[K:]), float(train_y.mean()))
        a, b = score(y[K:], p); maes.append(a); rmses.append(b)
    rows.append({"track": "trend_only_separate", "K": K, "model": "linear_trend", "variant": "support_soh_trend", "seed": -1, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "target_cells": len(maes), "source_budget": 0})

    result_path = OUT / "fair_results.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    with (OUT / "fair_cell_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cell_rows[0].keys()); writer.writeheader(); writer.writerows(cell_rows)
    grouped = {}
    for row in rows: grouped.setdefault(f"{row['track']}/{row['model']}/{row['variant']}", []).append(row)
    summary = []
    for setting, values in grouped.items():
        summary.append({"setting": setting, "runs": len(values), "mean_mae": float(np.mean([x["macro_mae"] for x in values])), "std_mae": float(np.std([x["macro_mae"] for x in values], ddof=1)) if len(values) > 1 else 0.0, "mean_rmse": float(np.mean([x["macro_rmse"] for x in values])), "std_rmse": float(np.std([x["macro_rmse"] for x in values], ddof=1)) if len(values) > 1 else 0.0})
    (OUT / "fair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({"device": str(device), "rows": len(rows), "results": str(result_path), "summary": str(OUT / 'fair_summary.json')}, indent=2))


if __name__ == "__main__":
    main()
