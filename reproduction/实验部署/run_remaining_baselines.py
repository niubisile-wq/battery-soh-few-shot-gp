#!/usr/bin/env python3
"""Simple and handcrafted-HI baselines for the fixed XJTU -> HUST K=10 task."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

import run_formal_protocol as formal
from degradation_baselines import exponential_trend

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / os.environ.get("HI_OUT", "remaining_baselines")
K = 10
SOURCE_BUDGET = 1000
SEED_START = int(os.environ.get("SEED_START", "0")); SEED_COUNT = int(os.environ.get("SEED_COUNT", "5"))
SEEDS = list(range(SEED_START, SEED_START + SEED_COUNT))


def features(x: np.ndarray) -> np.ndarray:
    """A locked, signal-only HI vector; no SOH or future information."""
    t = np.linspace(0, 1, x.shape[-1], dtype=np.float32)
    out = []
    for sample in x:
        v, c, tm = sample
        vals = [v.mean(), v.std(), v.min(), v.max(), c.mean(), c.std(), c.min(), c.max(),
                tm[-1] - tm[0], np.trapezoid(v, t), np.trapezoid(c, t),
                np.polyfit(t, v, 1)[0], np.polyfit(t, c, 1)[0],
                np.quantile(v, .25), np.quantile(v, .75), np.quantile(c, .75)]
        out.append(vals)
    return np.asarray(out, dtype="float32")


def metric(y, p):
    return float(mean_absolute_error(y, p)), float(mean_squared_error(y, p) ** .5)


def models(seed, source=False):
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=.002)),
        "RandomForest": RandomForestRegressor(n_estimators=100 if source else 25, min_samples_leaf=2, max_features=.8, random_state=seed, n_jobs=4),
        "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_lambda=1.0, objective="reg:squarederror", n_jobs=4, random_state=seed),
        "PLSR": PLSRegression(n_components=8, scale=True),
        "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF(1.0) + WhiteKernel(.01), normalize_y=True, random_state=seed, optimizer=None)),
    }


def record(rows, cell_rows, track, name, variant, seed, target, preds, budget):
    maes, rmses = [], []
    for cell_name, y, p in zip(target, preds["y"], preds["p"]):
        a, b = metric(y, p); maes.append(a); rmses.append(b)
        cell_rows.append({"track": track, "K": K, "model": name, "variant": variant, "seed": seed, "cell_id": cell_name, "query_count": len(y), "mae": a, "rmse": b, "source_budget": budget})
    rows.append({"track": track, "K": K, "model": name, "variant": variant, "seed": seed, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "target_cells": len(maes), "source_budget": budget})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, target = formal.read_cells(formal.SOURCE_ZIP), formal.read_cells(formal.TARGET_ZIP)
    src_train = [c for i, c in enumerate(source) if i % 5 != 4]
    train_x, train_y = np.concatenate([c[1] for c in src_train]), np.concatenate([c[2] for c in src_train])
    mu, sd = train_x.mean((0, 2), keepdims=True), train_x.std((0, 2), keepdims=True); sd[sd < 1e-6] = 1
    norm = lambda x: ((x - mu) / sd).astype("float32")
    train_x = norm(train_x); target = [(n, norm(x), y) for n, x, y in target]
    keep = np.linspace(0, len(train_x) - 1, min(SOURCE_BUDGET, len(train_x))).astype(int)
    train_x, train_y = train_x[keep], train_y[keep]
    train_h = features(train_x)
    target_h = [(n, features(x), y) for n, x, y in target]
    rows, cell_rows = [], []

    # Simple signal-free/trend baselines, with per-cell evidence.
    for variant in ("dummy_mean", "last_support", "linear_trend", "exponential_trend"):
        preds_y, preds_p = [], []
        for name, _, y in target:
            qy = y[K:]
            if variant == "dummy_mean": p = np.full(len(qy), float(train_y.mean()))
            elif variant == "last_support": p = np.full(len(qy), float(y[K - 1]))
            elif variant == "linear_trend": p = formal.linear_trend(None, y[:K], len(qy), float(train_y.mean()))
            else:
                # Legacy cache stores valid-cycle ordinal only. Corrected
                # protocol must supply original cycle numbers to this function.
                p = exponential_trend(np.arange(K), y[:K], np.arange(K, len(y)))
            preds_y.append(qy); preds_p.append(p)
        record(rows, cell_rows, "trend_only_separate", "simple", variant, -1, [n for n, _, _ in target], {"y": preds_y, "p": preds_p}, 0)

    # HI models: same feature vector, same source budget, all five seeds.
    model_source_budgets = {"GPR": 200, "RandomForest": SOURCE_BUDGET}
    for name in models(0, source=True):
        effective_budget = model_source_budgets.get(name, SOURCE_BUDGET)
        if effective_budget < len(train_x):
            model_keep = np.linspace(0, len(train_x) - 1, effective_budget).astype(int)
            model_train_h, model_train_y = train_h[model_keep], train_y[model_keep]
        else:
            model_train_h, model_train_y = train_h, train_y
        print(f"HI {name}: source_budget={effective_budget}", flush=True)
        for seed in SEEDS:
            est = models(seed, source=True)[name].fit(model_train_h, model_train_y)
            preds_y, preds_p = [], []
            for _, h, y in target_h:
                preds_y.append(y[K:]); preds_p.append(est.predict(h[K:]).reshape(-1))
            record(rows, cell_rows, "handcrafted_HI", name, "source_only", seed, [n for n, _, _ in target], {"y": preds_y, "p": preds_p}, effective_budget)
            preds_y, preds_p = [], []
            for _, h, y in target_h:
                local = models(seed, source=False)[name].fit(h[:K], y[:K])
                preds_y.append(y[K:]); preds_p.append(local.predict(h[K:]).reshape(-1))
            record(rows, cell_rows, "handcrafted_HI", name, "target_only", seed, [n for n, _, _ in target], {"y": preds_y, "p": preds_p}, K)

    with (OUT / "remaining_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (OUT / "remaining_cell_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cell_rows[0].keys()); w.writeheader(); w.writerows(cell_rows)
    grouped = {}
    for r in rows: grouped.setdefault(f"{r['track']}/{r['model']}/{r['variant']}", []).append(r)
    summary = [{"setting": k, "runs": len(v), "mean_mae": float(np.mean([r['macro_mae'] for r in v])), "std_mae": float(np.std([r['macro_mae'] for r in v], ddof=1)) if len(v) > 1 else 0.0, "mean_rmse": float(np.mean([r['macro_rmse'] for r in v]))} for k, v in grouped.items()]
    (OUT / "remaining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (OUT / "remaining_manifest.json").write_text(json.dumps({"task": "XJTU_to_HUST", "K": K, "source_budget": SOURCE_BUDGET, "seeds": SEEDS, "feature_count": 16, "query_labels_used_for_selection": False}, indent=2), encoding="utf8")
    print(json.dumps({"rows": len(rows), "cell_rows": len(cell_rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
