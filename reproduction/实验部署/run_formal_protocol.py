#!/usr/bin/env python3
"""Formal XJTU -> HUST few-shot protocol for paper 2.

The split is cell-level and is made before any window/sequence is used:
source cells are split into train/validation; every HUST cell contributes its
first K chronological cycles as support and all later cycles as query.
Query labels are used only for final scoring, never for model selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import random
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_ZIP = ROOT / "数据集/BatteryLife_v12_processed/XJTU.zip"
TARGET_ZIP = ROOT / "数据集/BatteryLife_v12_processed/HUST.zip"
OUT = HERE / "results" / "formal_protocol"
CACHE = OUT / "cache"
CACHE_VERSION = "v2_cycle_number"
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


def seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def read_cells(zip_path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Extract only charge-positive V/I/time and normalized capacity."""
    cache = CACHE / f"{zip_path.stem}_{CACHE_VERSION}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        names = z["names"].tolist()
        counts = z["counts"].astype(int).tolist()
        x, y, pos = z["x"], z["y"], 0
        cells = []
        for name, count in zip(names, counts):
            cells.append((str(name), x[pos:pos + count], y[pos:pos + count]))
            pos += count
        return cells

    cells = []
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".pkl"))
        for entry in names:
            with archive.open(entry) as f:
                obj = pickle.load(f)
            samples = []
            cycles = sorted(obj["cycle_data"], key=lambda c: float(c.get("cycle_number", 0)))
            cycle_numbers = [float(c.get("cycle_number", 0)) for c in cycles]
            if cycle_numbers != sorted(cycle_numbers):
                raise ValueError(f"cycle ordering failed for {entry}")
            for cycle in cycles:
                cur = np.asarray(cycle["current_in_A"], dtype=np.float32)
                vol = np.asarray(cycle["voltage_in_V"], dtype=np.float32)
                tim = np.asarray(cycle["time_in_s"], dtype=np.float32)
                cap = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=np.float32)
                positive = np.flatnonzero(cur > 0)
                if len(positive) < 32 or len(cap) == 0 or float(cap.max()) <= 0:
                    continue
                start, end = int(positive[0]), int(positive[-1]) + 1
                if end - start < 32:
                    continue
                grid = np.linspace(0, 1, 128, dtype=np.float32)
                old = np.linspace(0, 1, end - start, dtype=np.float32)
                xx = np.stack([np.interp(grid, old, a[start:end]) for a in (vol, cur, tim)])
                xx[2] = (xx[2] - xx[2, 0]) / (xx[2, -1] - xx[2, 0] + 1e-6)
                samples.append((xx, float(cap.max() / obj["nominal_capacity_in_Ah"])))
            if samples:
                cells.append((entry, np.stack([a for a, _ in samples]).astype("float32"),
                              np.asarray([b for _, b in samples], dtype="float32")))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, names=np.asarray([a[0] for a in cells], dtype=object),
                        counts=np.asarray([len(a[2]) for a in cells]),
                        x=np.concatenate([a[1] for a in cells]), y=np.concatenate([a[2] for a in cells]))
    return cells


class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Flatten(), nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)


class Conv(nn.Module):
    def __init__(self, dilated=False):
        super().__init__(); d = [1, 2, 4] if dilated else [1, 1, 1]
        self.body = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2*d[0], dilation=d[0]), nn.GELU(), nn.Conv1d(32, 64, 5, padding=2*d[1], dilation=d[1]), nn.GELU(), nn.Conv1d(64, 64, 5, padding=2*d[2], dilation=d[2]), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.body(x))


class LSTM(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn = nn.LSTM(3, 64, 2, batch_first=True, dropout=.1); self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.rnn(x.transpose(1, 2))[0][:, -1])


class GRU(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn = nn.GRU(3, 64, 2, batch_first=True, dropout=.1); self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.rnn(x.transpose(1, 2))[0][:, -1])


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 64, 5, padding=2), nn.GELU()); self.rnn = nn.LSTM(64, 64, 1, batch_first=True); self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.rnn(self.conv(x).transpose(1, 2))[0][:, -1])


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__(); self.body = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 64, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten()); self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.body(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__(); self.e = nn.Linear(3, 32); layer = nn.TransformerEncoderLayer(32, 4, 64, .1, batch_first=True, norm_first=True); self.t = nn.TransformerEncoder(layer, 2); self.head = nn.Linear(32, 1)
    def forward(self, x): return self.head(self.t(self.e(x.transpose(1, 2))).mean(1))


class PatchTSTLite(nn.Module):
    def __init__(self):
        super().__init__(); self.e = nn.Linear(48, 32); layer = nn.TransformerEncoderLayer(32, 4, 64, .1, batch_first=True, norm_first=True); self.t = nn.TransformerEncoder(layer, 2); self.head = nn.Linear(32, 1)
    def forward(self, x): return self.head(self.t(self.e(x.unfold(-1, 16, 8).transpose(1, 2).flatten(2))).mean(1))


NN_MODELS = {"MLP": MLP, "CNN1D": CNN1D, "LSTM": LSTM, "GRU": GRU, "CNN_LSTM": CNNLSTM, "TCN": lambda: Conv(True), "Transformer": Transformer, "PatchTSTLite": PatchTSTLite}


def cell_metrics(y, p):
    return float(mean_absolute_error(y, p)), float(mean_squared_error(y, p) ** .5)


def predict(model, x, device):
    model.eval()
    with torch.no_grad(): return model(torch.from_numpy(x).to(device)).cpu().numpy().ravel()


def fit_nn(ctor, x, y, xv, yv, device, epochs, value, early_stop=True):
    seed(value); model = ctor().to(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y[:, None])), batch_size=1024, shuffle=True)
    best, state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train:
            loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if not early_stop:
            continue
        score = float(np.mean((predict(model, xv, device) - yv) ** 2))
        if score < best:
            best, state, stale = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else: stale += 1
        if stale >= 7: break
    if early_stop:
        model.load_state_dict(state)
    return model


def last_block_parameters(model):
    """Select the complete final feature block and regression head."""
    if hasattr(model, "rnn"):
        layer = model.rnn.num_layers - 1
        selected = [p for n, p in model.rnn.named_parameters()
                    if n.endswith(f"_l{layer}") or n.endswith(f"_l{layer}_reverse")]
        return selected + list(model.head.parameters())
    if hasattr(model, "t"):
        return list(model.t.layers[-1].parameters()) + list(model.head.parameters())
    if hasattr(model, "body"):
        convolutions = [m for m in model.body.modules() if isinstance(m, nn.Conv1d)]
        return list(convolutions[-1].parameters()) + list(model.head.parameters())
    if hasattr(model, "net"):
        linear = [m for m in model.net.modules() if isinstance(m, nn.Linear)]
        return list(linear[-2].parameters()) + list(linear[-1].parameters())
    raise TypeError(f"No last-block definition for {type(model).__name__}")


def adapt(model, x, y, mode, device, steps):
    model = copy.deepcopy(model).to(device)
    if mode in ("head_only", "last_block"):
        for p in model.parameters(): p.requires_grad = False
        if mode == "head_only":
            head = getattr(model, "head", None) or getattr(model, "net")[-1]
            for p in head.parameters(): p.requires_grad = True
            params = list(head.parameters())
        else:
            params = last_block_parameters(model)
            for p in params: p.requires_grad = True
    else: params = list(model.parameters())
    if len(x) == 0: return model
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-5)
    xb, yb = torch.from_numpy(x).to(device), torch.from_numpy(y[:, None]).to(device)
    model.train()
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(xb), yb); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model


def linear_trend(x, y, n_query, fallback):
    if len(y) < 2: return np.full(n_query, y[-1] if len(y) else fallback, dtype="float32")
    coef = np.polyfit(np.arange(len(y), dtype=float), y, 1)
    return np.polyval(coef, np.arange(len(y), len(y) + n_query)).astype("float32")


def classical_models():
    return {"SVR_RBF": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=.002)),
            "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_lambda=1.0, objective="reg:squarederror", n_jobs=4, random_state=0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--adapt-steps", type=int, default=30); ap.add_argument("--ks", default="0,5,10,20"); ap.add_argument("--classical-max-samples", type=int, default=1000)
    args = ap.parse_args(); seeds = [int(x) for x in args.seeds.split(",")]; ks = [int(x) for x in args.ks.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    source, target = read_cells(SOURCE_ZIP), read_cells(TARGET_ZIP)
    src_train = [c for i, c in enumerate(source) if i % 5 != 4]; src_val = [c for i, c in enumerate(source) if i % 5 == 4]
    train_x, train_y = np.concatenate([c[1] for c in src_train]), np.concatenate([c[2] for c in src_train])
    val_x, val_y = np.concatenate([c[1] for c in src_val]), np.concatenate([c[2] for c in src_val])
    mu, sd = train_x.mean((0, 2), keepdims=True), train_x.std((0, 2), keepdims=True); sd[sd < 1e-6] = 1
    norm = lambda z: ((z - mu) / sd).astype("float32")
    train_x, val_x = norm(train_x), norm(val_x); target = [(n, norm(x), y) for n, x, y in target]
    manifest = {"task": "XJTU_to_HUST", "source_train_cells": [c[0] for c in src_train], "source_validation_cells": [c[0] for c in src_val], "target_cells": [c[0] for c in target], "support": "first K chronological valid cycles", "query": "all later cycles", "ks": ks, "primary_K": 10, "input": "partial-charge voltage,current,normalized-time resampled to 128", "split_before_window": True, "query_labels_used_for_model_selection": False, "seeds": seeds, "epochs": args.epochs, "adapt_steps": args.adapt_steps, "classical_max_samples": args.classical_max_samples, "note": "single-seed formal screening run; repeat K=10 with multiple seeds before paper tables"}
    (OUT / "formal_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for k in ks:
        if k == 0: continue
        for model_name, ctor in NN_MODELS.items():
            for s in seeds:
                print(f"RUN K={k} model={model_name} seed={s}", flush=True)
                base = fit_nn(ctor, train_x, train_y, val_x, val_y, device, args.epochs, s)
                for variant in ("full_finetune", "head_only", "target_only"):
                    maes, rmses = [], []
                    for _, x, y in target:
                        if len(y) <= k: continue
                        support_x, support_y, query_x, query_y = x[:k], y[:k], x[k:], y[k:]
                        if variant == "target_only": model = fit_nn(ctor, support_x, support_y, val_x, val_y, device, args.epochs, s + 1000, early_stop=False)
                        else: model = adapt(base, support_x, support_y, variant, device, args.adapt_steps)
                        a, b = cell_metrics(query_y, predict(model, query_x, device)); maes.append(a); rmses.append(b)
                    rows.append({"K": k, "model": model_name, "variant": variant, "seed": s, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "parameters": sum(p.numel() for p in base.parameters())})
        for variant in ("source_only", "linear_trend"):
            maes, rmses = [], []
            for _, x, y in target:
                if variant == "linear_trend": p = linear_trend(x, y[:k], len(y[k:]), float(train_y.mean()))
                else:
                    p = None
                if p is not None: a, b = cell_metrics(y[k:], p)
                else: a, b = 0., 0.
                maes.append(a); rmses.append(b)
            if variant == "linear_trend": rows.append({"K": k, "model": "linear_trend", "variant": variant, "seed": -1, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "parameters": 0})
    # K=0 source-only neural baselines and classical source/target baselines.
    for model_name, ctor in NN_MODELS.items():
        for s in seeds:
            print(f"RUN K=0 model={model_name} seed={s}", flush=True)
            base = fit_nn(ctor, train_x, train_y, val_x, val_y, device, args.epochs, s)
            maes, rmses = [], []
            for _, x, y in target:
                if len(y) <= k: continue
                a, b = cell_metrics(y, predict(base, x, device)); maes.append(a); rmses.append(b)
            rows.append({"K": 0, "model": model_name, "variant": "source_only", "seed": s, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "parameters": sum(p.numel() for p in base.parameters())})
    with (OUT / "formal_neural_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"NEURAL_BASELINES_COMPLETE rows={len(rows)}", flush=True)
    flat_train, flat_val = train_x.reshape(len(train_x), -1), val_x.reshape(len(val_x), -1)
    if len(flat_train) > args.classical_max_samples:
        keep = np.linspace(0, len(flat_train) - 1, args.classical_max_samples).astype(int)
        flat_train, classical_y = flat_train[keep], train_y[keep]
    else:
        classical_y = train_y
    for name, estimator in classical_models().items():
        print(f"RUN K=0 model={name} variant=source_only", flush=True)
        estimator.fit(flat_train, classical_y)
        maes, rmses = [], []
        for _, x, y in target:
            a, b = cell_metrics(y, estimator.predict(x.reshape(len(x), -1))); maes.append(a); rmses.append(b)
        rows.append({"K": 0, "model": name, "variant": "source_only", "seed": 0, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "parameters": -1})
    for k in ks:
        if k == 0: continue
        for name, template in classical_models().items():
            print(f"RUN K={k} model={name} variant=target_only", flush=True)
            maes, rmses = [], []
            for _, x, y in target:
                est = clone(template).fit(x[:k].reshape(k, -1), y[:k]); a, b = cell_metrics(y[k:], est.predict(x[k:].reshape(len(x)-k, -1))); maes.append(a); rmses.append(b)
            rows.append({"K": k, "model": name, "variant": "target_only", "seed": 0, "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)), "parameters": -1})
    with (OUT / "formal_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for row in rows: summary.setdefault(f"K{row['K']}/{row['model']}/{row['variant']}", []).append(row)
    compact = [{"setting": key, "macro_mae": float(np.mean([x["macro_mae"] for x in values])), "macro_rmse": float(np.mean([x["macro_rmse"] for x in values])), "runs": len(values)} for key, values in summary.items()]
    (OUT / "formal_summary.json").write_text(json.dumps(compact, indent=2), encoding="utf8")
    print(json.dumps({"device": str(device), "rows": len(rows), "output": str(OUT), "summary": str(OUT / 'formal_summary.json')}, indent=2))


if __name__ == "__main__": main()
