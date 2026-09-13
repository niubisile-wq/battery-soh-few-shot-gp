"""Corrected PINN4SOH-style baseline on the locked common raw input.

Unlike the legacy ``run_pinn_common_input.py``, this implementation keeps the
reference method's two networks, differentiable cycle-time derivative, PDE
residual, data loss, and degradation-order regularizer.  The only adaptation
is the common input representation: the 128-point V/I/elapsed-time curve is
flattened and a normalized cycle coordinate is appended.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import copy

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64, depth=3):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(input_dim if i == 0 else hidden, hidden))
            layers.append(Sin())
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PINN(nn.Module):
    def __init__(self, feature_dim=384):
        super().__init__()
        self.solution_u = MLP(feature_dim + 1, 1, hidden=64, depth=3)
        # [curve, cycle_t, u, du/dcurve, du/dt]
        self.dynamical_F = MLP(feature_dim + 1 + 1 + feature_dim + 1, 1, hidden=64, depth=4)

    def physics_forward(self, curve, cycle_t, create_graph=True):
        curve = curve.requires_grad_(True)
        cycle_t = cycle_t.requires_grad_(True)
        inp = torch.cat([curve, cycle_t], dim=1)
        u = self.solution_u(inp)
        du = torch.autograd.grad(u.sum(), [curve, cycle_t], create_graph=create_graph,
                                 only_inputs=True, allow_unused=False)
        u_curve, u_t = du
        f_in = torch.cat([curve, cycle_t, u, u_curve, u_t], dim=1)
        f = u_t - self.dynamical_F(f_in)
        return u, f

    def predict(self, curve, cycle_t):
        return self.solution_u(torch.cat([curve, cycle_t], dim=1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(base, split_dirs):
    rows = []
    for split in split_dirs:
        for path in sorted((base / split).glob("*.npz")):
            with np.load(path, allow_pickle=True) as z:
                x = z["x"].astype("float32")
                q = z["capacity_Ah"].astype("float32")
                cyc = z["cycle_number"].astype("float32")
                cid = str(z["cell_id"].item())
            order = np.argsort(cyc)
            x, q, cyc = x[order], q[order], cyc[order]
            rows.append({"id": cid, "x": x, "q": q, "cycle": cyc, "split": split})
    return rows


def arrays(rows, mu=None, sd=None, budget=None):
    xs, ts, ys, keys = [], [], [], []
    for index, r in enumerate(rows):
        q0 = float(r["q"][0]); t = r["cycle"] / max(float(r["cycle"].max()), 1.0)
        keep = np.arange(len(r["q"]))
        if budget is not None:
            quota = budget // len(rows) + int(index < budget % len(rows))
            keep = np.linspace(0, len(keep) - 1, min(quota, len(keep))).astype(int)
        xs.append(r["x"][keep]); ts.append(t[keep, None]); ys.append((r["q"][keep] / q0)[:, None])
        keys.extend((r["id"], float(r["cycle"][k])) for k in keep)
    x = np.concatenate(xs); t = np.concatenate(ts); y = np.concatenate(ys)
    if mu is None:
        mu = x.mean((0, 2), keepdims=True); sd = x.std((0, 2), keepdims=True); sd[sd < 1e-6] = 1
    return ((x - mu) / sd).reshape(len(x), -1).astype("float32"), t.astype("float32"), y.astype("float32"), mu, sd, keys


def train(model, x, t, y, device, epochs, lr, alpha, beta, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt_u = torch.optim.Adam(model.solution_u.parameters(), lr=lr)
    opt_f = torch.optim.Adam(model.dynamical_F.parameters(), lr=lr)
    xb = torch.from_numpy(x).to(device); tb = torch.from_numpy(t).to(device); yb = torch.from_numpy(y).to(device)
    for _ in range(epochs):
        model.train()
        u, f = model.physics_forward(xb, tb)
        data_loss = nn.functional.mse_loss(u, yb)
        pde_loss = nn.functional.mse_loss(f, torch.zeros_like(f))
        order = torch.argsort(tb[:, 0]); ordered = u[order]
        mono_loss = torch.relu(ordered[1:] - ordered[:-1]).mean()
        loss = data_loss + alpha * pde_loss + beta * mono_loss
        opt_u.zero_grad(set_to_none=True); opt_f.zero_grad(set_to_none=True)
        loss.backward(); opt_u.step(); opt_f.step()
    return model


def adapt(model, x, t, y, device, steps, lr, alpha, beta):
    local = copy.deepcopy(model).to(device)
    train(local, x, t, y, device, steps, lr, alpha, beta, 0)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--adapt-steps", type=int, default=50)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--output", default="corrected_pinn_xjtu_hust_v1")
    args = ap.parse_args()
    out = ROOT / "开发基线选择依据/results" / args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "实验部署/source_window_cache_v2/offset_03_01"
    target = ROOT / "实验部署/hust_target_cache_v1"
    tr = load(source, ["source_train"]); va = load(source, ["source_validation"]); te = load(target, ["."])
    tx, tt, ty, mu, sd, keys = arrays(tr, budget=1000)
    vx, vt, vy, _, _, _ = arrays(va, mu, sd)
    for r in te:
        r["x"] = ((r["x"] - mu) / sd).reshape(len(r["x"]), -1).astype("float32")
        r["t"] = (r["cycle"] / max(float(r["cycle"].max()), 1.0)).astype("float32")[:, None]
        r["y"] = (r["q"] / r["q"][0]).astype("float32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []; cells = []
    for seed in seeds:
        print(f"PINN4SOH-PDE seed={seed}", flush=True)
        torch.manual_seed(seed); model = PINN().to(device)
        train(model, tx, tt, ty, device, args.epochs, 2e-3, args.alpha, args.beta, seed)
        for r in te:
            k = args.K
            local = adapt(model, r["x"][:k], r["t"][:k], r["y"][:k, None], device,
                          args.adapt_steps, 5e-4, args.alpha, args.beta)
            with torch.no_grad():
                pred = local.predict(torch.from_numpy(r["x"][k:]).to(device), torch.from_numpy(r["t"][k:]).to(device)).cpu().numpy().ravel()
            y = r["y"][k:]
            mae = float(np.mean(abs(pred - y))); rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            cells.append({"model": "PINN4SOH_common_input_PDE", "seed": seed, "K": k, "cell_id": r["id"], "mae": mae, "rmse": rmse})
        print(f"finished seed={seed}", flush=True)
    by_seed = {}
    for c in cells: by_seed.setdefault(c["seed"], []).append(c)
    for seed, group in by_seed.items():
        rows.append({"model": "PINN4SOH_common_input_PDE", "seed": seed, "K": args.K,
                     "macro_mae": float(np.mean([c["mae"] for c in group])),
                     "macro_rmse": float(np.mean([c["rmse"] for c in group])), "target_cells": len(group)})
    for name, data in [("pinn_results.csv", rows), ("pinn_cell_results.csv", cells)]:
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    manifest = {"status": "CORRECTED_PINN_COMPLETE", "task": "XJTU_to_HUST", "input": "flattened partial-charge V/I/elapsed-time plus normalized cycle coordinate", "architecture": "solution_u + dynamical_F", "loss": "data + alpha PDE residual + beta degradation-order penalty", "source_cache": str(source.relative_to(ROOT)), "target_cache": str(target.relative_to(ROOT)), "source_cache_manifest_sha256": digest(ROOT / "实验部署/source_window_cache_v2/manifest.json"), "target_cache_manifest_sha256": digest(target / "manifest.json"), "K": args.K, "seeds": seeds, "epochs": args.epochs, "adapt_steps": args.adapt_steps, "query_labels_used_for_training": False, "reference": "PINN4SOH official Model.py structure with common-input adapter"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
