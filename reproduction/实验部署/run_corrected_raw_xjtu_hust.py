"""Corrected XJTU -> HUST raw-curve baseline benchmark.

This runner consumes only the versioned source/target caches.  It does not
reopen the legacy raw reader, so partial-charge extraction, measured-capacity
eligibility, relative voltage window, cycle identity, and support/query split
are all auditable from the manifests.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import run_formal_protocol as formal

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache(base, split_dirs):
    rows = []
    for split in split_dirs:
        for path in sorted((base / split).glob("*.npz")):
            with np.load(path, allow_pickle=True) as z:
                x = z["x"].astype("float32")
                q = z["capacity_Ah"].astype("float32")
                n = z["cycle_number"].astype("float32")
                cid = str(z["cell_id"].item() if z["cell_id"].ndim == 0 else z["cell_id"])
            if len(q) < 11 or len(x) != len(q) or len(n) != len(q):
                raise RuntimeError(f"invalid cache row: {path}")
            order = np.argsort(n)
            rows.append({"cell_id": cid, "x": x[order], "q": q[order], "cycle": n[order], "split": split})
    return rows


def balanced_source(rows, budget):
    xs, ys, keys = [], [], []
    for idx, row in enumerate(rows):
        q0 = float(row["q"][0])
        quota = budget // len(rows) + int(idx < budget % len(rows))
        keep = np.linspace(0, len(row["q"]) - 1, min(quota, len(row["q"]))).astype(int)
        xs.append(row["x"][keep]); ys.append(row["q"][keep] / q0)
        keys.extend((row["cell_id"], float(row["cycle"][k])) for k in keep)
    return np.concatenate(xs), np.concatenate(ys), keys


def macro_metrics(rows):
    mae, rmse, count = [], [], 0
    for row, pred in rows:
        y = row["q"][len(pred[0]):]
        p = pred[1]
        mae.append(float(np.mean(abs(y - p))))
        rmse.append(float(np.sqrt(np.mean((y - p) ** 2))))
        count += len(y)
    return float(np.mean(mae)), float(np.mean(rmse)), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--adapt-steps", type=int, default=10)
    ap.add_argument("--ks", default="0,1,5,10,20")
    ap.add_argument("--models", default=",".join(formal.NN_MODELS))
    ap.add_argument("--source-budget", type=int, default=1000)
    ap.add_argument("--output", default="corrected_raw_xjtu_hust_v1")
    args = ap.parse_args()
    out = ROOT / "开发基线选择依据/results" / args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_base = ROOT / "实验部署/source_window_cache_v2/offset_03_01"
    target_base = ROOT / "实验部署/hust_target_cache_v1"
    src_train = load_cache(source_base, ["source_train"])
    src_val = load_cache(source_base, ["source_validation"])
    target = load_cache(target_base, ["."])
    train_x, train_y, source_keys = balanced_source(src_train, args.source_budget)
    val_x = np.concatenate([r["x"][10:] for r in src_val])
    val_y = np.concatenate([r["q"][10:] / r["q"][0] for r in src_val])
    mu = train_x.mean((0, 2), keepdims=True); sd = train_x.std((0, 2), keepdims=True); sd[sd < 1e-6] = 1
    norm = lambda a: ((a - mu) / sd).astype("float32")
    train_x, val_x = norm(train_x), norm(val_x)
    for row in target:
        row["x"] = norm(row["x"])
        row["y"] = row["q"] / row["q"][0]
    for row in src_val:
        row["x"] = norm(row["x"])
        row["y"] = row["q"] / row["q"][0]

    seeds = [int(x) for x in args.seeds.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    models = [x for x in args.models.split(",") if x]
    manifest = {
        "status": "RUNNING",
        "task": "XJTU_to_HUST",
        "source_cache": str(source_base.relative_to(ROOT)),
        "target_cache": str(target_base.relative_to(ROOT)),
        "source_cache_manifest_sha256": digest(ROOT / "实验部署/source_window_cache_v2/manifest.json"),
        "source_selection_sha256": digest(ROOT / "实验部署/source_window_cache_v2/window_selection.json"),
        "target_cache_manifest_sha256": digest(target_base / "manifest.json"),
        "source_train_cells": [r["cell_id"] for r in src_train],
        "source_validation_cells": [r["cell_id"] for r in src_val],
        "target_cells": [r["cell_id"] for r in target],
        "source_sample_keys": source_keys,
        "support_policy": "first K chronological eligible measured target cycles",
        "query_policy": "all later eligible target cycles; query labels only for scoring",
        "ks": ks, "primary_K": 10, "seeds": seeds, "models": models,
        "epochs_max": args.epochs, "early_stopping_patience": args.patience,
        "adapt_steps": args.adapt_steps, "source_budget": args.source_budget,
        "normalization": "source-train signal mean/std only",
        "target_labels_used_for_selection": False,
        "runner_sha256": digest(Path(__file__)),
        "model_code_sha256": digest(Path(formal.__file__)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    cell_rows = []
    for model_name in models:
        for seed in seeds:
            print(f"PRETRAIN model={model_name} seed={seed}", flush=True)
            base = formal.fit_nn(formal.NN_MODELS[model_name], train_x, train_y, val_x, val_y, device,
                                 args.epochs, seed, early_stop=True)
            torch.save({"state_dict": base.state_dict(), "model": model_name, "seed": seed},
                       out / f"{model_name}_seed{seed}_source.pt")
            for k in ks:
                for variant in (["source_only"] if k == 0 else ["head_only", "last_block", "full_finetune", "target_only"]):
                    maes, rmses, nquery = [], [], 0
                    batch_cells = []
                    for target_row in target:
                        if len(target_row["y"]) <= k:
                            continue
                        sx, sy = target_row["x"][:k], target_row["y"][:k]
                        qx, qy = target_row["x"][k:], target_row["y"][k:]
                        if variant == "source_only":
                            model = base
                        elif variant == "target_only":
                            # Target-only has no independent target validation
                            # set. Use the declared support-adaptation budget,
                            # not the source pretraining maximum, to avoid an
                            # unreported 300-epoch target overfit.
                            model = formal.fit_nn(formal.NN_MODELS[model_name], sx, sy, val_x, val_y,
                                                  device, args.adapt_steps, seed + 10000, early_stop=False)
                        else:
                            model = formal.adapt(base, sx, sy, variant, device, args.adapt_steps)
                        pred = formal.predict(model, qx, device)
                        cell_mae = float(np.mean(abs(pred - qy)))
                        cell_rmse = float(np.sqrt(np.mean((pred - qy) ** 2)))
                        maes.append(cell_mae)
                        rmses.append(cell_rmse)
                        nquery += len(qy)
                        record = {"K": k, "model": model_name, "variant": variant,
                                  "seed": seed, "cell_id": target_row["cell_id"],
                                  "query_count": len(qy), "mae": cell_mae, "rmse": cell_rmse}
                        cell_rows.append(record); batch_cells.append(record)
                    all_abs = [r["mae"] * r["query_count"] for r in batch_cells]
                    all_sq = [r["rmse"] ** 2 * r["query_count"] for r in batch_cells]
                    rows.append({"K": k, "model": model_name, "variant": variant, "seed": seed,
                                 "macro_mae": float(np.mean(maes)), "macro_rmse": float(np.mean(rmses)),
                                 "micro_mae": float(np.sum(all_abs) / nquery),
                                 "micro_rmse": float(np.sqrt(np.sum(all_sq) / nquery)), "query_count": nquery})
            (out / "results.csv").write_text("") if False else None
            with (out / "results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows); f.flush()
            with (out / "cell_results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cell_rows[0].keys()); writer.writeheader(); writer.writerows(cell_rows); f.flush()
    manifest["status"] = "CORRECTED_RAW_BASELINES_COMPLETE"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"rows": len(rows), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
