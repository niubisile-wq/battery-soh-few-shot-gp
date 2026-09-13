"""Nested screen for a two-view adaptive GPR blend.

This is an isolated probe: it reuses the already frozen C00 and P02 fits,
selects the blend on inner validation only, and evaluates the selected blend
on the outer cells.  It does not alter the original champion or screen.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from data import load_cells, split, metrics


def macro(rows, key="mae"):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["domain"])].append(float(r[key]))
    by_dataset = defaultdict(list)
    for (ds, _), values in groups.items():
        by_dataset[ds].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in by_dataset.values()]))


def load_model(name, fold, root):
    path = root / "jobs" / f"{name}__{fold.replace(':', '__')}" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def make_cache(model, cells, modes):
    return {c.id: {mode: model.predict(c, mode) for mode in modes} for c in cells}


def evaluate(cells, cache_a, cache_b, alpha, mode_a, mode_b):
    rows = []
    for c in cells:
        pa = cache_a[c.id][mode_a]
        pb = cache_b[c.id][mode_b]
        pred = (1.0 - alpha) * pa + alpha * pb
        rows.append({"dataset": c.dataset, "domain": c.domain, "cell_id": c.id,
                     **metrics(c.y[10:], pred)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="XJTU,MATR,Tongji")
    ap.add_argument("--screen-dir", default="模块研发/results/m1_v1/screen_v1")
    ap.add_argument("--screen-dir-b", default=None)
    ap.add_argument("--model-a", default="C00_original")
    ap.add_argument("--model-b", default="P02_anchor_physics")
    ap.add_argument("--max-alpha", type=float, default=1.0)
    ap.add_argument("--out", default="模块研发/results/m1_v1/blend_v1.json")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    all_cells = [c for ds in datasets for c in load_cells(ds)]
    folds = sorted({f"{c.dataset}:{c.domain}" for c in all_cells})
    root = Path(args.screen_dir)
    root_b = Path(args.screen_dir_b) if args.screen_dir_b else root
    alphas = np.linspace(0.0, args.max_alpha, 9)
    modes = ["source_only", "source_bias", "posterior"]
    results = []

    for fold in folds:
        train, validation, target = split(all_cells, fold)
        # Models were trained with the exact source split and frozen protocol.
        base = load_model(args.model_a, fold, root)
        phys = load_model(args.model_b, fold, root_b)
        prediction_cells = validation + target
        cache_base = make_cache(base, prediction_cells, modes)
        cache_phys = make_cache(phys, prediction_cells, modes)
        trials = []
        for alpha in alphas:
            for mode_a in modes:
                for mode_b in modes:
                    rows = evaluate(validation, cache_base, cache_phys,
                                    float(alpha), mode_a, mode_b)
                    trials.append({"alpha": float(alpha), "mode_a": mode_a,
                                   "mode_b": mode_b, "inner_mae": macro(rows),
                                   "inner_rmse": macro(rows, "rmse"),
                                   "inner_p95_ae": macro(rows, "p95_ae")})
        chosen = min(trials, key=lambda r: (r["inner_mae"], r["inner_rmse"],
                                             r["inner_p95_ae"]))
        rows = evaluate(target, cache_base, cache_phys, chosen["alpha"],
                        chosen["mode_a"], chosen["mode_b"])
        results.append({"fold": fold, "chosen": chosen, "cells": len(rows),
                        "rows": rows, "mae": macro(rows),
                        "rmse": macro(rows, "rmse"),
                        "p95_ae": macro(rows, "p95_ae")})

    summary = []
    for ds in datasets:
        rows = [r for r in results if r["fold"].startswith(ds + ":")]
        cell_rows = [x for r in rows for x in r["rows"]]
        summary.append({"dataset": ds, "candidate": "adaptive_C00_P02",
                        "cells": len(cell_rows), "expected_cells": len([c for c in all_cells if c.dataset == ds]),
                        "mae": macro(cell_rows), "rmse": macro(cell_rows, "rmse"),
                        "p95_ae": macro(cell_rows, "p95_ae")})
    output = {"protocol": f"{args.model_a}/{args.model_b} frozen fits; blend and mode selected on inner validation",
              "alphas": alphas.tolist(), "summary": summary, "folds": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary, "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
