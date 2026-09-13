#!/usr/bin/env python3
"""Build the paper-facing baseline comparison table from final result directories."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "开发基线选择依据" / "results"
OUT_CSV = RESULTS / "baseline_comparison_K10.csv"
OUT_MD = RESULTS / "baseline_comparison_K10.md"
OUT_MANIFEST = RESULTS / "baseline_comparison_K10_manifest.json"

SOURCES = {
    "raw_transfer": RESULTS / "corrected_raw_xjtu_hust_v6" / "results.csv",
    "classical": RESULTS / "corrected_classical_xjtu_hust_v6" / "results.csv",
    "transfer_uda": RESULTS / "corrected_transfer_baselines_v1" / "transfer_results.csv",
    "pinn": RESULTS / "corrected_pinn_xjtu_hust_v1" / "pinn_results.csv",
}


def finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value}")
    return number


def aggregate(rows, keys, metric_names):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        item = dict(zip(keys, key))
        item["n_seeds"] = len({r["seed"] for r in group})
        item["n_rows"] = len(group)
        for metric in metric_names:
            values = [finite(r[metric]) for r in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_rows():
    final = []

    raw = [r for r in read(SOURCES["raw_transfer"]) if r["K"] == "10"]
    for r in aggregate(raw, ["model", "variant", "K"], ["macro_mae", "macro_rmse", "micro_mae", "micro_rmse"]):
        final.append({
            "track": "raw deep baseline",
            "comparison_group": "historical_inductive_training_needs_audit",
            "method": r["model"],
            "setting": r["variant"],
            "K": r["K"],
            "n_seeds": r["n_seeds"],
            "macro_mae_mean": r["macro_mae_mean"],
            "macro_mae_std": r["macro_mae_std"],
            "macro_rmse_mean": r["macro_rmse_mean"],
            "macro_rmse_std": r["macro_rmse_std"],
            "micro_mae_mean": r["micro_mae_mean"],
            "micro_mae_std": r["micro_mae_std"],
            "micro_rmse_mean": r["micro_rmse_mean"],
            "micro_rmse_std": r["micro_rmse_std"],
            "note": "historical result; stopping/budget evidence requires repair; not a champion ranking",
        })

    classical = [r for r in read(SOURCES["classical"]) if r["K"] == "10"]
    for r in aggregate(classical, ["model", "variant", "K"], ["macro_mae", "macro_rmse"]):
        reference = r["model"] == "full_target_upper_bound"
        final.append({
            "track": "classical/trend",
            "comparison_group": "reference only" if reference else "strict K=10",
            "method": r["model"],
            "setting": r["variant"],
            "K": r["K"],
            "n_seeds": r["n_seeds"],
            "macro_mae_mean": r["macro_mae_mean"],
            "macro_mae_std": r["macro_mae_std"],
            "macro_rmse_mean": r["macro_rmse_mean"],
            "macro_rmse_std": r["macro_rmse_std"],
            "micro_mae_mean": "",
            "micro_mae_std": "",
            "micro_rmse_mean": "",
            "micro_rmse_std": "",
            "note": "in-sample diagnostic: query labels included in fitting; not a held-out upper bound" if reference else "historical result; deterministic seed SD is unestimated; displayed zero may be rounding",
        })

    transfer = [r for r in read(SOURCES["transfer_uda"]) if r["K"] == "10"]
    for r in aggregate(transfer, ["model", "K"], ["macro_mae", "macro_rmse"]):
        final.append({
            "track": "meta-learning" if r["model"] == "MAML" else "UDA separate",
            "comparison_group": "historical_inductive_training_needs_audit" if r["model"] == "MAML" else "transductive_separate",
            "method": r["model"],
            "setting": "labeled support meta-adaptation" if r["model"] == "MAML" else "unsupervised transductive adaptation",
            "K": r["K"],
            "n_seeds": r["n_seeds"],
            "macro_mae_mean": r["macro_mae_mean"],
            "macro_mae_std": r["macro_mae_std"],
            "macro_rmse_mean": r["macro_rmse_mean"],
            "macro_rmse_std": r["macro_rmse_std"],
            "micro_mae_mean": "",
            "micro_mae_std": "",
            "micro_rmse_mean": "",
            "micro_rmse_std": "",
            "note": "10 seeds; first K target labels used for adaptation" if r["model"] == "MAML" else "10 seeds; unlabeled target query signals used during training; query labels excluded",
        })

    pinn = [r for r in read(SOURCES["pinn"]) if r["K"] == "10"]
    for r in aggregate(pinn, ["model", "K"], ["macro_mae", "macro_rmse"]):
        final.append({
            "track": "physics-informed",
            "comparison_group": "invalid_for_strict_early_life_ranking",
            "method": r["model"],
            "setting": "PDE + degradation-order penalty",
            "K": r["K"],
            "n_seeds": r["n_seeds"],
            "macro_mae_mean": r["macro_mae_mean"],
            "macro_mae_std": r["macro_mae_std"],
            "macro_rmse_mean": r["macro_rmse_mean"],
            "macro_rmse_std": r["macro_rmse_std"],
            "micro_mae_mean": "",
            "micro_mae_std": "",
            "micro_rmse_mean": "",
            "micro_rmse_std": "",
            "note": "historical PINN uses final target cycle for time scaling; strict early-life ranking invalid",
        })
    return final


def fmt(value):
    return "" if value in ("", None) else f"{float(value):.4f}"


def main():
    rows = make_rows()
    fields = ["track", "comparison_group", "method", "setting", "K", "n_seeds",
              "macro_mae_mean", "macro_mae_std", "macro_rmse_mean", "macro_rmse_std",
              "micro_mae_mean", "micro_mae_std", "micro_rmse_mean", "micro_rmse_std", "note"]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# XJTU→HUST 基线对比表（K=10）",
        "",
        "历史结果归档（冠军排名资格未通过）。数值按 seed 汇总；训练预算、PINN 未来循环信息、趋势坐标及信息轨道存在已确认问题，不能据此宣布冠军。全目标拟合行使用查询标签，仅为样本内诊断。新的公平选型见 `fair_selection_v1/`。",
        "",
        "| Track | Method | Setting | Seeds | Macro MAE | Macro RMSE | Micro MAE | Micro RMSE | 说明 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def pm(metric):
            mean = fmt(r[f"{metric}_mean"])
            std = fmt(r[f"{metric}_std"])
            return f"{mean} ± {std}" if mean else "—"
        lines.append("| " + " | ".join([
            r["track"], r["method"], r["setting"], str(r["n_seeds"]),
            pm("macro_mae"), pm("macro_rmse"), pm("micro_mae"), pm("micro_rmse"), r["note"]
        ]) + " |")
    lines += [
        "", "## 数据与结果来源", "",
        "- raw deep: `corrected_raw_xjtu_hust_v6/results.csv`（8 个深度基线，4 个 K=10 适配设置，10 seeds）。",
        "- classical/trend: `corrected_classical_xjtu_hust_v6/results.csv`。",
        "- transfer/UDA: `corrected_transfer_baselines_v1/transfer_results.csv`（MAML、MMD、DeepCORAL、DANN）。",
        "- physics-informed: `corrected_pinn_xjtu_hust_v1/pinn_results.csv`。",
        "- 中断、smoke test 和已判无效的 v1–v5 目录未纳入。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    manifest = {
        "status": "HISTORICAL_TABLE_NOT_VALIDATED_FOR_CHAMPION_SELECTION",
        "task": "XJTU_to_HUST",
        "K": 10,
        "aggregation": "mean and sample standard deviation over available seeds",
        "row_count": len(rows),
        "strict_rows": sum(r["comparison_group"] == "strict K=10" for r in rows),
        "reference_rows": sum(r["comparison_group"] == "reference only" for r in rows),
        "sources": {k: str(v.relative_to(ROOT)) for k, v in SOURCES.items()},
        "excluded": "interrupted, smoke-test, and invalid intermediate directories",
        "outputs": [str(OUT_CSV.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
