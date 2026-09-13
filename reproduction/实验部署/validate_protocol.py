#!/usr/bin/env python3
"""Validate the current XJTU -> HUST protocol before baseline execution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "开发基线选择依据/results/formal_protocol_fair/fair_manifest.json"
AUDIT = ROOT / "实验部署/data_audit/cell_audit.csv"
OUT = ROOT / "实验部署/protocol_validation.json"


def main() -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf8"))
    failures = []
    train, valid, target = set(m["source_train_cells"]), set(m["source_validation_cells"]), set(m["target_cells"])
    if train & valid: failures.append("source train/validation cell overlap")
    if train & target or valid & target: failures.append("source/target cell overlap")
    if m["split_before_window"] is not True: failures.append("split_before_window is not true")
    if m["query_labels_used_for_selection"] is not False: failures.append("query labels may have been used")
    if m["cycle_order"] != "sorted by numeric cycle_number before validity filtering": failures.append("cycle ordering rule mismatch")
    if m["charge_selector"] != "positive-current partial-charge voltage,current,normalized-time resampled to 128" and m["charge_selector"] != "positive current segment; resampled to 128 points":
        failures.append("charge selector mismatch")
    # The manifest is cell-level; cached arrays must have nonempty query for every target cell.
    cache = ROOT / "开发基线选择依据/results/formal_protocol/cache/HUST_v2_cycle_number.npz"
    if not cache.exists(): failures.append(f"missing cache: {cache}")
    else:
        z = np.load(cache, allow_pickle=True)
        if any(int(c) <= int(m["K"]) for c in z["counts"]): failures.append("target cell has empty K=10 query")
    result = {"status": "PROTOCOL_VALID" if not failures else "PROTOCOL_INVALID", "failures": failures, "manifest": str(MANIFEST), "audit": str(AUDIT), "target_cells": len(target), "source_train_cells": len(train), "source_validation_cells": len(valid)}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
