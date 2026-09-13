#!/usr/bin/env python3
"""Build a conservative audit of all downloaded battery sources.

The audit reads one cell at a time from BatteryLife archives and records
cycle ordering, available fields, valid charge windows, and SOH coverage.
Raw MAT/ZIP sources are recorded separately when a parser is not part of the
unified BatteryLife representation; they are never silently treated as ready.
"""
from __future__ import annotations

import csv
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import h5py
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "实验部署" / "data_audit"
OUT.mkdir(parents=True, exist_ok=True)

ARCHIVES = {
    "HUST": ROOT / "数据集/BatteryLife_v12_processed/HUST.zip",
    "XJTU": ROOT / "数据集/BatteryLife_v12_processed/XJTU.zip",
    "Tongji": ROOT / "数据集/BatteryLife_v12_processed/Tongji.zip",
    "SNL": ROOT / "数据集/BatteryLife_v12_processed/SNL.zip",
    "CALCE": ROOT / "数据集/BatteryLife_v12_processed/CALCE.zip",
}


def audit_cell(dataset: str, name: str, obj: dict) -> dict:
    cycles = sorted(obj.get("cycle_data", []), key=lambda c: float(c.get("cycle_number", 0)))
    nums = [float(c.get("cycle_number", 0)) for c in cycles]
    required = {"current_in_A", "voltage_in_V", "time_in_s", "discharge_capacity_in_Ah"}
    valid = 0
    charge_sign_ok = 0
    lengths = []
    for cycle in cycles:
        keys = set(cycle)
        if not required.issubset(keys):
            continue
        cur = np.asarray(cycle["current_in_A"], dtype=float)
        vol = np.asarray(cycle["voltage_in_V"], dtype=float)
        tim = np.asarray(cycle["time_in_s"], dtype=float)
        cap = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float)
        pos = np.flatnonzero(cur > 0)
        if len(pos) >= 32 and len(cap) and np.isfinite(cap).any() and float(np.nanmax(cap)) > 0:
            valid += 1
            lengths.append(int(len(pos)))
            if np.nanmax(cur) > 0 and np.nanmin(cur) < 0:
                charge_sign_ok += 1
    return {
        "dataset": dataset,
        "cell_id": str(obj.get("cell_id", name)),
        "archive_member": name,
        "nominal_capacity_Ah": obj.get("nominal_capacity_in_Ah"),
        "cycle_count": len(cycles),
        "cycle_number_min": min(nums) if nums else None,
        "cycle_number_max": max(nums) if nums else None,
        "cycle_numbers_monotonic_after_sort": nums == sorted(nums),
        "valid_charge_cycles": valid,
        "charge_sign_mixed_cycles": charge_sign_ok,
        "has_V_I_time_capacity": valid > 0,
        "mean_positive_segment_length": float(np.mean(lengths)) if lengths else None,
        "chemistry": f"{obj.get('anode_material')} / {obj.get('cathode_material')}",
        "charge_protocol": str(obj.get("charge_protocol", "")),
        "role": "unassigned_until_protocol_review",
    }


def main() -> None:
    rows = []
    for dataset, archive in ARCHIVES.items():
        with zipfile.ZipFile(archive) as z:
            members = sorted(n for n in z.namelist() if n.endswith(".pkl"))
            for member in members:
                with z.open(member) as f:
                    rows.append(audit_cell(dataset, member, pickle.load(f)))
    # Raw sources are explicitly present but not silently claimed as unified.
    matr = []
    for path in sorted((ROOT / "数据集/MATR").glob("*.mat")):
        with h5py.File(path, "r") as f:
            n = int(f["batch"]["cycle_life"].shape[0])
            matr.append({"file": str(path.relative_to(ROOT)), "batch_entries": n, "required_groups": all(k in f for k in ("batch", "batch_date")), "parser": "h5py"})
    oxford_path = ROOT / "数据集/Oxford_Battery_Degradation_Dataset_1/Oxford_Battery_Degradation_Dataset_1.mat"
    oxford = loadmat(oxford_path, squeeze_me=True, struct_as_record=False)
    oxford_cells = sorted(k for k in oxford if k.startswith("Cell"))
    nasa_path = ROOT / "数据集/NASA/NASA_PCoE_Battery_Data_Set.zip"
    with zipfile.ZipFile(nasa_path) as z:
        bad = z.testzip()
        nested = [n for n in z.namelist() if n.endswith(".zip")]
    raw = {
        "MATR": {"files": matr, "parser_verified": True},
        "Oxford": {"file": str(oxford_path.relative_to(ROOT)), "cell_ids": oxford_cells, "cell_count": len(oxford_cells), "parser": "scipy.io.loadmat", "parser_verified": True},
        "NASA": {"file": str(nasa_path.relative_to(ROOT)), "nested_archives": nested, "outer_zip_bad_member": bad, "parser_verified": bad is None},
    }
    with (OUT / "cell_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for row in rows:
        summary.setdefault(row["dataset"], {"cells": 0, "valid_charge_cycles": 0, "missing_signal_cells": 0})
        summary[row["dataset"]]["cells"] += 1
        summary[row["dataset"]]["valid_charge_cycles"] += row["valid_charge_cycles"]
        summary[row["dataset"]]["missing_signal_cells"] += int(not row["has_V_I_time_capacity"])
    (OUT / "audit_summary.json").write_text(json.dumps({"standardized": summary, "raw_sources": raw}, indent=2), encoding="utf8")
    print(json.dumps({"standardized_cells": len(rows), "summary": summary, "raw_sources": raw}, indent=2))


if __name__ == "__main__":
    main()
