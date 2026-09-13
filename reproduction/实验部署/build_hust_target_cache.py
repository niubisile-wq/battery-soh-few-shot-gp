"""Build the locked HUST target cache after source-only window selection.

The target archive is never consulted when selecting the window.  This script
only applies the already locked relative offsets (0.3--0.1 V below each
cell's recorded maximum voltage), audits measured-capacity eligibility, and
records the first ten eligible cycles as support.  Later eligible cycles are
query-only evaluation points.
"""
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "开发基线选择依据"))
from partial_charge_protocol import Window, extract


def measured_hust_capacity(cycle, cell):
    v = np.asarray(cycle["voltage_in_V"], dtype=float)
    i = np.asarray(cycle["current_in_A"], dtype=float)
    q = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float)
    if v.ndim != 1 or i.shape != v.shape or q.shape != v.shape:
        raise ValueError("capacity and signal shape mismatch")
    if not all(np.isfinite(a).all() for a in (v, i, q)):
        raise ValueError("nonfinite capacity measurement")
    nominal = float(cell["nominal_capacity_in_Ah"])
    cutoff = float(cell["min_voltage_limit_in_V"])
    discharge = i < -0.05 * nominal
    if not discharge.any() or float(v[discharge].min()) > cutoff + 0.02:
        raise ValueError("discharge did not reach recorded cutoff")
    capacity = float(q[discharge].max())
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("nonpositive measured capacity")
    return capacity


def main():
    source_selection = ROOT / "实验部署/source_window_cache_v2/window_selection.json"
    selection = json.loads(source_selection.read_text())
    if selection["status"] != "WINDOW_SELECTED_ON_SOURCE_VALIDATION_ONLY":
        raise RuntimeError("source-only window selection is not locked")
    if selection["selected"] != "offset_03_01":
        raise RuntimeError("unexpected selected source window")

    out = ROOT / "实验部署/hust_target_cache_v1"
    if (out / "manifest.json").exists():
        raise RuntimeError("refusing to overwrite target cache")
    out.mkdir(parents=True)
    archive = ROOT / "数据集/BatteryLife_v12_processed/HUST.zip"
    low_offset, high_offset = 0.3, 0.1
    records, errors = [], []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".pkl"))
        for index, name in enumerate(names, 1):
            cell = pickle.load(z.open(name))
            cutoff = float(cell["max_voltage_limit_in_V"])
            window = Window(cutoff - low_offset, cutoff - high_offset)
            cycles = sorted(cell["cycle_data"], key=lambda c: float(c["cycle_number"]))
            cycle_ids = [float(c["cycle_number"]) for c in cycles]
            if len(set(cycle_ids)) != len(cycle_ids):
                raise ValueError(f"duplicate cycle number: {name}")
            xs, caps, nums = [], [], []
            rejected = {}
            for cycle in cycles:
                try:
                    capacity = measured_hust_capacity(cycle, cell)
                    x, meta = extract(cycle, window)
                except (ValueError, KeyError) as exc:
                    rejected[str(exc)] = rejected.get(str(exc), 0) + 1
                    continue
                xs.append(x)
                caps.append(capacity)
                nums.append(meta["cycle_number"])
            if len(caps) < 11:
                raise RuntimeError(f"too few eligible target cycles: {name} ({len(caps)})")
            path = out / (Path(name).stem + ".npz")
            np.savez_compressed(
                path,
                x=np.stack(xs),
                capacity_Ah=np.asarray(caps, dtype="float32"),
                cycle_number=np.asarray(nums, dtype="float32"),
                nominal_capacity_Ah=float(cell["nominal_capacity_in_Ah"]),
                cutoff_voltage_V=cutoff,
                min_voltage_limit_V=float(cell["min_voltage_limit_in_V"]),
                cell_id=name,
            )
            records.append(
                {
                    "cell_id": name,
                    "valid_cycles": len(caps),
                    "total_cycles": len(cycles),
                    "support_cycle_numbers": nums[:10],
                    "query_cycle_numbers": nums[10:],
                    "support_policy": "first 10 chronological eligible measured cycles",
                    "rejections": rejected,
                    "path": str(path.relative_to(out)),
                }
            )
            print(f"{index}/{len(names)} {name} valid={len(caps)}", flush=True)

    manifest = {
        "status": "TARGET_CACHE_READY_EVALUATION_LOCKED",
        "dataset": "HUST",
        "source_window_selection": "offset_03_01",
        "source_window_selection_sha256": hashlib.sha256(source_selection.read_bytes()).hexdigest(),
        "relative_window_offsets_below_cell_max_voltage_V": [0.3, 0.1],
        "signal_units": ["V", "A", "seconds"],
        "capacity_unit": "Ah",
        "capacity_eligibility": "measured discharge current reaches recorded minimum voltage cutoff",
        "support_policy": "first 10 chronological eligible measured cycles",
        "query_policy": "later eligible measured cycles; labels unavailable to selection",
        "target_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "target_archives_read": ["HUST.zip"],
        "cells": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"cells": len(records), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
