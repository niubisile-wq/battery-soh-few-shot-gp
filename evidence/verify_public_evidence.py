"""Portable checks for the public frozen evidence released with the paper."""
from pathlib import Path
import csv, json, hashlib

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent

def rows(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def close(a, b, tol=1e-9):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError((a, b))

aggregate = rows("aggregate.csv")
assert len(aggregate) == 48
assert {r["dataset"] for r in aggregate} == {"XJTU", "MATR", "Tongji"}
assert len({r["group"] for r in aggregate}) == 16

original = {(r["dataset"], r["group"]): r for r in rows("full_ablation_original.csv")}
for r in aggregate:
    key = (r["dataset"], r["group"])
    assert key in original
    for metric in ("mae", "rmse", "p95_ae"):
        close(r[metric], original[key][metric])

effects = rows("module_effects.csv")
assert len(effects) == 36
for r in effects:
    assert r["dataset"] in {"XJTU", "MATR", "Tongji"}
    assert r["module"] in {"M1", "M2", "M3", "M4"}

intervals = rows("paired_intervals.csv")
assert len(intervals) > 0
for r in intervals:
    before, after = r["comparison"].split("-")
    lookup = {(x["dataset"], x["group"]): x for x in aggregate}
    close(float(lookup[r["dataset"], before]["mae"]) - float(lookup[r["dataset"], after]["mae"]), r["delta_pp"])

trajectories = rows("calce_210_trajectories.csv")
assert len(trajectories) == 210
assert {r["group"] for r in trajectories} == {"B", "B1", "B12", "B123", "B1234"}
assert len(rows("calce_protocols.csv")) == 4
assert len(rows("stage_cost.csv")) > 0

print("Public evidence checks passed:", len(aggregate), "aggregate rows,", len(trajectories), "trajectory rows.")
