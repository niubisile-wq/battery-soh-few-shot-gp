"""Legacy artifact-presence audit; explicitly NOT a scientific readiness gate.

The old check counted finite rows but did not verify training or data access.
Its previous CORE_R_AND_D_READY verdict is withdrawn. New development evidence
is recorded under results/fair_selection_v1 and requires a separate audit.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "开发基线选择依据/results"


def load_json(path): return json.loads(path.read_text())


def finite_csv(path, fields):
    rows = list(csv.DictReader(path.open()))
    bad = [r for r in rows if any(not math.isfinite(float(r[f])) for f in fields)]
    return rows, bad


def main():
    checks = []
    src = load_json(ROOT / "实验部署/source_window_cache_v2/manifest.json")
    sel = load_json(ROOT / "实验部署/source_window_cache_v2/window_selection.json")
    tgt = load_json(ROOT / "实验部署/hust_target_cache_v1/manifest.json")
    checks.append(("source window locked", src.get("status") == "SOURCE_CACHES_READY_WINDOW_SELECTED" and sel.get("status") == "WINDOW_SELECTED_ON_SOURCE_VALIDATION_ONLY" and sel.get("selected") == "offset_03_01"))
    checks.append(("target cache locked", tgt.get("status") == "TARGET_CACHE_READY_EVALUATION_LOCKED" and len(tgt.get("cells", [])) == 77))

    pinn = RESULTS / "corrected_pinn_xjtu_hust_v1"
    pd = load_json(pinn / "manifest.json") if (pinn / "manifest.json").exists() else {}
    pr = list(csv.DictReader((pinn / "pinn_results.csv").open())) if (pinn / "pinn_results.csv").exists() else []
    checks.append(("corrected PINN", pd.get("status") == "CORRECTED_PINN_COMPLETE" and len(pr) == 10 and all(math.isfinite(float(r["macro_mae"])) for r in pr)))

    raw = RESULTS / "corrected_raw_xjtu_hust_v6"
    rd = load_json(raw / "manifest.json") if (raw / "manifest.json").exists() else {}
    rr, rb = finite_csv(raw / "results.csv", ["macro_mae", "macro_rmse", "micro_mae", "micro_rmse"]) if (raw / "results.csv").exists() else ([], [1])
    rc = list(csv.DictReader((raw / "cell_results.csv").open())) if (raw / "cell_results.csv").exists() else []
    checks.append(("corrected raw baselines", rd.get("status") == "CORRECTED_RAW_BASELINES_COMPLETE" and len(rr) == 8 * 10 * 17 and not rb and len(rc) == len(rr) * 77))

    classical = RESULTS / "corrected_classical_xjtu_hust_v6"
    cd = load_json(classical / "manifest.json") if (classical / "manifest.json").exists() else {}
    cr, cb = finite_csv(classical / "results.csv", ["macro_mae", "macro_rmse"]) if (classical / "results.csv").exists() else ([], [1])
    checks.append(("corrected classical baselines", cd.get("status") == "CORRECTED_CLASSICAL_BASELINES_COMPLETE" and len(cr) == 75 and not cb))

    transfer = RESULTS / "corrected_transfer_baselines_v1"
    td = load_json(transfer / "manifest.json") if (transfer / "manifest.json").exists() else {}
    tr, tb = finite_csv(transfer / "transfer_results.csv", ["macro_mae", "macro_rmse"]) if (transfer / "transfer_results.csv").exists() else ([], [1])
    tc = list(csv.DictReader((transfer / "transfer_cell_results.csv").open())) if (transfer / "transfer_cell_results.csv").exists() else []
    checks.append(("corrected MAML/UDA", td.get("status") == "CORRECTED_TRANSFER_AND_UDA_COMPLETE" and len(tr) == 40 and not tb and len(tc) == len(tr) * 77))

    passed = all(x[1] for x in checks)
    result = {"status": "LEGACY_RESULTS_REQUIRE_PROTOCOL_REPAIR", "artifact_completeness_passed": passed,
              "scientific_readiness_passed": False,
              "reason": "Legacy audit only checks row presence/counts and finite values; it cannot validate convergence, information budgets or champion selection.",
              "replacement_evidence": "开发基线选择依据/results/fair_selection_v1",
              "checks": [{"name": n, "passed": p} for n, p in checks]}
    (ROOT / "实验部署/final_core_readiness_gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(1)


if __name__ == "__main__": main()
