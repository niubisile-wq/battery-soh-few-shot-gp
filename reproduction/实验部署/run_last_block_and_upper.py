#!/usr/bin/env python3
"""Complete the missing last-block and explicit oracle diagnostics for K=10."""
import csv, json
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import run_formal_protocol as formal

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / os.environ.get("LB_OUT", "last_block_upper")
K, EPOCHS, STEPS = 10, 25, 30
SEED_START = int(os.environ.get("SEED_START", "0")); SEED_COUNT = int(os.environ.get("SEED_COUNT", "5")); SEEDS = list(range(SEED_START, SEED_START + SEED_COUNT))

def score(y, p):
    return float(mean_absolute_error(y, p)), float(mean_squared_error(y, p) ** .5)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, target = formal.read_cells(formal.SOURCE_ZIP), formal.read_cells(formal.TARGET_ZIP)
    src_train = [c for i, c in enumerate(source) if i % 5 != 4]
    src_val = [c for i, c in enumerate(source) if i % 5 == 4]
    tx, ty = np.concatenate([c[1] for c in src_train]), np.concatenate([c[2] for c in src_train])
    vx, vy = np.concatenate([c[1] for c in src_val]), np.concatenate([c[2] for c in src_val])
    mu, sd = tx.mean((0,2), keepdims=True), tx.std((0,2), keepdims=True); sd[sd < 1e-6] = 1
    norm = lambda x: ((x-mu)/sd).astype("float32")
    tx, vx = norm(tx), norm(vx)
    target = [(n, norm(x), y) for n,x,y in target]
    keep = np.linspace(0, len(tx)-1, min(1000, len(tx))).astype(int); tx, ty = tx[keep], ty[keep]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cells = [], []
    for name, ctor in formal.NN_MODELS.items():
        for seed in SEEDS:
            print(f"LAST_BLOCK {name} seed={seed}", flush=True)
            base = formal.fit_nn(ctor, tx, ty, vx, vy, device, EPOCHS, seed)
            maes=[]; rmses=[]
            for cid, x, y in target:
                model = formal.adapt(base, x[:K], y[:K], "last_block", device, STEPS)
                a,b=score(y[K:], formal.predict(model, x[K:], device)); maes.append(a); rmses.append(b)
                cells.append({"track":"raw_V_I_time","model":name,"variant":"last_block","seed":seed,"K":K,"cell_id":cid,"mae":a,"rmse":b})
            rows.append({"track":"raw_V_I_time","model":name,"variant":"last_block","seed":seed,"K":K,"macro_mae":float(np.mean(maes)),"macro_rmse":float(np.mean(rmses)),"target_cells":len(maes),"source_budget":1000})
    # Deliberately non-comparable oracle: train on every target label and score on
    # those same cycles. It is only a ceiling/diagnostic, never a model ranking.
    oracle_mae=[]; oracle_rmse=[]
    for cid,x,y in target:
        oracle = formal.fit_nn(formal.MLP, x, y, vx, vy, device, EPOCHS, 9000, early_stop=False)
        a,b=score(y, formal.predict(oracle,x,device)); oracle_mae.append(a); oracle_rmse.append(b)
        cells.append({"track":"raw_V_I_time","model":"MLP","variant":"full_target_upper_bound","seed":-1,"K":K,"cell_id":cid,"mae":a,"rmse":b})
    rows.append({"track":"raw_V_I_time","model":"MLP","variant":"full_target_upper_bound","seed":-1,"K":K,"macro_mae":float(np.mean(oracle_mae)),"macro_rmse":float(np.mean(oracle_rmse)),"target_cells":len(oracle_mae),"source_budget":0})
    for fn, data in [("last_block_upper_results.csv",rows),("last_block_upper_cell_results.csv",cells)]:
        with (OUT/fn).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    (OUT/"manifest.json").write_text(json.dumps({"task":"XJTU_to_HUST","K":K,"seeds":SEEDS,"last_block":"freeze all but final named parameter groups","oracle":"MLP trained and scored on all target cycles; diagnostic only; query labels used"},indent=2),encoding="utf8")
    print(json.dumps({"rows":len(rows),"cell_rows":len(cells),"device":str(device),"output":str(OUT)},indent=2))
if __name__ == "__main__": main()
