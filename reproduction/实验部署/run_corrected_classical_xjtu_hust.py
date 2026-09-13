"""Corrected handcrafted-HI and simple baseline benchmark on locked caches."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(base, dirs):
    rows = []
    for d in dirs:
        for p in sorted((base / d).glob("*.npz")):
            with np.load(p, allow_pickle=True) as z:
                x, q, cyc = z["x"].astype("float32"), z["capacity_Ah"].astype("float32"), z["cycle_number"].astype("float32")
                cid = str(z["cell_id"].item())
            o = np.argsort(cyc); rows.append({"id": cid, "x": x[o], "q": q[o], "cycle": cyc[o]})
    return rows


def features(x):
    t = np.linspace(0, 1, x.shape[-1], dtype="float32")
    out = []
    for c in range(x.shape[1]):
        z = x[:, c, :]
        slope = ((z - z.mean(1, keepdims=True)) * (t - t.mean())).sum(1) / ((t - t.mean()) ** 2).sum()
        out.append(np.stack([z.mean(1), z.std(1), z.min(1), z.max(1), z[:, 0], z[:, -1], slope], 1))
    return np.concatenate(out, 1).astype("float32")


def model(name, seed):
    if name == "ridge": return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "plsr": return make_pipeline(StandardScaler(), PLSRegression(n_components=5, max_iter=500))
    if name == "svr_rbf": return make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=.002))
    if name == "random_forest": return RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=2, min_samples_leaf=2)
    if name == "xgboost": return XGBRegressor(n_estimators=50, max_depth=5, learning_rate=.03, subsample=.8, colsample_bytree=.8, reg_lambda=1.0, objective="reg:squarederror", n_jobs=2, random_state=seed)
    if name == "gpr": return GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(.01), normalize_y=True, random_state=seed, n_restarts_optimizer=0)
    raise KeyError(name)


def fit_predict(name, x, y, qx, seed):
    if len(y) == 0: return np.full(len(qx), .0, dtype="float32")
    if len(y) < 2 and name in {"svr_rbf", "gpr", "plsr"}:
        return np.full(len(qx), y[-1], dtype="float32")
    reg = model(name, seed)
    # GPR has a fixed effective budget; this is recorded instead of pretending
    # that its nominal fitting budget equals the neural source budget.
    if name == "gpr" and len(y) > 200:
        keep = np.linspace(0, len(y) - 1, 200).astype(int)
        x, y = x[keep], y[keep]
    reg.fit(x, y)
    return np.asarray(reg.predict(qx), dtype="float32")


def simple(name, support_y, n, fallback):
    if name == "last_support": return np.full(n, support_y[-1] if len(support_y) else fallback, dtype="float32")
    if len(support_y) < 2: return np.full(n, support_y[-1] if len(support_y) else fallback, dtype="float32")
    t = np.arange(len(support_y), dtype=float)
    if name == "linear_trend": coef = np.polyfit(t, support_y, 1); return np.polyval(coef, np.arange(len(support_y), len(support_y)+n)).astype("float32")
    if name == "exponential_trend":
        z = np.log(np.clip(support_y, 1e-6, None)); coef = np.polyfit(t, z, 1)
        return np.exp(np.polyval(coef, np.arange(len(support_y), len(support_y)+n))).astype("float32")
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ks", default="0,1,5,10,20"); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9"); ap.add_argument("--output", default="corrected_classical_xjtu_hust_v1"); ap.add_argument("--source-budget", type=int, default=1000)
    args = ap.parse_args(); out = ROOT / "开发基线选择依据/results" / args.output
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "实验部署/source_window_cache_v2/offset_03_01"; target = ROOT / "实验部署/hust_target_cache_v1"
    tr = load(source, ["source_train"]); te = load(target, ["."])
    xs=[]; ys=[]; keys=[]
    for i,r in enumerate(tr):
        keep=np.linspace(0,len(r["q"])-1,min(args.source_budget//len(tr)+int(i<args.source_budget%len(tr)),len(r["q"]))).astype(int)
        xs.append(r["x"][keep]); ys.append(r["q"][keep]/r["q"][0]); keys.extend((r["id"],float(r["cycle"][j])) for j in keep)
    sx=features(np.concatenate(xs)); sy=np.concatenate(ys); fallback=float(sy.mean())
    for r in te: r["xfeat"]=features(r["x"]); r["y"]=r["q"]/r["q"][0]
    seeds=[int(x) for x in args.seeds.split(",")]; ks=[int(x) for x in args.ks.split(",")]
    names=["dummy_mean","last_support","linear_trend","exponential_trend","ridge","plsr","svr_rbf","random_forest","xgboost","gpr","full_target_upper_bound"]
    rows=[]
    # Fixed deterministic models do not gain scientific information from
    # repeated identical fits.  Stochastic tree models retain the requested
    # seed repeats; GPR has a fixed effective budget and is run once.
    model_seeds = {n: ([0] if n in {"ridge", "plsr", "svr_rbf", "gpr"} else seeds[:3])
                   for n in {"ridge", "plsr", "svr_rbf", "random_forest", "xgboost", "gpr"}}
    source_fit = {}
    for name in model_seeds:
        for seed in model_seeds[name]:
            source_fit[(name, seed)] = model(name, seed)
            if name == "gpr":
                keep = np.linspace(0, len(sy) - 1, min(200, len(sy))).astype(int)
                source_fit[(name, seed)].fit(sx[keep], sy[keep])
            else:
                source_fit[(name, seed)].fit(sx, sy)
    for k in ks:
        for name in names:
            run_seeds = model_seeds[name] if name in model_seeds else [-1]
            for seed in run_seeds:
                maes=[]; rmses=[]; count=0
                for r in te:
                    qx=r["xfeat"][k:]; qy=r["y"][k:]; sy0=r["y"][:k]
                    if name == "full_target_upper_bound": pred=fit_predict("ridge",r["xfeat"],r["y"],qx,0)
                    elif name == "dummy_mean": pred=np.full(len(qx),fallback,dtype="float32")
                    elif name in {"last_support","linear_trend","exponential_trend"}: pred=simple(name,sy0,len(qx),fallback)
                    elif k == 0: pred=source_fit[(name, seed)].predict(qx).astype("float32")
                    else:
                        # Target-only uses exactly the K support observations.
                        # This avoids silently giving classical models a
                        # different source/sample budget from the declared
                        # target-only baseline and keeps the fit tractable.
                        support_x, support_y=r["xfeat"][:k],sy0
                        pred=fit_predict(name,support_x,support_y,qx,seed)
                    maes.append(float(np.mean(abs(pred-qy)))); rmses.append(float(np.sqrt(np.mean((pred-qy)**2)))); count += len(qy)
                rows.append({"K":k,"model":name,"variant":"source_only" if k==0 else ("support_or_trend" if name in {"dummy_mean","last_support","linear_trend","exponential_trend"} else "target_only"),"seed":seed,"macro_mae":float(np.mean(maes)),"macro_rmse":float(np.mean(rmses)),"query_count":count,"gpr_effective_budget":200 if name=="gpr" else None})
        with (out/"results.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    manifest={"status":"CORRECTED_CLASSICAL_BASELINES_COMPLETE","task":"XJTU_to_HUST","source_cache":str(source.relative_to(ROOT)),"target_cache":str(target.relative_to(ROOT)),"source_selection_sha256":digest(ROOT/"实验部署/source_window_cache_v2/window_selection.json"),"target_cache_manifest_sha256":digest(target/"manifest.json"),"feature_definition":"per-channel mean,std,min,max,first,last,slope over fixed 128-point partial-charge curve","ks":ks,"requested_seeds":seeds,"stochastic_tree_seeds":seeds[:3],"source_budget":args.source_budget,"gpr_effective_budget":200,"query_labels_used_for_selection":False,"classical_target_variant":"target_only for K>0; no source-plus-support hidden budget"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"rows":len(rows),"status":manifest["status"]}))
if __name__=="__main__": main()
