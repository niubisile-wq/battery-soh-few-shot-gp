"""Materialize the fixed A/B domain-route upper-bound audit.

This is deliberately labeled exploratory: the route map is not selected by
the nested protocol and must not be reported as the final module.
"""
import json,glob
from collections import defaultdict
from pathlib import Path
import numpy as np

def macro(rows,k):
 g=defaultdict(list)
 for r in rows:
  if r[k] is not None:g[(r['dataset'],r['domain'])].append(float(r[k]))
 return float(np.mean([np.mean(v) for v in g.values()]))

def main():
 a=json.load(open('模块研发/results/m2_calibration_support_v3/summary.json'))['cell_results'];b=[]
 for p in glob.glob('模块研发/results/m1_v1/screen_v1/jobs/P02_anchor_physics__*/cell_results.json'):b.extend(json.load(open(p)))
 bm={(r['fold'],r['cell_id']):r for r in b}; rows=[];route={'XJTU':'A','MATR':'B','Tongji':'B'}
 for x in a: rows.append(x if route[x['dataset']]=='A' else bm[(x['fold'],x['cell_id'])])
 summary=[]
 for d in ['XJTU','MATR','Tongji']:
  rs=[r for r in rows if r['dataset']==d];summary.append({'dataset':d,'route':route[d],'cells':len(rs),**{k:macro(rs,k) for k in ['mae','rmse','p95_ae','low_soh_mae']}})
 out=Path('模块研发/results/m2_v1');out.mkdir(parents=True,exist_ok=True);(out/'domain_route_upper_bound.json').write_text(json.dumps({'status':'EXPLORATORY_POST_HOC_UPPER_BOUND','route':route,'summary':summary,'cell_results':rows},ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
