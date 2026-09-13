"""Cache frozen M1/A/B predictions for cheap gate search."""
import json,sys
from pathlib import Path
import joblib,numpy as np
M1=Path(__file__).resolve().parents[1]/'m1';sys.path.insert(0,str(M1))
from data import load_cells,split  # noqa:E402
from meta_residual import K,episode_features,predict_mode  # noqa:E402

def main(out='模块研发/results/m2_prediction_cache_v1'):
    r7=Path('模块研发/results/m1_v1/screen_v2');r2=Path('模块研发/results/m1_v1/screen_v1');ra=Path('模块研发/results/m2_calibration_support_v3');out=Path(out);out.mkdir(parents=True,exist_ok=True)
    specs=json.loads((M1/'candidates_v1.json').read_text());cells=[c for d in ['XJTU','MATR','Tongji'] for c in load_cells(d)];folds=sorted({f'{c.dataset}:{c.domain}' for c in cells})
    for fold in folds:
        tr,val,te=split(cells,fold);name=fold.replace(':','__');j7=r7/'jobs'/f'P07_pls_concat__{name}';m7=joblib.load(j7/'model.joblib');mode7=json.loads((j7/'tuning.json').read_text())['chosen']['mode'];info=joblib.load(ra/'fold_models'/f'{name}.joblib');cal,alpha,beta=info['calibration'],info['alpha'],info['beta'];j2=r2/'jobs'/f'P02_anchor_physics__{name}';m2=joblib.load(j2/'model.joblib');mode2=json.loads((j2/'tuning.json').read_text())['chosen']['mode']
        rows=[]
        for split_name,cs in [('val',val),('test',te)]:
            for c in cs:
                p0=predict_mode(m7,c,mode7);X=episode_features(m7,c,mode7);pa=p0+alpha*(cal.predict(p0)-p0)+beta*X[:,0];pb=predict_mode(m2,c,mode2);q=np.arange(K,len(c.y));
                for i in range(len(q)): rows.append({'split':split_name,'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,'q':int(q[i]),'y':float(c.y[q[i]]),'p0':float(p0[i]),'pa':float(pa[i]),'pb':float(pb[i]),'resid':float(X[i,0]),'slope':float(X[i,1]),'std':float(X[i,2]),'dis_ab':float(abs(pa[i]-pb[i])),'dis_a0':float(abs(pa[i]-p0[i])),'dis_b0':float(abs(pb[i]-p0[i])),'base':float(X[i,3])})
        (out/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False))
    print('cached',len(folds),'folds')
if __name__=='__main__':main()
