#!/usr/bin/env python3
"""Run the pinned BatteryML MATR parser through a local filename adapter."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'数据集/MATR'
STAGE=ROOT/'实验部署/matr_batteryml_input'
OUT=ROOT/'数据集/MATR_BatteryML_processed'
FILES=[('2017-05-12_batchdata_updated_struct_errorcorrect.mat','MATR_batch_20170512.mat'),('2017-06-30_batchdata_updated_struct_errorcorrect.mat','MATR_batch_20170630.mat'),('2018-04-12_batchdata_updated_struct_errorcorrect.mat','MATR_batch_20180412.mat'),('2019-01-24_batchdata_updated_struct_errorcorrect.mat','MATR_batch_20190124.mat')]
def main():
    STAGE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for src,dst in FILES:
        p=STAGE/dst
        if not p.exists():p.symlink_to(RAW/src)
    sys.path.insert(0,str(ROOT/'基线模型/参考实现/BatteryML'))
    from batteryml.preprocess.preprocess_MATR import MATRPreprocessor
    pre=MATRPreprocessor(str(OUT),silent=False); pre.process(STAGE)
    files=sorted(p.name for p in OUT.glob('*.pkl'))
    (ROOT/'实验部署/matr_processed_manifest.json').write_text(json.dumps({'parser':'BatteryML MATRPreprocessor','source_files':[str(RAW/s) for s,_ in FILES],'output_dir':str(OUT),'cell_files':files,'cell_count':len(files)},indent=2),encoding='utf8')
    print(json.dumps({'cell_count':len(files),'output':str(OUT)},indent=2))
if __name__=='__main__':main()
