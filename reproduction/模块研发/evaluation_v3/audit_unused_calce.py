"""Cell-level exposure audit of local raw CALCE archives; no capacity reads."""
import csv
import json
from pathlib import Path
import re
import subprocess
import zipfile
from evaluate import sa, HERE


def main():
    root=sa.ROOT
    out=root/'模块研发/results/calce_unused_cohort_audit_v1'
    out.mkdir(parents=True,exist_ok=False)
    historical=root/'开发基线选择依据/results/multidataset_hi_matrix/matrix_cell_results.csv'
    with historical.open() as f:scored=sorted({r['cell_id'] for r in csv.DictReader(f) if r['target']=='CALCE'})
    exposed={re.search(r'(C[XS]2_\d+)',cid).group(1) for cid in scored}
    roots=[root/'开发基线选择依据',root/'实验部署',root/'模块研发']
    # Capture searchable file inventory before writing new candidate evidence.
    inventory=subprocess.run(['rg','--files',*[str(p) for p in roots]],check=True,capture_output=True,text=True).stdout.splitlines()
    skip=[str(HERE),str(out)]
    textpaths=[Path(p) for p in inventory if Path(p).suffix in {'.json','.csv','.py','.md','.txt','.log'}
               and not any(p.startswith(s+'/') for s in skip)]
    candidates=[]
    for folder in ['CS2','CX2']:
        for archive in sorted((root/'数据集/CALCE'/folder).glob('*.zip')):
            cid=archive.stem
            if cid in exposed:continue
            pattern=re.compile(r'(?<![A-Za-z0-9])'+re.escape(cid)+r'(?!\d)')
            matches=[]
            for p in textpaths:
                for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                    if pattern.search(line):matches.append(dict(path=str(p.relative_to(root)),line=i,text=line[:250]))
            with zipfile.ZipFile(archive) as z:
                members=[n for n in z.namelist() if not n.endswith('/')]
            candidates.append(dict(cell_id=cid,archive=str(archive.relative_to(root)),sha256=sa.digest(archive),
                member_count=len(members),formats=sorted({Path(n).suffix for n in members}),members=members,
                historical_text_matches=matches,exposure_status='NO_MATCH_IN_AUDITED_HISTORY' if not matches else 'REQUIRES_REVIEW'))
    result=dict(status='CELL_INVENTORY_AUDITED_PROTOCOL_COMPATIBILITY_PENDING',
        code_sha256=sa.digest(Path(__file__)),scored_cell_ids=scored,scored_file_sha256=sa.digest(historical),
        searched_text_files=len(textpaths),candidate_cells=len(candidates),candidates=candidates,
        limits=['Absence of matching IDs is evidence within audited workspace, not proof of global never-use.',
                'CALCE is a previously evaluated source; a new cell cohort is not a new laboratory or new chemistry.',
                'No final model predictions or target capacity trajectories read by this audit.',
                'Archive member names and historical IDs do not establish signal/label compatibility.',
                'PL pouch archives remain a separate possible cohort, not assessed here.'])
    sa.write_json(out/'audit.json',result)
    print(json.dumps({k:result[k] for k in ['status','searched_text_files','candidate_cells']},indent=2))
    print([(r['cell_id'],r['exposure_status']) for r in candidates])


if __name__=='__main__':main()
