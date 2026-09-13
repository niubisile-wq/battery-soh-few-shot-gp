"""Small, isolated IO utilities; never modify old manuscript/model artifacts."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parent
BUNDLE = ROOT / '模块研发/results/m4_random_function_candidate_v1'
PROTOCOL_PATH = HERE / 'protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Refuse an empty evidence table')
    path = Path(path)
    assert path.resolve().is_relative_to(HERE), path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def preservation():
    dest = HERE / 'preservation.json'
    if dest.exists():
        record = json.loads(dest.read_text())
        for p, sha in record['protected_files'].items():
            if digest(p) != sha:
                raise RuntimeError('Protected original changed: ' + p)
        return record
    files = subprocess.check_output(['rg', '--files', '-g', '*.tex', '-g', '*.pdf', '-g', '*.png',
         '-g', '*.svg', '-g', '*.json', '-g', '*.py',
         str(WORK/'论文修订版_实验问题修复/正文'), str(BUNDLE/'source_snapshot')], text=True).splitlines()
    files += [str(WORK/'main.tex'), str(BUNDLE/'candidate.json'), str(BUNDLE/'acceptance.json'),
              str(BUNDLE/'comparison.csv'), str(BUNDLE/'cells.csv'), str(PROTOCOL_PATH)]
    record = dict(created_unix=time.time(), protected_files={p:digest(p) for p in sorted(set(files))},
                  model_manifest_sha256=digest(BUNDLE/'candidate.json'),
                  scope='Manuscript assets and frozen model code/data summaries; model artifacts checked against candidate manifest in final audit')
    write_json(dest, record)
    return record


if __name__ == '__main__':
    print('Protected files verified:', len(preservation()['protected_files']))
