"""Copy the preceding manuscript revision; never modify the source version."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parent
previous = root.parent / '论文修订版_证据收束_20260911'
if (root / '正文').exists():
    raise SystemExit('Version already prepared; refusing to overwrite edited files.')
shutil.copytree(previous / '正文', root / '正文')
for name in ['build_evidence.py', 'audit_revision.py', 'evidence_manifest.json', '图表目录.md']:
    shutil.copy2(previous / name, root / name)
manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(previous.rglob('*')) if p.is_file()}
(root / 'previous_version_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('Prepared independent narrative revision; previous files protected:', len(manifest))
