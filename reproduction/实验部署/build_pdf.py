"""Compile existing manuscript assets; do not regenerate evidence or train models."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
logs = root / '核验'
logs.mkdir(exist_ok=True)
for i in range(1, 4):
    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                            cwd=root/'正文', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (logs/f'pdflatex_pass_{i}.txt').write_bytes(result.stdout)
    if result.returncode:
        print(result.stdout.decode(errors='replace')[-5000:])
        raise SystemExit(result.returncode)
    print(f'LaTeX pass {i}: success', flush=True)
print(root/'正文/main.pdf')
