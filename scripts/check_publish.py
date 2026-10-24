"""Fail before publication if sensitive/bulky files escaped ignore rules.

When Git exists, inspect its tracked/indexed paths (including staged additions).
Before Git initialization, inspect distributable source paths only. This does
not print file contents or secrets. It is not a substitute for human review.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
forbidden_suffixes={'.pkl','.pickle','.pt','.pth','.joblib','.npy','.npz','.whl','.zip'}
forbidden_names={'kaggle.json','credentials.json','.env'}
try:
    output=subprocess.check_output(['git','ls-files','-z','--cached'],cwd=ROOT,stderr=subprocess.DEVNULL)
    paths=[ROOT/name for name in output.decode('utf-8').split('\0') if name]
except (subprocess.CalledProcessError,FileNotFoundError):
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(part in {'__pycache__','.pytest_cache','.git','data','runs','outputs','.venv','build','dist'} for part in p.relative_to(ROOT).parts)]
issues=[]
for path in paths:
    if path.suffix.lower() in forbidden_suffixes or path.name in forbidden_names or path.name.startswith('.env.'):
        issues.append((str(path.relative_to(ROOT)),'disallowed artifact/credential filename'))
    if path.exists() and path.stat().st_size>10*1024*1024:
        issues.append((str(path.relative_to(ROOT)),'larger than 10 MiB'))
if issues:
    for name,reason in issues:print(f'{reason}: {name}',file=sys.stderr)
    raise SystemExit(1)
print(f'Publication path check passed ({len(paths)} files; no data/model/credential files detected).')
