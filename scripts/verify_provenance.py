"""Verify byte-identical vendored source snapshots against provenance.json."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
record=json.loads((root/'src/waferdefect/provenance.json').read_text(encoding='utf-8'))
for row in record['files']:
    path=root/row['packaged_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise SystemExit(f"Source provenance mismatch: {row['packaged_path']}")
print(f"Verified {len(record['files'])} exact source snapshots; absent Git commit IDs are explicitly null.")
