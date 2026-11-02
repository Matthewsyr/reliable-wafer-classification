"""Build the private prior-exposure registry once on a hosted CPU, before registration."""
import argparse
import csv
import gzip
import json
import sqlite3
import tempfile
from pathlib import Path

from prospective import atomic_dir, sha, checked, module_file, REPO


def build(raw, extra, out):
    code = REPO / "cloud_review/extension/shift_audit_v1.py"
    checked(code, "786692eaea13c1b0a6d52d821e443e00f546735aa353a24d034caec94638b7e2")
    index = module_file("historical_wafer_indexer", code)
    index.require_cloud()
    with atomic_dir(out) as tmp, tempfile.TemporaryDirectory() as scratch:
        target = tmp / "prior.csv.gz"
        with sqlite3.connect(Path(scratch) / "source.db") as db, gzip.open(target, "wt", newline="", encoding="utf8") as file:
            flags = index.index_maps(raw, db)  # Checks the entire source hash before unpickling.
            total, good = db.execute("SELECT COUNT(*),SUM(good) FROM m").fetchone()
            if (total, good) != (811457, 811457) or flags:
                raise ValueError("Unexpected original WM-811K registry")
            writer = csv.writer(file)
            cols = ["wafer_id", "lot", "map_sha256"]
            writer.writerow(cols)
            writer.writerows(db.execute("SELECT 'WM811K:'||id,lot,LOWER(HEX(digest)) FROM m ORDER BY id"))
            extra_n = 0
            if extra:
                with extra.open(newline="", encoding="utf-8-sig") as more:
                    reader = csv.DictReader(more)
                    if reader.fieldnames != cols:
                        raise ValueError(f"Expected prior exposure columns {cols}")
                    for row in reader:
                        vals = [row[c].strip() for c in cols]
                        if not all(vals) or len(bytes.fromhex(vals[2])) != 32:
                            raise ValueError("Invalid additional historical record")
                        writer.writerow([vals[0], vals[1], vals[2].lower()])
                        extra_n += 1
        receipt = dict(original_rows=total, extra_rows=extra_n, original_raw_sha256=index.SHA,
                       extra_sha256=sha(extra) if extra else None, registry_sha256=sha(target))
        (tmp/"receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--extra", type=Path, help="Prior pilots / other historical sources sharing the canonical ID namespace")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    build(args.raw, args.extra, args.out)
