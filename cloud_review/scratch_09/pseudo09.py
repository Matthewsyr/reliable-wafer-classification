"""Development-only Scratch pseudo-label gate for hosted Google Colab.

python pseudo09.py qualify --repo REPO --out NEW_GATE_DIR
python pseudo09.py select --repo REPO --qualification GATE_DIR --out NEW_SELECTION_DIR \
    --model RF77_S42 --raw LSWMD.pkl --pack OFFSETS_DIR --pool ssl_pool.csv.gz
python pseudo09.py self-test
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys


CLASSES = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none")
BANDS = ("0", "1-24", "25+")
THRESHOLDS = (.50, .60, .70, .80, .90, .95)
MANIFEST = "c86c2c3d77762d6d8083545bbb3377b09c13eb09d7119c17426b933602d0a0a7"
DEV = "4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879"
SPLIT_SALT = "wm811k-decision-component-20261101-v1|"
STUDENT_FIELDS = ("source_row", "lot", "height", "width", "map_sha256", "raw_data_offset",
                  "raw_data_nbytes", "order", "p_scratch", "size_band", "threshold", "pseudo_label")
RULE = dict(model="rf_linearity77", seed=42, target="Scratch", prediction="raw nine-way argmax",
            design="historical tune/calibration components", confirmation="historical tune/selection components",
            size_support="number of real supervised training normals of identical ordered (height,width)",
            size_bands=BANDS, threshold_candidates=THRESHOLDS, min_wafers=12, min_lots=5,
            min_precision=.90, min_wilson_lower_one_sided_95=.75, confirm_min_leave_one_lot_out_precision=.80,
            threshold_choice="max accepted design wafers, tie highest threshold; each band confirmed once",
            student_max_unique=80, student_max_per_lot=2, pseudo_classes=["Scratch"])


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned(path, expected):
    check(digest(path) == expected, f"SHA-256 mismatch: {path.name}")
    return path


def write_json(path, obj):
    temp = path.with_suffix(".part")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def write_csv(path, fields, rows):
    temp = path.with_suffix(".part")
    with temp.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_gz(path):
    with gzip.open(path, "rt", newline="", encoding="utf8") as stream:
        yield from csv.DictReader(stream)


def hosted_only():
    import importlib.util
    check(os.name == "posix" and Path("/content").is_dir()
          and importlib.util.find_spec("google.colab") is not None,
          "Real-row commands require the hosted Google Colab runtime")


def band(count):
    return "0" if count == 0 else "1-24" if count < 25 else "25+"


def inputs(repo):
    ref = pinned(repo / "data-reference/development_reference.csv.gz", DEV)
    archive = repo / "results/probabilities"
    manifest = json.loads(pinned(archive/"manifest.json", MANIFEST).read_text())
    check(tuple(manifest["probability_columns"]) == tuple(manifest["class_order"]) == CLASSES, "Class axis")
    spec, = [s for s in manifest["probability_files"] if
             (s["model"], s["seed"], s["role"]) == ("rf_linearity77", 42, "tune")]
    scores = pinned(archive / spec["file"], spec["sha256"])
    refs = list(read_gz(ref))
    check(len(refs) == len({r["source_row"] for r in refs}) == 54355, "Reference identities")
    support = Counter((int(r["height"]), int(r["width"])) for r in refs if
                      r["role"] == "train" and r["label"] == "none")
    check(sum(support.values()) == 20780, "Original training normal support")
    seen = {key: {} for key in ("lot", "map_sha256", "component")}
    tune = {}
    counts = Counter()
    for r in refs:
        role = r["role"]
        check(role in ("train", "tune", "audit"), "Invalid role")
        if role == "tune":
            design = hashlib.sha256((SPLIT_SALT + r["component"]).encode()).digest()[0] < 128
            role = "design" if design else "confirmation"
            r["partition"] = role
            r["band"] = band(support[int(r["height"]), int(r["width"])])
            tune[r["source_row"]] = r
        counts[role] += 1
        for key, assignment in seen.items():
            check(r[key] and assignment.setdefault(r[key], role) == role, "Cross-partition " + key)
    check(dict(counts) == dict(train=31457, design=5637, confirmation=5670, audit=11591), "Role counts")
    loaded = set()
    for p in read_gz(scores):
        sid = p["source_row"]
        check(sid in tune and sid not in loaded and p["truth"] == tune[sid]["label"], "Tune score identity")
        loaded.add(sid)
        raw = tuple(float(p[c]) for c in CLASSES)
        check(all(0 <= v <= 1 for v in raw) and abs(sum(raw)-1) < 2e-6, "Invalid RF votes")
        r = tune[sid]
        r["prob"] = raw
        r["pred"] = CLASSES[max(range(len(raw)), key=raw.__getitem__)]
    check(len(loaded) == len(tune) == spec["rows"], "Incomplete tune scores")
    return list(tune.values()), support, seen, spec["sha256"]


def ratio(num, den):
    return num / den if den else None


def wilson(tp, n):
    if not n:
        return None
    z = 1.6448536269514722
    p = tp/n
    return (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)


def metrics(rows, threshold, label="Scratch"):
    ix = CLASSES.index(label)
    accepted = [r for r in rows if r["pred"] == label and r["prob"][ix] >= threshold]
    truth = sum(r["label"] == label for r in rows)
    n = len(accepted)
    tp = sum(r["label"] == label for r in accepted)
    fp_normal = sum(r["label"] == "none" and r["label"] != label for r in accepted)
    per_lot = defaultdict(list)
    by_image = defaultdict(list)
    for r in accepted:
        hit = r["label"] == label
        per_lot[r["lot"]].append(hit)
        by_image[r["map_sha256"]].append(hit)
    loo = [(tp-sum(hits))/(n-len(hits)) for hits in per_lot.values() if len(hits) < n]
    return dict(band_rows=len(rows), truth=truth, accepted=n, tp=tp, fp=n-tp,
                normal_as_target=fp_normal, other_as_target=n-tp-fp_normal,
                precision=ratio(tp,n), lower95_one_sided=wilson(tp,n), recall=ratio(tp,truth),
                coverage=ratio(n,len(rows)), accepted_lots=len(per_lot),
                accepted_components=len({r["component"] for r in accepted}),
                accepted_unique_images=len(by_image),
                dedup_precision_conservative=ratio(sum(all(hits) for hits in by_image.values()),len(by_image)),
                worst_leave_one_lot_out_precision=min(loo,default=None))


def passed(m, confirm=False):
    basic = (m["accepted"] >= RULE["min_wafers"] and m["accepted_lots"] >= RULE["min_lots"]
             and m["precision"] >= RULE["min_precision"]
             and m["lower95_one_sided"] >= RULE["min_wilson_lower_one_sided_95"])
    return basic and (not confirm or m["worst_leave_one_lot_out_precision"] >=
                       RULE["confirm_min_leave_one_lot_out_precision"])


def design_rule(rows):
    grid = []
    chosen = {}
    for size in BANDS:
        group = [r for r in rows if r["band"] == size]
        for t in THRESHOLDS:
            m = metrics(group,t)
            grid.append(dict(band=size, threshold=t, passed=passed(m), **m))
        valid = [m for m in grid if m["band"] == size and m["passed"]]
        if valid:
            chosen[size] = max(valid, key=lambda m: (m["accepted"],m["threshold"]))["threshold"]
    return grid, chosen


def save_student(out, qualification, selected, receipt):
    csv_path = out/"student_pseudo.csv"
    q = json.loads(qualification.read_text())
    check(bool(q["qualified_bands"]) or not selected, "Unqualified pseudo labels")
    write_csv(csv_path, STUDENT_FIELDS, selected)
    write_json(out/"select_receipt.json", dict(status="complete", rows=len(selected), pseudo_train=bool(selected),
               qualified_bands=q["qualified_bands"], qualification_sha256=digest(qualification),
               student_pseudo_sha256=digest(csv_path), **receipt))


def qualify(args):
    tune, _, seen, score_digest = inputs(args.repo)
    design = [r for r in tune if r["partition"] == "design"]
    confirm = [r for r in tune if r["partition"] == "confirmation"]
    args.out.mkdir(parents=True, exist_ok=True)
    grid, chosen = design_rule(design)
    freeze = args.out/"design_freeze.json"
    write_json(freeze, dict(rule=RULE, score_sha256=score_digest, thresholds=chosen))
    write_csv(args.out/"design_grid.csv", list(grid[0]), grid)
    confirmed, qualified = [], {}
    for size, threshold in chosen.items():
        m = metrics([r for r in confirm if r["band"] == size], threshold)
        ok = passed(m, confirm=True)
        confirmed.append(dict(band=size, threshold=threshold, passed=ok, **m))
        if ok:
            qualified[size] = threshold
    write_csv(args.out/"confirmation.csv", list(grid[0]), confirmed)
    descriptive = []
    for part, rows in (("design",design), ("confirmation",confirm)):
        for size in BANDS:
            members = [r for r in rows if r["band"] == size]
            for label in CLASSES:
                descriptive.append(dict(partition=part, band=size, label=label,
                                        cutoff="argmax", **metrics(members,0,label)))
            descriptive.append(dict(partition=part, band=size, label="Scratch", cutoff="0.50",
                                    **metrics(members,.5)))
    write_csv(args.out/"descriptive_by_size.csv",list(descriptive[0]),descriptive)
    partitions = {part: {k: sum(val == part for val in lookup.values()) for k,lookup in seen.items()}
                  for part in ("train","design","confirmation","audit")}
    q = args.out/"qualification.json"
    write_json(q, dict(status="complete", rule=RULE, source_sha256=digest(Path(__file__)),
                       reference_sha256=DEV, manifest_sha256=MANIFEST, score_sha256=score_digest,
                       unique_counts_by_partition=partitions, partition_crossings=dict.fromkeys(seen,0),
                       design_freeze_sha256=digest(freeze), design_thresholds=chosen,
                       confirmed=confirmed, qualified_bands=qualified, needs_unlabeled_scoring=bool(qualified),
                       artifacts={n:digest(args.out/n) for n in
                                  ("design_grid.csv","confirmation.csv","descriptive_by_size.csv")}))
    if not qualified:
        save_student(args.out, q, [], dict(reason="No size band passed the frozen Scratch gate."))
    print(json.dumps(dict(design_thresholds=chosen, qualified_bands=qualified)))


def self_test():
    rows = [dict(label="Scratch", pred="Scratch", band="25+", prob=(0,0,0,0,0,0,0,.8,.2),
                 lot=str(i%6), component=str(i%6), map_sha256=str(i)) for i in range(12)]
    grid, chosen = design_rule(rows)
    check(len(grid)==18 and chosen=={"25+":.80}, "Design tie or absent bands")
    m = metrics(rows,.8)
    check(passed(m,True) and m["tp"]==12 and m["worst_leave_one_lot_out_precision"]==1, "Perfect gate")
    for r in rows[:2]:
        r["label"] = "none"
    m = metrics(rows,.8)
    check(not passed(m,True) and m["normal_as_target"] == 2, "False positive gate")
    check([band(v) for v in (0,1,24,25)]==["0","1-24","1-24","25+"], "Band edges")
    print("synthetic self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    q, s = sub.add_parser("qualify"), sub.add_parser("select")
    for p in (q,s):
        p.add_argument("--repo",type=Path,required=True)
        p.add_argument("--out",type=Path,required=True)
    for name in ("qualification","model","raw","pack","pool","registry"):
        s.add_argument("--"+name,type=Path,required=name=="qualification")
    args = parser.parse_args()
    if args.command == "self-test":
        return self_test()
    hosted_only()
    if args.command == "qualify":
        return qualify(args)
    from select_pseudo09 import select
    select(args)


if __name__ == "__main__":
    main()
