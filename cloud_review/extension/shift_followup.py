"""Hosted-CPU follow-up for the frozen WM-811K cost/shift study; no model fitting.

Inputs: previous Colab result folder, pinned development reference, seed-42 RF71
audit CSV, original Kaggle v1 pickle, and original shift_audit.py (or its notebook).
Use --stage cost / pool / decision to resume each item on mounted persistent storage.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sqlite3
import tempfile
from collections import Counter


REFERENCE = "4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879"
INDEXER = "786692eaea13c1b0a6d52d821e443e00f546735aa353a24d034caec94638b7e2"
RAW = "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65"
ANCHOR = "3b35158896762752e0f34e43894830f45ce782d4bd1f1b799a9fd00b06747fbc"
PROTOCOL = "9e8411c52e0e11357547353a21621d956c049439daaafbf4a1b6daf3896753af"
FREEZE = "f95fde21e70b8bf0fc582c8953bedf8825994c45c47032a7debd3a90a25b4a54"
SHIFT = "a87f50f5edfbaa886b950f75ac22ef0cc203733f390ba093e641ad6d0d013378"
LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
SEEDS = (42, 123, 2026)
MODELS = ("RF71", "RF77")
FLAGGED = "simulated_flagged_candidates"
CASE, POLICY = "scratch_5x", "benefit_fixed"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_digest(path, expected):
    check(digest(path) == expected, f"Unexpected SHA-256: {path}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def cloud():
    hosted = (Path("/content").is_dir() and any(k.startswith("COLAB_") for k in os.environ)
              or Path("/kaggle/working").is_dir() and any(k.startswith("KAGGLE_") for k in os.environ))
    check(platform.system() == "Linux" and hosted, "Hosted Colab/Kaggle only; use --self-test for tiny synthetic checks.")


def load_indexer(path):
    if path.suffix == ".ipynb":
        matches = ["".join(c.get("source", "")) for c in read_json(path)["cells"]]
        source, = [s.split("\n", 1)[1] for s in matches if s.startswith("%%writefile shift_audit.py\n")]
    else:
        source = path.read_text(encoding="utf-8")
    check(hashlib.sha256(source.encode()).hexdigest() == INDEXER, "Unrecognized earlier raw-data indexer")
    scope = {"__name__": "original_shift_audit"}
    exec(compile(source, str(path), "exec"), scope)
    return scope


def load_reference(path):
    import pandas as pd
    require_digest(path, REFERENCE)
    ref = pd.read_csv(path, usecols=["source_row", "lot", "height", "width", "label", "role"],
                      dtype={"lot": str}, keep_default_na=False).set_index("source_row", verify_integrity=True)
    check(ref.role.value_counts().to_dict() == {"train": 31457, "tune": 11307, "audit": 11591}, "Split counts")
    return ref


def geometry(ref):
    import numpy as np
    # Keep the original orientation of (height, width), before resizing any map.
    ref = ref.copy()
    n = ref[(ref.role == "train") & (ref.label == "none")].groupby(["height", "width"]).size()
    ref["train_normal_n"] = n.reindex(list(zip(ref.height, ref.width)), fill_value=0).to_numpy()
    ref["coverage"] = np.select([ref.train_normal_n == 0, ref.train_normal_n < 25], ["0", "1-24"], default="25+")
    side = ref[["height", "width"]].max(axis=1)
    ref["size_family"] = np.select([side <= 32, side <= 64], ["<=32", "33-64"], default="65+")
    ref["truth_group"] = np.select([ref.label == "Scratch", ref.label == "none"],
                                   ["Scratch", "normal"], default="other_defect")
    return ref[ref.role == "audit"].copy()


def prior_cost_inputs(args):
    root = args.previous / "cost"
    for name, sha in (("protocol.json", PROTOCOL), ("tune_calibration_freeze.json", FREEZE)):
        require_digest(root / name, sha)
    require_digest(args.anchor_probs, ANCHOR)
    require_digest(args.reference, REFERENCE)
    files = [root / "gate.csv", root / "summary.csv"]
    files += [root / "evidence" / FLAGGED / f"s{s}" / m / "selected_rows.json.gz" for s in SEEDS for m in MODELS]
    return {str(p.relative_to(root)): digest(p) for p in files}


def cost_breakdown(frame, selected, scenario):
    import numpy as np
    frame = frame.copy()
    label_idx = frame.label.map(dict(zip(LABELS, range(9))))
    loss = np.asarray(scenario["missed_loss"])[label_idx]
    effect = np.asarray(scenario["effect"])[label_idx]
    harm = np.asarray(scenario["review_harm"])[label_idx]
    reviewed = frame.index.isin(selected)
    flagged = frame.flagged.to_numpy()
    check(not np.any(reviewed & ~flagged), "Reviewed row outside original common gate")
    frame["rows"] = 1
    frame["gate_flagged"] = flagged.astype(int)
    frame["reviewed"] = reviewed.astype(int)
    defect = frame.label != "none"
    frame["normal_flagged"] = (~defect & flagged).astype(int)
    frame["normal_reviewed"] = (~defect & reviewed).astype(int)
    frame["upstream_missed_defects"] = (defect & ~flagged).astype(int)
    frame["flagged_unreviewed_defects"] = (defect & flagged & ~reviewed).astype(int)
    frame["reviewed_defects"] = (defect & reviewed).astype(int)
    frame["upstream_missed_loss"] = loss * ~flagged
    frame["capacity_missed_loss"] = loss * (flagged & ~reviewed)
    frame["review_residual_loss"] = loss * reviewed * (1.0 - effect)
    frame["residual_loss"] = frame[["upstream_missed_loss", "capacity_missed_loss", "review_residual_loss"]].sum(axis=1)
    frame["review_fees_and_harm"] = reviewed * (scenario["review_fee"] + harm)
    frame["total_cost"] = frame.residual_loss + frame.review_fees_and_harm
    return frame


def run_cost(args, out):
    import numpy as np
    import pandas as pd
    root = args.previous / "cost"
    protocol = read_json(root / "protocol.json")
    cfg = read_json(root / "tune_calibration_freeze.json")
    frame = geometry(load_reference(args.reference))
    anchor = pd.read_csv(args.anchor_probs).set_index("source_row", verify_integrity=True).reindex(frame.index)
    check((anchor.truth == frame.label).all(), "Seed-42 archive and audit identities disagree")
    z = np.log(np.maximum(anchor[LABELS].to_numpy(), cfg["probability_floor_for_log"]))
    z *= cfg["configurations"]["rf_s42"]["inverse_temperature"]
    p = np.exp(z - z.max(axis=1, keepdims=True))
    frame["flagged"] = 1 - p[:, -1] / p.sum(axis=1) >= protocol["gate"]["min_any_defect_probability"]
    gate = pd.read_csv(root / "gate.csv").set_index("label")
    check((frame.groupby("label").flagged.sum().reindex(gate.index) == gate.simulated_flagged).all(), "Frozen gate mismatch")
    old = pd.read_csv(root / "summary.csv")
    tables, shapes, seen = [], [], []
    for seed in SEEDS:
        for model in MODELS:
            path = root / "evidence" / FLAGGED / f"s{seed}" / model / "selected_rows.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as f:
                chosen = list(map(int, json.load(f)[CASE][POLICY]))
            check(len(set(chosen)) == len(chosen) == 728 and set(chosen) <= set(frame.index), "Invalid original slots")
            rows = cost_breakdown(frame, chosen, protocol["scenarios"][CASE])
            expected, = old[(old.model_seed == seed) & (old.model == model) & (old.scenario == CASE) &
                            (old.policy == POLICY) & (old.pool_kind == FLAGGED)].to_dict("records")
            check(np.isclose(rows.total_cost.sum(), expected["total_cost"] + expected["loss_from_excluded_candidates"])
                  and np.isclose(rows.residual_loss.sum(), expected["residual_loss"] + expected["loss_from_excluded_candidates"])
                  and rows.reviewed_defects.sum() == expected["true_defect_reviewed"]
                  and rows.normal_reviewed.sum() == expected["true_normal_reviewed"], "Original cost does not reconcile")
            metrics = list(rows.columns[rows.columns.get_loc("rows"):])
            def aggregate(keys):
                result = rows.groupby(keys)[metrics].sum()
                result["lots"] = rows.groupby(keys).lot.nunique()
                result["model"], result["seed"] = model, seed
                result["cost_per_1000_subgroup"] = 1000 * result.total_cost / result.rows
                result["share_all_residual_loss"] = result.residual_loss / rows.residual_loss.sum()
                return result.reset_index()
            for keys in (["coverage"], ["size_family"], ["truth_group"], ["coverage", "size_family", "truth_group"]):
                tables.append(aggregate(keys).assign(grouping="|".join(keys)))
            shapes.append(aggregate(["height", "width", "train_normal_n", "coverage", "size_family", "truth_group"]))
            seen.append(dict(seed=seed, model=model, arrivals=len(rows), flagged=int(rows.flagged.sum()),
                             reviewed=len(chosen), full_arrival_cost=float(rows.total_cost.sum()),
                             residual_loss=float(rows.residual_loss.sum())))
    pd.concat(tables).to_csv(out / "audit_cost_strata.csv", index=False)
    pd.concat(shapes).to_csv(out / "audit_cost_shapes.csv.gz", index=False)
    write_json(out / "cost_summary.json", dict(
        policy=POLICY, scenario=CASE, interpretation="Relative hypothetical units. Mutually exclusive strata within each grouping; seeds reuse wafers.",
        missing_group="Omitted group cells have zero rows; rates have no denominator.",
        population="All fixed development audit arrivals, including common gate misses.", results=seen))


class LotGroups:
    def __init__(self, lots):
        self.parent = {x: x for x in lots}

    def root(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        a, b = self.root(x), self.root(y)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def table_csv(db, path, query):
    stream = gzip.open if path.suffix == ".gz" else open
    cur = db.execute(query)
    with stream(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c[0] for c in cur.description])
        writer.writerows(cur)


def connected(db, out):
    scalar = lambda sql: db.execute(sql).fetchone()[0]
    # The pinned source has a real lot and digest on every row. Missing either would
    # make full isolation unknowable: fail closed instead of treating null as safe.
    check(not scalar("SELECT count(*) FROM m WHERE lot IS NULL OR lot='' OR digest IS NULL OR length(digest)!=32"),
          "Incomplete graph keys on the source; full isolation cannot be established")
    db.execute("CREATE INDEX digest_lot ON m(digest,lot)")
    groups = LotGroups(lot for lot, in db.execute("SELECT DISTINCT lot FROM m"))
    previous, first = None, None
    for h, lot in db.execute("SELECT DISTINCT digest,lot FROM m ORDER BY digest,lot"):
        if h != previous:
            previous, first = h, lot
        else:
            groups.union(first, lot)
    db.execute("CREATE TABLE link(lot TEXT PRIMARY KEY, component TEXT NOT NULL)")
    db.executemany("INSERT INTO link VALUES(?,?)", ((lot, groups.root(lot)) for lot in groups.parent))
    db.execute("""CREATE TABLE component AS SELECT component,count(*) AS rows,count(DISTINCT lot) AS lots,
               sum(g='development') AS labeled_dev,sum(g='test') AS labeled_test,sum(g='unlabeled') AS unlabeled,
               sum(coalesce(role='train',0)) AS train,sum(coalesce(role='tune',0)) AS tune,
               sum(coalesce(role='audit',0)) AS audit,max(g!='unlabeled') AS protected
               FROM m JOIN link USING(lot) LEFT JOIN dev USING(id) GROUP BY component""")
    db.execute("CREATE UNIQUE INDEX component_id ON component(component)")
    db.execute("""CREATE TABLE eligible AS SELECT m.id,m.digest FROM m JOIN link USING(lot)
               JOIN component USING(component) WHERE protected=0 AND g='unlabeled' AND good=1 AND src IN ('','train')""")
    db.execute("CREATE INDEX elig_hash ON eligible(digest)")
    db.execute("CREATE TABLE keep AS SELECT min(id) AS id,count(*) AS copies FROM eligible GROUP BY digest")
    n = scalar("SELECT count(*) FROM keep")
    query = """SELECT m.id AS source_row,lot,component,h AS height,w AS width,lower(hex(digest)) AS map_sha256,
               copies AS eligible_rows_with_identical_map FROM keep JOIN m USING(id) JOIN link USING(lot)"""
    table_csv(db, out / "ssl_pool.csv.gz", query + " ORDER BY m.id")
    table_csv(db, out / "components.csv.gz", "SELECT * FROM component ORDER BY component")
    table_csv(db, out / "lot_components.csv.gz", "SELECT * FROM link ORDER BY lot")
    table_csv(db, out / "eligible_shapes.csv", """SELECT h AS height,w AS width,count(*) AS unique_maps,
              count(DISTINCT component) AS components FROM keep JOIN m USING(id) JOIN link USING(lot)
              GROUP BY h,w ORDER BY h,w""")
    check(not scalar("SELECT count(*) FROM keep JOIN m USING(id) JOIN link USING(lot) JOIN component USING(component) WHERE protected"),
          "Protected component in manifest")
    return dict(unique_candidate_maps=n, eligible_before_dedup=scalar("SELECT count(*) FROM eligible"),
                components=scalar("SELECT count(*) FROM component"), protected_components=scalar("SELECT sum(protected) FROM component"),
                labeled_component_unlabeled_rows=scalar("SELECT sum(unlabeled) FROM component WHERE protected"),
                all_rows=scalar("SELECT count(*) FROM m"), candidate_components=scalar(
                    "SELECT count(DISTINCT component) FROM keep JOIN m USING(id) JOIN link USING(lot)"))


def run_pool(args, out):
    old = read_json(args.previous / "shift" / "shift_summary.json")
    scan = load_indexer(args.indexer)
    _, available = scan["require_cloud"]()
    check(shutil.disk_usage(tempfile.gettempdir()).free > 2 * 2**30, "Need >2 GiB free local cloud scratch")
    ref = load_reference(args.reference)
    with tempfile.TemporaryDirectory() as tmp, sqlite3.connect(Path(tmp) / "links.sqlite") as db:
        db.execute("PRAGMA cache_size=-49152")
        db.execute("PRAGMA temp_store=FILE")
        # Original pinned function verifies SHA-256 before it unpickles, on cloud only.
        checks = scan["index_maps"](args.raw, db)
        check(checks == old["quality_flags"] and dict(db.execute("SELECT g,count(*) FROM m GROUP BY g")) == old["counts"],
              "Prior whole-source counts or quality checks not reproducible")
        db.execute("CREATE TABLE dev(id INTEGER PRIMARY KEY, role TEXT, orig_lot TEXT, orig_label TEXT, h0 INT, w0 INT)")
        db.executemany("INSERT INTO dev VALUES(?,?,?,?,?,?)",
                       ((r.Index, r.role, r.lot, r.label.lower(), r.height, r.width) for r in ref.itertuples()))
        check(db.execute("""SELECT count(*) FROM dev JOIN m USING(id) WHERE g='development' AND lot=orig_lot
                         AND label=orig_label AND h=h0 AND w=w0""").fetchone()[0] == len(ref), "Development identity mismatch")
        result = connected(db, out)
    direct = old["conservative_pretraining_pool"]
    result.update(direct_filter_unique_maps=direct["unique_map_rows"],
                  unique_maps_removed_by_multihop=direct["unique_map_rows"] - result["unique_candidate_maps"],
                  raw_sha256=RAW, indexer_sha256=INDEXER, initial_free_gib=available,
                  map_hash="SHA256(int64 little-endian [height,width] followed by C-order uint8 pixels)",
                  component_id="Lexicographically smallest original lot in connected component",
                  admission="Only unlabeled rows with valid maps, raw split empty/train, and no labeled row anywhere in their lot/pixel connected component. Retain lowest raw source_row per identical map.",
                  labels_assigned=False)
    check(result["unique_maps_removed_by_multihop"] >= 0, "Multihop pool exceeds the direct-only pool")
    write_json(out / "pool_summary.json", result)


def run_decision(args, out):
    import pandas as pd
    audit = pd.read_csv(args.out / "cost" / "audit_cost_shapes.csv.gz")
    shapes = pd.read_csv(args.out / "pool" / "eligible_shapes.csv")
    joined = audit.merge(shapes, how="left", on=["height", "width"], validate="many_to_one")
    joined["pool_support"] = joined.unique_maps.fillna(0).gt(0)
    rows = joined.groupby(["model", "seed", "coverage", "pool_support"], as_index=False)[
        ["rows", "upstream_missed_defects", "upstream_missed_loss", "capacity_missed_loss", "residual_loss",
         "normal_flagged", "normal_reviewed"]].sum()
    rows.to_csv(out / "pool_support_and_cost.csv", index=False)
    write_json(out / "decision.json", dict(
        status="defer_self_supervised_training", trained=False, pseudo_labels=False,
        evidence="Read pool_support_and_cost.csv for actual development-audit loss and counts at dimensions available in the isolated unlabeled pool. The same rows recur across seed/model combinations.",
        rationale="Shape availability and residual loss can locate a possible pilot cohort. They cannot identify unlabeled defect morphology or demonstrate learnable benefit beyond the supervised alternative. Capacity losses may persist even with perfect rankings.",
        pilot_prerequisites=["Reserve a new representative lot-independent evaluation cohort with labeled Scratch and normals at target dimensions.",
                             "On identical train/validation lots and budget, compare random-init CNN with self-supervised initialization; freeze the same review gate, capacity, Scratch recall and total normal false-positive constraints.",
                             "Measure paired loss and uncertainty by lot; also report queue allocation and upstream misses before claiming incremental benefit."],
        test_use="No final-test predictions or tuning. All labeled components, including original test, are excluded from pretraining."))


def stage_receipt(folder):
    receipt = read_json(folder / "receipt.json")
    for name, expected in receipt["outputs"].items():
        require_digest(folder / name, expected)
    return receipt


def run_stage(args, name):
    folder = args.out / name
    if name == "cost":
        check(args.previous and args.reference and args.anchor_probs, "cost needs --previous, --reference, --anchor-probs")
        inputs = dict(reference=REFERENCE, anchor=ANCHOR, protocol=PROTOCOL, freeze=FREEZE, evidence=prior_cost_inputs(args))
    elif name == "pool":
        check(args.previous and args.reference, "pool needs --previous and --reference")
        require_digest(args.previous / "shift" / "shift_summary.json", SHIFT)
        require_digest(args.reference, REFERENCE)
        inputs = dict(reference=REFERENCE, raw=RAW, shift=SHIFT, indexer=INDEXER)
    else:
        inputs = {k: stage_receipt(args.out / k) for k in ("cost", "pool")}
    lineage = dict(script=digest(Path(__file__)), inputs=inputs)
    if (folder / "receipt.json").is_file():
        check(stage_receipt(folder)["lineage"] == lineage, f"{name}: changed inputs/code; choose a fresh output root")
        print(f"{name}: verified existing result", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    if name == "pool":
        check(args.raw and args.indexer, "pool needs --raw and --indexer for first run")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp)
        {"pool": run_pool, "cost": run_cost, "decision": run_decision}[name](args, local)
        outputs = {}
        for p in sorted(local.iterdir()):
            dest = folder / p.name
            partial = dest.with_suffix(dest.suffix + ".partial")
            shutil.copyfile(p, partial)
            outputs[p.name] = digest(p)
            require_digest(partial, outputs[p.name])
            partial.replace(dest)
        write_json(folder / "receipt.json", dict(stage=name, lineage=lineage, outputs=outputs))
    print(f"{name}: saved and verified {folder}", flush=True)


def self_test():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp, sqlite3.connect(":memory:") as db:
        out = Path(tmp)
        db.execute("CREATE TABLE m(id INTEGER PRIMARY KEY,g TEXT,lot TEXT,digest BLOB,h INT,w INT,src TEXT,good INT)")
        db.execute("CREATE TABLE dev(id INTEGER PRIMARY KEY,role TEXT)")
        db.execute("INSERT INTO dev VALUES(0,'audit')")
        for i,(lot, image, group) in enumerate([("L","a","development"),("U","a","unlabeled"),("U","b","unlabeled"),
                ("V","b","unlabeled"),("V","c","unlabeled"),("W","d","unlabeled"),("X","d","unlabeled"),
                ("Y","e","test"),("Z","e","unlabeled"),("Z","f","unlabeled")]):
            db.execute("INSERT INTO m VALUES(?,?,?,?,1,2,'',1)", (i,group,lot,hashlib.sha256(image.encode()).digest()))
        result = connected(db, out)
        check(result["eligible_before_dedup"] == 2 and result["unique_candidate_maps"] == 1, "Synthetic graph")
        with gzip.open(out / "ssl_pool.csv.gz", "rt") as f:
            wafer, = csv.DictReader(f)
        check(wafer["source_row"] == "5" and wafer["component"] == "W" and wafer["eligible_rows_with_identical_map"] == "2",
              "Synthetic deterministic retention")
    frame = pd.DataFrame(dict(label=["Scratch", "Scratch", "Center", "none"], flagged=[False,True,True,True]))
    r = cost_breakdown(frame, [1,3], dict(missed_loss=[1]*7+[5,0], effect=[.9]*8+[0], review_fee=.1, review_harm=[0]*9))
    check(abs(r.total_cost.sum()-6.7) < 1e-10 and r.normal_reviewed.sum()==1 and r.upstream_missed_loss.sum()==5,
          "Synthetic loss accounting")
    print("Synthetic transitive protection, deduplication, audit losses: passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("previous", "reference", "anchor-probs", "raw", "indexer", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--stage", choices=("cost", "pool", "decision", "all"), default="all")
    parser.add_argument("--minutes", type=int, default=25, choices=range(1,31), metavar="1..30")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    cloud()
    check(args.out, "Specify a persistent --out directory")
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args.out.mkdir(parents=True, exist_ok=True)
    def expire(*_):
        raise TimeoutError("Hosted CPU wall-time cap reached; resume remaining stages separately.")
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(60 * args.minutes)
    for name in (("cost", "pool", "decision") if args.stage == "all" else (args.stage,)):
        run_stage(args, name)


if __name__ == "__main__":
    main()
