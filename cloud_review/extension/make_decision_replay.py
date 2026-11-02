"""Generate one Snowflake SELECT to audit the new hypothetical USD decision experiment.

python make_decision_replay.py --run DECISION_OR_AGGREGATE_DIR --sql replay.sql
On the hosted run, optionally pass --export-batches batches.csv to reduce the two
private membership files to auditable per-queue counts. Later pass --batches
batches.csv with the published aggregate directory for a full replay.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sqlite3
from collections import Counter, defaultdict

AUDIT = "audit_exploratory"
POLICIES = ("argmax", "scratch_joint", "f1", "cap_threshold", "net_value_zero", "min_cost",
            "queue_random", "queue_probability", "queue_expected_value")
QUEUES = POLICIES[-3:]
SCENARIOS = ("uniform_1000", "scratch_3000")
CLASSES = {"Center","Donut","Edge-Loc","Edge-Ring","Loc","Near-full","Random","Scratch","none"}
KEY = ("model", "seed", "scenario", "policy")
BATCH_FIELDS = (*KEY, "queue", "reviewed")


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_inputs(root):
    if (root / "run_status.json").is_file():
        p = root / "run_status.json"
        status = read_json(p)
        require(status["status"] == "complete", "Decision run incomplete")
        files, suffix, lineage = status["sha256"], ".csv", sha(p)
    else:
        status = read_json(root / "manifest.json")
        files, suffix, lineage = status["files"], ".csv.gz", status["source_receipt_sha256"]
    def checked(name):
        p = root / name
        require(sha(p) == files[name], f"Input SHA mismatch: {name}")
        return p
    freeze = read_json(checked("frozen_validation_policy.json"))
    expected = {(p["model"], int(p["seed"])) for p in freeze["params"]}
    require(expected and len(expected) == len(freeze["params"]), "Unexpected frozen model-seed keys")
    tables = {}
    for name in ("routing", "inspection_by_class", "queue_inventory"):
        p = checked(name + suffix)
        stream = gzip.open if suffix.endswith("gz") else open
        with stream(p, "rt", encoding="utf-8-sig", newline="") as f:
            tables[name] = [row for row in csv.DictReader(f) if row["phase"] == AUDIT]
    return tables, expected, lineage, checked


def key(r):
    return r["model"], int(r["seed"]), r["scenario"], r["policy"]


def compact(tables, fits):
    expected = {(*fit, s, p) for fit, s, p in itertools.product(fits, SCENARIOS, POLICIES)}
    inspection = defaultdict(dict)
    for row in tables["inspection_by_class"]:
        subgroup = inspection[key(row)]
        require(row["label"] not in subgroup, "Duplicate class evidence")
        subgroup[row["label"]] = row
    values, seen = [], set()
    for r in tables["routing"]:
        k = key(r)
        require(k not in seen, "Duplicate routing row")
        seen.add(k)
        c = inspection[k]
        require(set(c) == CLASSES, "Incomplete class evidence")
        for v in c.values():
            require(int(v["not_reviewed"]) + int(v["reviewed"]) == int(v["total"])
                    and min(int(v[n]) for n in ("not_reviewed", "reviewed", "total", "candidates")) >= 0,
                    "Bad class panel")
        misses = sum(int(v["not_reviewed"]) for name, v in c.items() if name not in ("none","Scratch"))
        values.append((*k, int(r["arrivals"]), int(r["candidates"]), int(r["reviewed"]), int(r["fn"]), int(r["fp"]),
                       int(r["cost_dollars"]), misses, int(c["Scratch"]["not_reviewed"]), int(c["none"]["reviewed"]),
                       sum(int(v["reviewed"]) for v in c.values()), sum(int(v["total"]) for v in c.values())))
    require(seen == expected == set(inspection), "Missing/extra decision panels versus frozen fits")
    inventory = sorted((int(r["queue"]), int(r["candidates"]), int(r["slots"])) for r in tables["queue_inventory"])
    require(inventory and [q for q, _, _ in inventory] == list(range(len(inventory))), "Duplicate/missing queue IDs")
    return sorted(values), inventory


def export_batches(output, checked, lineage, fits, inventory):
    # Only the resulting aggregate counts leave the hosted runtime.
    hosted = (Path("/content").is_dir() and any(k.startswith("COLAB_") for k in os.environ)
              or Path("/kaggle/working").is_dir() and any(k.startswith("KAGGLE_") for k in os.environ)
              or os.environ.get("GITHUB_ACTIONS") == "true")
    require(platform.system() == "Linux" and hosted, "Build private-to-aggregate batch evidence only on the hosted run")
    path = checked("risk_scores.csv.gz")
    membership = checked("review_membership.csv.gz")
    queues = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            require(r["phase"] == AUDIT, "Unexpected risk score phase")
            if int(r["historical_gate"]):
                sid, qid = int(r["source_row"]), int(r["queue"])
                require(queues.setdefault(sid, qid) == qid, "Queue differs across frozen seeds")
    require(Counter(queues.values()) == {q:n for q,n,_ in inventory}, "Risk membership and queue inventory differ")
    totals, seen = Counter(), set()
    with gzip.open(membership, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fit = r["model"], int(r["seed"])
            sc, sid = r["scenario"], int(r["source_row"])
            require(r["phase"] == AUDIT and fit in fits and sc in SCENARIOS and sid in queues, "Unexpected membership")
            rowkey = (*fit, sc, sid)
            require(rowkey not in seen, "Duplicate review membership")
            seen.add(rowkey)
            for p in QUEUES:
                v = int(r[p.removeprefix("queue_")])
                require(v in (0, 1), "Invalid review indicator")
                totals[(*fit, sc, p, queues[sid])] += v
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BATCH_FIELDS)
        for fit, sc, p, (q,_,_) in itertools.product(sorted(fits), SCENARIOS, QUEUES, inventory):
            k = (*fit, sc, p, q)
            writer.writerow((*k, totals[k]))
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(dict(
        source_receipt_sha256=lineage, batch_aggregate_sha256=sha(output)), indent=2), encoding="utf-8")


def read_batches(path, lineage, fits, inventory):
    meta = read_json(path.with_suffix(path.suffix + ".json"))
    require(meta["source_receipt_sha256"] == lineage and meta["batch_aggregate_sha256"] == sha(path), "Wrong batch receipt")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [(*key(r), int(r["queue"]), int(r["reviewed"])) for r in csv.DictReader(f)]
    expected = {(*fit,s,p,q) for fit,s,p,(q,_,_) in itertools.product(fits,SCENARIOS,QUEUES,inventory)}
    require(len(expected) == len(rows) and {r[:-1] for r in rows} == expected, "Incomplete per-queue observed review counts")
    return sorted(rows)


def quote(val):
    return str(val) if isinstance(val, int) else "'" + val.replace("'", "''") + "'"


def cte(name, fields, rows):
    body = ",\n".join("(" + ",".join(map(quote, row)) + ")" for row in rows)
    return f"{name}({fields}) AS (SELECT * FROM (VALUES\n{body}\n))"


def statement(data, inv, batches, lineage):
    inputs = cte("metric", "model,seed,scenario,policy,n,candidates,reviewed,fn,fp,reported_cost,other_fn,scratch_fn,normal_fp,class_reviewed,class_n", data)
    inventory = cte("inventory", "qid,candidates,slots", inv)
    if batches:
        observed = cte("observed", "model,seed,scenario,policy,qid,nreviewed", batches)
    else:
        observed = "observed(model,seed,scenario,policy,qid,nreviewed) AS (SELECT '',0,'','',0,0 WHERE 1=0)"
    return f"""-- WM-811K hypothetical USD decision replay. Source receipt SHA-256: {lineage}
-- All inputs are aggregates. TOTAL_ONLY means actual per-batch usage was not supplied.
WITH {inputs}, {inventory}, {observed},
budget AS (SELECT SUM(candidates) AS flagged, SUM(FLOOR(candidates/5.0)) AS cap,
    COUNT(*) AS nq, SUM(CASE WHEN candidates<1 OR candidates>1000 OR slots<>FLOOR(candidates/5.0)
                             THEN 1 ELSE 0 END) AS inventory_errors FROM inventory),
by_batch AS (SELECT model,seed,scenario,policy,COUNT(*) AS batches_checked,SUM(nreviewed) AS batch_reviews,
    SUM(CASE WHEN nreviewed<0 OR nreviewed>FLOOR(i.candidates/5.0) THEN 1 ELSE 0 END) AS batch_overages
    FROM observed o JOIN inventory i ON o.qid=i.qid GROUP BY model,seed,scenario,policy),
checked AS (SELECT m.*, 1000*other_fn + CASE WHEN m.scenario='scratch_3000' THEN 3000 ELSE 1000 END*scratch_fn
    + 5*normal_fp AS recomputed_cost, b.*, x.batches_checked, x.batch_reviews, x.batch_overages,
    CASE WHEN m.policy IN ('queue_expected_value','queue_probability','queue_random') THEN 1 ELSE 0 END AS queued
    FROM metric m CROSS JOIN budget b LEFT JOIN by_batch x
    ON m.model=x.model AND m.seed=x.seed AND m.scenario=x.scenario AND m.policy=x.policy)
SELECT scenario, policy, COUNT(*) AS panels_checked,
    SUM(CASE WHEN reported_cost<>recomputed_cost THEN 1 ELSE 0 END) AS cost_mismatches,
    SUM(CASE WHEN fn<>other_fn+scratch_fn OR fp<>normal_fp OR reviewed<>class_reviewed OR n<>class_n
        THEN 1 ELSE 0 END) AS count_mismatches,
    SUM(CASE WHEN queued=1 AND (candidates<>flagged OR reviewed>cap OR inventory_errors<>0)
        THEN 1 ELSE 0 END) AS total_quota_mismatches,
    SUM(COALESCE(batches_checked,0)) AS observed_batches_checked,
    SUM(COALESCE(batch_overages,0)) AS observed_batch_overages,
    SUM(CASE WHEN batches_checked IS NOT NULL AND (batches_checked<>nq OR batch_reviews<>reviewed)
        THEN 1 ELSE 0 END) AS batch_reconciliation_mismatches,
    CASE WHEN MAX(queued)=0 THEN 'NO_HARD_QUEUE'
         WHEN COUNT(batches_checked)=COUNT(*) THEN 'PER_BATCH_AND_TOTAL' ELSE 'TOTAL_ONLY' END AS quota_check_scope
FROM checked GROUP BY scenario, policy ORDER BY scenario, policy;
"""


def self_test():
    # Execute the generated SQL, including deliberate cost, count and quota violations.
    data = [("synthetic",42,"uniform_1000","queue_random",11,11,3,3,1,3005,2,1,1,3,11),
            ("synthetic",42,"scratch_3000","queue_probability",11,11,2,2,0,5001,2,1,0,2,11)]
    inv = [(0,6,1),(1,5,1)]
    obs = [(*data[0][:4],0,2),(*data[0][:4],1,1),(*data[1][:4],0,1),(*data[1][:4],1,1)]
    with sqlite3.connect(":memory:") as db:
        x = db.execute(statement(data,inv,obs,"synthetic")).fetchall()
        require(x[0][3:9] == (1,1,0,2,0,0) and x[1][3:9] == (0,0,1,2,1,0), "Violation checks")
        x = db.execute(statement(data,inv,[],"synthetic")).fetchall()
        require(x[0][-1] == "TOTAL_ONLY" and x[0][6] == 0, "Aggregate-only scope")
    print("Synthetic SQL execution detected seeded arithmetic, counts, batch and total quota mismatches.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--sql", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--export-batches", type=Path)
    group.add_argument("--batches", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.run and args.sql, "Specify --run and --sql")
    tables, fits, lineage, checked = get_inputs(args.run)
    data, inv = compact(tables, fits)
    if args.export_batches:
        export_batches(args.export_batches, checked, lineage, fits, inv)
    batch_file = args.export_batches or args.batches
    batches = read_batches(batch_file, lineage, fits, inv) if batch_file else []
    sql = statement(data, inv, batches, lineage)
    args.sql.write_text(sql, encoding="utf-8")
    print(f"Prepared {len(data)} audit panels, {len(batches)} observed queue counts for Snowflake.")


if __name__ == "__main__":
    main()
