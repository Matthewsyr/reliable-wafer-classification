"""Export private wafer-level RF77/RF71 membership changes from completed Colab evidence."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile


POOL = "simulated_flagged_candidates"
LABELS = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none")
SEEDS = (42, 123, 2026)
CASES = ("equal", "scratch_5x", "near_full_5x")
POLICIES = ("benefit_fixed", "benefit_positive_cap")
SOURCE = {
    "06_seed42_wafer_replay.sql": "4be9525569be1a291afd6ac2d2d79f159843392f778524fc6c0c38c9246ea783",
    "cost/protocol.json": "9e8411c52e0e11357547353a21621d956c049439daaafbf4a1b6daf3896753af",
    "cost/summary.csv": "d21bbb9e3b42bd5b258ed2cb82e0355324cb0168eb5a36c7d534388834c169ee",
    "cost/paired_models.csv": "737f63921faf76ba902be248b8178f7cb5236ba6c43c48fd0692e4a0bfaaebad",
    "cost/slot_swaps.csv": "86eb4604648b3a8fa0d2e9cc62a52d1839f0fa252355d65e10545667e68edaec",
}
LIST_HASHES = {
    (42, "RF71"): "28a46f8ae478802d0ae650da712262daa9c6ed6313250241e46a6f3bd6aec5f5",
    (42, "RF77"): "46d9e16684e0a2ca641b328694cd067feb9e1647c96bdd1d0b872324085a35f1",
    (123, "RF71"): "db51326245a3f157c9febbf4b11e6a22672b0933ad2a5024ab00e85eb3d3daa5",
    (123, "RF77"): "7a0c96ba4f67a4fd98731292707e0f8a0768a25717f5eb1a099e30b8b03a4713",
    (2026, "RF71"): "ab965adaa65e41ebec9b0b876a846049d5c2b460d0dd02ca0338b42b7dbb6618",
    (2026, "RF77"): "f4a17f9636582ceeb017a83f758e2074b6ec1fa0e1fbad7af25e980ec4bf3c1a",
}


def check(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def checked(path, expected):
    data = path.read_bytes()
    check(hashlib.sha256(data).hexdigest() == expected, f"Source checksum mismatch: {path.name}")
    return data


def parse_bridge(text):
    # Parse only the VALUES block of the checksum-pinned, privately archived SQL. Never execute it.
    block = text.split("  FROM VALUES\n", 1)[1].split("\n), scores AS (", 1)[0]
    lookup = {}
    for row in block.splitlines():
        record = row.removesuffix(",")
        check(record.startswith("(") and record.endswith(")"), "Invalid frozen VALUES row")
        fields = next(csv.reader([record[1:-1]], quotechar="'", escapechar="\\", strict=True))
        sid, lot, queue, cap, _, label = fields[:6]
        check(len(fields) == 14 and sid not in lookup and label in LABELS, "Duplicate ID or malformed frozen lookup")
        lookup[sid] = (lot, int(queue), int(cap), label, int(fields[9]), int(fields[13]))
    return lookup


def keyed(data, fields, **filters):
    parsed = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    rows = [r for r in parsed if all(r[k] == v for k, v in filters.items())]
    result = {tuple(r[k] for k in fields): r for r in rows}
    check(len(result) == len(rows), "Nonunique archived summary")
    return result


def run(previous, out):
    data = {name: checked(previous / name, sha) for name, sha in SOURCE.items()}
    lookup = parse_bridge(data["06_seed42_wafer_replay.sql"].decode("utf-8"))
    queues = Counter(v[1] for v in lookup.values())
    check(queues == {0:1000, 1:1000, 2:1000, 3:642} and len(set(v[0] for v in lookup.values())) == 903,
          "Size or lot mismatch in frozen common pool")
    capacities = {q: n // 5 for q, n in queues.items()}
    check(all(v[2] == capacities[v[1]] for v in lookup.values()), "Frozen capacity mismatch")
    costs = json.loads(data["cost/protocol.json"])["scenarios"]
    keys = ("model_seed", "scenario", "policy")
    summary = keyed(data["cost/summary.csv"], (*keys, "model"), pool_kind=POOL)
    pairs = keyed(data["cost/paired_models.csv"], keys, pool_kind=POOL, new_model="RF77", base_model="RF71")
    classes = keyed(data["cost/slot_swaps.csv"], (*keys, "label"), pool_kind=POOL, new_model="RF77", baseline_model="RF71")
    lists = {}
    for (seed, model), sha in LIST_HASHES.items():
        f = previous / "cost" / "evidence" / POOL / f"s{seed}" / model / "selected_rows.json.gz"
        lists[seed, model] = json.loads(gzip.decompress(checked(f, sha)))
    private, panels = [], []
    for seed in SEEDS:
        for case in CASES:
            spec = costs[case]
            missed = dict(zip(LABELS, spec["missed_loss"]))
            reviewed = {k: missed[k]*(1-spec["effect"][i])+spec["review_fee"]+spec["review_harm"][i]
                        for i,k in enumerate(LABELS)}
            for policy in POLICIES:
                key = (str(seed), case, policy)
                masks, model_totals = [], []
                for model, flag_idx in (("RF71", 4), ("RF77", 5)):
                    frozen = lists[seed, model][case][policy]
                    selected = set(frozen)
                    check(len(selected) == len(frozen) and selected.issubset(lookup), "Invalid frozen member")
                    if seed == 42 and case == "scratch_5x" and policy == "benefit_fixed":
                        check(selected == {s for s,r in lookup.items() if r[flag_idx]}, "Seed-42 SQL/list disagree")
                    spent = Counter(lookup[i][1] for i in selected)
                    check(all(n <= capacities[q] for q,n in spent.items()), "Queue overflow")
                    if policy == "benefit_fixed":
                        check(spent == capacities, "Fixed policy did not fill every queue")
                    ref = summary[key + (model,)]
                    total = math.fsum(reviewed[r[3]] if sid in selected else missed[r[3]] for sid,r in lookup.items())
                    check(len(selected) == int(ref["reviewed"]) and math.isclose(total,float(ref["total_cost"]),abs_tol=1e-7,rel_tol=0),
                          "Frozen selection disagrees with original reviewed count or label-settled cost")
                    masks.append(selected)
                    model_totals.append(total)
                a, b = masks
                entered, displaced = b-a, a-b
                pair = pairs[key]
                savings = model_totals[0] - model_totals[1]
                scaled = savings*1000/len(lookup)
                check(len(entered) == int(pair["entered"]) and len(displaced) == int(pair["displaced"])
                      and math.isclose(scaled,float(pair["paired_savings_per_1000"]),abs_tol=1e-7,rel_tol=0), "Paired summary mismatch")
                tally = Counter()
                cohort = dict(model_seed=seed,scenario=case,policy=policy)
                for sid in sorted(entered|displaced, key=lambda i: (lookup[i][1], int(i))):
                    lot, queue, cap, truth, _, _ = lookup[sid]
                    change = "entered_RF77" if sid in entered else "displaced_by_RF77"
                    v71 = reviewed[truth] if sid in a else missed[truth]
                    v77 = reviewed[truth] if sid in b else missed[truth]
                    tally[change,truth] += 1
                    private.append(dict(**cohort, queue=queue, capacity=cap, source_row=sid, lot=lot, truth=truth,
                                        direction=change, picked_rf71=int(sid in a), picked_rf77=int(sid in b),
                                        cost_rf71_units=v71, cost_rf77_units=v77, savings_units=v71-v77))
                for lab in LABELS:
                    row = classes[key + (lab,)]
                    en, dis = tally["entered_RF77",lab], tally["displaced_by_RF77",lab]
                    check((en,dis) == (int(row["entered"]),int(row["displaced"])) and
                          math.isclose((en-dis)*(missed[lab]-reviewed[lab]),float(row["net_savings_units"]),
                                       abs_tol=1e-7,rel_tol=0), "Per-class swap count or signed loss mismatch")
                panels.append(dict(**cohort, candidates=len(lookup), rf71_reviewed=len(a), rf77_reviewed=len(b),
                                   entered=len(entered), displaced=len(displaced), scratch_net=tally["entered_RF77","Scratch"]-tally["displaced_by_RF77","Scratch"],
                                   normal_net=tally["entered_RF77","none"]-tally["displaced_by_RF77","none"],
                                   savings_units=round(savings,7), savings_per_1000_candidates=round(scaled,7)))
    check(len(panels) == 18, "Incomplete comparison")
    out.parent.mkdir(parents=True, exist_ok=True)
    check(not out.exists(), "Choose a fresh output directory")
    # Publish only after every aggregate agrees. A partial output is never marked complete.
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp)/"complete"
        stage.mkdir()
        for name, rows in (("PRIVATE_dev_swaps.csv.gz", private),("dev_swap_panels.csv",panels)):
            path = stage / name
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path,"wt",encoding="utf-8",newline="") as f:
                writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}
        receipt = dict(status="complete",panels=18,private_rows=len(private),verified_class_panels=18*9,
                       archived_inputs=SOURCE,sha256=files,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       interpretation="Positive savings are relative simulated cost RF71 minus RF77. No artificial one-to-one pairing of wafers. Each side is identified within the same queue.")
        (stage/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        stage.rename(out)
    print(f"Verified 18 policy panels and 162 class panels. Private changed-membership rows: {len(private)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--previous",type=Path,required=True,help="Original Drive run root (contains cost/ and 06_seed42_wafer_replay.sql)")
    ap.add_argument("--out",type=Path,required=True,help="Fresh directory in private Drive")
    args = ap.parse_args()
    check(Path("/content").is_dir() and any(k.startswith("COLAB_") for k in os.environ), "Use hosted Colab for private wafer rows")
    run(args.previous,args.out)
