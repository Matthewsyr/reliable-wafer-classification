"""Make a compact, denominator-checked readout from four aggregate tables of the historical hosted run."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

HASHES = {
    "protocol.json": "9e8411c52e0e11357547353a21621d956c049439daaafbf4a1b6daf3896753af",
    "summary.csv": "d21bbb9e3b42bd5b258ed2cb82e0355324cb0168eb5a36c7d534388834c169ee",
    "paired_models.csv": "737f63921faf76ba902be248b8178f7cb5236ba6c43c48fd0692e4a0bfaaebad",
    "slot_swaps.csv": "86eb4604648b3a8fa0d2e9cc62a52d1839f0fa252355d65e10545667e68edaec",
}
MODELS = ("RF71", "RF77", "CNN")
SEEDS = ("42", "123", "2026")
SCENARIOS = {"equal":"Equal losses", "scratch_5x":"Scratch 5x", "near_full_5x":"Near-full 5x"}
POLICIES = ("random_fixed", "any_defect_fixed", "benefit_fixed", "benefit_positive_cap")
POOL = "simulated_flagged_candidates"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def read_aggregates(source):
    result = {}
    for name, expected in HASHES.items():
        raw = (source/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==expected, f"Unexpected archive member {name}")
        result[name] = json.loads(raw) if name.endswith(".json") else list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    return result


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def triple(values):
    display = [f"{float(v):.3f}" for v in values]
    return display[0] if len(set(display)) == 1 else " / ".join(display)


def generate(source, destination):
    data = read_aggregates(source)
    info = data["protocol.json"]
    arrivals, candidates = info["gate"]["rows_total"], info["gate"]["rows_selected"]
    require((arrivals, candidates) == (11591,3642) and info["model_seeds"] == [42,123,2026], "Unexpected cohort")
    cols = ("model_seed","model","scenario","policy","rows","lots","reviewed","capacity","true_normal_reviewed",
            "true_defect_reviewed","true_defect_not_reviewed","total_cost","cost_per_1000",
            "loss_from_excluded_candidates","cost_per_1000_simulated_arrivals")
    matrix = [{k:r[k] for k in cols} for r in data["summary.csv"]
              if r["pool_kind"]==POOL and r["policy"] in POLICIES]
    ref = {(r["model"],r["scenario"],r["model_seed"],r["policy"]):r for r in matrix}
    require(len(matrix)==len(ref)==108, "Incomplete coverage")
    for r in matrix:
        cost = float(r["total_cost"])
        require(int(r["rows"])==candidates and int(r["lots"])==903 and int(r["reviewed"])==int(r["capacity"])==728,
                "Unexpected common pool or utilization")
        require(math.isclose(cost*1000/candidates,float(r["cost_per_1000"]),abs_tol=1e-8) and
                math.isclose((cost+float(r["loss_from_excluded_candidates"]))*1000/arrivals,
                             float(r["cost_per_1000_simulated_arrivals"]),abs_tol=1e-8), "Denominator mismatch")
    destination.mkdir(parents=True,exist_ok=True)
    write_rows(destination/"relative_policy_matrix.csv",matrix)
    swaps = {}
    for r in data["slot_swaps.csv"]:
        if r["pool_kind"]==POOL and r["new_model"]=="RF77" and r["baseline_model"]=="RF71" and r["policy"] in POLICIES[2:]:
            swaps.setdefault((r["model_seed"],r["scenario"],r["policy"]),[]).append(r)
    pairs = []
    for r in data["paired_models.csv"]:
        if (r["pool_kind"],r["new_model"],r["base_model"])!=(POOL,"RF77","RF71") or r["policy"] not in POLICIES[2:]:
            continue
        seed, case, pol = r["model_seed"], r["scenario"], r["policy"]
        labels = swaps[seed,case,pol]
        by_label = {x["label"]:x for x in labels}
        require(len(by_label)==len(labels)==9 and sum(int(x["entered"]) for x in labels)==int(r["entered"])
                and sum(int(x["displaced"]) for x in labels)==int(r["displaced"]), "Unreconciled membership changes")
        a,b = ref["RF71",case,seed,pol], ref["RF77",case,seed,pol]
        savings = float(r["paired_savings_per_1000"])
        arrival_savings = float(a["cost_per_1000_simulated_arrivals"])-float(b["cost_per_1000_simulated_arrivals"])
        require(abs(float(a["cost_per_1000"])-float(b["cost_per_1000"])-savings)<1e-8 and
                abs(savings*candidates/arrivals-arrival_savings)<1e-8, "Unreconciled paired cost")
        pairs.append(dict(model_seed=seed,scenario=case,policy=pol,entered=r["entered"],displaced=r["displaced"],
                          scratch_entered=by_label["Scratch"]["entered"],scratch_displaced=by_label["Scratch"]["displaced"],
                          normal_entered=by_label["none"]["entered"],normal_displaced=by_label["none"]["displaced"],
                          savings_per_1000_candidates=savings,ci_low=r["ci_low"],ci_high=r["ci_high"],
                          savings_per_1000_all_arrivals=arrival_savings))
    require(len(pairs)==18,"Missing paired panel")
    write_rows(destination/"relative_geometry_swaps.csv",pairs)
    lines = [
        "# Inspection ranking in a fixed development alert pool",
        "",
        "This retrospective simulation uses the original labels of the development audit, which had been inspected during research. Lots and exact-image components were separated across training, tuning and audit.",
        "",
        "After calibration on tuning batches, RF71 seed 42 screened 3,642 of 11,591 arrivals at an any-defect probability of at least 0.5. The 903 contributing lots formed three full 1,000-alert queues and a final queue of 642. Their capacities were 200, 200, 200 and 128, for 728 total inspections. Every model used the same alerts, order and capacity.",
        "",
        "An unreviewed defect incurs a loss of 1, or 5 for Scratch or Near-full in the corresponding scenario. Normal loss is 0. Each review costs 0.1 and is assumed to prevent 90% of the defect loss. Reviewed wafers cost 0.1 + 0.1L. Expected net benefit is 0.9 times probability-weighted loss minus 0.1. Outcomes use observed labels. Costs are hypothetical relative units.",
        "",
        "## Simulated cost per 1,000 alerted candidates",
        "",
        "Cells list seeds 42, 123 and 2026 in order. Identical rounded results are shown once. Lower cost is preferable. The random reference is the same frozen draw.",
        "",
        "| Scenario | Ranker | Random, fill capacity | Any-defect probability, fill | Net benefit, fill | Positive net benefit, capped |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case,title in SCENARIOS.items():
        for model in MODELS:
            cells=[triple(ref[model,case,s,pol]["cost_per_1000"] for s in SEEDS) for pol in POLICIES]
            lines.append("| "+" | ".join([title,model,*cells])+" |")
    lines += ["", "All queues contained enough positive-benefit candidates to fill all 728 slots. The [full matrix](relative_policy_matrix.csv) gives 108 combinations with number reviewed, normals reviewed, unreviewed defects, loss outside the alert pool and whole-stream cost per 1,000 arrivals.", "",
              "## Which wafers changed places with RF77?", "",
              "The table holds full-capacity benefit ranking and fivefold Scratch loss fixed. RF71 and RF77 share an upstream miss cost. Including these misses changes the denominator from 3,642 alerts to 11,591 arrivals. Intervals resample paired lots conditional on the models, queues and selected lists.",
              "", "| Seed | Entered / displaced | Scratch entered / displaced | Normal entered / displaced | Savings per 1,000 alerts [95% interval] | Savings per 1,000 arrivals |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in pairs:
        if r["scenario"]=="scratch_5x" and r["policy"]=="benefit_fixed":
            lines.append(f"| {r['model_seed']} | {r['entered']} / {r['displaced']} | {r['scratch_entered']} / {r['scratch_displaced']} | {r['normal_entered']} / {r['normal_displaced']} | {r['savings_per_1000_candidates']:.3f} [{float(r['ci_low']):.3f}, {float(r['ci_high']):.3f}] | {r['savings_per_1000_all_arrivals']:.3f} |")
    lines+=["", "Under equal losses or fivefold Near-full loss the realized RF77 versus RF71 cost differences were zero to numerical precision. Each seed uses the same cohort. [Per-seed entries, displacements and uncertainty](relative_geometry_swaps.csv)", ""]
    (destination/"relative_readout.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    generate(args.cost,args.out)
