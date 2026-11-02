"""Publish verified aggregate tables from a completed decision_study.py run.

python render_evidence.py --run /content/decision_full_v1 --out /content/decision_evidence_v1
The output directory must be new. Uses only the Python standard library.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


AGGREGATES = ("split_support", "classification", "per_class", "confusion", "scratch_guard", "calibration",
              "reliability", "routing", "inspection_by_class", "paired_cost", "queue_inventory", "threshold_curves",
              "five_stages")
FIGURES = ("threshold_precision_recall_cost.png", "calibration_reliability.png")
SEEDS = {42,123,2026}
RF = "rf_original71"
MODELS = {RF:"RF71", "rf_linearity77":"RF77", "cnn_bn_natural_7ep":"CNN (archived)",
          "ensemble_rf77_cnn":"RF77 + CNN (50/50)", "cnn_ce":"CNN CE", "cnn_weighted":"CNN weighted loss",
          "cnn_sampled":"CNN weighted sampling", "cnn_rot180":"CNN 180° rotation", "cnn_focal":"CNN focal"}
AUDIT = "audit_exploratory"
BOUNDARY = "Previously inspected development audit. Each row averages three training seeds on the same wafers. Mean counts can be fractional."
STAGES = {
    "1_RF71_archived_argmax": "Archived RF71",
    "1b_CNN_matched_CE_control": "CNN CE control",
    "2_validation_selected_training_arm": "Selected training arm",
    "2_archived_CNN_proxy_no_new_training": "Archived CNN comparator",
    "3_same_fit_F1_threshold": "F1 threshold",
    "4_same_fit_minimum_cost": "Cost threshold",
    "5_same_fit_fixed_gate_and_slots": "Capacity within the gate",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def match(table, **conditions):
    return [r for r in table if all(r[k] == v for k,v in conditions.items())]


def seed_mean(rows, field):
    require(len(rows)==3 and {int(r["seed"]) for r in rows}==SEEDS, f"Incomplete seed group for {field}")
    return mean(float(r[field]) for r in rows)


def fmt(value, decimals=1):
    return f"{value:,.{decimals}f}"


def avg(rows,field,decimals=1):
    return fmt(seed_mean(rows,field),decimals)


def pct(rows,field):
    return fmt(100*seed_mean(rows,field),2)+"%"


def md_table(header, rows):
    return "\n".join("| " + " | ".join(map(str,r)) + " |" for r in [header, ["---"]*len(header), *rows])


def summarize(tables, freeze, extended):
    model = freeze["downstream_model"]
    mlabel = MODELS[model]
    title = "Development experiment results"
    sections = [f"# WM-811K · {title}",BOUNDARY,
        "Disjoint tune components fit the temperature and select policies. The training arm and thresholds were frozen before loading audit scores."]
    if not extended:
        sections.append("This run replays archived scores. Five-arm retraining results are pending.")
    sections.append("## "+"Sample composition")
    rows = []
    for phase in ("train","calibration","selection","audit"):
        group = match(tables["split_support"],phase=phase)
        n = int(group[0]["total"])
        norm = int(match(group,label="none")[0]["rows"])
        scr = int(match(group,label="Scratch")[0]["rows"])
        require(sum(int(r["rows"]) for r in group)==n, "Support totals disagree")
        rows.append([phase,fmt(n,0),fmt(norm,0),fmt(n-norm,0),fmt(scr,0),fmt(100*norm/n,2)+"%"])
    sections.append(md_table(["Partition","N","Normal","Defect","Scratch","Normal share"],rows))
    sections.append("These proportions describe labeled wafers in each development partition. The raw archive, the full labeled population, and production arrivals have distinct denominators.")
    cl = match(tables["classification"],phase=AUDIT,rule="argmax")
    rows = []
    for name, title in MODELS.items():
        group = match(cl,model=name)
        if group:
            rows.append([title,pct(group,"macro_f1"),pct(group,"scratch_precision"),pct(group,"scratch_recall"),
                         pct(group,"scratch_ap"),pct(group,"normal_any_fpr"),avg(group,"normal_scratch_fp")])
    sections.extend(["## "+"Training and nine-class predictions",
            md_table(["Model","Macro-F1","Scratch P","Scratch R","Scratch AP",
                      "Normal FPR","Normal→Scratch"],rows)])
    sections.append(f"Validation chose {mlabel}. The five new CNN arms share architecture, split and seeds; assess weighting, sampling, rotation or focal loss against the fresh CE control."
                    if extended else f"Downstream policies use the prespecified archived CNN ({mlabel}).")
    guard = match(tables["scratch_guard"],phase=AUDIT,model=model)
    passed = sum(r["all_three_hold"]=="True" for r in guard)
    sections.append(f"For {mlabel}, the Scratch-priority rule changed true Scratch hits by {avg(guard,'scratch_tp_delta')}, total false alarms on normal wafers by {avg(guard,'normal_any_fp_delta')}, and normal-to-Scratch predictions by {avg(guard,'normal_scratch_fp_delta')} versus argmax. All three guards held on audit for {passed}/3 seeds.")
    identity = sum(c["model"] == model and c["scratch"] is None for c in freeze["params"])
    sections.append(f"Validation retained the original argmax rule for {identity}/3 seeds.")
    raw,cal = [match(tables["calibration"],phase=AUDIT,model=model,probabilities=kind) for kind in ("raw","calibrated")]
    sections.append(f"Before and after temperature scaling, multiclass NLL was {avg(raw,'multiclass_nll',3)} → {avg(cal,'multiclass_nll',3)}, multiclass Brier {avg(raw,'multiclass_brier',3)} → {avg(cal,'multiclass_brier',3)}, and any-defect ECE {pct(raw,'defect_ece')} → {pct(cal,'defect_ece')}.")
    sections.append("For exact zero raw probabilities, NLL uses a numerical floor of 1e-300. Temperature scaling first floors input probabilities at 1e-12, so the NLL comparison includes smoothing of zeros.")
    sections.append("## "+"Thresholds, inspection slots and costs")
    sections.append("Both hypothetical scenarios assign $1,000 per missed ordinary defect and $5 per inspected normal wafer with perfect human inspection. The cost columns assume $1,000 and $3,000 per missed Scratch. All-arrival denominators include wafers excluded by the upstream gate.")
    rows = []
    for stage, name in STAGES.items():
        specific = match(tables["five_stages"],stage=stage)
        if not specific:
            continue
        uniform, diff = [match(specific,scenario=sc) for sc in ("uniform_1000","scratch_3000")]
        rows.append([name,avg(diff,"reviewed"),avg(diff,"fn"),avg(diff,"fp"),
                     avg(uniform,"cost_per_1000_arrivals",0),avg(diff,"cost_per_1000_arrivals",0)])
    sections.append(md_table(["Stage","Reviewed¹","FN¹","FP¹",
                              "USD/1k ($1k Scratch)","USD/1k ($3k Scratch)"],rows))
    sections.append("¹ Reviews, FN and FP use the $3,000 Scratch scenario. Thresholds can route different numbers under different cost assumptions. Capped rankings enforce a 20% limit in each simulated batch.")
    routing = match(tables["routing"],phase=AUDIT,model=model,scenario="scratch_3000")
    rows = []
    for pol, en in (("queue_random", "Random"), ("queue_probability", "Defect probability"),
                    ("queue_expected_value", "Expected net benefit")):
        g=match(routing,policy=pol)
        rows.append([en,avg(g,"candidates",0),avg(g,"reviewed",0),avg(g,"scratch_captured"),
                     avg(g,"upstream_fn"),avg(g,"candidate_unreviewed_fn"),avg(g,"fp"),
                     avg(g,"cost_per_1000_arrivals",0)])
    sections.append(md_table(["Common-pool rank","Flagged","Reviewed",
                "Scratch found","Gate FN","In-pool FN","FP","USD/1k"],rows))
    sections.append("All rankings share the historical RF71 gate and inspect up to 200 of 1,000 flagged wafers per batch, rounding down in the final partial batch. The random comparator hashes wafer IDs. Actual arrival times, inspection efficacy and monetary costs have not been observed.")
    pair = match(tables["paired_cost"],comparator=RF,candidate=model)
    if pair:
        r=pair[0]
        sections.append(f"At fixed capacity, replacing RF71 with {mlabel} saved a simulated ${fmt(float(r['savings_per_1000_arrivals']))} per 1,000 arrivals. The paired component bootstrap 95% interval was [{fmt(float(r['lo_95']))}, {fmt(float(r['hi_95']))}], conditional on fitted models and these queues.")
    sections += ["![Threshold, precision, recall and simulated cost](threshold_precision_recall_cost.png)",
                 "![Calibration reliability](calibration_reliability.png)",
                 "Per-class metrics and confusions are in per_class.csv.gz and confusion.csv.gz; routing.csv.gz gives per-seed cost results. The frozen policy file records the fitted choices."]
    return "\n\n".join(sections)+"\n"


def build(source, out):
    receipt = json.loads((source/"run_status.json").read_text(encoding="utf8"))
    require(receipt["status"]=="complete", "Require a completed hosted experiment")
    def verified(name):
        b = (source/name).read_bytes()
        require(sha(b)==receipt["sha256"][name], f"Receipt mismatch for {name}")
        return b

    policy = "frozen_validation_policy.json"
    b = verified(policy)
    require(sha(b)==receipt["validation_policy_sha256"], "Freeze mismatch")
    freeze = json.loads(b)
    require(len(freeze["params"])==receipt["fits_scored"], "Unexpected number of scored models")
    data = {policy:b}
    tables = {}
    # This is an allow-list. Row identities, input probabilities, membership and raw
    # hashes for individual wafers are excluded regardless of the input directory.
    for name in AGGREGATES:
        raw = verified(name+".csv")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"),newline=""))
        require(not ({"source_row","lot","map_sha256","component"} & set(reader.fieldnames)),
                "Aggregate table unexpectedly contains row identifiers")
        tables[name] = list(reader)
        data[name+".csv.gz"] = gzip.compress(raw, compresslevel=9,mtime=0)
    for name in FIGURES:
        data[name] = verified(name)
    extended = bool(receipt["training_extension"])
    data["SUMMARY.md"] = summarize(tables, freeze, extended).encode("utf8")
    note = dict(scope="Previously inspected development audit. No new official test. Aggregate-only evidence.",
                training_extension=extended, model_seed_pairs=receipt["fits_scored"],
                source_receipt_sha256=sha((source/"run_status.json").read_bytes()),
                source_policy_sha256=receipt["validation_policy_sha256"],
                render_script_sha256=sha(Path(__file__).read_bytes()),
                files={n:sha(b) for n,b in sorted(data.items())})
    data["manifest.json"]=(json.dumps(note,indent=2)+"\n").encode("utf8")
    out.mkdir(parents=True,exist_ok=False)
    for n,b in data.items():
        (out/n).write_bytes(b)
    # Fixed ZIP metadata and sorted members make the bundle reproducible on this runtime.
    with ZipFile(out/"aggregate_evidence.zip","w") as z:
        for n,b in sorted(data.items()):
            z.writestr(ZipInfo(n,(1980,1,1,0,0,0)),b,compress_type=ZIP_DEFLATED,compresslevel=6)
    print(f"Published {len(data)} verified aggregate artifacts. ZIP size {(out/'aggregate_evidence.zip').stat().st_size:,} bytes.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    build(args.run,args.out)


if __name__=="__main__":
    main()
