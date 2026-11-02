"""Prespecified supplementary arm selection using aggregate 9-class argmax results.

The main study maximizes mean selection Macro-F1 over all five new CNN arms.
Here every seed must first preserve Scratch TP and both normal FP counts versus
matched CE. Audit never selects the arm. No scores or wafer-level files are read.
"""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from zipfile import ZipFile


ARMS=("cnn_ce","cnn_weighted","cnn_sampled","cnn_rot180","cnn_focal")
SEEDS=(42,123,2026)
CLASSES=("Center","Donut","Edge-Loc","Edge-Ring","Loc","Near-full","Random","Scratch","none")
SELECT,AUDIT="selection","audit_exploratory"
RULE="9-class argmax. A candidate must meet all three inequalities versus seed-matched cnn_ce on every selection seed: Scratch TP >= CE, normal-to-any-defect FP <= CE, normal-to-Scratch FP <= CE. Rank eligible arms by highest exact three-seed mean nine-class Macro-F1. Break ties by declared arm order. Freeze before interpreting audit."
TABLES=("classification","confusion","per_class")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def json_file(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def load_inputs(read, published):
    receipt_name="manifest.json" if published else "run_status.json"
    receipt_b=read(receipt_name)
    receipt=json.loads(receipt_b)
    check(receipt["training_extension"] is True and (published or receipt["status"]=="complete"),
          "Require completed five-arm evidence, not the archive-only run")
    valid=receipt["files"] if published else receipt["sha256"]
    def verified(name):
        b=read(name)
        check(sha(b)==valid[name],f"Input digest mismatch: {name}")
        return b
    policy=json.loads(verified("frozen_validation_policy.json"))
    check(policy["downstream_model"] in ARMS,"Unexpected main model")
    data={}
    for name in TABLES:
        suffix=".csv.gz" if published else ".csv"
        raw=verified(name+suffix)
        data[name]=gzip.decompress(raw) if published else raw
    return data,policy,{"source_receipt_sha256":sha(receipt_b),
            "main_policy_sha256":valid["frozen_validation_policy.json"], "source_table_sha256":{k:sha(v) for k,v in data.items()}}


def phase_rows(raw, phase):
    reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig"),newline=""))
    check(not {"source_row","map_sha256","component","lot"}.intersection(reader.fieldnames or []),"Expected aggregate table")
    return [r for r in reader if (r["phase"],r["rule"])==(phase,"argmax") and r["model"] in ARMS]


def indexed(rows, fields):
    out={tuple(r[k] for k in fields):r for r in rows}
    check(len(out)==len(rows),"Duplicate aggregate records")
    return out


def stats(data, phase):
    rows={k:phase_rows(raw,phase) for k,raw in data.items()}
    summary=indexed(rows["classification"],("model","seed"))
    cells=indexed(rows["confusion"],("model","seed","actual"))
    labels=indexed(rows["per_class"],("model","seed","label"))
    fits={(m,str(s)) for m in ARMS for s in SEEDS}
    triples={(*pair,c) for pair in fits for c in CLASSES}
    check(set(summary)==fits and set(cells)==set(labels)==triples,"Require five complete arms, three seeds, nine classes")
    common_support=None
    result={}
    for m in ARMS:
        for seed in SEEDS:
            key=(m,str(seed))
            s=summary[key]
            cm=[[int(cells[(*key,actual)][pred]) for pred in CLASSES] for actual in CLASSES]
            support=[sum(r) for r in cm]
            check(min(min(r) for r in cm)>=0 and sum(support)==int(s["n"]), "Invalid 9-class confusion matrix")
            if common_support is None:
                common_support=support
            check(common_support==support and min(support[7:])>0, "Different evaluation populations or missing Scratch/normal")
            per_class=[]
            for j,c in enumerate(CLASSES):
                classified=sum(row[j] for row in cm)
                denominator=support[j]+classified
                f=Fraction(2*cm[j][j],denominator) if denominator else Fraction(0)
                label=labels[(*key,c)]
                check(int(label["support"])==support[j] and int(label["predictions"])==classified
                      and math.isclose(float(label["f1"]),float(f),abs_tol=1e-12,rel_tol=0), "Per-class/CM mismatch")
                per_class.append(f)
            macro=sum(per_class)/9
            counts=dict(scratch_tp=cm[7][7],normal_any_fp=sum(cm[8][:8]),normal_scratch_fp=cm[8][7])
            check(all(int(s[k])==v for k,v in counts.items()) and math.isclose(float(s["macro_f1"]),float(macro),
                  abs_tol=1e-12,rel_tol=0) and int(s["scratch_n"])==support[7] and int(s["normal_n"])==support[8],
                  "Summary/CM mismatch")
            result[m,seed]=dict(**counts,macro=macro,scratch_n=support[7],normal_n=support[8])
    return result


def compare(stage, results):
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            ce,model=results["cnn_ce",seed],results[arm,seed]
            d_tp=model["scratch_tp"]-ce["scratch_tp"]
            d_any=model["normal_any_fp"]-ce["normal_any_fp"]
            d_sc=model["normal_scratch_fp"]-ce["normal_scratch_fp"]
            ok=d_tp>=0 and d_any<=0 and d_sc<=0
            rows.append(dict(phase=stage,arm=arm,seed=seed,scratch_n=model["scratch_n"],normal_n=model["normal_n"],
                ce_scratch_tp=ce["scratch_tp"],scratch_tp=model["scratch_tp"],scratch_tp_delta=d_tp,
                ce_normal_any_fp=ce["normal_any_fp"],normal_any_fp=model["normal_any_fp"],normal_any_fp_delta=d_any,
                ce_normal_scratch_fp=ce["normal_scratch_fp"],normal_scratch_fp=model["normal_scratch_fp"],
                normal_scratch_fp_delta=d_sc,all_three_hold=ok,strict_gain=ok and (d_tp>0 or d_any<0 or d_sc<0),
                macro_f1=float(model["macro"])))
    all_seeds={a:all(r["all_three_hold"] for r in rows if r["arm"]==a) for a in ARMS}
    means={a:sum(results[a,s]["macro"] for s in SEEDS)/len(SEEDS) for a in ARMS}
    return rows,all_seeds,means


def analyze(data, main, source, out):
    chosen_rows,eligible,means=compare(SELECT,stats(data,SELECT))
    candidates=[a for a in ARMS if eligible[a]]
    check(ARMS[0] in candidates,"CE identity did not satisfy its own constraint")
    chosen=max(candidates,key=means.get)
    # Validate the main study too, without using its downstream choice as a new vote.
    for arm in ARMS:
        check(math.isclose(float(means[arm]),float(main["mean_selection_scores"][arm]["macro_f1"]),
                           rel_tol=0,abs_tol=1e-12),"Inconsistent main selection source")
    freeze=dict(rule=RULE,arm_order=list(ARMS),seeds=list(SEEDS),selection_macro_f1={a:float(v) for a,v in means.items()},
        selection_all_three_all_seeds=eligible,eligible_candidates=candidates,frozen_arm=chosen,
        main_unconstrained_choice=main["downstream_model"],source_script_sha256=sha(Path(__file__).read_bytes()),**source)
    out.mkdir(parents=True,exist_ok=False)
    json_file(out/"selection_freeze.json",freeze)
    audit_rows,holds,_=compare(AUDIT,stats(data,AUDIT))
    selected=[r for r in audit_rows if r["arm"]==chosen]
    conclusion=dict(**freeze,audit_all_three_all_seeds=holds,audit_passing_seeds=[r["seed"] for r in selected if r["all_three_hold"]],
        audit_all_three_for_frozen_arm=holds[chosen],
        audit_any_strict_gain_for_frozen_arm=any(r["strict_gain"] for r in selected),
        audit_strict_gain_on_every_seed_for_frozen_arm=all(r["strict_gain"] for r in selected),
        scope="Existing development audit; same wafers across seeds; exploratory. Audit was never used to replace the frozen selection.")
    json_file(out/"guarded_summary.json",conclusion)
    with (out/"paired_guard_deltas.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,list(chosen_rows[0]))
        writer.writeheader()
        writer.writerows(chosen_rows+audit_rows)
    passed=len(conclusion["audit_passing_seeds"])
    if chosen == "cnn_ce":
        statement = "CE is the retained control. Its paired differences against itself are zero."
    else:
        statement = f"The selected arm met all three audit constraints in {passed}/3 seeds."
    note = [
        "# CNN selection with joint Scratch and false-alarm constraints",
        "Each of the three seeds must preserve the matched CE Scratch hit count and increase neither "
        "total false alarms on normal wafers nor normal-to-Scratch errors. Among eligible arms, selection "
        "uses mean nine-class Macro-F1 across seeds.",
        f"Eligible arms: {', '.join(candidates)}. The selected arm was {chosen} "
        f"({float(means[chosen]):.2%} Macro-F1). The unrestricted comparison selected {main['downstream_model']}.",
        statement,
        "| Audit seed | Scratch hit change | Normal false-alarm change | Normal-to-Scratch change | All conditions held |\n"
        "|---|---:|---:|---:|---|\n" + "\n".join(
            f"| {r['seed']} | {r['scratch_tp_delta']:+d} | {r['normal_any_fp_delta']:+d} | "
            f"{r['normal_scratch_fp_delta']:+d} | {'Yes' if r['all_three_hold'] else 'No'} |" for r in selected),
        f"All three seeds saw the same {selected[0]['scratch_n']} scratches and {selected[0]['normal_n']:,} normal wafers. "
        "See paired_guard_deltas.csv for every arm. Independent validation requires new lots.",
    ]
    (out/"BRIEF.md").write_text("\n\n".join(note)+"\n",encoding="utf8")
    return conclusion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",type=Path,help="Completed full run, published aggregate folder, or aggregate_evidence.zip")
    p.add_argument("--out",type=Path,help="New directory")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.evidence or not a.out:
        p.error("Use --evidence and --out, or --self-test")
    if a.evidence.is_dir():
        inputs=load_inputs(lambda n:(a.evidence/n).read_bytes(),(a.evidence/"manifest.json").is_file())
    else:
        with ZipFile(a.evidence) as f:
            inputs=load_inputs(f.read,True)
    summary=analyze(*inputs,a.out)
    print("Frozen arm:",summary["frozen_arm"],"| all three audit guards on every seed:",summary["audit_all_three_for_frozen_arm"])


def self_test():
    raw={k:[] for k in TABLES}
    for phase in (SELECT,AUDIT):
        for arm in ARMS:
            for seed in SEEDS:
                tp,fp_sc,fp_other=12,3,4
                if arm=="cnn_weighted":
                    tp=19
                    fp_other=5 if seed==123 else 4  # Fails one seed, despite high mean F1.
                if arm in ("cnn_sampled","cnn_rot180"):
                    tp,fp_sc=15,2  # Tied eligible scores; sampled precedes rotation.
                if arm=="cnn_focal":
                    fp_sc,fp_other=1,7  # Moves Scratch alarms to other defects.
                if phase==AUDIT and arm=="cnn_sampled" and seed==2026:
                    tp=11  # Audit does not switch to rotation.
                cm=[[int(j==k)*20 for k in range(9)] for j in range(9)]
                cm[7][7],cm[7][8]=tp,20-tp
                cm[8][0],cm[8][7],cm[8][8]=fp_other,fp_sc,20-fp_other-fp_sc
                key=dict(model=arm,seed=seed,phase=phase,rule="argmax")
                macro=0
                for j,c in enumerate(CLASSES):
                    predicted=sum(row[j] for row in cm)
                    f1=2*cm[j][j]/(20+predicted)
                    macro+=f1/9
                    raw["confusion"].append(dict(**key,actual=c,**dict(zip(CLASSES,cm[j]))))
                    raw["per_class"].append(dict(**key,label=c,support=20,predictions=predicted,f1=f1))
                raw["classification"].append(dict(**key,n=180,macro_f1=macro,scratch_tp=tp,scratch_n=20,normal_n=20,
                                                  normal_scratch_fp=fp_sc,normal_any_fp=fp_sc+fp_other))
    data={}
    for k,rows in raw.items():
        f=io.StringIO()
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        data[k]=f.getvalue().encode("utf8")
    _,_,means=compare(SELECT,stats(data,SELECT))
    policy=dict(downstream_model=max(ARMS,key=means.get),mean_selection_scores={a:{"macro_f1":float(v)} for a,v in means.items()})
    with tempfile.TemporaryDirectory() as td:
        s=analyze(data,policy,{},Path(td)/"new")
        check(s["eligible_candidates"]==["cnn_ce","cnn_sampled","cnn_rot180"] and s["frozen_arm"]=="cnn_sampled",
              "All-seed filtering or exact tie order failed")
        check(s["audit_passing_seeds"]==[42,123] and s["audit_all_three_all_seeds"]["cnn_rot180"],
              "Audit must not replace the frozen winner")
        check(s["main_unconstrained_choice"]=="cnn_weighted","Expected unconstrained comparator")
    print("Synthetic one-seed veto, shifted FP, exact ties, independent freeze, and audit failure checks passed.")


if __name__=="__main__":
    main()
