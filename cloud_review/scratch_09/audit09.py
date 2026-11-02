"""Read-only hosted replication and paired component bootstrap; exports aggregates only."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


STUDY = "8b05f0bdcc13a20ac032e58eecd7862cba085cd56be9ec121d54cebcd40a8c67"
REF = "4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879"
HISTORY = "bc2bb70d78bf551f400448b0ddf49e7e849c8942b39d7a953bee46d2e34c3108"
SEEDS, DRAWS = (42, 123, 2026), 1000


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def maphash(a):
    return hashlib.sha256(np.array(a.shape, dtype="<i8").tobytes()+np.ascontiguousarray(a).tobytes()).hexdigest()


def table(p):
    with (gzip.open if p.suffix == ".gz" else open)(p, "rt", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def output(p, rows):
    with p.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loadj(p):
    return json.loads(p.read_text(encoding="utf8"))


def spot(path, r):
    check(sha(path) == r["spot_file_sha256"], "Spot archive changed")
    with np.load(path, allow_pickle=False) as z:
        donor, bg, mixed = (z[k] for k in ("donor","background","synthetic"))
    native = [maphash(a) for a in (donor,bg,mixed)]
    check(native == [r["donor_sha256"],r["normal_sha256"],r["synthetic_sha256"]], "Native spot or donor mismatch")
    changed = np.argwhere(mixed != bg)
    rc = tuple(changed.T)
    orig = changed-np.array(r["offset_rc"])
    check(np.all((orig>=0)&(orig<np.array(donor.shape))) and np.all(donor[tuple(orig.T)]==2),
          "Inserted failure unsupported by real donor")
    check(np.array_equal(mixed!=0,bg!=0) and np.all(bg[rc]==1) and np.all(mixed[rc]==2)
          and len(changed)==r["added_dies"] and int((bg==2).sum())==r["old_failed_dies"], "Invalid die edit")


def quality(repo, training):
    p = repo/"data-reference/development_reference.csv.gz"
    check(sha(p)==REF, "Development reference changed")
    rows = sorted(table(p), key=lambda r:int(r["source_row"]))
    train = {int(r["source_row"]):r for r in rows if r["role"]=="train"}
    ids = np.array(list(train))
    originals = {r["map_sha256"] for r in rows}
    limit = loadj(training/"template_qa.json")["max_failure_fraction"]
    qa = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        receipts = []
        for arm in ("real_repeat","synthetic"):
            d = training/"fits"/f"{arm}_s{seed}"
            rec = loadj(d/"run.json")
            check(rec["model"]==arm and rec["seed"]==seed and len(rec["epochs"])==7, "Incomplete epoch receipts")
            receipts.append((d,rec))
        for epoch in range(7):
            order = rng.permutation(ids)
            logs = []
            for d,rec in receipts:
                h = rec["epochs"][epoch]
                file = d/h["provenance_file"]
                check(h["epoch"]==epoch+1 and sha(file)==h["provenance_sha256"], "Changed epoch provenance")
                with gzip.open(file,"rt",encoding="utf8") as f:
                    logs.append([json.loads(line) for line in f])
            b,c = logs
            check(len(b)==len(c)==80 and len({v["slot"] for v in b})==80, "Invalid sample budget")
            spots=0
            for j,(repeat,r) in enumerate(zip(b,c)):
                slot = r["slot"]
                check(0<=slot<len(ids) and slot==repeat["slot"] and order[slot]==r["removed_normal_row"]==
                      repeat["removed_normal_row"] and train[int(order[slot])]["label"]=="none", "Shifted normal slot")
                source,target = train[r["donor_row"]],train[r["normal_row"]]
                check(train[repeat["repeat_row"]]["label"]=="Scratch" and repeat["repeat_row"]==r["paired_repeat_row"]
                      and source["label"]=="Scratch" and target["label"]=="none" and
                      (source["map_sha256"],target["map_sha256"])==(r["donor_sha256"],r["normal_sha256"]),
                      "Wrong role, label or real source identity")
                h,w = int(target["height"]),int(target["width"])
                check(r["native_hw"]==[h,w] and r["donor_hw"]==[int(source["height"]),int(source["width"])]
                      and r["added_dies"]>=max(3,.5*r["trace_dies"]) and
                      r["trace_dies"]>=r["retained_trace_dies"]>=max(r["added_dies"],.9*r["trace_dies"])
                      and (r["old_failed_dies"]+r["added_dies"])/r["valid_dies"]<=limit, "Invalid reported synthesis")
                if epoch==0 and j<4:
                    spot(receipts[1][0]/f"spot{j}.npz",r)
                    spots+=1
            qa.append(dict(seed=seed,epoch=epoch+1,paired_normal_slots=80,valid_edit_records=80,native_spots=spots,
                same_donor=sum(r["paired_repeat_row"]==r["donor_row"] for r in c),
                exact_native_size=sum(r["native_hw"]==r["donor_hw"] for r in c),
                novel_vs_labeled_development=sum(r["synthetic_sha256"] not in originals for r in c),
                distinct_synthetic=len({r["synthetic_sha256"] for r in c}),distinct_donors=len({r["donor_row"] for r in c}),
                distinct_backgrounds=len({r["normal_row"] for r in c}),added_die_median=float(np.median([r["added_dies"] for r in c])),
                min_retained_fraction=min(r["retained_trace_dies"]/r["trace_dies"] for r in c)))
    return qa


def bootstrap(truth, baseline, candidate, groups, weights):
    result=[]
    for label,metric in ((7,"scratch_recall"),(8,"normal_any_fpr")):
        mask=truth==label
        b=baseline[mask]==label if label==7 else baseline[mask]!=label
        c=candidate[mask]==label if label==7 else candidate[mask]!=label
        chunks=groups[mask]
        den=np.bincount(chunks,minlength=weights.shape[1])
        diff=np.bincount(chunks,weights=c.astype(int)-b.astype(int),minlength=len(den))
        bd=weights@den
        valid=bd>0
        rates=(weights@diff)[valid]/bd[valid]
        lo,hi=np.quantile(rates,[.025,.975])
        result.append(dict(metric=metric,eligible_wafers=int(mask.sum()),eligible_components=int((den>0).sum()),
              baseline_events=int(b.sum()),candidate_events=int(c.sum()),delta_rate=float(diff.sum()/den.sum()),
              percentile_95_low=float(lo),percentile_95_high=float(hi),replicates_used=int(valid.sum())))
    return result


def run(args):
    out,training,ev=args.out,args.training,args.eval
    p=args.repo/"cloud_review/extension/decision_study.py"
    check(sha(p)==STUDY,"Frozen grouping or alignment changed")
    spec=importlib.util.spec_from_file_location("audit_decision",p)
    study=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    _,split=study.inputs(args.repo/"results/probabilities",None)
    mr=ev/"run_receipt.json"
    er=loadj(mr)
    scores=training/"probabilities"
    manifest=scores/"manifest.json"
    check(er["complete"] and sha(manifest)==er["training_manifest_sha256"] and
          sha(ev/"confusion.csv")==er["files"]["confusion.csv"],"Evaluation bindings differ")
    files={(r["model"],r["seed"],r["role"]):r for r in loadj(manifest)["files"]}
    arms=sorted({a for a,s,r in files if a!="baseline"})
    cm={(r["phase"],r["arm"],int(r["seed"]),r["truth"]):[int(r[c]) for c in study.CLASSES] for r in table(ev/"confusion.csv")}
    intervals=[]
    for role,phase in (("tune","selection"),("audit","audit_exploratory")):
        refs=[r for r in split if r["role"]==role and r["partition"]==("selection" if role=="tune" else "audit")]
        g,gi=np.unique([r["component"] for r in refs],return_inverse=True)
        lots={}
        for r in refs:
            check(lots.setdefault(r["lot"],r["component"])==r["component"], "Lot split across bootstrap components")
        rng=np.random.default_rng(901 if role=="tune" else 902)
        weights=rng.multinomial(len(g),np.full(len(g),1/len(g)),size=DRAWS)
        y=np.array([study.CLASSES.index(r["truth"]) for r in refs])
        for seed in SEEDS:
            outputs={}
            for a in ["baseline",*arms]:
                f=files[a,seed,role]
                pred={r["source_row"]:r for r in table(study.checked_file(scores,f))}
                check(all(pred[r["source_row"]]["truth"]==r["truth"] for r in refs), "Labels or identities differ")
                labels=np.argmax([[float(pred[r["source_row"]][c]) for c in study.CLASSES] for r in refs],axis=1)
                check(study.confusion(y,labels).tolist()==[cm[phase,a,seed,c] for c in study.CLASSES],
                      "Per-row argmax differs from reported matrix")
                outputs[a]=labels
            for a in arms:
                for r in bootstrap(y,outputs["baseline"],outputs[a],gi,weights):
                    intervals.append(dict(phase=phase,seed=seed,arm=a,reference="baseline",total_lots=len(lots),
                        total_components=len(g),**r))
    old=args.repo/"cloud_review/results/full/confusion.csv.gz"
    check(sha(old)==HISTORY,"Historical aggregates changed")
    prior={(r["phase"],int(r["seed"]),r["actual"]):[int(r[c]) for c in study.CLASSES] for r in table(old)
           if r["model"]=="cnn_ce" and r["rule"]=="argmax"}
    replication=[]
    for phase in ("selection","audit_exploratory"):
        for seed in SEEDS:
            left=np.array([cm[phase,"baseline",seed,c] for c in study.CLASSES])
            right=np.array([prior[phase,seed,c] for c in study.CLASSES])
            replication.append(dict(phase=phase,seed=seed,identical_cells=int((left==right).sum()),cells=81,
                exact_confusion_match=bool(np.array_equal(left,right)),max_abs_cell_change=int(np.abs(left-right).max()),
                scratch_tp_delta=int(left[7,7]-right[7,7]),normal_any_fp_delta=int(left[8,:8].sum()-right[8,:8].sum())))
    qa=quality(args.repo,training)
    out.mkdir(parents=True,exist_ok=False)
    for name,data in (("paired_component_intervals",intervals),("synthesis_epoch_qa",qa),("historical_ce_replication",replication)):
        output(out/(name+".csv"),data)
    receipt={"complete":True,"selection_changes":False,"draws":DRAWS,"estimand":"paired per-wafer rate difference within each fixed seed; resample complete lot/image-linked components with replacement; percentile intervals conditional on the trained fits, no independence across seeds or multiplicity adjustment",
        "official_test_used":False,"history_confusion_sha256":HISTORY,"training_manifest_sha256":sha(manifest),
        "eval_receipt_sha256":sha(mr),"script_sha256":sha(Path(__file__)),"files":{p.name:sha(p) for p in out.iterdir()}}
    (out/"audit_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf8")
    print("Replication, paired component intervals and native die checks complete.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","training","eval","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    check(os.name=="posix" and Path("/content").is_dir() and
          (os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_BACKEND_VERSION")), "Real inputs require hosted Colab")
    run(args)
