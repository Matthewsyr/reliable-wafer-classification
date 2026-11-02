"""Post-hoc paired argmax transitions on identical labeled development rows."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


STUDY_SHA = "8b05f0bdcc13a20ac032e58eecd7862cba085cd56be9ec121d54cebcd40a8c67"
S,N = 7,8


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(v,msg):
    if not v:
        raise ValueError(msg)


def transition(y,a,b):
    normal = y==N
    fine = np.bincount(a[normal]*9+b[normal],minlength=81).reshape(9,9)
    bands = np.array([2]*7+[1,0])  # normal, Scratch, any other defect
    coarse = np.bincount(bands[a[normal]]*3+bands[b[normal]],minlength=9).reshape(3,3)
    scratch = y==S
    aa,bb = (a[scratch]==S).astype(int),(b[scratch]==S).astype(int)
    hits = np.bincount(aa*2+bb,minlength=4).reshape(2,2)
    require(fine.sum()==coarse.sum()==normal.sum() and hits.sum()==scratch.sum(),"Cohort sizes changed")
    return fine,coarse,hits


def run(args):
    require(os.name=="posix" and Path("/content").is_dir() and
            (os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_BACKEND_VERSION")),"Requires hosted Colab")
    p=args.repo/"cloud_review/extension/decision_study.py"
    require(sha(p)==STUDY_SHA,"Frozen decision module changed")
    spec=importlib.util.spec_from_file_location("study_transitions",p)
    study=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    _,split=study.inputs(args.repo/"results/probabilities",None)
    prob=args.training/"probabilities"
    mp=prob/"manifest.json"
    receipt=args.eval/"run_receipt.json"
    er=json.loads(receipt.read_text())
    cm_path=args.eval/"confusion.csv"
    require(er["complete"] and er["training_manifest_sha256"]==sha(mp) and
            er["files"]["confusion.csv"]==sha(cm_path),"Mismatch against frozen evaluation")
    with cm_path.open(newline="",encoding="utf8") as f:
        cms={(r["phase"],r["arm"],int(r["seed"]),r["truth"]):[int(r[c]) for c in study.CLASSES] for r in csv.DictReader(f)}
    files={(r["model"],r["seed"],r["role"]):r for r in json.loads(mp.read_text())["files"]}
    output=[]
    for role,partition,phase in (("tune","selection","selection"),("audit","audit","audit_exploratory")):
        full=[r for r in split if r["role"]==role]
        include=np.array([r["partition"]==partition for r in full])
        truth=np.array([study.CLASSES.index(r["truth"]) for r in full])[include]
        for seed in study.SEEDS:
            predictions={}
            for arm in ("baseline","real_repeat","synthetic"):
                fs=files[arm,seed,role]
                header,rows=study.read_table(study.checked_file(prob,fs))
                byid={r["source_row"]:r for r in rows}
                require(header==study.HEAD and len(byid)==len(rows)==len(full)==fs["rows"] and
                        set(byid)=={r["source_row"] for r in full}, "IDs or columns changed")
                aligned=[byid[r["source_row"]] for r in full]
                require(all(t["truth"]==r["truth"] for t,r in zip(full,aligned)), "Labels disagree")
                pred=np.argmax([[float(r[c]) for c in study.CLASSES] for r in aligned],axis=1)[include]
                require(study.confusion(truth,pred).tolist()==[cms[phase,arm,seed,c] for c in study.CLASSES],
                        "Marginals differ from frozen evaluation")
                predictions[arm]=pred
            for arm in ("real_repeat","synthetic"):
                fine,coarse,hits=transition(truth,predictions["baseline"],predictions[arm])
                output.append(dict(phase=phase,seed=seed,candidate=arm,normal_transition9=fine.tolist(),
                    normal_transition3=coarse.tolist(),scratch_hits={"both_missed":int(hits[0,0]),
                    "newly_hit":int(hits[0,1]),"lost_hit":int(hits[1,0]),"both_hit":int(hits[1,1])}))
    result=dict(complete=True,train_manifest_sha256=sha(mp),eval_receipt_sha256=sha(receipt),
        script_sha256=sha(Path(__file__)),exploratory=True,official_test_used=False,
        matrix_orientation="rows: baseline A argmax, columns: candidate argmax, true normal wafers only",
        keys9=study.CLASSES,keys3=["normal","Scratch","other_defect"],records=output)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"paired_transitions.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("Verified per-wafer paired transitions against every evaluation confusion matrix.",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","training","eval","out"):
        p.add_argument("--"+name,required=True,type=Path)
    run(p.parse_args())
