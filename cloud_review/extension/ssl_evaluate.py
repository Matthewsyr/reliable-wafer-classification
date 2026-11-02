"""Paired seed-42 SSL pilot evaluation. All decisions reproduce the frozen main study.

python ssl_evaluate.py --archive ARCHIVE --control TRAINING_ROOT --pilot SSL_GPU_OUT --out NEW_DIRECTORY
python ssl_evaluate.py --self-test
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import decision_study as d


CONTROL, PILOT, SEED = "cnn_ce", "ssl_masked_s42", 42
MAIN_SHA = "8b05f0bdcc13a20ac032e58eecd7862cba085cd56be9ec121d54cebcd40a8c67"
REL = Path("fits/cnn_ce_s42/run.json")
GUARD_FIELDS = ("scratch_tp", "normal_any_fp", "normal_scratch_fp")


def load_receipts(args, records):
    receipt_path=args.pilot/"ssl_fit.json"
    pilot=json.loads(receipt_path.read_bytes())
    d.require((pilot["status"],pilot["model"],pilot["seed"])==("complete",PILOT,SEED), "Pilot is incomplete")
    d.require(pilot["supervised_fit_receipt"]=="supervised/fits/cnn_ce_s42/run.json", "Unrecognized fine-tune receipt")
    for field in ("pool_receipt_sha256","pack_receipt_sha256","script_sha256"):
        value=pilot[field]
        d.require(isinstance(value,str) and len(value)==64 and set(value)<=set("0123456789abcdef"),f"Invalid {field}")
    budgets=[]
    for model,base,expected in ((CONTROL,args.control,pilot["control_run_sha256"]),
                                (PILOT,args.pilot/"supervised",pilot["train_run_sha256"])):
        path=base/REL
        d.require(d.digest(path)==expected,f"Training receipt mismatch for {model}")
        fit=json.loads(path.read_bytes())
        d.require(fit["model"]==CONTROL and fit["seed"]==SEED and len(fit["files"])==2,"Not a seed-42 CE fit")
        d.require({f["role"] for f in fit["files"]}=={"tune","audit"}, "Incomplete scoring roles")
        for f in fit["files"]:
            d.require(f["model"]==CONTROL and f["seed"]==SEED and f["file"]==f"cnn_ce_s42_{f['role']}.csv.gz",
                      "Unexpected fine-tune score file")
            records[model,SEED,f["role"]]=base/"probabilities",f
        budgets.append([[r[k] for k in ("epoch","steps","examples","unique_examples","class_draws","rotated")]
                        for r in fit["epochs"]])
    d.require(budgets[0]==budgets[1] and [r[0] for r in budgets[0]]==list(range(1,8)), "Supervised budget differs")
    d.require(all(steps==246 and n==unique==31457 and sum(draws)==n and rotated==0
                  for _,steps,n,unique,draws,rotated in budgets[0]), "Unexpected matched CE training budget")
    return {"pilot_receipt_sha256":d.digest(receipt_path), **{k:v for k,v in pilot.items() if k.endswith("_sha256")}}


def phase_scores(records, refs, role):
    ids=[r["source_row"] for r in refs]
    truth=[r["truth"] for r in refs]
    result={}
    for model in (d.RF,CONTROL,PILOT):
        base,spec=records[model,SEED,role]
        fields,rows=d.read_table(d.checked_file(base,spec))
        by_id={r["source_row"]:r for r in rows}
        d.require(fields==d.HEAD and len(by_id)==len(rows)==spec["rows"]==len(refs) and set(ids)==by_id.keys(),
                  f"Score identity or schema mismatch: {model} {role}")
        rows=[by_id[i] for i in ids]
        d.require([r["truth"] for r in rows]==truth,"Label alignment failure")
        p=np.array([[float(r[k]) for k in d.CLASSES] for r in rows])
        d.require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                  and np.allclose(p.sum(1),1,atol=2e-6,rtol=0),"Invalid probabilities")
        result[model,SEED]=p
    return result


def compare(a,b):
    delta={field+"_delta":int(b[field])-int(a[field]) for field in GUARD_FIELDS}
    favorable=(delta["scratch_tp_delta"],-delta["normal_any_fp_delta"],-delta["normal_scratch_fp_delta"])
    ok=all(v>=0 for v in favorable)
    return {**delta,"all_three_hold":ok,"strict_improvement":ok and any(v>0 for v in favorable)}


def guards(tables, phase):
    rows=tables["classification"]
    output=[]
    for rule in ("argmax","scratch_joint"):
        a,b=[next(r for r in rows if (r["phase"],r["model"],r["rule"])==(phase,m,rule)) for m in (CONTROL,PILOT)]
        d.require(a["scratch_n"]==b["scratch_n"] and a["normal_n"]==b["normal_n"],"Different subgroup denominators")
        out=dict(phase=phase,rule=rule,seed=SEED,scratch_n=int(a["scratch_n"]),normal_n=int(a["normal_n"]),
                 **{prefix+"_"+k:int(row[k]) for row,prefix in [(a,"control"),(b,"pilot")] for k in GUARD_FIELDS})
        output.append({**out,**compare(a,b)})
    return output


def paired_cost(refs, traces):
    _,g=np.unique([r["component"] for r in refs],return_inverse=True)
    n=np.bincount(g)
    saved=np.bincount(g, weights=traces[CONTROL,SEED]-traces[PILOT,SEED])
    rng=np.random.default_rng(d.QUEUE_SEED)
    draws=[]
    for _ in range(500):
        idx=rng.integers(len(n),size=len(n))
        draws.append(1000*saved[idx].sum()/n[idx].sum())
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(seed=SEED,scenario="scratch_3000",rule="queue_expected_value",components=len(n),reps=len(draws),
                savings_per_1000_arrivals=1000*saved.sum()/n.sum(),lo_95=lo,hi_95=hi,
                interpretation="Positive saves simulated dollars. Conditional on seed 42, fixed models, gate and queues.")


def run(args):
    hosted=(Path("/content").is_dir() and any(k.startswith("COLAB_") for k in os.environ)) or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    d.require(hosted,"Real-data SSL evaluation requires a hosted Colab or Kaggle runtime")
    d.require(d.digest(Path(d.__file__))==MAIN_SHA,"Decision protocol source mismatch")
    records,split=d.inputs(args.archive,None)
    lineage=load_receipts(args,records)
    args.out.mkdir(parents=True,exist_ok=False)
    d.write_json(args.out/"run_status.json",dict(status="running"))
    refs=[r for r in split if r["role"]=="tune"]
    cfg,p,subset=d.fit(phase_scores(records,refs,"tune"),refs)
    tables=defaultdict(list)
    d.evaluate_phase(p,subset,cfg,"selection",tables)
    val=guards(tables,"selection")
    freeze=dict(seed=SEED,primary_rule="argmax",secondary_rule="scratch_joint",
        criterion="Compare SSL with seed-42 supervised CE under each matched rule. Primary argmax must retain true Scratch hits and not increase either kind of normal false positive. Report strict improvement separately from ties; never switch the primary rule using audit.",
        scope="Single-seed exploratory pilot. The original 15-fit selection remains frozen. Shadow audit even if selection guards fail.",
        gate=dict(model=d.RF,seed=SEED,beta=d.LEGACY_BETA,p_any_cutoff=.5),component_split_salt=d.SPLIT_SALT,
        decision_source_sha256=MAIN_SHA,evaluator_sha256=d.digest(Path(__file__)),archive_manifest_sha256=d.PIN,
        selection=val,params=[dict(model=k[0],seed=k[1],**v) for k,v in cfg.items()],**lineage)
    d.write_json(args.out/"pilot_validation_freeze.json",freeze)
    print("Seed-42 pilot policies frozen; scoring the previously seen development audit.",flush=True)
    refs=[r for r in split if r["role"]=="audit"]
    # A distinct phase name leaves private per-wafer risk/membership exports disabled.
    phase="ssl_audit_exploratory"
    _,traces=d.evaluate_phase(phase_scores(records,refs,"audit"),refs,cfg,phase,tables,require_legacy=True)
    audit=guards(tables,phase)
    tables["paired_scratch_guard"]=val+audit
    tables["paired_capacity_cost"]=[paired_cost(refs,traces)]
    for name,rows in tables.items():
        tables[name]=[r for r in rows if r.get("model")!=d.RF]
    d.write_tables(args.out,tables)
    summary=dict(scope=freeze["scope"],primary_rule="argmax",primary_validation=val[0],primary_audit=audit[0],
        secondary_audit=audit[1],held_primary_in_both=val[0]["all_three_hold"] and audit[0]["all_three_hold"],
        strict_primary_in_both=val[0]["strict_improvement"] and audit[0]["strict_improvement"],
        paired_capacity_cost=tables["paired_capacity_cost"][0],
        next_step="Multi-seed replication and fresh lot-independent labels required for a generalization claim.")
    d.write_json(args.out/"pilot_summary.json",summary)
    outputs={p.name:d.digest(p) for p in sorted(args.out.iterdir()) if p.name!="run_status.json"}
    d.write_json(args.out/"run_status.json",dict(status="complete",sha256=outputs,primary_rule="argmax",seed=SEED))
    print(json.dumps({"guard_held_both":summary["held_primary_in_both"],"strict_gain_both":summary["strict_primary_in_both"]}))


def self_test():
    y=np.arange(450)%9
    refs=[dict(source_row=str(i),truth=d.CLASSES[j],lot=str(i//10),component=str(i//10),
              partition="selection" if i//10%2==0 else "calibration") for i,j in enumerate(y)]
    rng=np.random.default_rng(123)
    p=d.softmax(rng.normal(size=(450,9))+1.8*np.eye(9)[y])
    raw={(key,SEED):p for key in (d.RF,CONTROL,PILOT)}
    cfg,sel,sub=d.fit(raw,refs)
    tt=defaultdict(list)
    d.evaluate_phase(sel,sub,cfg,"selection",tt)
    _,traces=d.evaluate_phase(raw,refs,cfg,"ssl_audit_exploratory",tt)
    for phase in ("selection","ssl_audit_exploratory"):
        require_equal=guards(tt,phase)
        assert all(r["all_three_hold"] and not r["strict_improvement"] for r in require_equal)
    x=dict(scratch_tp=20,normal_any_fp=8,normal_scratch_fp=3)
    assert not compare(x,{**x,"normal_any_fp":10,"normal_scratch_fp":1})["all_three_hold"]
    assert not compare(x,{**x,"scratch_tp":19})["all_three_hold"]
    assert compare(x,{**x,"normal_scratch_fp":1})["strict_improvement"]
    assert paired_cost(refs,traces)["savings_per_1000_arrivals"]==0
    assert "review_membership" not in tt and "risk_scores" not in tt
    print("Synthetic identical-fit pair, displaced false alarms, Scratch loss, capacity and interval checks passed.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("archive","control","pilot","out"):
        p.add_argument("--"+arg,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
    elif all((a.archive,a.control,a.pilot,a.out)):
        run(a)
    else:
        p.error("Pass --archive, --control, --pilot and --out, or --self-test")


if __name__=="__main__":
    main()
