"""Rebuild the frozen RF77 decisions, then settle only the A/B disagreement.

Real data run on hosted Colab CPU. Use --self-test for a synthetic-only check.
The two input ZIPs are the private, hash-verified 07 outputs and inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

COMMIT = "4b2f0123ea429fa7b1b5e5e2139148dc789c3b2d"
PRIVATE = "3d0660a6a956762d979cff978f7620642d2da08edb074cbf595e0a18b6b4c56b"
INPUTS = "54a2a6269136af87939eb553e2805f18f8443bb2a91d9814066bf9a5dec4b421"
SEEDS = (42, 123, 2026)
POLICY = {"A": "any_defect_fixed", "B": "benefit_positive_cap"}
BOUNDS = [0, .01, .05, .10, .25, .50, .75, 1]
BINS = ["0-.01", ".01-.05", ".05-.10", ".10-.25", ".25-.50", ".50-.75", ".75-1"]
SUMCOL = ["n", "sum_p_scratch", "sum_p_any", "sum_raw_p_scratch", "sum_raw_p_any"]
KEYS = ["seed", "side", "size_support"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_out(frame, path):
    frame.to_csv(path, index=False, float_format="%.17g")


def extract_needed(path, names, root):
    with zipfile.ZipFile(path) as z:
        for n in names:
            if z.namelist().count(n) != 1:
                raise ValueError(f"Missing or duplicate archived member {n}")
            dest = root/n
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))


def membership(d, ids, lots, p):
    """Same numeric algorithms, queue boundary, seed, ties and cutoff as 04/07.

    IDs/lot/scores are the only inputs. Labels and shape groups are joined later.
    """
    order, chunks, inventory, _, _, tie = d._plans(ids, lots, d.SEED)
    x, ordered_ids = p[order], ids[order]
    scenario = d.SCENARIOS["scratch_5x"]
    loss = np.asarray(scenario.missed_loss)
    efficacy = np.asarray(scenario.effect)
    harm = np.asarray(scenario.review_harm)
    value = x @ (loss * efficacy - harm) - scenario.review_fee
    a = np.zeros(len(p), dtype=bool)
    b = a.copy()
    a[order] = d._top(1-x[:, -1], ordered_ids, tie, chunks)
    b[order] = d._top(value, ordered_ids, tie, chunks, positive=True)
    queue = np.empty(len(p), dtype=int)
    for j, (lo, hi) in enumerate(chunks):
        queue[order[lo:hi]] = j
    return a, b, queue, inventory


def seed_cost(seed, name, truth, gate, selected, observed):
    """All-arrival total, with three unavoided-loss categories and review fees."""
    loss = np.where(truth == "Scratch", 5., np.where(truth == "none", 0., 1.))
    masks = {"upstream": ~gate, "capacity": gate & ~selected}
    result = dict(seed=seed, arm=name, arrivals=len(truth), candidates=int(gate.sum()),
                  reviewed=int(selected.sum()), reviewed_normal=int((selected & (truth=="none")).sum()),
                  reviewed_scratch=int((selected & (truth=="Scratch")).sum()))
    for key, mask in masks.items():
        result[key+"_missed_scratch"] = int((mask & (truth=="Scratch")).sum())
        result[key+"_missed_other"] = int((mask & (truth!="none") & (truth!="Scratch")).sum())
        result[key+"_missed_loss"] = float(loss[mask].sum())
    result["missed_scratch"] = result["capacity_missed_scratch"] + result["upstream_missed_scratch"]
    result["missed_other"] = result["capacity_missed_other"] + result["upstream_missed_other"]
    result["review_residual_loss"] = .1 * loss[selected].sum()
    result["review_fees"] = .1 * result["reviewed"]
    result["total_cost"] = sum(result[k] for k in ("upstream_missed_loss","capacity_missed_loss",
                                                  "review_residual_loss","review_fees"))
    result["cost_per_1000_arrivals"] = 1000*result["total_cost"]/len(truth)
    if observed is not None:
        require((result["reviewed"], result["reviewed_normal"], result["reviewed_scratch"]) ==
                (observed.reviewed, observed.normal_reviewed, observed.scratch_reviewed), "07 arm counts")
        for k in ("total_cost","upstream_missed_loss","cost_per_1000_arrivals"):
            require(abs(result[k]-getattr(observed,k)) < 1e-7, "07 arm accounting mismatch: "+k)
    return result


def exchange_summary(x, accounts, seed, n, candidates, common):
    a, b = (x.loc[x.side==side] for side in ("A_only","B_only"))
    def diff(label):
        return int(b.truth.eq(label).sum() - a.truth.eq(label).sum())
    ds, dn, dnum = diff("Scratch"), diff("none"), len(b)-len(a)
    other = dnum-ds-dn
    realized = .1*dnum-.9*(other+5*ds)
    predicted_other = (b.p_any-b.p_scratch).sum() - (a.p_any-a.p_scratch).sum()
    predicted_scratch = b.p_scratch.sum()-a.p_scratch.sum()
    predicted = .1*dnum-.9*(predicted_other+5*predicted_scratch)
    require(abs(realized-(accounts[1]["total_cost"]-accounts[0]["total_cost"]))<1e-6,
            "XOR and all-arrival cost contrasts disagree")
    return dict(seed=seed,arrivals=n,candidates=candidates,a_reviewed=accounts[0]["reviewed"],
                b_reviewed=accounts[1]["reviewed"],both_selected=common,a_only=len(a),b_only=len(b),
                delta_reviewed_b_a=dnum,delta_normal_b_a=dn,delta_scratch_b_a=ds,delta_other_defects_b_a=other,
                a_total_cost=accounts[0]["total_cost"],b_total_cost=accounts[1]["total_cost"],
                observed_cost_gap_b_a=realized,predicted_cost_gap_b_a=predicted,
                observed_cost_gap_per_1000_arrivals=1000*realized/n,
                predicted_cost_gap_per_1000_arrivals=1000*predicted/n,
                delta_predicted_scratch_b_a=predicted_scratch,delta_predicted_other_b_a=predicted_other,
                fixed_lists_break_even_scratch_weight=(.1*dnum/.9-other)/ds if ds else None,
                b_cheaper_condition=("weight>break_even" if ds>0 else "weight<break_even" if ds<0
                                     else "all_weights" if realized<0 else "equal_at_all_weights" if realized==0
                                     else "no_weight"))


def aggregate(x):
    x=x.assign(n=1,sum_p_scratch=x.p_scratch,sum_p_any=x.p_any,
               sum_raw_p_scratch=x.raw_p_scratch,sum_raw_p_any=x.raw_p_any)
    cells=x.groupby(KEYS+["truth","scratch_bin"],as_index=False,sort=True)[SUMCOL].sum()
    sql=cells.groupby(KEYS+["truth"],as_index=False,sort=True)[SUMCOL].sum()
    cal=cells.assign(scratch_count=np.where(cells.truth=="Scratch",cells.n,0)).groupby(
        KEYS+["scratch_bin"],as_index=False,sort=True)[["n","scratch_count","sum_p_scratch","sum_p_any"]].sum()
    sizes=cal.groupby(KEYS,as_index=False,sort=True)[["n","scratch_count","sum_p_scratch","sum_p_any"]].sum()
    for f in (cal,sizes):
        f["mean_p_scratch"]=f.sum_p_scratch/f.n
        f["observed_scratch_rate"]=f.scratch_count/f.n
        f["predicted_minus_observed_rate"]=f.mean_p_scratch-f.observed_scratch_rate
    return dict(exchange_cells=cells,exchange_sql_cells=sql,reliability_cells=cal,
                exchange_size_calibration=sizes)


def run(args, dest, work):
    repo=args.repo.resolve(); rd=repo/"cloud_review/roadmap_07"; results=rd/"results"
    require(subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==COMMIT,
            "Use the pinned 07 project checkout")
    subprocess.run(["git","-C",str(repo),"diff","--quiet","HEAD","--","cloud_review/roadmap_07",
                    "data-reference/development_reference.csv.gz"],check=True)
    sys.path.insert(0,str(rd))
    import replay_target as replay
    d=replay.load_decision(rd/"decision.py")
    require(digest(args.inputs)==INPUTS and digest(args.evidence)==PRIVATE, "07 input/evidence ZIP digest")
    extract_needed(args.evidence,["scores/score_receipt.json","scores/test_meta.csv.gz",
                  "scores/anchor_rf71_s42.npz",*(f"scores/scores_s{s}.npz" for s in SEEDS)],work)
    extract_needed(args.inputs,["inputs/lot_components.csv.gz","previous/cost/protocol.json",
                                "previous/cost/tune_calibration_freeze.json"],work)
    score=work/"scores"
    require(replay.read_receipt(score,d.CLASS_ORDER)==digest(results/"score_receipt.json"),"07 scoring receipt")
    opt=argparse.Namespace(repo=repo,input=score,lot_components=work/"inputs/lot_components.csv.gz",
                           cost=work/"previous/cost")
    meta,pops,target,_,gate,calibrated=replay.load_inputs(opt,d)
    require(pops["original_full"].all() and int(gate.sum())==49229,"07 original cohort/gate")
    ref=pd.read_csv(repo/"data-reference/development_reference.csv.gz",usecols=["role","label","height","width"])
    train=ref[ref.role=="train"]
    normal=train[train.label=="none"]
    require(len(train)==31457 and len(normal)==20780,"Training-normal size support")
    counts=normal.groupby(["height","width"]).size().to_dict()
    ci=meta.loc[gate,["source_row","lot","truth","height","width"]].copy().reset_index(drop=True)
    ci["target34"]=target[gate].astype(int)
    ci["train_normal_count"]=[counts.get((h,w),0) for h,w in zip(ci.height,ci.width)]
    ci["size_support"]=np.where(ci.train_normal_count==0,"0",np.where(ci.train_normal_count<25,"1-24","25+"))
    require(not ((ci.target34==1)&(ci.train_normal_count>0)).any(),"Target-34 definition drift")
    yy=meta.truth.to_numpy()
    ids=ci.source_row.astype(str).to_numpy(); lots=ci.lot.astype(str).to_numpy()

    policy=pd.read_csv(results/"policy_costs.csv").query("population=='original_full' and model=='RF77' and scenario=='scratch_5x'").set_index(["seed","policy"])
    quotas=pd.read_csv(results/"queue_usage.csv.gz").query("population=='original_full' and model=='RF77' and scenario=='scratch_5x'").set_index(["seed","policy","queue"])
    target07=pd.read_csv(results/"target_occupancy.csv").query("population=='original_full' and model=='RF77' and scenario=='scratch_5x' and subgroup=='target_normal'").set_index(["seed","policy"])
    public=dest/"public"; private=dest/"private";public.mkdir();private.mkdir()
    panels=[]; account_rows=[]; union_rows=[]; detail=[]; candidate_ids=ci.source_row.to_numpy()
    for seed in SEEDS:
        p=calibrated["RF77",seed]
        with np.load(score/f"scores_s{seed}.npz",allow_pickle=False) as z:
            require(np.array_equal(z["source_row"],candidate_ids),"Candidate score order changed")
            raw=z["RF77"]
        a,b,queue,inventory=membership(d,ids,lots,p)
        accounts=[]
        for name, m in (("A",a),("B",b)):
            key=(seed,POLICY[name]); old=policy.loc[key]
            require(len(inventory)==50 and sum(q["capacity"] for q in inventory)==9845,"Historical quota")
            used=np.bincount(queue[m],minlength=len(inventory))
            require(np.array_equal(used,quotas.loc[key].sort_index().reviewed.to_numpy()),"07 per-queue selections")
            normal34=(ci.target34.to_numpy()==1)&ci.truth.eq("none").to_numpy()
            require(int((m&normal34).sum())==int(target07.loc[key].reviewed),"07 target occupancy")
            selected=np.zeros(len(yy),bool);selected[gate]=m
            accounts.append(seed_cost(seed,name,yy,gate,selected,old))
        account_rows.extend(accounts)
        f=ci.assign(seed=seed,queue=queue,raw_p_scratch=raw[:,7],raw_p_any=1-raw[:,8],
                    p_scratch=p[:,7],p_any=1-p[:,8],in_a=a.astype(int),in_b=b.astype(int))
        union_rows.append(f.loc[a|b,["seed","source_row","queue","truth","height","width","train_normal_count",
                                     "p_scratch","p_any","in_a","in_b"]])
        ex=f.loc[a^b].copy()
        ex["side"]=np.where(ex.in_a==1,"A_only","B_only")
        ex["scratch_bin"]=np.asarray(BINS)[np.searchsorted(BOUNDS[1:-1],ex.p_scratch.to_numpy(),side="right")]
        summary=exchange_summary(ex,accounts,seed,len(meta),len(ci),int((a&b).sum()))
        if seed==42:
            require((summary["delta_scratch_b_a"],summary["delta_normal_b_a"],summary["delta_other_defects_b_a"])==
                    (9,3352,-3361),"Historical seed-42 invariant")
        detail.append(ex);panels.append(summary)

    combined=pd.concat(detail,ignore_index=True)
    tables=aggregate(combined)
    tables["exchange_seed_summary"]=pd.DataFrame(panels)
    tables["arm_cost_components"]=pd.DataFrame(account_rows)
    tables["target34_exchange"]=combined.groupby(["seed","side","target34","truth"],as_index=False).agg(
                n=("source_row","size"),sum_p_scratch=("p_scratch","sum"),sum_p_any=("p_any","sum"))
    for name,table in tables.items(): csv_out(table,public/(name+".csv"))
    csv_out(pd.concat(union_rows,ignore_index=True),private/"selected_union.csv.gz")
    csv_out(combined,private/"exclusive_wafers.csv.gz")
    receipt=dict(status="complete",repo_pin=COMMIT,script_sha256=digest(Path(__file__)),private07_sha256=PRIVATE,
                 inputs07_sha256=INPUTS,score_receipt_sha256=digest(score/"score_receipt.json"),
                 calibration_sha256=replay.FREEZE,decision_sha256=replay.DECISION,
                 development_reference_sha256=replay.REFERENCE,model_seeds=list(SEEDS),
                 original_full_arrivals=len(yy),candidates=len(ci),lots=int(meta.lot.nunique()),capacity=9845,
                 ranking_A=POLICY["A"],ranking_B=POLICY["B"],strata="normal count at exact ordered height,width in train only (31,457 rows; 20,780 normals): 0 / 1-24 / 25+",
                 target34="Separate historical test-normal geometry target absent from all development normals; never used to rank",
                 diagnostics="Every score-table diagnostic is conditional on XOR of selected A and B lists for each seed. Joint selections cancel. Bins are left-closed, final upper bound inclusive. Repeated seeds reuse the same arrivals.",
                 cost_gap="B minus A. Price sensitivity holds memberships fixed. True Scratch/other reviewed deltas price observed losses; this does not identify factory prices or establish global calibration.",
                 historical_reconciliation="Candidate gate, both policies' per-queue counts, target-normal occupancy, Scratch/normal counts, upstream losses and total costs reproduced for each seed.",
                 outputs={str(f.relative_to(dest)):digest(f) for f in sorted(dest.rglob("*")) if f.is_file()})
    (public/"exchange_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({"status":"complete","seeds":list(SEEDS),"exclusive_rows":len(combined),
                      "union_rows":sum(map(len,union_rows)),"aggregates":len(tables)}))


def self_test():
    # At equal review counts, choosing an extra Scratch but losing three other defects
    # becomes advantageous only above the other-defect loss ratio of three.
    x=pd.DataFrame({"side":["A_only"]*3+["B_only"]*3,"truth":["Center"]*3+["Scratch","none","none"],
                    "p_scratch":[.01]*3+[.9]*3,"p_any":[.95]*3+[.94]*3})
    accounts=[{"reviewed":4,"total_cost":10.},{"reviewed":4,"total_cost":8.2}]
    s=exchange_summary(x,accounts,42,12,8,1)
    require(np.isclose(s["observed_cost_gap_b_a"],-1.8) and s["fixed_lists_break_even_scratch_weight"]==3,"fixed-list sensitivity")
    x=x.assign(seed=42,size_support="0",scratch_bin=np.where(x.side=="A_only",".01-.05",".75-1"),
               raw_p_scratch=x.p_scratch,raw_p_any=x.p_any)
    out=aggregate(x)
    require(out["exchange_cells"].n.sum()==6 and out["reliability_cells"].scratch_count.sum()==1,"aggregate bins")
    y=np.array(["none","Center","Scratch","Scratch"])
    a=seed_cost(42,"A",y,np.array([True,True,True,False]),np.array([True,False,True,False]),None)
    require(np.isclose(a["total_cost"],6.7) and a["upstream_missed_scratch"]==1 and a["capacity_missed_other"]==1,"cost components")
    print("PASS synthetic cancellation, calibration groups, break-even and whole-stream costing")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","evidence","inputs","out"): parser.add_argument("--"+k,type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: return self_test()
    require(all([args.repo,args.evidence,args.inputs,args.out]),"Provide --repo --evidence --inputs --out")
    require(platform.system()=="Linux" and Path("/content").is_dir() and any(k.startswith("COLAB_") for k in os.environ),
            "Run real-data derivation on hosted Colab")
    require(not args.out.exists(),"Output must be fresh")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        work=Path(tmp);dest=work/"result";dest.mkdir()
        run(args,dest,work)
        dest.rename(args.out)


if __name__=="__main__":
    main()
