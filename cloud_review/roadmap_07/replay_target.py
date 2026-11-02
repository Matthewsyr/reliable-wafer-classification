"""Replay the frozen 04 triage policies on the previously examined original test.

No fitting. All real-data execution must use hosted Colab or Kaggle. Public output
contains aggregate counts/costs. The separate private/ directory contains swap IDs.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


SEEDS = (42, 123, 2026)
MODELS = {"RF71": "rf", "RF77": "rf_geometry", "CNN": "cnn"}
DECISION = "7a89039475d80a9417407cd4d1c63ddeebda85ea9217518439448185c962164c"
PROTOCOL = "9e8411c52e0e11357547353a21621d956c049439daaafbf4a1b6daf3896753af"
FREEZE = "f95fde21e70b8bf0fc582c8953bedf8825994c45c47032a7debd3a90a25b4a54"
REFERENCE = "4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879"
LINKS = "ae0587ed311b19d702d5e61f200aaeae4e8e49acd47cc557cbe15f78d71600d9"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(path, expected):
    check(sha(path) == expected, f"Changed evidence: {path.name}")


def load_decision(path):
    verify(path, DECISION)
    spec = importlib.util.spec_from_file_location("recorded_04_decision", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def scaled(raw, beta, epsilon):
    # Identical arithmetic to 04 score_cloud.scaled and its calling log floor.
    p = np.asarray(raw, dtype=np.float64)
    check(p.ndim == 2 and p.shape[1] == 9 and np.isfinite(p).all() and
          (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, rtol=0, atol=2e-5),
          "Invalid raw nine-class probabilities")
    z = np.log(np.maximum(p, epsilon)) * beta
    z -= np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    return z/z.sum(axis=1, keepdims=True)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")


def ratio(a, b):
    return float(a / b) if b else None


def prepare_populations(meta, ref, links):
    check(links.lot.is_unique and set(ref.lot) <= set(links.lot) and set(meta.lot) <= set(links.lot),
          "Missing or repeated lot-component mapping")
    check(set(ref.lot).isdisjoint(meta.lot), "Direct development/test lot overlap")
    src = dict(zip(links.lot, links.component))
    dev_shapes = set(zip(ref.loc[ref.label == "none", "height"], ref.loc[ref.label == "none", "width"]))
    pairs = list(zip(meta.height, meta.width))
    target = set(p for t, p in zip(meta.truth, pairs) if t == "none") - dev_shapes
    normal = (meta.truth == "none").to_numpy()
    novelty = np.array([p in target for p in pairs])
    meta["component"] = meta.lot.map(src)
    connected_dev = frozenset(ref.lot.map(src))
    isolated = ~meta.component.isin(connected_dev).to_numpy()
    check(not np.any(isolated & meta.duplicate_dev.to_numpy()), "Duplicate survived transitive exclusion")
    masks = {"original_full": np.ones(len(meta), bool), "no_direct_dev_duplicate": ~meta.duplicate_dev.to_numpy(),
             "no_transitive_dev_component": isolated}
    return novelty, masks, sorted(target), int((normal & novelty).sum())


def read_receipt(root, class_order):
    receipt = json.loads((root/"score_receipt.json").read_text())
    check(receipt.get("status") in ("complete", "completed"), "Score stage is incomplete")
    check(tuple(receipt["class_order"]) == tuple(class_order) and receipt["dev_sha256"] == REFERENCE and
          receipt["temperature_freeze_sha256"] == FREEZE and receipt["rows"] == 118595 and
          receipt["cross_split_lots"] == 0 and receipt["duplicate_dev"] == 3158, "Score provenance mismatch")
    files = receipt.get("outputs", receipt.get("sha256", {}))
    for name in ["test_meta.csv.gz", "anchor_rf71_s42.npz", *(f"scores_s{s}.npz" for s in SEEDS)]:
        entry = files.get(name)
        digest = entry.get("sha256") if isinstance(entry, dict) else entry
        check(isinstance(digest, str) and len(digest) == 64, f"Missing score receipt hash for {name}")
        verify(root/name, digest)
    return sha(root/"score_receipt.json")


def load_inputs(args, d):
    path = args.repo/"data-reference/development_reference.csv.gz"
    verify(path, REFERENCE)
    verify(args.lot_components, LINKS)
    verify(args.cost/"protocol.json", PROTOCOL)
    verify(args.cost/"tune_calibration_freeze.json", FREEZE)
    protocol = json.loads((args.cost/"protocol.json").read_text())
    freeze = json.loads((args.cost/"tune_calibration_freeze.json").read_text())
    check(protocol["decision_sha256"] == DECISION and protocol["queue_seed"] == d.SEED and
          protocol["gate"]["min_any_defect_probability"] == .5, "Historical protocol mismatch")
    meta = pd.read_csv(args.input/"test_meta.csv.gz", dtype={"lot":str, "truth":str, "duplicate_dev":str},
                       keep_default_na=False)
    check(meta.source_row.is_unique and meta.source_row.is_monotonic_increasing and
          meta.lot.nunique() == 4953 and len(meta) == 118595 and set(meta.truth) == set(d.CLASS_ORDER),
          "Test metadata integrity")
    check(meta.duplicate_dev.str.lower().isin(["false","true","0","1"]).all(), "Invalid duplicate flags")
    meta["duplicate_dev"] = meta.duplicate_dev.str.lower().isin(["true","1"])
    ref = pd.read_csv(path, usecols=["source_row", "lot", "label", "height", "width"], dtype={"lot":str})
    check(len(ref) == 54355 and set(ref.source_row).isdisjoint(meta.source_row), "Source row leak")
    links = pd.read_csv(args.lot_components, dtype=str, keep_default_na=False)
    target, populations, shapes, n = prepare_populations(meta, ref, links)
    check(n == 78158 and (meta.truth == "none").sum() == 110701 and
          meta.duplicate_dev.sum() == 3158, "Historical reference count mismatch")
    epsilon = freeze["probability_floor_for_log"]
    config = freeze["configurations"]
    with np.load(args.input/"anchor_rf71_s42.npz", allow_pickle=False) as a:
        check(np.array_equal(a["source_row"], meta.source_row), "Unaligned full anchor")
        p = scaled(a["RF71"], config["rf_s42"]["inverse_temperature"], epsilon)
    gate = 1 - p[:,-1] >= .5
    check(gate.any(), "Empty full-test gate")
    fixed_anchor = p[gate]
    probabilities = {}
    for seed in SEEDS:
        with np.load(args.input/f"scores_s{seed}.npz", allow_pickle=False) as panels:
            check(np.array_equal(panels["source_row"], meta.source_row.to_numpy()[gate]), "Candidate ID mismatch")
            for model, alias in MODELS.items():
                x = scaled(panels[model], config[f"{alias}_s{seed}"]["inverse_temperature"], epsilon)
                check(x.shape == fixed_anchor.shape, "Candidate probability shape changed")
                if seed == 42 and model == "RF71":
                    check(np.array_equal(x, fixed_anchor), "Candidate anchor scores disagree with full anchor")
                probabilities[model, seed] = x
    return meta, populations, target, shapes, gate, probabilities


def replay(d, meta, populations, target, gate, probs, out, seeds=SEEDS, reps=500):
    rows = {k:[] for k in ("populations","queues","queue_usage","policy_costs","target_occupancy","swaps_by_class","model_pairs")}
    private = out/"private"
    private.mkdir()
    all_y = d._labels(meta.truth.to_numpy())
    ids = meta.source_row.astype(str).to_numpy()
    lots = meta.lot.to_numpy()
    with gzip.open(private/"rf77_vs_rf71_scratch5_slot_changes.csv.gz","wt",encoding="utf8",newline="") as sw:
        writer = csv.writer(sw)
        writer.writerow(["population","seed","policy","queue","source_row","lot","component","truth","height","width",
                         "target_size","change","rf71_defect_p","rf77_defect_p","rf71_scratch_p","rf77_scratch_p"])
        for population, use in populations.items():
            keep = use & gate
            n, m = int(use.sum()), int(keep.sum())
            inventory = dict(population=population, arrivals=n, lots=int(meta.lot[use].nunique()),
                             components=int(meta.component[use].nunique()), candidates=m,
                             true_defects=int((use & (all_y!=8)).sum()), true_scratch=int((use & (all_y==7)).sum()),
                             upstream_fn=int((use & ~gate & (all_y!=8)).sum()), target_arrivals=int((use & target).sum()),
                             target_normals=int((use & target & (all_y==8)).sum()), capacity=0)
            rows["populations"].append(inventory)
            if not m:
                continue
            loc = np.flatnonzero(keep)
            ci, cl = ids[keep], lots[keep]
            order, chunks, queues, _, _, _ = d._plans(ci, cl, d.SEED)
            queue = np.empty(m,int)
            for j,(a,b) in enumerate(chunks):
                queue[order[a:b]]=j
                rows["queues"].append(dict(population=population, **queues[j]))
            capacity = sum(x["capacity"] for x in queues)
            inventory["capacity"] = capacity
            addr = {i:j for j,i in enumerate(ci)}
            groups = {"all":np.ones(len(meta),bool),"target_any":target,
                      "target_normal":target & (all_y==8),"target_scratch":target & (all_y==7)}
            for seed in seeds:
                baseline = None
                for model in MODELS:
                    p = probs[model,seed][use[gate]]
                    result = d.evaluate(ci, cl, all_y[keep], p, model=model, pool_kind="simulated_flagged_candidates",
                                        seed=d.SEED, bootstrap_reps=0)
                    if model=="RF71":
                        baseline=result
                    for line in result["summary"]:
                        case, policy = line["scenario"],line["policy"]
                        sc = d.SCENARIOS[case]
                        L,e,harm = map(np.asarray,(sc.missed_loss,sc.effect,sc.review_harm))
                        index = np.fromiter((addr[i] for i in result["selected_rows"][case][policy]),dtype=int)
                        selected=np.zeros(len(meta),bool)
                        selected[loc[index]]=True
                        occupied=np.bincount(queue[index],minlength=len(chunks))
                        quotas=np.array([q["capacity"] for q in queues])
                        check((occupied<=quotas).all() and (policy not in ("random_fixed","any_defect_fixed","benefit_fixed")
                              or np.array_equal(occupied,quotas)), "Queue quota failure")
                        cost=L[all_y]*(1-e[all_y]*selected)+selected*(sc.review_fee+harm[all_y])
                        check(np.isclose(cost[keep].sum(),line["total_cost"]), "Candidate settlement mismatch")
                        key=dict(population=population,seed=seed,model=model,scenario=case,policy=policy)
                        rows["queue_usage"].extend(dict(**key,queue=q["queue"],candidates=q["rows"],
                            capacity=q["capacity"],reviewed=int(used)) for q,used in zip(queues,occupied))
                        rows["policy_costs"].append({**key,"arrivals":n,"candidates":m,"capacity":capacity,
                            "reviewed":len(index),"normal_reviewed":line["true_normal_reviewed"],
                            "scratch_reviewed":int((selected & (all_y==7)).sum()),
                            "candidate_cost":line["total_cost"],"upstream_missed_loss":float(cost[use & ~gate].sum()),
                            "total_cost":float(cost[use].sum()),"cost_per_1000_candidates":line["cost_per_1000"],
                            "cost_per_1000_arrivals":float(1000*cost[use].sum()/n)})
                        for group, g in groups.items():
                            arrived, flagged, chosen = int((use&g).sum()), int((keep&g).sum()), int((selected&g).sum())
                            rows["target_occupancy"].append(dict(**key,subgroup=group,arrivals=arrived,candidates=flagged,
                                reviewed=chosen,all_capacity=capacity,all_used_slots=len(index),
                                share_of_arrivals=ratio(arrived,n),share_of_candidates=ratio(flagged,m),
                                share_of_capacity=ratio(chosen,capacity),share_of_used_slots=ratio(chosen,len(index)),
                                within_group_review_rate=ratio(chosen,arrived),group_cost=float(cost[use&g].sum()),
                                group_candidate_cost=float(cost[keep&g].sum())))
                    if model!="RF77":
                        continue
                    for policy in ("benefit_fixed","benefit_positive_cap"):
                        for pair in d.compare_models(result,baseline,policy=policy,bootstrap_reps=reps,seed=d.SEED):
                            pair.update(population=population,seed=seed,per_1000_all_arrivals=pair["paired_savings_per_1000"]*m/n)
                            rows["model_pairs"].append(pair)
                        for swap in d.swap_by_class(result,baseline,ci,all_y[keep],policy=policy):
                            rows["swaps_by_class"].append(dict(population=population,seed=seed,**swap))
                        a=set(baseline["selected_rows"]["scratch_5x"][policy]); b=set(result["selected_rows"]["scratch_5x"][policy])
                        old=probs["RF71",seed][use[gate]]
                        for wafer in sorted(a ^ b,key=int):
                            j=addr[wafer]; i=loc[j]
                            writer.writerow([population,seed,policy,queue[j],wafer,lots[i],meta.component.iat[i],
                                d.CLASS_ORDER[all_y[i]],meta.height.iat[i],meta.width.iat[i],bool(target[i]),
                                "entered" if wafer in b else "displaced",1-old[j,8],1-p[j,8],old[j,7],p[j,7]])
                print("Finished",population,"seed",seed,flush=True)
    for name, table in rows.items():
        # Queue occupancy is recorded for every realized model/seed/strategy panel.
        filename = "queue_usage.csv.gz" if name == "queue_usage" else name+".csv"
        pd.DataFrame(table).to_csv(out/filename,index=False)
    return rows


def self_test(d):
    import tempfile
    rng=np.random.default_rng(3447)
    n=1300
    y=np.array(d.CLASS_ORDER)[np.arange(n)%9]
    m=pd.DataFrame(dict(source_row=np.arange(n),lot=np.array([str(x//9) for x in range(n)]),truth=y,
                        height=2+np.arange(n)%2,width=3,duplicate_dev=np.arange(n)<5))
    ref=pd.DataFrame(dict(lot=["D"],label=["none"],height=[2],width=[3]))
    links=pd.DataFrame(dict(lot=["D",*m.lot.unique()],component=["C",*["C" if i=="0" else i for i in m.lot.unique()]]))
    t,pops,shapes,_=prepare_populations(m,ref,links)
    check(shapes==[(3,3)] and pops["no_transitive_dev_component"].sum()==1291,"Wrong synthetic target or isolation")
    gate=np.arange(n)%5!=0
    p={key:rng.dirichlet(np.ones(9),gate.sum()) for key in [(name,42) for name in MODELS]}
    with tempfile.TemporaryDirectory() as tmp:
        r=replay(d,m,pops,t,gate,p,Path(tmp),seeds=[42],reps=12)
        check(len(r["policy_costs"])==3*3*3*5 and len(r["model_pairs"])==18,"Wrong panel counts")
        fixed=[x for x in r["policy_costs"] if x["policy"]=="benefit_fixed"]
        check(all(x["reviewed"]==x["capacity"] for x in fixed),"Quota test")
        used=pd.read_csv(Path(tmp)/"queue_usage.csv.gz")
        check(len(used)==len(r["queues"])*3*3*5 and (used.reviewed<=used.capacity).all() and
              (used.groupby(["population","model","scenario","policy"]).reviewed.sum().sort_index().to_numpy()==
               pd.DataFrame(r["policy_costs"]).set_index(["population","model","scenario","policy"]).sort_index().reviewed.to_numpy()).all(),
              "Saved slot inventory does not reconcile")
        print("PASS synthetic target sizes, three rebuilt cohorts, quotas and accounting.")


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--decision",type=Path,required=True)
    for k in ("input","repo","cost","lot-components","out"):
        a.add_argument("--"+k,type=Path)
    a.add_argument("--self-test",action="store_true")
    args=a.parse_args(); d=load_decision(args.decision)
    if args.self_test:
        self_test(d);return
    if not(platform.system()=="Linux" and (Path("/content").is_dir() and any(k.startswith("COLAB_") for k in os.environ) or
                                          Path("/kaggle/working").is_dir() and any(k.startswith("KAGGLE_") for k in os.environ))):
        a.error("Use hosted Colab/Kaggle for original data.")
    if not all((args.input,args.repo,args.cost,args.lot_components,args.out)):
        a.error("Pass input, repo, cost, lot-components, out and decision")
    receipt=read_receipt(args.input, d.CLASS_ORDER)
    meta,pops,target,shapes,gate,probs=load_inputs(args,d)
    args.out.mkdir(parents=True,exist_ok=False)
    print("Test rows",len(meta),"frozen candidates",int(gate.sum()),flush=True)
    rows=replay(d,meta,pops,target,gate,probs,args.out)
    pd.DataFrame(shapes,columns=["height","width"]).to_csv(args.out/"target_shapes.csv",index=False)
    save_json(args.out/"run_receipt.json",dict(status="complete",test_status="Retrospective replay of previously inspected test labels.",
        candidate_rule="Original 04 calibrated RF71 seed 42 P(any defect)>=0.5. No threshold or model was selected on test.",
        scope="Each retained population forms its own simulated flagged queues, cap=floor(N/5) per 1000. Upstream misses included in full-arrival costs.",
        costs="Relative hypothetical units, label-settled. Review fee .1, 90% loss avoidance, equal, Scratch 5x, or Near-full 5x.",
        model_pair_intervals="500 paired lot bootstrap samples, candidate-normalized, conditional on fitted models/gate/fixed selections. Not a new blind test.",
        target="Exact ordered (height,width) present among original-test normals and absent from the normal class within all 54,355 labeled development wafers.",
        score_receipt_sha256=receipt,protocol_sha256=PROTOCOL,calibration_sha256=FREEZE,
        decision_sha256=DECISION,development_reference_sha256=REFERENCE,lot_components_sha256=LINKS,
        source_sha256=sha(Path(__file__)),populations=rows["populations"],panels=len(rows["policy_costs"]),
        sha256={str(f.relative_to(args.out)):sha(f) for f in sorted(args.out.rglob("*")) if f.is_file()}))


if __name__=="__main__":
    main()
