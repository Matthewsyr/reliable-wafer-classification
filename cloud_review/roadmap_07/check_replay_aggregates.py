"""Verify the 07 public aggregate bundle without wafer IDs, images or checkpoints."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


KEY = ("population","seed","model","scenario","policy")
FIXED = {"any_defect_fixed","benefit_fixed","random_fixed"}
PUBLIC = {"populations.csv","queues.csv","queue_usage.csv.gz","policy_costs.csv","target_occupancy.csv",
          "swaps_by_class.csv","model_pairs.csv","target_shapes.csv"}


def test(v, message):
    if not v:
        raise ValueError(message)


def read(root, name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(root/name, "rt",encoding="utf-8-sig",newline="") as f:
        table = csv.DictReader(f)
        test(not set(table.fieldnames) & {"source_row","lot","component","truth","wafer_index","map_sha256",
              "waferMap","raw_data_offset"},f"Private field in {name}")
        return list(table)


def near(x,y):
    return math.isclose(float(x),float(y),abs_tol=1e-6,rel_tol=1e-10)


def key(r, fields=KEY):
    return tuple(r[k] for k in fields)


def check(root, actual=True):
    pops={r["population"]:r for r in read(root,"populations.csv")}
    panels=read(root,"policy_costs.csv")
    by_key={key(r):r for r in panels}
    test(len(by_key)==len(panels),"Duplicate policy panels")
    inventory={(r["population"],int(r["queue"])):(int(r["rows"]),int(r["capacity"])) for r in read(root,"queues.csv")}
    sums=defaultdict(Counter)
    queue_keys=set()
    for r in read(root,"queue_usage.csv.gz"):
        k=key(r); q=int(r["queue"]); n=int(r["candidates"]); c=int(r["capacity"]); u=int(r["reviewed"])
        test((k,q) not in queue_keys and k in by_key,"Orphaned/repeated queue panel")
        queue_keys.add((k,q))
        test(inventory[r["population"],q]==(n,c) and 0<n<=1000 and c==n//5 and 0<=u<=c,"Exceeded queue capacity")
        if r["policy"] in FIXED: test(u==c,"Unexplained empty slot in fixed policy")
        if r["policy"]=="no_review": test(u==0,"Unexpected review")
        sums[k].update({"candidates":n,"capacity":c,"reviewed":u})
    for k,r in by_key.items():
        cohort=pops[r["population"]]
        for name in ("candidates","capacity","reviewed"):
            test(int(r[name])==sums[k][name],"Realized queues do not sum to policy")
        n,m,u,nrm,scr=map(int,(r["arrivals"],r["candidates"],r["reviewed"],r["normal_reviewed"],r["scratch_reviewed"]))
        total=float(r["total_cost"])
        test(0<=nrm<=u and 0<=scr<=u-nrm,"Impossible selected classes")
        test(near(total,float(r["candidate_cost"])+float(r["upstream_missed_loss"])) and
             near(total/n*1000,r["cost_per_1000_arrivals"]) and near(float(r["candidate_cost"])/m*1000,r["cost_per_1000_candidates"]),
             "Candidate and full denominators disagree")
        if r["scenario"] in ("equal","scratch_5x"):
            value=int(cohort["true_defects"])-.9*(u-nrm)+.1*u
            if r["scenario"]=="scratch_5x":
                value+=4*(int(cohort["true_scratch"])-.9*scr)
            test(near(value,total),"Cost does not match observed class counts")
    target={(*key(r),r["subgroup"]):r for r in read(root,"target_occupancy.csv")}
    for k,r in by_key.items():
        n=int(r["arrivals"]); m=int(r["candidates"]); u=int(r["reviewed"]); c=int(r["capacity"])
        for g in ("all","target_any","target_normal","target_scratch"):
            s=target[(*k,g)]
            na,nc,nu=map(int,(s["arrivals"],s["candidates"],s["reviewed"]))
            test(n>=na>=nc>=nu>=0 and nc<=m and nu<=u,"Impossible target group counts")
            if c: test(near(s["share_of_capacity"],nu/c),"Wrong capacity denominator")
            if u: test(near(s["share_of_used_slots"],nu/u),"Wrong used-slot denominator")
            if g=="all":
                test((na,nc,nu)==(n,m,u) and near(s["group_cost"],r["total_cost"]),"Full group mismatch")
        a,b,t=(target[(*k,g)] for g in ("target_normal","target_scratch","target_any"))
        for field in ("arrivals","candidates","reviewed"):
            test(int(a[field])+int(b[field])<=int(t[field]),"Target normal and Scratch overcount")
    swaps=read(root,"swaps_by_class.csv"); pairs=read(root,"model_pairs.csv")
    for p in pairs:
        pk=(p["population"],p["seed"],p["scenario"],p["policy"])
        ss=[x for x in swaps if key(x,("population","seed","scenario","policy"))==pk]
        test(len(ss)==9 and sum(int(x["entered"]) for x in ss)==int(p["entered"]) and
             sum(int(x["displaced"]) for x in ss)==int(p["displaced"]),"Swap label inventory mismatch")
        a,b=(by_key[(pk[0],pk[1],model,pk[2],pk[3])] for model in ("RF71","RF77"))
        dif=float(a["total_cost"])-float(b["total_cost"])
        test(near(sum(float(x["net_savings_units"]) for x in ss),dif) and
             near(dif*1000/int(a["candidates"]),p["paired_savings_per_1000"]) and
             near(dif*1000/int(a["arrivals"]),p["per_1000_all_arrivals"]),"Pairing units or swap accounting")
    if actual:
        receipt=json.loads((root/"run_receipt.json").read_text())
        test(set(receipt["sha256"]) >= PUBLIC and receipt["status"]=="complete","Incomplete receipt")
        for name in PUBLIC:
            test(hashlib.sha256((root/name).read_bytes()).hexdigest()==receipt["sha256"][name],f"Digest mismatch {name}")
        test(sorted((int(pops["original_full"]["arrivals"]),int(pops["no_direct_dev_duplicate"]["arrivals"])))==[115437,118595] and
             int(pops["original_full"]["target_normals"])==78158 and len(read(root,"target_shapes.csv"))==34 and
             int(pops["no_transitive_dev_component"]["arrivals"])<=115437, "Historical support mismatch")
        test(len(by_key)==sum(int(x["candidates"])>0 for x in pops.values())*3*3*3*5,"Missing real panel")
    print(json.dumps(dict(status="PASS",populations=len(pops),policy_panels=len(by_key),
                 individual_queue_checks=len(queue_keys),target_panels=len(target),paired_model_contrasts=len(pairs),
                 source="public aggregates only"),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    check(p.parse_args().results)
