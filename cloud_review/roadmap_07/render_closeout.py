"""Render the historical queue report from checksum-verified, already aggregated cloud outputs."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path


POPS = {"original_full":"Full original test", "no_direct_dev_duplicate":"Exclude exact development duplicates",
        "no_transitive_dev_component":"Exclude development-linked components"}
MAIN = ("original_full", "no_transitive_dev_component")
MODELS = ("RF71", "RF77", "CNN")
SEEDS = ("42", "123", "2026")
COST = "scratch_5x"
RISK = "benefit_positive_cap"
POLICIES = {"random_fixed":"Fixed random sample", "any_defect_fixed":"Any-defect probability", RISK:"Positive net benefit"}


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def text_table(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def index(rows, *fields):
    lookup = {tuple(r[k] for k in fields):r for r in rows}
    require(len(lookup)==len(rows), "Duplicate aggregate keys")
    return lookup


def integer(value):
    return f"{int(value):,}"


def decimal(value):
    x=float(value)
    return f"{0 if abs(x)<.0005 else x:,.3f}"


def ordered(values, fmt=integer):
    return " / ".join(fmt(v) for v in values)


def percent(n, d):
    return f"{100*n/d:.1f}%" if d else "No capacity"


def read_run(root):
    receipt=json.loads((root/"run_receipt.json").read_bytes())
    require(receipt.get("status")=="complete", "Replay has not completed")
    result={}
    for name in ("populations","queues","policy_costs","target_occupancy","swaps_by_class","model_pairs"):
        member=name+".csv"
        path=root/member
        data=path.read_bytes() if path.exists() else gzip.decompress((root/(member+".gz")).read_bytes())
        require(hashlib.sha256(data).hexdigest()==receipt["sha256"][member], f"Changed aggregate: {member}")
        result[name]=text_table(data)
    return result,receipt


def report(run, support, repo):
    tables, receipt=read_run(run)
    pop={r["population"]:r for r in tables["populations"]}
    require(set(pop)==set(POPS) and len(pop)==len(tables["populations"]), "Unexpected populations")
    active=[p for p in MAIN if int(pop[p]["candidates"])>0]
    k=("population","seed","model","scenario","policy")
    metrics=index(tables["policy_costs"],*k)
    target=index(tables["target_occupancy"],*k,"subgroup")
    paired=index(tables["model_pairs"],"population","seed","scenario","policy")
    swaps=index(tables["swaps_by_class"],"population","seed","scenario","policy","label")
    require(len(metrics)==receipt["panels"]==45*3*sum(int(r["candidates"])>0 for r in pop.values()),
            "Incomplete policy table")
    require(int(pop[MAIN[0]]["target_normals"])==78158 and int(pop[MAIN[0]]["arrivals"])==118595,
            "Original exposure differs from historical report")
    for policy_key,r in metrics.items():
        p=pop[r["population"]]
        cap,chosen=int(p["capacity"]),int(r["reviewed"])
        n,m=int(p["arrivals"]),int(p["candidates"])
        require(0<=chosen<=cap and m<=n, "Capacity or denominator invalid")
        require(math.isclose(float(r["candidate_cost"])+float(r["upstream_missed_loss"]),float(r["total_cost"]),
                             rel_tol=0,abs_tol=1e-5), "Source settlement incomplete")
        require(math.isclose(float(r["total_cost"])*1000/n,float(r["cost_per_1000_arrivals"]),rel_tol=0,abs_tol=1e-5)
                and math.isclose(float(r["candidate_cost"])*1000/m,float(r["cost_per_1000_candidates"]),rel_tol=0,abs_tol=1e-5),
                "Cost normalization mismatch")
        g=target[policy_key+("target_normal",)]
        require(int(g["reviewed"])<=int(r["normal_reviewed"]) and int(g["all_capacity"])==cap
                and int(g["all_used_slots"])==chosen and int(g["arrivals"])==int(p["target_normals"]),
                "Target dimensions/denominators mismatch")
    def rows(p,model,table,policy=RISK,subgroup=None):
        keys=[(p,s,model,COST,policy) for s in SEEDS]
        return [table[key+((subgroup,) if subgroup else ())] for key in keys]
    lines=["# Which wafers received the inspection slots?", ""]
    if MAIN[0] in active:
        pop0=pop[MAIN[0]]
        t=rows(MAIN[0],"RF77",target,subgroup="target_normal")[0]
        c=rows(MAIN[0],"RF77",metrics)[0]
        share=percent(int(t["reviewed"]),int(t["all_capacity"]))
        lines += [f"On the original test, 78,158 of 110,701 normal wafers (70.60%) had native dimensions absent from development normals. Call these dimensions **T**. With fivefold Scratch loss and RF77 seed 42, the positive-net-benefit policy allocated **{integer(t['reviewed'])} of {integer(t['all_capacity'])} available inspections ({share})** to normal wafers in T. It used {integer(c['reviewed'])} slots in total.", ""]
    lines += ["### Allocation to target sizes and realized cost", "",
              "Cells list seeds 42, 123 and 2026 in order. The policy reviews positive-net-benefit alerts in descending order. Scratch counts include all dimensions. Costs use observed labels and hypothetical relative units.", "",
              "| Population | Ranker | Reviewed normals in T / total capacity | True scratches reviewed | Cost per 1,000 arrivals | Cost per 1,000 alerts |",
              "|---|---|---|---|---|---|"]
    for p in active:
        for model in MODELS:
            m=rows(p,model,metrics);t=rows(p,model,target,subgroup="target_normal")
            capacity=int(pop[p]["capacity"])
            normal=" / ".join(integer(r["reviewed"]) for r in t)+f" (each out of {integer(capacity)})"
            lines.append(f"| {POPS[p]} | {model} | {normal} | {ordered(r['scratch_reviewed'] for r in m)} | {ordered((r['cost_per_1000_arrivals'] for r in m),decimal)} | {ordered((r['cost_per_1000_candidates'] for r in m),decimal)} |")
    for p in MAIN:
        if p not in active:
            lines.append(f"\n{POPS[p]} had zero alerts. Within-queue rankings and shares are undefined.")
    lines+=["", "## Shared alert pool and capacity", "",
            "RF71 seed 42 supplied the common upstream screen with a development-calibrated probability of any defect of at least 0.5. Frozen lot and wafer hashes determine the simulated order. Each queue of 1,000 alerts permits up to 200 inspections. A partial queue of n alerts permits floor(n/5). Queues are rebuilt separately for each population. These historical labels were previously examined.", "",
            "| Population | Labeled arrivals | Lots | Alerts | Available slots | Upstream missed defects | Normal arrivals in T |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for p,title in POPS.items():
        r=pop[p]
        lines.append("| "+" | ".join([title]+[integer(r[k]) for k in ("arrivals","lots","candidates","capacity","upstream_fn","target_normals")])+" |")
    lines += ["", "An unreviewed wafer incurs its class loss L. A reviewed wafer incurs 0.1 + 0.1L. Base defect loss is 1 and normal loss is 0. Scenarios assign equal losses, fivefold Scratch loss or fivefold Near-full loss. Each assumes a review fee of 0.1 and a 90% avoidable loss fraction. Whole-stream cost includes upstream misses.", "",
              "### Compare rankings at the same capacity", "",
              "RF77 is fixed below and a missed Scratch costs five times an ordinary missed defect. The random policy uses one frozen sample. Defect probability is one minus the normal probability. Net benefit weights probabilities by class loss and subtracts the review fee.", "",
              "| Population | Ranking | Slots used | True scratches reviewed | Cost per 1,000 arrivals |",
              "|---|---|---|---|---|"]
    for p in active:
        for pol,title in POLICIES.items():
            rr=rows(p,"RF77",metrics,policy=pol)
            lines.append(f"| {POPS[p]} | {title} | {ordered(r['reviewed'] for r in rr)} | {ordered(r['scratch_reviewed'] for r in rr)} | {ordered((r['cost_per_1000_arrivals'] for r in rr),decimal)} |")
    if len(active)==2:
        # Hold RF77 and the loss schedule fixed. A paired model comparison answers a different question.
        gaps = []
        for p in MAIN:
            prob, risk = rows(p,"RF77",metrics,policy="any_defect_fixed"), rows(p,"RF77",metrics)
            gaps.append([float(b["cost_per_1000_arrivals"])-float(a["cost_per_1000_arrivals"]) for a,b in zip(prob,risk)])
        if all(d>1e-8 for panel in gaps for d in panel):
            lines += ["",f"**Ranking with RF77 held fixed.** Any-defect ranking had lower realized cost in both populations for every seed. Positive-net-benefit ranking increased cost per 1,000 arrivals by {ordered(gaps[0],decimal)} units in the full population and {ordered(gaps[1],decimal)} after removing development-linked components."]
        a=metrics[MAIN[0],"42","RF77",COST,"any_defect_fixed"]
        b=metrics[MAIN[0],"42","RF77",COST,RISK]
        if a["reviewed"]==b["reviewed"]:
            sa,sb=int(a["scratch_reviewed"]),int(b["scratch_reviewed"])
            na,nb=int(a["normal_reviewed"]),int(b["normal_reviewed"])
            oa=int(a["reviewed"])-sa-na
            ob=int(b["reviewed"])-sb-nb
            # The common gate cancels and equal numbers of reviews imply equal fees.
            extra_cost=0.9*((oa-ob)-5*(sb-sa))
            require(math.isclose(extra_cost,float(b["total_cost"])-float(a["total_cost"]),rel_tol=0,abs_tol=1e-6),
                    "RF77 policy difference does not reconcile with observed class substitutions")
            if sb>sa and nb>na and oa>ob and extra_cost>0:
                lines += ["",f"Both lists reviewed {integer(a['reviewed'])} wafers in the full population at seed 42. Switching to expected net benefit increased scratches reviewed from {integer(sa)} to {integer(sb)} (+{integer(sb-sa)}), increased normal reviews from {integer(na)} to {integer(nb)} (+{integer(nb-na)}) and reduced reviews of other true defects from {integer(oa)} to {integer(ob)} (-{integer(oa-ob)}). Review fees were equal. The net substitution increased simulated loss by {extra_cost:,.1f} units under the frozen weights."]
    lines+=["","## Effect of the six RF77 geometry features", "",
            "Hold full-capacity net-benefit ranking fixed and change the model from RF71 to RF77. Positive savings per 1,000 arrivals favor RF77. Direction is called consistent only if all three point estimates are positive in both populations.", "",
            "| Scenario | Full population savings (42 / 123 / 2026) | Component-isolated savings (42 / 123 / 2026) | Positive for both populations and all seeds |",
            "|---|---|---|---|"]
    for case,title in (("equal","Equal losses"),(COST,"Scratch 5x"),("near_full_5x","Near-full 5x")):
        vv=[[float(paired[p,s,case,"benefit_fixed"]["per_1000_all_arrivals"]) for s in SEEDS] if p in active else [] for p in MAIN]
        consistent=all(len(v)==3 and all(x>1e-8 for x in v) for v in vv)
        row=[title,*[ordered(v,decimal) if v else "No candidates" for v in vv],"Yes" if consistent else "No"]
        lines.append("| "+" | ".join(row)+" |")
    lines+=["", "Entries and displacements below compare RF77 with RF71 under fivefold Scratch loss. Intervals for savings per 1,000 alerted candidates resample paired lots conditional on the models, queues and selected lists.", "",
            "| Population | Seed | Entered / displaced | Net scratches reviewed | Net normals reviewed | Savings per 1,000 alerts [95% interval] |",
            "|---|---:|---:|---:|---:|---:|"]
    for p in active:
        for s in SEEDS:
            r=paired[p,s,COST,"benefit_fixed"]
            def net(lab):
                v=swaps[p,s,COST,"benefit_fixed",lab]
                return int(v["entered"])-int(v["displaced"])
            lines.append(f"| {POPS[p]} | {s} | {integer(r['entered'])} / {integer(r['displaced'])} | {net('Scratch'):+d} | {net('none'):+d} | {decimal(r['paired_savings_per_1000'])} [{decimal(r['ci_low'])}, {decimal(r['ci_high'])}] |")
    dev=text_table((support/"relative_geometry_swaps.csv").read_bytes())
    samples=sorted([r for r in dev if r["scenario"]==COST and r["policy"]=="benefit_fixed"],key=lambda r:SEEDS.index(r["model_seed"]))
    require([r["model_seed"] for r in samples]==list(SEEDS), "Missing 04 development evidence")
    pool=json.loads((repo/"cloud_review/results/shift/pool_summary.json").read_bytes())
    pilot=json.loads((repo/"cloud_review/results/ssl_paired/pilot_summary.json").read_bytes())["primary_audit"]
    require((support/"shift_and_ssl.md").is_file(), "Missing unlabeled audit account")
    lines += ["", "## Development audit and the unlabeled-map pilot", "",
              f"On the earlier development audit, switching to RF77 under fivefold Scratch loss saved {ordered((r['savings_per_1000_all_arrivals'] for r in samples),decimal)} units per 1,000 arrivals. All seeds used the same 11,591 development audit wafers. Population differences limit comparisons with original test costs.", "",
              f"The 638,507 unlabeled wafers supplied information about dimensions, lot overlap and duplicate maps. Removing all components linked to labeled lots or maps yielded {integer(pool['unique_candidate_maps'])} unique candidate maps. A paired single-seed masked-pretraining pilot used 50,000 isolated unlabeled maps and one extra epoch before the same supervised training. Audit Scratch hits rose from {pilot['control_scratch_tp']} to {pilot['pilot_scratch_tp']}, while total normal false alarms rose from {pilot['control_normal_any_fp']} to {pilot['pilot_normal_any_fp']}. The joint criteria failed. [Unlabeled coverage and failed-die distributions](shift_and_ssl.md)", "",
              "A prospective evaluation still needs independent lots, actual alert and arrival timing, realized review effectiveness and manufacturing losses.", ""]
    return "\n".join(lines)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",type=Path,required=True,help="07 aggregate directory with run_receipt.json")
    ap.add_argument("--support",type=Path,required=True,help="Directory with aggregate 04 swap CSV and shift_and_ssl.md")
    ap.add_argument("--repo",type=Path,required=True,help="Portfolio repository root")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    text=report(args.run,args.support,args.repo)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(text,encoding="utf-8")
