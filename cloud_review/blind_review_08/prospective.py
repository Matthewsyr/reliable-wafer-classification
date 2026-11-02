"""Register an unseen cohort, freeze blinded packets, then settle A/B costs. See PROTOCOL.zh-CN.md."""
import argparse
import csv
import gzip
import hashlib
import hmac
import importlib.util
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import secrets
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = REPO / "cloud_review/roadmap_07"
POLICY = HERE / "frozen_policy.json"
CFG = json.loads(POLICY.read_text(encoding="utf8"))
CLASSES = CFG["class_order"]
ARRIVALS = ["wafer_id", "lot", "arrival_utc", "map_file"]
PRIOR = ["wafer_id", "lot", "map_sha256"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    require(sha(path) == expected, f"Checksum mismatch: {path.name}")
    return path


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decisions():
    return module_file("wm08_decision", checked(OLD / "decision.py",
        "7a89039475d80a9417407cd4d1c63ddeebda85ea9217518439448185c962164c"))


@contextmanager
def atomic_dir(path):
    require(not path.exists(), f"Choose a new output directory: {path}")
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp) / "stage"
        stage.mkdir()
        yield stage
        stage.rename(path)


def rows(path, header):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == header, f"{path.name}: expected only {header}")
        for row in reader:
            require(None not in row and all(row.get(k) and row[k] == row[k].strip() for k in header),
                    f"{path.name}: blank, malformed, or padded field")
            yield row


def save_csv(path, header, data):
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(data)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf8")


def utc(s):
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    require(d.tzinfo is not None, "Timestamp needs an explicit UTC offset")
    return d.astimezone(timezone.utc)


def register(a):
    now = datetime.now(timezone.utc)
    start = utc(a.start)
    require(start > now and a.budget > 0 and a.site.strip(), "Specify a future start, one site/stream and positive audit budget")
    hist = json.loads((a.registry/"receipt.json").read_text(encoding="utf8"))
    require(hist["original_rows"] == 811457 and hist["original_raw_sha256"] ==
            "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65", "Incomplete source registry")
    checked(a.registry/"prior.csv.gz", hist["registry_sha256"])
    plan = dict(site=a.site, registered=now.isoformat(), start=start.isoformat(),
                end=(start+timedelta(days=CFG["cohort_days_utc"])).isoformat(),
                budget_cases=a.budget, policy_sha256=sha(POLICY), program_sha256=sha(Path(__file__)),
                registry=hist, random_key=secrets.token_hex(32))
    with atomic_dir(a.out) as tmp:
        save_json(tmp/"registration.json", plan)
    print("Archive private registration; distribute this commitment:", sha(a.out/"registration.json"))


def image(row, base):
    file = (base / row["map_file"]).resolve()
    require(file.is_relative_to(base) and file.is_file(), "Image outside private input directory or absent")
    x = np.load(file, allow_pickle=False)
    require(x.ndim == 2 and x.size and x.dtype.kind in "iuf" and np.isin(x, (0,1,2)).all()
            and np.any(x != 0), "Invalid original categorical wafer")
    x = np.ascontiguousarray(x, dtype=np.uint8)
    h = hashlib.sha256(np.asarray(x.shape, dtype="<i8").tobytes())
    h.update(x.tobytes(order="C"))
    return x, h.hexdigest()


def independent(records, historical):
    """Every historical node is protected. Propagate prospective lot/hash links to closure."""
    parent = {r["lot"]:r["lot"] for r in records}

    def find(lot):
        while parent[lot] != lot:
            parent[lot] = parent[parent[lot]]
            lot = parent[lot]
        return lot

    hashes, ids = {}, {}
    for r in records:
        lot, h = r["lot"], r["map_sha256"]
        if h in hashes:
            a, b = sorted((find(lot), find(hashes[h])))
            parent[b] = a
        else:
            hashes[h] = lot
        ids[r["wafer_id"]] = lot
    blocked = set()
    for old in historical:
        for lot in (old["lot"], hashes.get(old["map_sha256"]), ids.get(old["wafer_id"])):
            if lot in parent:
                blocked.add(find(lot))
    for r in records:
        r["component"] = find(r["lot"])
    return [r for r in records if r["component"] not in blocked]


def scaled(p, beta):
    require(p.ndim==2 and p.shape[1]==9 and np.isfinite(p).all() and (p>=0).all()
            and (p<=1).all() and np.allclose(p.sum(1), 1, rtol=0, atol=1e-10), "Invalid raw class probabilities")
    z = np.log(np.maximum(p.astype(np.float64), CFG["probability_floor"])) * beta
    z = np.exp(z-z.max(1,keepdims=True))
    return z/z.sum(1,keepdims=True)


def extractors():
    manifest = checked(REPO/"src/waferdefect/provenance.json",
        "6a74ec2d8588c2d92179552ff024c7f089ea98acab23465e5216f1c664830372")
    frozen = {r["packaged_path"]:r["sha256"] for r in json.loads(manifest.read_bytes())["files"]}
    source = "src/waferdefect/features.py"
    features = module_file("wm08_features", checked(REPO/source, frozen[source]))
    strip = module_file("wm08_strip", checked(OLD/"strip_portability.py",
        "c1f471b0769cbd78f6d7b534da7ebe333dd7b0d64389b9dbdbaa00ca18ea870e"))
    require(strip.GEOMETRY == frozen["src/waferdefect/geometry.py"], "Geometry source references disagree")
    geometry = strip.windows_extractor(REPO, OLD/"windows_strip_trig.json")
    return features.extract_features, geometry


def infer(records, base, models):
    import sklearn
    import joblib
    require(sklearn.__version__ == "1.6.1", "Use the original scikit-learn 1.6.1")
    base_features, geometry = extractors()
    forests = []
    for key, n in (("upstream",71), ("ranker",77)):
        cfg = CFG[key]
        m = joblib.load(checked(models/(cfg["model"]+".joblib"), cfg["sha256"]))
        require(m.n_features_in_ == n and np.array_equal(m.classes_, np.arange(9)), "Unsupported model classes")
        m.n_jobs = 1
        forests.append(m)
    gate_batches, prob_batches = [], []
    for i in range(0,len(records),256):
        batch = records[i:i+256]
        images = []
        for r in batch:
            x, hh = image(r,base)
            require(hh == r["map_sha256"], "Image changed after enrollment screening")
            images.append(x)
        x = np.stack([base_features(im) for im in images])
        gate = 1-scaled(forests[0].predict_proba(x),CFG["upstream"]["inverse_temperature"])[:,-1] >= 0.5
        gate_batches.append(gate)
        if gate.any():
            extra = np.stack([geometry(im) for im, include in zip(images,gate) if include])
            prob_batches.append(scaled(forests[1].predict_proba(np.column_stack((x[gate],extra))),
                                       CFG["ranker"]["inverse_temperature"]))
    return np.concatenate(gate_batches), np.concatenate(prob_batches) if prob_batches else np.empty((0,9))


def token(key, domain, value):
    return hmac.new(key, (domain+"|"+value).encode("utf8"), hashlib.sha256).hexdigest()[:32]


def allocate(records, gate, p, key):
    d = decisions()
    require((d.QUEUE_SIZE,d.MAX_REVIEW,d.SEED) == (CFG["queue"]["alerted"],CFG["queue"]["maximum_reviews"],
            CFG["queue"]["tie_seed"]), "Queue definition drift")
    ids = np.array([r["wafer_id"] for r in records], dtype=str)[gate]
    chunks = [(i,min(i+d.QUEUE_SIZE,len(ids))) for i in range(0,len(ids),d.QUEUE_SIZE)]
    tie = d._keys(ids, d.SEED, "common-tie-break")
    av = d._top(1-p[:,-1],ids,tie,chunks)
    sc = d.SCENARIOS["scratch_5x"]
    value = p @ (np.array(sc.missed_loss)*np.array(sc.effect)) - sc.review_fee
    bv = d._top(value,ids,tie,chunks,positive=True)
    strata = defaultdict(list)
    j = 0
    for r,flagged in zip(records,gate):
        a = bool(flagged and av[j])
        b = bool(flagged and bv[j])
        stratum = "both" if a and b else "A_only" if a else "B_only" if b else "alert_neither" if flagged else "off"
        r.update(case_id=token(key,"case",r["wafer_id"]),queue=j//d.QUEUE_SIZE if flagged else -1,
                 gate=int(flagged),a=int(a),b=int(b),stratum=stratum)
        strata[r["lot"],stratum].append(r)
        j += int(flagged)
    for (_,name), members in strata.items():
        n = len(members)
        design = CFG["sampling"]
        rate = design["unflagged"] if name=="off" else design["alerted_selected_by_neither"] if name=="alert_neither" else design["selected_by_either"]
        k = min(n,max(2,math.ceil(n*rate)))
        members.sort(key=lambda r: token(key,"audit-sample",r["wafer_id"]))
        for i,r in enumerate(members):
            r.update(pi=k/n, drawn=int(i<k))
    require(len({r["case_id"] for r in records}) == len(records), "Case ID collision")


LEDGER = ["wafer_id","lot","component","arrival_utc","map_sha256","case_id","queue","gate","a","b","stratum","pi","drawn"]


def enroll(a):
    plan = json.loads(checked(a.plan, a.plan_sha).read_bytes())
    checked(POLICY, plan["policy_sha256"])
    checked(Path(__file__), plan["program_sha256"])
    checked(a.registry/"prior.csv.gz", plan["registry"]["registry_sha256"])
    require(datetime.now(timezone.utc) >= utc(plan["end"]), "The fixed observation window has not closed")
    original = list(rows(a.arrivals, ARRIVALS))
    require(len(original) == a.mes_count and a.mes_count>0 and len({r["wafer_id"] for r in original})==len(original),
            "Independent MES total mismatch, empty window or duplicate arrival ID")
    base = a.maps.resolve()
    for r in original:
        t = utc(r["arrival_utc"])
        require(utc(plan["start"]) <= t < utc(plan["end"]), "Arrival outside the prespecified cohort")
        r["arrival_utc"] = t.isoformat(timespec="microseconds")
        _, r["map_sha256"] = image(r,base)
    records = independent(original,rows(a.registry/"prior.csv.gz",PRIOR))
    require(records,"All prospective lots overlap prior exposure")
    records.sort(key=lambda r:(r["arrival_utc"],r["wafer_id"]))
    gate, p = infer(records,base,a.models)
    allocate(records,gate,p,bytes.fromhex(plan["random_key"]))
    draw = sorted((r for r in records if r["drawn"]),key=lambda r:r["case_id"])
    require(len(draw)<=plan["budget_cases"], f"Blind-label budget exceeded ({len(draw)} cases). Stop before labels.")
    with atomic_dir(a.out) as tmp:
        packet = tmp/"blind"
        packet.mkdir()
        packet_hashes = {}
        for r in draw:
            x, hh = image(r,base)
            require(hh==r["map_sha256"],"A selected original changed")
            filename=r["case_id"]+".npy"
            np.save(packet/filename,x,allow_pickle=False)
            packet_hashes[filename]=sha(packet/filename)
        save_csv(packet/"cases.csv",["case_id"],({"case_id":r["case_id"]} for r in draw))
        save_json(packet/"file_hashes.json",packet_hashes)
        save_csv(tmp/"private_ledger.csv",LEDGER,({k:r[k] for k in LEDGER} for r in records))
        save_json(tmp/"enrollment.json",dict(site=plan["site"],start=plan["start"],end=plan["end"],
                  plan_sha256=a.plan_sha,policy_sha256=sha(POLICY),
                  program_sha256=sha(Path(__file__)),arrivals_sha256=sha(a.arrivals),mes_total=a.mes_count,
                  eligible=len(records),quarantined=len(original)-len(records),sampled=len(draw),
                  status="scores, queues and sample sealed before label release",
                  file_sha256={k:sha(tmp/k) for k in ["private_ledger.csv","blind/cases.csv","blind/file_hashes.json"]}))
    print("Lock this enrollment receipt hash with the custodian before sending blind/ to readers:",sha(a.out/"enrollment.json"))


def labels(path):
    seq = list(rows(path, ["case_id","label"]))
    out = {r["case_id"]:r["label"] for r in seq}
    require(len(out)==len(seq) and all(v in CLASSES for v in out.values()), "Duplicate or invalid reader labels")
    return out


def settle_rows(records, truth):
    """Horvitz–Thompson label-dependent losses, plus known fees for every proposed review."""
    d = decisions()
    n = len(records)
    group_names = sorted({r["component"] for r in records})
    group_index = {g:i for i,g in enumerate(group_names)}
    group_n = np.zeros(len(group_names),int)
    for r in records:
        group_n[group_index[r["component"]]] += 1
    results = {}
    for scenario, scratch_loss in (("scratch_5x",5),("equal_fixed_lists",1)):
        metrics = {}
        paired = np.zeros(len(group_names))
        for which, sign in (("a",-1),("b",1)):
            reviewed=sum(int(r[which]) for r in records)
            stage = {k:{"scratch":0.0,"other_defect":0.0,"loss_units":0.0} for k in ("upstream","capacity")}
            residual=normal_reviews=0.0
            design = defaultdict(list)
            for r in records:
                sel, g = int(r[which]),group_index[r["component"]]
                paired[g] += sign*0.1*sel
                if not int(r["drawn"]):
                    continue
                pi=float(r["pi"])
                lab=truth[r["case_id"]]
                loss=scratch_loss if lab=="Scratch" else 0 if lab=="none" else 1
                v=loss*(1-.9*sel)
                paired[g]+=sign*v/pi
                design[r["lot"],r["stratum"],pi].append(v)
                if sel:
                    residual += v/pi
                    normal_reviews += int(lab=="none")/pi
                elif loss:
                    m=stage["capacity" if int(r["gate"]) else "upstream"]
                    m["scratch" if lab=="Scratch" else "other_defect"]+=1/pi
                    m["loss_units"]+=loss/pi
            var=0.0
            for (_,_,pi),vals in design.items():
                if pi < 1:
                    ns=len(vals)
                    require(ns>=2,"Unsampled stratum variance cannot be estimated")
                    var += (ns/pi)**2*(1-pi)*float(np.var(vals,ddof=1))/ns
            fee=.1*reviewed
            total=fee+residual+sum(m["loss_units"] for m in stage.values())
            metrics[which.upper()]=dict(proposed_reviews=reviewed,unreviewed=stage,
                total_unreviewed_scratch=sum(m["scratch"] for m in stage.values()),
                total_unreviewed_other_defects=sum(m["other_defect"] for m in stage.values()),
                treated_residual_loss=residual,review_fee=fee,total=total,cost_per_1000=1000*total/n,
                finite_cohort_verification_se_per_1000=1000*math.sqrt(var)/n,normal_reviewed=normal_reviews)
        delta=1000*paired.sum()/n
        require(math.isclose(delta,metrics["B"]["cost_per_1000"]-metrics["A"]["cost_per_1000"],abs_tol=1e-8),
                "Paired and absolute cost ledgers disagree")
        ci=d._cluster_intervals(group_n,paired[:,None],CFG["inference"]["replicates"],CFG["inference"]["resampling_seed"])[0]
        adequate = len(group_n)>=CFG["inference"]["minimum_clusters_for_recommendation"]
        verdict="insufficient_independent_groups" if not adequate else "favor_B" if ci[1]<0 else "favor_A" if ci[0]>0 else "inconclusive"
        results[scenario]=dict(metrics=metrics,B_minus_A_per_1000=delta,paired_cluster_interval_95=ci,
                               primary_decision=verdict if scenario=="scratch_5x" else "sensitivity_only")
    return dict(eligible_arrivals=n,eligible_lots=len({r["lot"] for r in records}),components=len(group_names),
                audited=len(truth),assumptions="Label-settled hypothetical loss units. Fixed queues and fitted models.",scenarios=results)


def settle(a):
    rec=json.loads(checked(a.enrollment/"enrollment.json",a.enrollment_sha).read_bytes())
    checked(Path(__file__),rec["program_sha256"])
    checked(POLICY,rec["policy_sha256"])
    for k,v in rec["file_sha256"].items():
        checked(a.enrollment/k,v)
    packet=a.enrollment/"blind"
    for k,v in json.loads((packet/"file_hashes.json").read_bytes()).items():
        checked(packet/k,v)
    ledger=list(rows(a.enrollment/"private_ledger.csv",LEDGER))
    sample={r["case_id"] for r in ledger if int(r["drawn"])}
    require(len(ledger)==rec["eligible"] and sample=={r["case_id"] for r in rows(a.enrollment/"blind/cases.csv",["case_id"])},
            "Sealed packet/sample mismatch")
    m1,m2=labels(a.reader1),labels(a.reader2)
    require(sample==set(m1)==set(m2),"Missing or unexpected blinded labels. Do not drop missing cases.")
    disputes={k for k in sample if m1[k]!=m2[k]}
    arb=labels(a.adjudication) if a.adjudication else {}
    require(set(arb)==disputes,"Every disagreement requires exactly one adjudication")
    result=settle_rows(ledger,{**m1,**arb})
    result.update(site=rec["site"],start=rec["start"],end=rec["end"],mes_total=rec["mes_total"],
                  quarantined=rec["quarantined"],enrollment_sha256=a.enrollment_sha,
                  plan_sha256=rec["plan_sha256"],disagreements=len(disputes),
                  reader1_sha256=sha(a.reader1),reader2_sha256=sha(a.reader2),
                  adjudication_sha256=sha(a.adjudication) if a.adjudication else None)
    with atomic_dir(a.out) as tmp:
        save_json(tmp/"settlement.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="step",required=True)
    r=subs.add_parser("register")
    for name,typ in (("registry",Path),("out",Path),("site",str),("start",str),("budget",int)):
        r.add_argument("--"+name,required=True,type=typ)
    e=subs.add_parser("enroll")
    for name,typ in (("plan",Path),("plan-sha",str),("registry",Path),("arrivals",Path),("maps",Path),
                     ("models",Path),("mes-count",int),("out",Path)):
        e.add_argument("--"+name,required=True,type=typ)
    q=subs.add_parser("settle")
    for name,typ in (("enrollment",Path),("enrollment-sha",str),("reader1",Path),("reader2",Path),("out",Path)):
        q.add_argument("--"+name,required=True,type=typ)
    q.add_argument("--adjudication",type=Path)
    a=p.parse_args()
    {"register":register,"enroll":enroll,"settle":settle}[a.step](a)
