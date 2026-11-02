"""No factory inputs. Test prospective blinding, leakage screening, sampling and settlement."""
import csv
import argparse
import gzip
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import prospective as v


def main(models=None):
    base_features, geometry=v.extractors()
    small=np.array([[0,1,1],[1,2,1],[1,1,2]],dtype=np.uint8)
    assert base_features(small).shape==(71,) and geometry(small).shape==(6,)
    v.checked(v.REPO/"cloud_review/extension/shift_audit_v1.py",
        "786692eaea13c1b0a6d52d821e443e00f546735aa353a24d034caec94638b7e2")
    assert tuple(v.CFG["class_order"])==tuple(v.decisions().CLASS_ORDER)
    # Private weights are checked against their archived public receipt without shipping binaries.
    evidence=json.loads((v.REPO/"cloud_review/roadmap_07/results/score_receipt.json").read_bytes())
    for role in ("upstream","ranker"):
        cfg=v.CFG[role]
        assert cfg["sha256"]==evidence["checkpoint_sha256"][cfg["model"]+".joblib"]
    if models is not None:
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve()
            rng=np.random.default_rng(817)
            imgs=[]
            for size in (16,32,48):
                yy,xx=np.ogrid[:size,:size]
                r=np.hypot(yy-(size-1)/2,xx-(size-1)/2)/(size/2)
                disk=r<1
                for failed in (disk,r<.42,abs(r-.80)<.10,abs(yy-xx)<=1,rng.random((size,size))<.45):
                    x=np.where(disk,np.where(failed,2,1),0).astype(np.uint8)
                    row={"map_file":f"sim_{len(imgs)}.npy"}
                    np.save(base/row["map_file"],x,allow_pickle=False)
                    _,row["map_sha256"]=v.image(row,base)
                    imgs.append(row)
            gate,p=v.infer(imgs,base,models)
            assert gate.any() and p.shape==(int(gate.sum()),9)
            assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-12)
            print(f"PASS frozen seed-42 forests, synthetic maps={len(imgs)}, alerted={int(gate.sum())}, nine-class scores verified.")
    rows=[dict(wafer_id=str(i),lot=lot,map_sha256=h) for i,(lot,h) in
          enumerate([("a","x"),("a","y"),("b","y"),("b","z"),("c","z"),("d","w")])]
    keep=v.independent(rows,[dict(wafer_id="old",lot="prior",map_sha256="x")])
    assert [r["lot"] for r in keep]==["d"]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        maps=root/"maps"
        maps.mkdir()
        records=[]
        start=datetime.now(timezone.utc)-timedelta(days=31)
        for i in range(1310):
            x=np.ones((4,4),dtype=np.uint8)
            for j in range(12):
                x.flat[j]=1+((i>>j)&1)
            f=f"w{i}.npy"
            np.save(maps/f,x,allow_pickle=False)
            records.append(dict(wafer_id=f"new{i:04}",lot=f"L{i//30:02}",arrival_utc=(start+timedelta(seconds=i)).isoformat(),map_file=f))
        v.save_csv(root/"arrivals.csv",v.ARRIVALS,records)
        with gzip.open(root/"prior.csv.gz","wt",newline="") as h:
            csv.writer(h).writerows([v.PRIOR,["old","past","a"*64]])
        # Explicit synthetic registration bypass: no 811,457-row file, wait period or fitted forest is needed.
        plan=dict(site="SYNTHETIC",start=start.isoformat(),end=(start+timedelta(days=30)).isoformat(),
                  program_sha256=v.sha(Path(v.__file__)),policy_sha256=v.sha(v.POLICY),budget_cases=1310,
                  registry=dict(registry_sha256=v.sha(root/"prior.csv.gz")),random_key="a1"*32)
        reg=root/"registration.json"
        v.save_json(reg,plan)
        args=SimpleNamespace(plan=reg,plan_sha=v.sha(reg),arrivals=root/"arrivals.csv",registry=root,
                             maps=maps,models=None,mes_count=1310,out=root/"enrolled")
        def fake_infer(items, *_):
            ix=np.arange(len(items))
            gate=ix>=100
            px=np.zeros((gate.sum(),9))
            fraction=(ix[gate]%80)/80
            px[:,8]=.1+.6*fraction
            px[:,7]=np.where(ix[gate]%2==1,.25,.005)
            px[:,0]=1-px[:,8]-px[:,7]
            return gate,px
        with patch.object(v,"infer",side_effect=fake_infer):
            v.enroll(args)
        packet=args.out/"blind"
        ledger=list(v.rows(args.out/"private_ledger.csv",v.LEDGER))
        chosen=[r for r in ledger if int(r["drawn"])]
        assert set(x["stratum"] for x in chosen)=={"both","A_only","B_only","alert_neither","off"}
        for r in ledger:
            assert 0<float(r["pi"])<=1 and (r["a"]==r["b"]=="0" or float(r["pi"])==1)
        for q,cap in [(0,200),(1,42)]:
            assert sum(int(r["a"]) for r in ledger if int(r["queue"])==q)==cap
            assert sum(int(r["b"]) for r in ledger if int(r["queue"])==q)<=cap
        cases={r["case_id"] for r in v.rows(packet/"cases.csv",["case_id"])}
        assert cases=={r["case_id"] for r in chosen} and len(list(packet.glob("*.npy")))==len(cases)
        assert all(len(c)==32 for c in cases)
        # A constant label within each sampling stratum lets us check HT against a known complete synthetic census.
        truth={r["case_id"]:v.CLASSES[(int(r["lot"][1:])+len(r["stratum"]))%9] for r in ledger}
        reader=[dict(case_id=c,label=truth[c]) for c in sorted(cases)]
        v.save_csv(root/"one.csv",["case_id","label"],reader)
        conflict=reader[0].copy()
        reader[0]=dict(case_id=conflict["case_id"],label="Loc" if conflict["label"]!="Loc" else "Scratch")
        v.save_csv(root/"two.csv",["case_id","label"],reader)
        a=SimpleNamespace(enrollment=args.out,enrollment_sha=v.sha(args.out/"enrollment.json"),
                          reader1=root/"one.csv",reader2=root/"two.csv",adjudication=None,out=root/"report")
        try:
            v.settle(a)
            raise AssertionError("Disagreement disappeared without adjudication")
        except ValueError as err:
            assert "disagreement" in str(err)
        a.adjudication=root/"resolver.csv"
        v.save_csv(a.adjudication,["case_id","label"],[conflict])
        v.settle(a)
        ans=json.loads((a.out/"settlement.json").read_bytes())
        for scenario,sw in (("scratch_5x",5),("equal_fixed_lists",1)):
            for policy in ("a","b"):
                exact=sum(.1*int(r[policy])+(sw if truth[r["case_id"]]=="Scratch" else int(truth[r["case_id"]]!="none"))
                          *(1-.9*int(r[policy])) for r in ledger)
                m=ans["scenarios"][scenario]["metrics"][policy.upper()]
                assert abs(m["total"]-exact)<1e-7 and m["finite_cohort_verification_se_per_1000"]<1e-10
        # Empty candidate pools and strict positive-benefit filtering keep their denominators.
        few=[dict(wafer_id=str(i),lot="z") for i in range(9)]
        p=np.zeros((9,9)); p[:,8]=1
        v.allocate(few,np.ones(9,bool),p,b"seed")
        assert sum(r["a"] for r in few)==1 and sum(r["b"] for r in few)==0
        v.allocate(few,np.zeros(9,bool),np.empty((0,9)),b"seed")
        assert all(r["queue"]==-1 and not r["a"] and not r["b"] for r in few)
        print("PASS synthetic connected-lot quarantine, five strata, blinding, 1000+tail, off-gate sampling, adjudication, two cost ledgers and zero-alert handling.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,help="Also run both original frozen forests on generated wafers")
    main(parser.parse_args().models)
