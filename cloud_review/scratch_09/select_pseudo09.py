"""Conditional RF77 inference. Invoked by pseudo09.py only for qualified Scratch strata."""
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from pseudo09 import (BANDS, CLASSES, DEV, MANIFEST, RULE, STUDENT_FIELDS, band, check, digest,
                      inputs, passed, pinned, read_gz, save_student)

RAW = "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65"
PACK = "cdf38b57a875fe22d903576f660f0b4483fa4534d9908fe785908317733fad68"
FOREST = "09edefdb9ebbbc847a2817d10e826b1d999251b8d388a2010b472ac67167ffa2"
ISOLATION = "534e62fb56a4ccff0a49d5bb9ca762b48b8a9fb5013e59e19479fb71625656a5"
REGISTRY_ZIP = "7cbb78655053dfd47f72308b408063f6508dc6febaa834a02ba2d8b6c039e2ed"
REGISTRY = "fc68aaeab0844f1265ed71ceb15e93522cce8986b7bcf8db36aa3c9e8ecb1d7d"
FEATURES = "0dc546f82fe12a1ec054db26692597ae727c8195e24c428eb0b9c0afceb3e0f4"
PORTABILITY = "c1f471b0769cbd78f6d7b534da7ebe333dd7b0d64389b9dbdbaa00ca18ea870e"
TRAIN = "3982640b77227e34ebbf69854c2b61f8246840b051aa3291deab8c5b3a086661"
PILOT = "501ba3d765e41dbe06a9e7be597dd6449eac73ece0dfe00399c9238e2fc0a0b1"
DOMAIN = b"wm811k-study09-qualified-scratch-v1|"


def eligible_rules(folder):
    q = json.loads((folder/"qualification.json").read_text())
    check(q["status"] == "complete" and q["rule"] == json.loads(json.dumps(RULE))
          and q["source_sha256"] == digest(Path(__file__).with_name("pseudo09.py"))
          and q["reference_sha256"] == DEV and q["manifest_sha256"] == MANIFEST,
          "Incompatible qualification")
    frozen = json.loads(pinned(folder/"design_freeze.json",q["design_freeze_sha256"]).read_text())
    check(frozen["thresholds"] == q["design_thresholds"] and frozen["score_sha256"] == q["score_sha256"],
          "Incomplete design freeze")
    for name in ("design_grid.csv","confirmation.csv","descriptive_by_size.csv"):
        pinned(folder/name,q["artifacts"][name])
    expected = {}
    check({r["band"] for r in q["confirmed"]} == set(frozen["thresholds"]), "Missing confirmation band")
    for result in q["confirmed"]:
        size, cutoff = result["band"], result["threshold"]
        check(size in BANDS and frozen["thresholds"][size] == cutoff and result["passed"] == passed(result,True),
              "Unconfirmed rule")
        if result["passed"]:
            expected[size] = cutoff
    check(q["qualified_bands"] == expected, "Unverified student stratum")
    return expected


def attach_lots(rows, args, pool_digest):
    by_id = {r["source_row"]:r for r in rows}
    seen = set()
    def attach(records, registry=False):
        for r in records:
            key = r["wafer_id"].removeprefix("WM811K:") if registry else r["source_row"]
            if key in by_id:
                target = by_id[key]
                check(key not in seen and r["lot"] and r["map_sha256"] == target["map_sha256"],
                      "Lot/offset identity mismatch")
                if not registry:
                    check(all(r[k] == target[k] for k in ("height","width")), "Pool geometry mismatch")
                target["lot"] = r["lot"]
                seen.add(key)
    check(bool(args.pool) != bool(args.registry), "Specify one of --pool or --registry")
    if args.pool:
        attach(read_gz(pinned(args.pool,pool_digest)))
        origin = pool_digest
    else:
        with ZipFile(pinned(args.registry,REGISTRY_ZIP)) as archive:
            compressed = archive.read("prior.csv.gz")
        check(hashlib.sha256(compressed).hexdigest() == REGISTRY, "Registry member mismatch")
        with gzip.open(io.BytesIO(compressed),"rt",encoding="utf8",newline="") as stream:
            attach(csv.DictReader(stream),True)
        origin = REGISTRY
    check(seen == set(by_id), "Missing lots for frozen 50k")
    return origin


def pick(candidates, limit=80):
    candidates = sorted(candidates,key=lambda r:(hashlib.sha256(DOMAIN+bytes.fromhex(r["map_sha256"])).digest(),
                                                int(r["source_row"])))
    tally = Counter()
    selected = []
    for r in candidates:
        if len(selected) == limit:
            break
        if tally[r["lot"]] < RULE["student_max_per_lot"]:
            selected.append({name:r[name] for name in STUDENT_FIELDS})
            tally[r["lot"]] += 1
    return selected


def select(args):
    rules = eligible_rules(args.qualification)
    args.out.mkdir(parents=True, exist_ok=True)
    qualification = args.qualification/"qualification.json"
    if not rules:
        save_student(args.out,qualification,[],dict(reason="No size band passed the frozen Scratch gate."))
        return
    check(all((args.pack,args.model,args.raw)), "Missing pack, forest or raw maps")
    import joblib
    import numpy as np
    import sklearn
    check(sklearn.__version__ == "1.6.1","Frozen forest needs scikit-learn 1.6.1")
    ext, geom = args.repo/"cloud_review/extension", args.repo/"cloud_review/roadmap_07"
    pinned(ext/"ssl_pilot.py",PILOT)
    pinned(ext/"train_imbalance.py",TRAIN)
    pinned(geom/"strip_portability.py",PORTABILITY)
    pinned(args.repo/"src/waferdefect/features.py",FEATURES)
    sys.path[:0] = [str(ext),str(geom),str(args.repo/"src")]
    from ssl_pilot import validate_pack
    from strip_portability import windows_extractor
    from train_imbalance import indexed_map
    from waferdefect.features import extract_features
    strip = windows_extractor(args.repo,geom/"windows_strip_trig.json")
    forest = joblib.load(pinned(args.model,FOREST))
    check(np.array_equal(forest.classes_,np.arange(9)) and forest.n_features_in_ == 77,"Wrong classifier")
    forest.n_jobs = 1
    pinned(args.raw,RAW)
    isolation_dir = args.repo/"cloud_review/results/shift"
    isolation = json.loads(pinned(isolation_dir/"pool_receipt.json",ISOLATION).read_text())
    summary = json.loads(pinned(isolation_dir/"pool_summary.json",isolation["outputs"]["pool_summary.json"]).read_text())
    check(isolation["stage"] == "pool" and isolation["lineage"]["inputs"]["raw"] == RAW
          and summary["unique_candidate_maps"] == 434498 and summary["unique_maps_removed_by_multihop"] == 31247
          and summary["labels_assigned"] is False,"Pool isolation lineage")
    pool = validate_pack(args.pack,PACK)
    lot_origin = attach_lots(pool,args,isolation["outputs"]["ssl_pool.csv.gz"])
    tune, support, forbidden, _ = inputs(args.repo)
    for key in ("map_sha256","lot"):
        check(not any(r[key] in forbidden[key] for r in pool),"Protected development overlap: "+key)
    for r in pool:
        r["size_band"] = band(support[int(r["height"]), int(r["width"])])
    all_by_band = Counter(r["size_band"] for r in pool)
    to_score = [r for r in pool if r["size_band"] in rules]
    candidates = []
    def predict(stream, rows):
        maps = [indexed_map(stream,r) for r in rows]
        x = np.column_stack((np.stack([extract_features(a) for a in maps]),np.stack([strip(a) for a in maps])))
        p = forest.predict_proba(x)
        check(p.shape == (len(rows),9) and np.isfinite(p).all() and (p>=0).all()
              and np.allclose(p.sum(1),1,rtol=0,atol=1e-10),"Invalid RF77 scores")
        return p
    with args.raw.open("rb") as stream:
        parity = []
        for label in CLASSES:
            parity.extend(sorted((r for r in tune if r["label"] == label),key=lambda r:int(r["source_row"]))[:8])
        p = predict(stream,parity)
        known = np.array([r["prob"] for r in parity])
        error = float(np.max(np.abs(known-p)))
        check(len(parity) == 72 and error < 1e-10 and np.array_equal(known.argmax(1),p.argmax(1)),
              f"Frozen tune parity failed: {error}")
        for offset in range(0,len(to_score),128):
            batch = to_score[offset:offset+128]
            for r, pred in zip(batch,predict(stream,batch)):
                t = rules[r["size_band"]]
                if CLASSES[pred.argmax()] == "Scratch" and pred[7] >= t:
                    candidates.append(dict(r,p_scratch=float(pred[7]),threshold=t,pseudo_label="Scratch"))
    selected = pick(candidates, RULE["student_max_unique"])
    save_student(args.out,qualification,selected,dict(
        source_sha256=digest(Path(__file__)), forest_sha256=FOREST, raw_sha256=RAW,
        pack_receipt_sha256=PACK, lot_lookup_sha256=lot_origin, isolation_receipt_sha256=ISOLATION,
        development_direct_lot_or_hash_overlaps=0, all_labeled_including_test_isolation=summary["admission"],
        parity_rows=len(parity), parity_max_abs_error=error, parity_argmax_identical=True,
        sampled_unlabeled=all_by_band, scored=len(to_score), scored_by_band=Counter(r["size_band"] for r in to_score),
        accepted_before_cap=Counter(r["size_band"] for r in candidates), selected_by_band=Counter(r["size_band"] for r in selected),
        selected_lots=len({r["lot"] for r in selected}), selected_unique_maps=len({r["map_sha256"] for r in selected}),
        pick="SHA-256 seeded order, max 80 across all eligible bands, up to 2 per lot",
        unknown_ground_truth=True))
    print(json.dumps(dict(accepted=len(candidates),student_rows=len(selected))))
