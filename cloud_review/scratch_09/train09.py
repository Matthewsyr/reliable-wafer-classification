"""Frozen development-only Scratch augmentation study. Run actual data on hosted Colab GPU.

python train09.py --repo REPO --raw LSWMD.pkl --cache CACHE --out OUT
python train09.py --repo REPO --raw LSWMD.pkl --cache CACHE --out OUT --arms student --pseudo GATE_DIR
Use --self-test for synthetic-only checks. One writer per output directory.
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
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F


CORE_SHA = "3982640b77227e34ebbf69854c2b61f8246840b051aa3291deab8c5b3a086661"
ARMS = ("baseline", "real_repeat", "synthetic", "student")
SEEDS = (42, 123, 2026)
EXTRA = 80
EPOCHS, BATCH = 7, 128
SCRATCH, NORMAL = 7, 8
PSEUDO_WEIGHT = 0.25


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_core(repo):
    path = repo / "cloud_review/extension/train_imbalance.py"
    if sha_bytes(path.read_bytes()) != CORE_SHA:
        raise ValueError("Stage 05 dependency checksum mismatch")
    spec = importlib.util.spec_from_file_location("verified_imbalance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_once(core, path, payload):
    if path.exists() and core.read_json(path) != payload:
        raise ValueError(f"Incompatible existing result: {path}")
    core.atomic_json(path, payload)


def rng_for(seed, epoch, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, stream]))


def linear_trace(wafer):
    """Select an occupied narrow band of actual failed dies, keeping their gaps."""
    pts = np.argwhere(wafer == 2)
    if len(pts) < 4:
        return None
    best, result = 0., None
    for angle in np.arange(0, 180, 5):
        theta = np.deg2rad(angle)
        along = np.array([np.cos(theta), np.sin(theta)])
        across = np.array([-along[1], along[0]])
        cross = pts @ across
        bins = np.rint(cross).astype(int)
        # Three neighbouring native die pitches, with ties resolved deterministically.
        shifted = bins - bins.min() + 1
        hist = np.bincount(shifted, minlength=int(shifted.max()) + 2)
        density = hist[:-2] + hist[1:-1] + hist[2:]
        center = int(density.argmax()) + int(bins.min())
        mask = np.abs(bins - center) <= 1
        chosen = pts[mask]
        if len(chosen) < 4:
            continue
        position = chosen @ along
        # Drop remote chance alignments beyond a gap of four native pitches.
        order = np.argsort(position, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(position[order]) > 4) + 1)
        for group in groups:
            if len(group) < 4:
                continue
            span = float(np.ptp(position[group]))
            width = float(np.ptp(cross[mask][group]))
            if span < max(4, min(wafer.shape) * .15) or span < 3 * max(width, 1):
                continue
            score = len(group) * span / (1 + width)
            if score > best:
                best = score
                result = (chosen[group], {"angle_deg": int(angle), "span_native": span,
                                          "width_native": width, "trace_dies": len(group)})
    return result


def transplant(background, donor_shape, points, jitter):
    # No stretching, interpolation or invented line pixels. Only functional dies can change.
    offset = (np.array(background.shape) - donor_shape) // 2 + jitter
    projected = points + offset
    inside = ((projected >= 0) & (projected < np.array(background.shape))).all(axis=1)
    p = projected[inside]
    p = p[background[p[:, 0], p[:, 1]] != 0]
    if len(p) < .9 * len(points):
        return None
    added = p[background[p[:, 0], p[:, 1]] == 1]
    if len(added) < max(3, .5 * len(points)):
        return None
    mixed = background.copy()
    mixed[added[:, 0], added[:, 1]] = 2
    if not np.array_equal(mixed > 0, background > 0) or np.any((background == 2) & (mixed != 2)):
        raise AssertionError("Scratch synthesis violated the native wafer masks")
    return mixed, {"offset_rc": offset.tolist(), "retained_trace_dies": len(p), "added_dies": len(added),
                   "old_failed_dies": int((background == 2).sum()), "valid_dies": int((background > 0).sum())}


class SourcePool:
    def __init__(self, raw, records, y, train, core):
        self.raw, self.records, self.y, self.core = raw, records, y, core
        self.scratch = train[y[train] == SCRATCH]
        normals = train[y[train] == NORMAL]
        self.normal_shapes = {}
        for i in normals:
            row = records[i]
            self.normal_shapes.setdefault((int(row["height"]), int(row["width"])), []).append(int(i))
        self.templates, rates = {}, []
        for i in self.scratch:
            wafer = self.wafer(i)
            rates.append(float((wafer == 2).sum() / (wafer != 0).sum()))
            result = linear_trace(wafer)
            if result is None:
                continue
            shape = np.array(wafer.shape)
            compatible = [s for s in self.normal_shapes if np.all(np.array(s) >= .8 * shape)
                          and np.all(np.array(s) <= 1.25 * shape)]
            if not compatible:
                continue
            # Nearest available native geometry, exact shape if present.
            closest = min(compatible, key=lambda s: (np.abs(np.log(np.array(s) / shape)).sum(), s))
            self.templates[int(i)] = (shape, result, self.normal_shapes[closest])
        if not self.templates:
            raise ValueError("No compatible genuine Scratch traces in train")
        self.accepted = np.array(sorted(self.templates))
        self.max_fail_fraction = float(np.quantile(rates, .95))

    def wafer(self, idx):
        row = self.records[idx]
        if row["role"] != "train":
            raise ValueError("Augmentation source outside train")
        return self.core.indexed_map(self.raw, row)

    def generate(self, donor, rng):
        for attempt in range(300):
            i = donor if not attempt and donor in self.templates else int(rng.choice(self.accepted))
            shape, (points, geometry), compatible = self.templates[i]
            normal = int(rng.choice(compatible))
            background = self.wafer(normal)
            jitter = rng.integers(-2, 3, size=2)
            result = transplant(background, shape, points, jitter)
            if result is None:
                continue
            mixed, qa = result
            if (qa["old_failed_dies"] + qa["added_dies"]) / qa["valid_dies"] > self.max_fail_fraction:
                continue
            source, recipient = self.records[i], self.records[normal]
            meta = {"donor_row": int(source["source_row"]), "donor_sha256": source["map_sha256"],
                    "normal_row": int(recipient["source_row"]), "normal_sha256": recipient["map_sha256"],
                    "native_hw": list(mixed.shape), "donor_hw": shape.tolist(), "attempts": attempt + 1,
                    "synthetic_sha256": sha_bytes(np.array(mixed.shape, dtype="<i8").tobytes() + mixed.tobytes()),
                    **geometry, **qa}
            return mixed, meta, (self.wafer(i), background)
        raise ValueError("No physically admissible transplant after 300 attempts")


def epoch_plan(y, order, seed, epoch, arm, scratch, pseudo_count=0):
    n = 0 if arm == "baseline" else pseudo_count if arm == "student" else EXTRA
    # Pick the same positions for paired arms, including the m<=80 prefix for a student.
    candidates = np.flatnonzero(y[order] == NORMAL)
    slots = rng_for(seed, epoch, 91).choice(candidates, EXTRA, replace=False)[:n]
    donors = rng_for(seed, epoch, 92).choice(scratch, n, replace=True)
    return slots, donors


def write_rows(core, path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as dst, gzip.GzipFile(fileobj=dst, mode="wb", filename="", mtime=0) as compressed:
        for r in rows:
            compressed.write((json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode())
    tmp.replace(path)
    return core.digest(path)


def load_student(core, cnn, raw, records, gate):
    csvpath, receipt = gate / "student_pseudo.csv", gate / "select_receipt.json"
    cfg = core.read_json(receipt)
    core.require_hash(csvpath, cfg["student_pseudo_sha256"])
    with csvpath.open(encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    if cfg.get("rows") != len(items) or cfg.get("pseudo_train") is not bool(items):
        raise ValueError("Gate receipt does not authorize precisely these pseudo labels")
    if not items:
        return None
    from collections import Counter
    if len(items) > EXTRA or max(Counter(i["lot"] for i in items).values()) > 2:
        raise ValueError("Pseudo-label selection cap exceeded")
    for key in ("source_row", "lot", "map_sha256"):
        recorded = {str(r[key]) for r in records}
        proposed = [str(r[key]) for r in items]
        if not all(proposed) or recorded.intersection(proposed) or (key != "lot" and len(proposed) != len(set(proposed))):
            raise ValueError(f"Non-isolated or duplicate pseudo-label {key}")
    maps = []
    for r in items:
        if r["pseudo_label"] != "Scratch" or not (0 < float(r["threshold"]) <= float(r["p_scratch"]) <= 1):
            raise ValueError("Unexpected pseudo target or out-of-gate score")
        maps.append(cnn.preprocess_map(core.indexed_map(raw, r)))
    return {"x": np.stack(maps), "items": items, "receipt_sha256": core.digest(receipt),
            "table_sha256": core.digest(csvpath)}


def materialize(core, cnn, arm, seed, epoch, order, slots, donor, records, pool, pseudo, folder):
    provenance, transformed = [], []
    mix_rng = rng_for(seed, epoch, 93)
    pseudo_order = rng_for(seed, epoch, 94).permutation(len(slots))
    for j, pos in enumerate(slots):
        mark = {"slot": int(pos), "removed_normal_row": int(records[order[pos]]["source_row"])}
        if arm == "real_repeat":
            order[pos] = donor[j]
            mark["repeat_row"] = int(records[donor[j]]["source_row"])
        elif arm == "synthetic":
            mixed, details, originals = pool.generate(int(donor[j]), mix_rng)
            mark.update(details)
            mark["paired_repeat_row"] = int(records[donor[j]]["source_row"])
            transformed.append(cnn.preprocess_map(mixed))
            order[pos] = -1 - j
            if epoch == 1 and j < 4:
                spot = folder / f"spot{j}.npz"
                np.savez_compressed(spot, donor=originals[0], background=originals[1], synthetic=mixed)
                mark["spot_file_sha256"] = core.digest(spot)
        elif arm == "student":
            k = pseudo_order[j]
            order[pos] = -1 - k
            mark["pseudo_row"] = int(pseudo["items"][k]["source_row"])
        provenance.append(mark)
    extra_x = np.stack(transformed) if transformed else pseudo["x"] if arm == "student" else None
    return extra_x, provenance


def fit(core, cnn, arm, seed, x, y, idx, records, device, out, binding, pool, pseudo):
    folder = out / "fits" / f"{arm}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = folder / "checkpoint.pt"
    fit_id = core.fingerprint([binding, arm, seed])
    if (folder / "run.json").exists():
        receipt = core.read_json(folder / "run.json")
        if receipt["fit_id"] != fit_id or receipt["binding"] != binding:
            raise ValueError("Incompatible completed fit")
        core.require_hash(ckpt, receipt["checkpoint_sha256"])
        for f in receipt["files"]:
            core.require_hash(out / "probabilities" / f["file"], f["sha256"])
        for h in receipt["epochs"]:
            core.require_hash(folder / h["provenance_file"], h["provenance_sha256"])
        return receipt
    cnn.seed_everything(seed)
    model = cnn.build_model("medium").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, foreach=False)
    train = idx["train"]
    scratch = train[y[train] == SCRATCH]
    labels = torch.from_numpy(y).to(device)
    order_rng = np.random.default_rng(seed)
    history = []
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if state["fit_id"] != fit_id:
            raise ValueError("Incompatible checkpoint; runtime, source or gate changed")
        history = state["history"]
        if [h["epoch"] for h in history] != list(range(1, len(history) + 1)) or len(history) > EPOCHS:
            raise ValueError("Invalid epoch history")
        for h in history:
            core.require_hash(folder / h["provenance_file"], h["provenance_sha256"])
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        order_rng.bit_generator.state = json.loads(state["order_rng"])
    for epoch in range(len(history) + 1, EPOCHS + 1):
        started = time.monotonic()
        order = core.order_for_epoch(order_rng, train, None)
        slots, donors = epoch_plan(y, order, seed, epoch, arm, scratch, len(pseudo["items"]) if pseudo else 0)
        extra_x, provenance = materialize(core, cnn, arm, seed, epoch, order, slots, donors,
                                         records, pool, pseudo, folder)
        if extra_x is not None:
            extra_x = torch.from_numpy(extra_x).to(device)
        prov_file = f"train_sources_epoch{epoch}.jsonl.gz"
        prov_sha = write_rows(core, folder / prov_file, provenance)
        batch_labels = np.where(order < 0, SCRATCH, y[np.maximum(order, 0)])
        counts = np.bincount(batch_labels, minlength=9)
        expected = np.bincount(y[train], minlength=9)
        expected[[NORMAL, SCRATCH]] += [-len(slots), len(slots)]
        if not np.array_equal(counts, expected):
            raise AssertionError("Unexpected train class exposure")
        model.train()
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        steps = 0
        for start in range(0, len(order), BATCH):
            batch = order[start:start+BATCH]
            safe = np.maximum(batch, 0)
            xb = core.input_batch(x, safe, device)
            targets = labels.index_select(0, torch.as_tensor(safe, device=device))
            changed = np.flatnonzero(batch < 0)
            if len(changed):
                pos = torch.as_tensor(changed, device=device)
                xb.index_copy_(0, pos, core.input_batch(extra_x, -1-batch[changed], device))
                targets[pos] = SCRATCH
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if arm == "student" and len(changed):
                w = torch.ones(len(batch), device=device)
                w[pos] = PSEUDO_WEIGHT
                num, den = (F.cross_entropy(logits, targets, reduction="none") * w).sum(), w.sum()
            else:
                num, den = core.loss_for_arm(logits, targets, "cnn_ce", None)
            (num / den).backward()
            optimizer.step()
            totals[0] += num.detach()
            totals[1] += den
            steps += 1
        num, den = totals.cpu().tolist()
        if not np.isfinite(num) or den <= 0 or steps != (len(train) + BATCH - 1) // BATCH:
            raise ValueError("Unexpected training budget or nonfinite objective")
        h = {"epoch": epoch, "steps": steps, "examples": len(order), "class_draws": counts.tolist(),
             "substitutions": len(slots), "objective": num/den, "seconds": time.monotonic()-started,
             "provenance_file": prov_file, "provenance_sha256": prov_sha}
        if arm == "synthetic":
            h["qa"] = {"added_range": [min(p["added_dies"] for p in provenance), max(p["added_dies"] for p in provenance)],
                       "unique_synthetic": len({p["synthetic_sha256"] for p in provenance}),
                       "unique_donors": len({p["donor_row"] for p in provenance}),
                       "exact_size_pairs": sum(p["native_hw"] == p["donor_hw"] for p in provenance),
                       "min_retention": min(p["retained_trace_dies"]/p["trace_dies"] for p in provenance)}
        history.append(h)
        tmp = ckpt.with_suffix(".part")
        torch.save({"fit_id": fit_id, "history": history, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "order_rng": json.dumps(order_rng.bit_generator.state)}, tmp)
        tmp.replace(ckpt)
        print(f"{arm} s{seed} epoch {epoch}/{EPOCHS} loss={num/den:.5f} {h['seconds']:.1f}s", flush=True)
    files = [core.write_predictions(model, x, idx[r], records, device, out / "probabilities", arm, seed, r)
             for r in ("tune", "audit")]
    receipt = {"model": arm, "seed": seed, "fit_id": fit_id, "binding": binding, "epochs": history,
               "checkpoint_sha256": core.digest(ckpt), "files": files}
    core.atomic_json(folder / "run.json", receipt)
    return receipt


def self_test(core, cnn):
    a = np.ones((25, 31), dtype=np.uint8)
    a[[0, -1], :] = 0
    a[:, [0, -1]] = 0
    donor = a.copy()
    donor[10, 4:22] = 2
    donor[10, 13:15] = 1
    donor[21, 3] = 2
    points, geometry = linear_trace(donor)
    a[20, 5] = 2
    mixed, qa = transplant(a, np.array(a.shape), points, np.array([1, 1]))
    assert np.array_equal(a > 0, mixed > 0) and mixed[20, 5] == 2 and qa["added_dies"] >= 14
    assert np.all(donor[points[:, 0], points[:, 1]] == 2) and geometry["span_native"] >= 15
    assert transplant(a, np.array(a.shape), points, np.array([80, 80])) is None
    assert cnn.preprocess_map(mixed).shape == (2, 64, 64)
    y = np.array([NORMAL] * 90 + [SCRATCH] * 10)
    order = np.random.default_rng(42).permutation(len(y))
    a_slots, b_donors = epoch_plan(y, order, 42, 1, "real_repeat", np.arange(90, 100))
    c_slots, c_donors = epoch_plan(y, order, 42, 1, "synthetic", np.arange(90, 100))
    d_slots, _ = epoch_plan(y, order, 42, 1, "student", np.arange(90, 100), 12)
    assert np.array_equal(a_slots,c_slots) and np.array_equal(b_donors,c_donors)
    assert len(set(a_slots)) == EXTRA and np.all(y[order[a_slots]] == NORMAL) and np.array_equal(a_slots[:12], d_slots)
    print("PASS paired normal slots, repeat donors, partial student budget, authentic line gaps, valid die masks, preserved normal failures.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--raw", type=Path)
    p.add_argument("--cache", type=Path, default=Path("/content/wm811k_cache"))
    p.add_argument("--out", type=Path)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS[:3]))
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument("--device", choices=["cuda"], default="cuda")
    p.add_argument("--pseudo", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    core = load_core(args.repo)
    cnn = core.source_module(args.repo)
    if args.self_test:
        self_test(core, cnn)
        return
    if (platform.system() != "Linux" or not Path("/content").is_dir()
        or not (os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_BACKEND_VERSION"))
        or not torch.cuda.is_available()):
        p.error("Actual runs require an existing hosted Colab GPU.")
    if not args.raw or not args.out or ("student" in args.arms and not args.pseudo):
        p.error("Need --raw, --out and, for student, --pseudo.")
    out = args.out
    (out / "probabilities").mkdir(parents=True, exist_ok=True)
    cnn.seed_everything(42)
    records, y, idx = core.load_records(args.repo)
    x, cache_info = core.prepare(args.raw, args.cache, records, cnn)
    device = torch.device(args.device)
    protocol = {"schema": 1, "stage": "09", "raw_sha256": core.RAW, "reference_sha256": core.REF,
                "cnn_sha256": core.CNN, "stage05_trainer_sha256": CORE_SHA, "script_sha256": core.digest(Path(__file__)),
                "class_order": core.CLASSES, "architecture": "medium", "epochs": EPOCHS, "batch_size": BATCH,
                "seeds": list(SEEDS), "arms": list(ARMS), "optimizer": {"name":"AdamW","lr":.001,"weight_decay":.0001,"foreach":False},
                "training_rows": len(idx["train"]), "roles": core.ROLES, "cache_sha256": cache_info["sha256"],
                "class_counts": np.bincount(y[idx["train"]], minlength=9).tolist(), "extra_train_slots": EXTRA,
                "procedure": "80 train-normal slots/epoch replaced by genuine Scratch draws or train-only native trace transplants; D uses m<=80 gated Scratch targets",
                "synthesis": "5-degree band search; max 4-pitch gaps; span>=max(4,.15*min(H,W)); aspect>=3; nearest compatible H,W in [.8,1.25] donor; integer center-align/jitter +/-2; native pitch preserved; retention>=.90; newly failed>=max(3,.50*trace); train-Scratch q95 failure-density cap",
                "student_weight": PSEUDO_WEIGHT, "audit": "historically inspected development role", "predicted_roles":["tune","audit"]}
    write_once(core, out / "protocol.json", protocol)
    binding = {"protocol_sha256": core.fingerprint(protocol), "runtime": core.runtime_info(device)}
    write_once(core, out / "runtime.json", binding)
    source = args.cache / "verified_LSWMD.pkl" if Path("/content/drive") in args.raw.resolve().parents else args.raw
    x = core.device_cache(x, device)
    with source.open("rb") as stream:
        pool = SourcePool(stream, records, y, idx["train"], core) if "synthetic" in args.arms else None
        if pool:
            write_once(core, out / "template_qa.json", {"eligible_donors": len(pool.templates),
                       "train_scratch": len(pool.scratch), "max_failure_fraction": pool.max_fail_fraction})
        pseudo = load_student(core, cnn, stream, records, args.pseudo) if "student" in args.arms else None
        if "student" in args.arms and pseudo is None:
            core.atomic_json(out / "student_skipped.json", {"reason": "gate_returned_no_admissible_pseudo_rows",
                    "receipt_sha256":core.digest(args.pseudo/"select_receipt.json")})
        for seed in args.seeds:
            for arm in args.arms:
                if arm == "student" and pseudo is None:
                    continue
                fit_binding = {**binding, "gate_sha256": pseudo["receipt_sha256"]} if arm=="student" else binding
                fit(core, cnn, arm, seed, x, y, idx, records, device, out, fit_binding, pool, pseudo)
                files = []
                for path in sorted((out / "fits").glob("*/run.json")):
                    receipt = core.read_json(path)
                    if receipt["binding"]["protocol_sha256"] != binding["protocol_sha256"]:
                        raise ValueError("Mixed protocols in probability manifest")
                    files.extend(receipt["files"])
                core.atomic_json(out / "probabilities/manifest.json", {"schema":1, "files":files,
                    "raw_sha256":core.RAW,"reference_sha256":core.REF,"cnn_sha256":core.CNN,
                    "protocol_sha256":binding["protocol_sha256"],"class_order":core.CLASSES,
                    "probability_columns":core.CLASSES})


if __name__ == "__main__":
    main()
