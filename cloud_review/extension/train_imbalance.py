"""Five matched CNN fits per seed on the frozen WM-811K development roles.

Hosted Colab: python train_imbalance.py --repo REPO --raw LSWMD.pkl
    --cache /content/wm811k_cache --out /content/drive/MyDrive/wm811k_imbalance
Use --prepare-only for indexed extraction or --arms cnn_ce --seeds 42 for a shard.
Rerun the same command to resume. Use a single writer per output directory.
The real-data CLI requires a hosted Colab runtime. --self-test uses synthetic inputs only.
"""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import time
from collections import Counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F


RAW = "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65"
REF = "4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879"
CNN = "88671864f38eb400c9ee9678a71cfa3abf8015c74e5fa6b48b911cf00c71582d"
COMMIT = "aaa667c0421ece13e3a71fa616fb10287b4453ff"
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
ROLES = {"train": 31457, "tune": 11307, "audit": 11591}
SEEDS = [42, 123, 2026]
ARMS = {
    "cnn_ce": "unweighted cross-entropy and shuffled natural sampling",
    "cnn_weighted": "cross-entropy weighted by inverse square root of train class count",
    "cnn_sampled": "unweighted CE, replacement draws proportional to inverse square root of train class count",
    "cnn_rot180": "unweighted CE, train-only 180 degree two-channel rotation with probability 0.5",
    "cnn_focal": "natural sampling and focal loss with gamma 2 and alpha 1",
}
EPOCHS, BATCH = 7, 128
PROTOCOL = {
    "schema": 1, "raw_sha256": RAW, "reference_sha256": REF, "cnn_sha256": CNN,
    "reference_commit": COMMIT, "class_order": CLASSES, "roles": ROLES,
    "seeds": SEEDS, "arms": ARMS, "model": "medium", "batch_size": BATCH, "epochs": EPOCHS,
    "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "foreach": False,
    "precision": "float32 deterministic, TF32 off", "examples_per_epoch": ROLES["train"],
    "steps_per_epoch": (ROLES["train"] + BATCH - 1) // BATCH,
    "epoch_selection": "fixed final epoch", "trained_role": "train",
    "predicted_roles": ["tune", "audit"], "audit_interpretation": "previously inspected, exploratory",
    "augmentation_assumption": "The defect label is stable under a 180-degree change in wafer-map reading orientation.",
    "rng": "NumPy PCG64(seed) order generator continued across epochs; independent seed/epoch rotation stream",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_module(repo):
    path = repo / "src/waferdefect/cnn.py"
    require_hash(path, CNN)
    spec = importlib.util.spec_from_file_location("frozen_wafer_cnn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.CLASSES != CLASSES:
        raise ValueError("Unexpected frozen class order")
    return module


def load_records(repo):
    ref = repo / "data-reference/development_reference.csv.gz"
    require_hash(ref, REF)
    with gzip.open(ref, "rt", encoding="utf-8-sig", newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["source_row"]))
    if Counter(r["role"] for r in rows) != ROLES or {r["source_split"] for r in rows} != {"train"}:
        raise ValueError("Unexpected frozen development roles")
    if len({int(r["source_row"]) for r in rows}) != len(rows):
        raise ValueError("Repeated source_row")
    for column in ("lot", "component", "map_sha256"):
        bindings = {}
        for row in rows:
            key = row[column]
            if not key or bindings.setdefault(key, row["role"]) != row["role"]:
                raise ValueError(f"Missing or cross-role group for {column}")
    labels = np.array([CLASSES.index(r["label"]) for r in rows], dtype=np.int64)
    roles = np.array([r["role"] for r in rows])
    indices = {name: np.flatnonzero(roles == name) for name in ROLES}
    return rows, labels, indices


def verified_source(raw, cache):
    # Scanning the whole file verifies identity. Only indexed development payloads are decoded.
    # On Drive, copy during that same sequential pass to avoid 54k remote seeks.
    if Path("/content/drive") not in raw.resolve().parents:
        require_hash(raw, RAW)
        return raw
    staged = cache / "verified_LSWMD.pkl"
    checksum = hashlib.sha256()
    tmp = staged.with_suffix(".part")
    with raw.open("rb") as src, tmp.open("wb") as dst:
        for block in iter(lambda: src.read(8 << 20), b""):
            checksum.update(block)
            dst.write(block)
    if checksum.hexdigest() != RAW:
        tmp.unlink()
        raise ValueError("Original data checksum mismatch")
    tmp.replace(staged)
    return staged


def indexed_map(stream, row):
    shape = int(row["height"]), int(row["width"])
    size, offset = int(row["raw_data_nbytes"]), int(row["raw_data_offset"])
    if min(shape) < 1 or size != shape[0] * shape[1] or offset < 0 or row["order"] not in ("C", "F"):
        raise ValueError(f"Invalid index for source_row {row['source_row']}")
    stream.seek(offset)
    wafer = np.frombuffer(stream.read(size), dtype=np.uint8).reshape(shape, order=row["order"])
    checksum = hashlib.sha256(np.asarray(shape, dtype="<i8").tobytes())
    checksum.update(np.ascontiguousarray(wafer).tobytes())
    if checksum.hexdigest() != row["map_sha256"]:
        raise ValueError(f"Indexed map checksum mismatch at source_row {row['source_row']}")
    return wafer


def prepare(raw, cache, rows, cnn):
    cache.mkdir(parents=True, exist_ok=True)
    raw = verified_source(raw, cache)
    path, receipt = cache / "development_f32.npy", cache / "cache.json"
    shape = (len(rows), 2, 64, 64)
    binding = {"raw_sha256": RAW, "reference_sha256": REF, "cnn_sha256": CNN,
               "dtype": "<f4", "shape": list(shape), "order": "ascending source_row"}
    if receipt.exists() and path.exists():
        info = read_json(receipt)
        if info["binding"] != binding:
            raise ValueError("Incompatible cache receipt; use another cache directory")
        require_hash(path, info["sha256"])
    else:
        start = time.monotonic()
        tmp = path.with_suffix(".part")
        tensor = np.lib.format.open_memmap(tmp, mode="w+", dtype="<f4", shape=shape)
        with raw.open("rb") as stream:
            for i, row in enumerate(rows):
                tensor[i] = cnn.preprocess_map(indexed_map(stream, row))
                if (i + 1) % 10000 == 0:
                    print(f"Prepared {i + 1}/{len(rows)} development wafers", flush=True)
        tensor.flush()
        del tensor
        tmp.replace(path)
        info = {"binding": binding, "sha256": digest(path), "seconds": time.monotonic() - start}
        atomic_json(receipt, info)
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != shape or x.dtype != np.dtype("<f4"):
        raise ValueError("Unexpected raster cache layout")
    return x, info


def runtime_info(device):
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "device": str(device), "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.machine(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "threads": torch.get_num_threads()}


def order_for_epoch(rng, train, sampling_probs):
    if sampling_probs is None:
        return rng.permutation(train)
    return rng.choice(train, size=len(train), replace=True, p=sampling_probs)


def rotate_both(inputs, mask):
    if np.any(mask):
        selected = torch.as_tensor(np.flatnonzero(mask), device=inputs.device)
        inputs.index_copy_(0, selected, inputs.index_select(0, selected).flip((-2, -1)))
    return inputs


def loss_for_arm(logits, targets, arm, weights):
    if arm == "cnn_focal":
        per_item = F.cross_entropy(logits, targets, reduction="none")
        numerator = ((-torch.expm1(-per_item)).square() * per_item).sum()
    else:
        numerator = F.cross_entropy(logits, targets, weight=weights, reduction="sum")
    denominator = targets.numel() if weights is None else weights[targets].sum()
    return numerator, denominator


def input_batch(x, indices, device):
    if torch.is_tensor(x):
        return x.index_select(0, torch.as_tensor(indices, device=device))
    return torch.from_numpy(np.ascontiguousarray(x[indices])).to(device)


def device_cache(x, device):
    if device.type != "cuda" or torch.cuda.mem_get_info()[0] < x.nbytes + (5 << 30):
        return x
    gpu = torch.empty(x.shape, dtype=torch.float32, device=device)
    for start in range(0, len(x), 512):
        end = min(start + 512, len(x))
        gpu[start:end].copy_(torch.from_numpy(np.array(x[start:end], copy=True)))
    return gpu


def write_predictions(model, x, indices, records, device, folder, arm, seed, role):
    file = f"{arm}_s{seed}_{role}.csv.gz"
    target = folder / file
    tmp = folder / (file + ".part")
    model.eval()
    with tmp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as writer, torch.inference_mode():
            csvout = csv.writer(writer, lineterminator="\n")
            csvout.writerow(["source_row", "truth", *CLASSES])
            for start in range(0, len(indices), 256):
                batch = indices[start:start + 256]
                # Float64 softmax over the float32 model logits for stable downstream calibration.
                probs = model(input_batch(x, batch, device)).double().softmax(1).cpu().numpy()
                if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1., rtol=0., atol=1e-12):
                    raise ValueError("Invalid predicted probability vector")
                for idx, p in zip(batch, probs):
                    row = records[idx]
                    csvout.writerow([row["source_row"], row["label"], *[format(v, ".17g") for v in p]])
    tmp.replace(target)
    return {"model": arm, "seed": seed, "role": role, "file": file,
            "rows": len(indices), "bytes": target.stat().st_size, "sha256": digest(target)}


def fit(arm, seed, cnn, x, y, indices, records, weights, dev, out, binding):
    start = time.monotonic()
    folder = out / "fits" / f"{arm}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    progress = folder / "checkpoint.pt"
    manifest = out / "probabilities"
    fit_id = fingerprint([binding, arm, seed])
    cnn.seed_everything(seed)
    model = cnn.build_model("medium").to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, foreach=False)
    train = indices["train"]
    weighted_ce = torch.as_tensor(weights, device=dev) if arm == "cnn_weighted" else None
    sampling = None
    if arm == "cnn_sampled":
        sampling = weights[y[train]].astype(np.float64)
        sampling /= sampling.sum()
    labels = torch.from_numpy(y).to(dev)
    order_rng = np.random.default_rng(seed)
    history = []
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=True)
        if state["fit_id"] != fit_id:
            raise ValueError("Incompatible checkpoint, runtime or protocol")
        history = state["history"]
        if len(history) > EPOCHS or [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
            raise ValueError("Invalid checkpoint epoch history")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        order_rng.bit_generator.state = json.loads(state["order_rng"])
        print(f"Resumed {arm} seed {seed} after epoch {len(history)}", flush=True)
        del state
    for epoch in range(len(history) + 1, EPOCHS + 1):
        begin = time.monotonic()
        order = order_for_epoch(order_rng, train, sampling)
        aug_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 180]))
        model.train()
        totals = torch.zeros(2, device=dev, dtype=torch.float64)
        rotated, steps = 0, 0
        for begin_row in range(0, len(order), BATCH):
            batch = order[begin_row:begin_row + BATCH]
            xb = input_batch(x, batch, dev)
            if arm == "cnn_rot180":
                mask = aug_rng.random(len(batch)) < 0.5
                rotate_both(xb, mask)
                rotated += int(mask.sum())
            target = labels.index_select(0, torch.as_tensor(batch, device=dev))
            optimizer.zero_grad(set_to_none=True)
            numerator, denominator = loss_for_arm(model(xb), target, arm, weighted_ce)
            loss = numerator / denominator
            loss.backward()
            optimizer.step()
            totals[0] += numerator.detach()
            totals[1] += denominator
            steps += 1
        total_loss, total_weight = totals.cpu().tolist()
        if not np.isfinite(total_loss) or total_weight <= 0 or steps != PROTOCOL["steps_per_epoch"]:
            raise RuntimeError("Nonfinite objective or unexpected training budget")
        stats = {"epoch": epoch, "steps": steps, "examples": len(order), "unique_examples": len(np.unique(order)),
                 "class_draws": np.bincount(y[order], minlength=9).tolist(), "rotated": rotated,
                 "objective": total_loss / total_weight, "seconds": time.monotonic() - begin}
        history.append(stats)
        state = {"fit_id": fit_id, "history": history, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "order_rng": json.dumps(order_rng.bit_generator.state)}
        tmp = progress.with_suffix(".part")
        torch.save(state, tmp)
        tmp.replace(progress)
        print(f"{arm} seed={seed} epoch={epoch}/{EPOCHS} objective={stats['objective']:.6f} "
              f"seconds={stats['seconds']:.1f}", flush=True)
    files = [write_predictions(model, x, indices[r], records, dev, manifest, arm, seed, r)
             for r in ("tune", "audit")]
    receipt = {"model": arm, "seed": seed, "fit_id": fit_id, "binding": binding, "epochs": history,
               "checkpoint_sha256": digest(progress), "files": files, "this_session_seconds": time.monotonic() - start}
    atomic_json(folder / "run.json", receipt)
    return receipt


def scan_finished(out, binding):
    finished = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = out / "fits" / f"{arm}_s{seed}"
            receipt = folder / "run.json"
            if not receipt.exists():
                continue
            known = read_json(receipt)
            if known["fit_id"] != fingerprint([binding, arm, seed]) or known["binding"] != binding:
                raise ValueError(f"Incompatible completed fit: {arm} seed={seed}")
            if [f["role"] for f in known["files"]] != ["tune", "audit"]:
                raise ValueError("Incomplete role outputs in run receipt")
            require_hash(folder / "checkpoint.pt", known["checkpoint_sha256"])
            for f in known["files"]:
                if (f["file"] != f"{arm}_s{seed}_{f['role']}.csv.gz" or f["model"] != arm
                        or f["seed"] != seed or f["rows"] != ROLES[f["role"]]):
                    raise ValueError("Invalid probability receipt")
                require_hash(out / "probabilities" / f["file"], f["sha256"])
            finished[arm, seed] = known
    return finished


def update_status(out, finished, binding):
    files = [f for arm in ARMS for seed in SEEDS if (arm, seed) in finished for f in finished[arm, seed]["files"]]
    complete = len(finished) == len(ARMS) * len(SEEDS)
    manifest = {"schema": 1, "complete": complete, "class_order": CLASSES, "probability_columns": CLASSES,
                "raw_sha256": RAW, "reference_sha256": REF, "cnn_sha256": CNN,
                "protocol_sha256": binding["protocol_sha256"], "files": files}
    atomic_json(out / "probabilities/manifest.json", manifest)
    atomic_json(out / "run_status.json", {"complete": complete, "completed_fits": len(finished),
                "expected_fits": len(ARMS) * len(SEEDS), "missing": [
                    f"{a}_s{s}" for a in ARMS for s in SEEDS if (a, s) not in finished]})


def self_test(repo):
    cnn = source_module(repo)
    wafer = np.array([[0, 2, 0], [1, 1, 1]], dtype=np.uint8)
    map_hash = hashlib.sha256(np.asarray(wafer.shape, dtype="<i8").tobytes() + wafer.tobytes()).hexdigest()
    record = {"height": 2, "width": 3, "raw_data_nbytes": 6, "raw_data_offset": 1,
              "order": "F", "map_sha256": map_hash, "source_row": 7}
    raw = io.BytesIO(b"X" + wafer.tobytes(order="F"))
    assert np.array_equal(indexed_map(raw, record), wafer)
    sample = torch.from_numpy(cnn.preprocess_map(wafer))
    pair = torch.stack([sample, sample])
    rotate_both(pair, np.array([True, False]))
    assert torch.equal(pair[0], sample.flip((-2, -1))) and torch.equal(pair[1], sample)
    logits = torch.tensor([[1., 0.], [-0.5, 0.1]])
    targets, weights = torch.tensor([0, 1]), torch.tensor([0.5, 1.5])
    for arm in ARMS:
        num, den = loss_for_arm(logits, targets, arm, weights if arm == "cnn_weighted" else None)
        ce = F.cross_entropy(logits, targets, reduction="none")
        if arm == "cnn_focal":
            expected = (ce * (1 - torch.exp(-ce)).square()).mean()
        elif arm == "cnn_weighted":
            expected = F.cross_entropy(logits, targets, weight=weights)
        else:
            expected = ce.mean()
        assert torch.allclose(num / den, expected)
    train = np.arange(9, dtype=np.int64)
    natural = order_for_epoch(np.random.default_rng(42), train, None)
    sampled = order_for_epoch(np.random.default_rng(42), train, np.full(9, 1 / 9))
    assert np.array_equal(np.sort(natural), train) and len(sampled) == len(train)
    assert len(np.unique(sampled)) < len(train)
    print("PASS synthetic offset/hash, paired rotation, five objectives, matching sampler budgets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/content/wm811k_cache"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.repo)
        return
    if (not Path("/content").is_dir() or platform.system() != "Linux"
            or not (os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))):
        parser.error("Actual wafer data may be processed only on a hosted Google Colab runtime.")
    if not args.raw or not args.out:
        parser.error("--raw and --out are required for a hosted run")
    if not args.prepare_only and args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select a free GPU in hosted Colab or explicitly choose hosted --device cpu")
    out = args.out
    cnn = source_module(args.repo)
    cnn.seed_everything(42, num_threads=2)
    records, labels, indices = load_records(args.repo)
    print("Frozen development reference verified. Checking source and cache...", flush=True)
    x, cache_info = prepare(args.raw, args.cache, records, cnn)
    out.mkdir(parents=True, exist_ok=True)
    weights, counts = cnn.class_weights(labels, indices["train"], "sqrt_inverse")
    protocol = {**PROTOCOL, "training_class_counts": counts.tolist(), "sqrt_inverse_weights": weights.tolist(),
                "script_sha256": digest(Path(__file__)), "cache_sha256": cache_info["sha256"]}
    protocol_sha = fingerprint(protocol)
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Protocol, source or cache changed; use a fresh output directory")
    atomic_json(protocol_path, protocol)
    if args.prepare_only:
        print(f"Prepared and verified {len(records)} development maps; protocol_sha256={protocol_sha}", flush=True)
        return
    device = torch.device(args.device)
    binding = {"protocol_sha256": protocol_sha, "runtime": runtime_info(device)}
    runtime_path = out / "runtime.json"
    if runtime_path.exists() and read_json(runtime_path) != binding:
        raise ValueError("Hardware/software changed; use a fresh output directory for matched fits")
    atomic_json(runtime_path, binding)
    (out / "probabilities").mkdir(exist_ok=True)
    finished = scan_finished(out, binding)
    update_status(out, finished, binding)
    pending = [(a, s) for s in args.seeds for a in args.arms if (a, s) not in finished]
    if pending:
        x = device_cache(x, device)
    for arm, seed in pending:
        print(f"Starting {arm} seed={seed} on {device}", flush=True)
        finished[arm, seed] = fit(arm, seed, cnn, x, labels, indices, records, weights, device, out, binding)
        update_status(out, finished, binding)
    print(f"Completed and verified {len(finished)}/15 fits; outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
