"""Optional single-seed masked-density pilot on the isolated WM-811K pool.

CPU (hosted Colab): python ssl_pilot.py pack --pool FROZEN_POOL --raw LSWMD.pkl --out PACK
GPU (same runtime type as control): python ssl_pilot.py train --pack PACK
    --pack-sha SHA_PRINTED_BY_CPU --control TRAINING --repo PINNED_REPO
    --raw LSWMD.pkl --cache /content/wm811k_xcache --out PILOT
Only the gzip byte index and its receipt move from CPU to GPU. Keep both private.
Pair the two seed-42 runs using ssl_evaluate.py after training. No pseudo-labels.
"""

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import pickletools
import platform
import signal
import time


RAW = "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65"
POOL_RECEIPT = "534e62fb56a4ccff0a49d5bb9ca762b48b8a9fb5013e59e19479fb71625656a5"
TRAIN_SOURCE = "3982640b77227e34ebbf69854c2b61f8246840b051aa3291deab8c5b3a086661"
REF = "4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879"
INDEXER = "786692eaea13c1b0a6d52d821e443e00f546735aa353a24d034caec94638b7e2"
POOL_SIZE, SAMPLES, SEED, BATCH = 434498, 50000, 42, 128
MAP_START, SOURCE_ROWS = 12983935, 811457
FIELDS = ["source_row", "height", "width", "map_sha256", "raw_data_offset", "raw_data_nbytes", "order"]
POLICY = {
    "sample": "first 50000 by SHA256(domain || bytes.fromhex(map_sha256)), tie by source_row",
    "domain": "wm811k-isolated-ssl-pilot-v1-seed42|",
    "seed": SEED, "maps": SAMPLES, "ssl_epochs": 1, "ssl_steps": (SAMPLES+BATCH-1)//BATCH,
    "batch_size": BATCH, "mask_grid": [8,8], "masked_fraction": 0.25,
    "objective": "BCE with valid-die-weighted held-out per-cell failure density; keep valid channel visible",
    "encoder_transfer": "all frozen-architecture convolution and BN parameters and BN buffers",
    "supervised": "same seed42 fresh classifier head + seven natural CE epochs in frozen train_imbalance.fit",
    "comparison": "equal supervised budget; SSL receives one additional unlabeled epoch",
    "cpu_minutes": 12, "gpu_minutes": 50,
}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    check(sha(path) == expected, f"SHA-256 mismatch: {path}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, obj):
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_sha():
    return sha(Path(__file__))


@contextmanager
def budget(out, stage, minutes):
    # Conservative absolute deadline persists across restarts, including disconnected time.
    # This caps wall time even if the job is interrupted before updating a progress file.
    receipt = out / "budget.json"
    if not receipt.exists():
        save_json(receipt, {"stage": stage, "script_sha256": source_sha(), "deadline": time.time()+60*minutes})
    cached = read_json(receipt)
    check(cached["stage"] == stage and cached["script_sha256"] == source_sha(), "Budget binding mismatch")
    seconds = cached["deadline"] - time.time()
    check(0 < seconds <= 60 * minutes + 1, "This pilot's wall-clock budget expired")
    def stop(signum, frame):
        raise TimeoutError(f"{stage} exceeded the fixed {minutes}-minute budget")
    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sample_rows(pool):
    require_hash(pool / "receipt.json", POOL_RECEIPT)
    saved = read_json(pool / "receipt.json")
    check(saved["stage"] == "pool" and saved["lineage"]["inputs"]["raw"] == RAW
          and saved["lineage"]["inputs"]["reference"] == REF
          and saved["lineage"]["inputs"]["indexer"] == INDEXER, "Incompatible isolated pool lineage")
    needed = {"pool_summary.json", "ssl_pool.csv.gz", "components.csv.gz", "lot_components.csv.gz", "eligible_shapes.csv"}
    check(needed <= set(saved["outputs"]), "Missing isolated-pool artifacts")
    for name in needed:
        require_hash(pool / name, saved["outputs"][name])
    summary = read_json(pool / "pool_summary.json")
    check(summary["unique_candidate_maps"] == POOL_SIZE and summary["unique_maps_removed_by_multihop"] == 31247
          and summary["labels_assigned"] is False and summary["raw_sha256"] == RAW, "Unexpected safe pool summary")
    rows_seen = 0
    def candidates():
        nonlocal rows_seen
        with gzip.open(pool / "ssl_pool.csv.gz", "rt", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_seen += 1
                key = hashlib.sha256(POLICY["domain"].encode() + bytes.fromhex(row["map_sha256"])).digest()
                yield key, int(row["source_row"]), int(row["height"]), int(row["width"]), row["map_sha256"]
    selected = heapq.nsmallest(SAMPLES, candidates())
    result = {sid: (h,w,digest) for _,sid,h,w,digest in selected}
    check(rows_seen == POOL_SIZE and len(result) == SAMPLES and len({r[2] for r in result.values()}) == SAMPLES,
          "Invalid population or duplicate selected identities")
    return result


def scan_offsets(raw, selected):
    # Bounded-memory adaptation of work/diagnosis/extract_raw_maps.py's protocol-2
    # layout. GLOBAL/REDUCE/BUILD are only recognized as tokens, never executed.
    import numpy as np
    with raw.open("rb") as f:
        f.seek(MAP_START)
        ops = pickletools.genops(f)
        def take(names, value=None):
            op, arg, pos = next(ops)
            accepted = (names,) if isinstance(names, str) else names
            check(op.name in accepted and (value is None or arg == value), f"Unexpected pickle layout at {pos}")
            return op.name, arg, pos
        for sid in range(SOURCE_ROWS):
            first, arg, _ = take(("APPENDS", "BINGET"))
            if first == "APPENDS":
                take("MARK")
                take("BINGET", 16)
            else:
                check(arg == 16, "Unexpected reconstruction entry")
            for name, value in (("BINGET",17),("BININT1",0),("TUPLE1",None),("SHORT_BINSTRING","b"),
                                ("TUPLE3",None),("REDUCE",None)):
                take(name, value)
            take(("BINPUT", "LONG_BINPUT"))
            take("MARK"); take("BININT1", 1)
            h = take(("BININT1","BININT2","BININT"))[1]
            w = take(("BININT1","BININT2","BININT"))[1]
            take("TUPLE2")
            if sid == 0:
                for name,value in (("BINGET",19),("SHORT_BINSTRING","u1"),("BININT1",0),("BININT1",1),
                                   ("TUPLE3",None),("REDUCE",None),("BINPUT",42),("MARK",None),
                                   ("BININT1",3),("SHORT_BINSTRING","|"),("NONE",None),("NONE",None),
                                   ("NONE",None),("BININT",-1),("BININT",-1),("BININT1",0),("TUPLE",None),("BUILD",None)):
                    take(name,value)
            else:
                take("BINGET",42)
            order = "F" if take(("NEWTRUE","NEWFALSE"))[0] == "NEWTRUE" else "C"
            name, data, pos = take(("BINSTRING","SHORT_BINSTRING"))
            check(h > 0 and w > 0 and len(data) == h*w, "Invalid wafer payload")
            take("TUPLE"); take("BUILD")
            if sid in selected:
                ht, wt, digest = selected[sid]
                check((ht,wt) == (h,w), "Selected geometry mismatch")
                arr = np.frombuffer(data.encode("latin1"), np.uint8).reshape(h,w,order=order)
                hh = hashlib.sha256(np.asarray((h,w),dtype="<i8").tobytes())
                hh.update(arr.tobytes(order="C"))
                check(hh.hexdigest() == digest and arr.max() <= 2 and arr.any(), "Selected raw payload mismatch")
                yield [sid,h,w,digest,pos + (5 if name == "BINSTRING" else 2),h*w,order]
        take("GLOBAL","numpy.core.multiarray scalar")


def pack(args):
    out, raw = args.out, args.raw
    receipt_path, csv_path = out/"receipt.json", out/"seed42_offsets.csv.gz"
    if receipt_path.exists():
        validate_pack(out, sha(receipt_path))
        print("Existing pack verified; trusted transfer receipt SHA256", sha(receipt_path), flush=True)
        return
    with budget(out, "pack", POLICY["cpu_minutes"]):
        selected = sample_rows(args.pool)
        require_hash(raw, RAW)
        tmp = csv_path.with_suffix(".part")
        with tmp.open("wb") as rawout, gzip.GzipFile(fileobj=rawout, filename="", mode="wb", mtime=0) as wrapped:
            with io.TextIOWrapper(wrapped, encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n"); w.writerow(FIELDS)
                count = 0
                for row in scan_offsets(raw, selected):
                    w.writerow(row); count += 1
        check(count == SAMPLES, "Selected maps missing from source")
        tmp.replace(csv_path)
        save_json(receipt_path, {"status":"complete", "maps":count, "raw_sha256":RAW, "pool_receipt_sha256":POOL_RECEIPT,
                  "script_sha256":source_sha(), "policy":POLICY, "index_sha256":sha(csv_path)})
        print("Pass both private pack files; trusted transfer receipt SHA256", sha(receipt_path), flush=True)


def validate_pack(folder, expected):
    require_hash(folder / "receipt.json", expected)
    info = read_json(folder / "receipt.json")
    check(info["status"] == "complete" and info["pool_receipt_sha256"] == POOL_RECEIPT
          and info["raw_sha256"] == RAW and info["policy"] == POLICY and info["maps"] == SAMPLES
          and info["script_sha256"] == source_sha(), "Incompatible offset pack")
    path = folder / "seed42_offsets.csv.gz"
    require_hash(path, info["index_sha256"])
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        check(reader.fieldnames == FIELDS, "Unexpected index columns")
        rows = list(reader)
    ids = [int(r["source_row"]) for r in rows]
    check(len(rows) == SAMPLES and ids == sorted(set(ids)) and 0 <= ids[0] <= ids[-1] < SOURCE_ROWS
          and len({r["map_sha256"] for r in rows}) == SAMPLES, "Invalid selected IDs or hashes")
    return rows


def control_info(control, frozen, device):
    protocol = frozen.read_json(control/"protocol.json")
    check(all(protocol.get(k) == v for k,v in frozen.PROTOCOL.items()), "Different supervised protocol")
    check(protocol["script_sha256"] == TRAIN_SOURCE, "Different control trainer")
    runtime = frozen.read_json(control/"runtime.json")
    check(runtime == {"protocol_sha256": frozen.fingerprint(protocol), "runtime": frozen.runtime_info(device)},
          "Cannot pair control across different hardware or software")
    run_path = control / "fits/cnn_ce_s42/run.json"
    run = frozen.read_json(run_path)
    check(run["model"] == "cnn_ce" and run["seed"] == SEED and run["binding"] == runtime
          and run["fit_id"] == frozen.fingerprint([runtime, "cnn_ce", SEED]), "Incorrect CE control lineage")
    epochs = run["epochs"]
    check([r["epoch"] for r in epochs] == list(range(1,8)) and
          all(r["examples"] == 31457 and r["unique_examples"] == 31457 and r["steps"] == 246 for r in epochs),
          "Unmatched supervised exposure")
    require_hash(run_path.parent / "checkpoint.pt", run["checkpoint_sha256"])
    check([(p["role"],p["rows"]) for p in run["files"]] == [("tune",11307),("audit",11591)], "Control predictions incomplete")
    for p in run["files"]:
        check(p["file"] == f"cnn_ce_s42_{p['role']}.csv.gz", "Unexpected control prediction file")
        require_hash(control / "probabilities" / p["file"], p["sha256"])
    return protocol, runtime, sha(run_path)


def get_images(raw, rows, cache, frozen, cnn, pack_sha):
    import numpy as np
    path, receipt = cache / "ssl_50k.npy", cache / "ssl_50k_cache.json"
    if path.exists() and receipt.exists():
        info = read_json(receipt)
        check(info["pack_sha256"] == pack_sha and info["cnn_sha256"] == frozen.CNN, "Wrong SSL image cache")
        require_hash(path, info["sha256"])
    else:
        tmp = path.with_suffix(".part")
        image = np.lib.format.open_memmap(tmp, "w+", "<f4", shape=(len(rows),2,64,64))
        with raw.open("rb") as stream:
            for i,row in enumerate(rows):
                image[i] = cnn.preprocess_map(frozen.indexed_map(stream, row))
        image.flush()
        del image
        tmp.replace(path)
        save_json(receipt, {"pack_sha256":pack_sha,"cnn_sha256":frozen.CNN,"sha256":sha(path)})
    return np.load(path, mmap_mode="r", allow_pickle=False)


def masked_targets(x, rng):
    import torch
    from torch.nn import functional as F
    # Region density is observed failed-die mass / observed valid-die mass before masking.
    v, f = F.avg_pool2d(x,8).unbind(1)
    target = f / v.clamp_min(1e-12)
    values = torch.as_tensor(rng.random(tuple(v.shape)), dtype=x.dtype, device=x.device)
    active = v > 0
    selected = (values < POLICY["masked_fraction"]) & active
    # Each wafer contributes at least one valid cell. The rule only uses coverage and RNG.
    fallback = values.masked_fill(~active, 2.).flatten(1).argmin(1)
    extra = F.one_hot(fallback,64).bool().reshape_as(selected) & ~selected.any((1,2))[:,None,None]
    selected |= extra & active
    x[:,1].masked_fill_(selected.repeat_interleave(8,1).repeat_interleave(8,2), 0)
    return x, target, selected * v


def pretrain(x, out, cnn, frozen, device, binding):
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    cnn.seed_everything(SEED)
    model = cnn.build_model("medium").to(device)
    # Same feature blocks as the classifier; leave its unused 4x4 pooling and head out of this objective.
    encoder = model.features[:-2]
    decoder = nn.Conv2d(128,1,1).to(device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=.001, weight_decay=.0001, foreach=False)
    order = np.random.default_rng(SEED).permutation(len(x))
    steps = POLICY["ssl_steps"]
    ck = out / "pretraining.pt"
    start, total = 0, 0.
    if ck.exists():
        state = torch.load(ck,map_location="cpu",weights_only=True)
        check(state["binding"] == binding and 0 <= state["completed"] <= steps, "Different SSL checkpoint")
        encoder.load_state_dict(state["encoder"]); decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        start, total = state["completed"], state["objective_sum"]
        del state
    for n in range(start, steps):
        batch = order[n*BATCH:(n+1)*BATCH]
        x_batch = frozen.input_batch(x, batch, device)
        inputs, target, weights = masked_targets(x_batch, np.random.default_rng([SEED, n, 51]))
        prediction = decoder(encoder(inputs)).squeeze(1)
        loss = (F.binary_cross_entropy_with_logits(prediction, target, reduction="none") * weights).sum() / weights.sum()
        check(bool(torch.isfinite(loss)), "Nonfinite masked-density loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
        if (n+1) % 32 == 0 or n+1 == steps:
            tmp = ck.with_suffix(".part")
            torch.save({"binding":binding,"completed":n+1,"objective_sum":total,"encoder":encoder.state_dict(),
                        "decoder":decoder.state_dict(),"optimizer":optimizer.state_dict()},tmp)
            tmp.replace(ck)
            print(f"SSL updates {n+1}/{steps}; mean masked-cell BCE {total/(n+1):.6f}",flush=True)
    return {key:value.detach().cpu() for key,value in encoder.state_dict().items()}


def train(args):
    # Loading the frozen trainer first sets the cuBLAS reproducibility environment before CUDA initialization.
    require_hash(Path(__file__).with_name("train_imbalance.py"), TRAIN_SOURCE)
    import train_imbalance as frozen
    from types import SimpleNamespace
    import torch
    check(torch.cuda.is_available(), "A matching free hosted GPU is required")
    device = torch.device("cuda")
    cnn = frozen.source_module(args.repo)
    cnn.seed_everything(SEED)
    control_protocol, runtime, control_sha = control_info(args.control, frozen, device)
    rows = validate_pack(args.pack, args.pack_sha)
    binding = {"script_sha256":source_sha(), "pack_receipt_sha256":args.pack_sha, "control_run_sha256":control_sha,
               "pool_receipt_sha256":POOL_RECEIPT, "runtime":runtime, "policy":POLICY}
    token = frozen.fingerprint(binding)
    final = args.out/"ssl_fit.json"
    supervised_out = args.out/"supervised"
    run_path = supervised_out/"fits/cnn_ce_s42/run.json"
    if final.exists():
        completed = read_json(final)
        check(completed["fit_binding"] == token and completed["status"] == "complete", "Different finished run")
        require_hash(run_path, completed["train_run_sha256"])
        r = read_json(run_path)
        require_hash(run_path.parent/"checkpoint.pt", r["checkpoint_sha256"])
        for f in r["files"]:
            require_hash(supervised_out/"probabilities"/f["file"],f["sha256"])
        print("Completed SSL run and predictions verified",flush=True)
        return
    with budget(args.out,"train",POLICY["gpu_minutes"]):
        args.cache.mkdir(parents=True,exist_ok=True)
        # Whole-source SHA is checked before the first selected payload read. KaggleHub files
        # are already local to Colab. Use a local cache path if the raw source is on Drive.
        raw = frozen.verified_source(args.raw,args.cache)
        x_unlab = get_images(raw,rows,args.cache,frozen,cnn,args.pack_sha)
        x_unlab = frozen.device_cache(x_unlab,device)
        encoder_state = pretrain(x_unlab,args.out,cnn,frozen,device,token)
        del x_unlab
        torch.cuda.empty_cache()
        records, labels, idx = frozen.load_records(args.repo)
        x, cache_info = frozen.prepare(raw,args.cache,records,cnn)
        check(cache_info["sha256"] == control_protocol["cache_sha256"], "Supervised images differ from matched CE")
        weights, counts = cnn.class_weights(labels,idx["train"],"sqrt_inverse")
        check(counts.tolist() == control_protocol["training_class_counts"], "Different supervised labels")
        def initialized(size):
            model = cnn.build_model(size)
            model.features[:-2].load_state_dict(encoder_state,strict=True)
            return model
        # fit internally resets seed 42 before construction. The nine-class head is drawn
        # exactly as in its CE control; only encoder parameters/BN buffers transfer.
        factory = SimpleNamespace(seed_everything=cnn.seed_everything, build_model=initialized)
        (supervised_out/"probabilities").mkdir(parents=True,exist_ok=True)
        fit = frozen.fit("cnn_ce",SEED,factory,frozen.device_cache(x,device),labels,idx,records,
                         weights,device,supervised_out,{"ssl_binding":token, "control":runtime})
        control_epochs = read_json(args.control/"fits/cnn_ce_s42/run.json")["epochs"]
        for a,b in zip(fit["epochs"],control_epochs,strict=True):
            keys = ("epoch","steps","examples","unique_examples","class_draws","rotated")
            check(all(a[k] == b[k] for k in keys),"Supervised sample count/order protocol diverged")
        save_json(final, {"status":"complete","model":"ssl_masked_s42","seed":SEED,"fit_binding":token,
                  "control_run_sha256":control_sha,"train_run_sha256":sha(run_path),"pool_receipt_sha256":POOL_RECEIPT,
                  "pack_receipt_sha256":args.pack_sha,"script_sha256":source_sha(),"pretraining_sha256":sha(args.out/"pretraining.pt"),
                  "supervised_fit_receipt":"supervised/fits/cnn_ce_s42/run.json","policy":POLICY,"runtime":runtime,
                  "scope":"single-seed, previously inspected development audit, no new independent test"})
        print("Single-seed SSL fine-tune complete. Private paired-evaluation input:", final, flush=True)


def self_test():
    import numpy as np
    import torch
    x = torch.zeros(2,2,64,64)
    x[:,0,12:44,12:44] = 1
    x[0,1,20:24,20:24] = 1
    original = x.clone()
    masked,y,w = masked_targets(x,np.random.default_rng(SEED))
    check(torch.equal(masked[:,0],original[:,0]), "Valid channel changed")
    cells = (w>0).repeat_interleave(8,1).repeat_interleave(8,2)
    check((masked[:,1][cells] == 0).all() and torch.equal(masked[:,1][~cells],original[:,1][~cells]), "Mask leakage")
    check((w.sum((1,2)) > 0).all() and y.min()>=0 and y.max()<=1, "Wrong density support")
    check(torch.equal(masked_targets(original.clone(),np.random.default_rng(SEED))[2],w), "Masks not reproducible")
    print("PASS synthetic valid-die masking, prediction targets and seeded mask reproducibility.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage",required=True)
    cpu = stages.add_parser("pack")
    for name in ("pool","raw","out"):
        cpu.add_argument("--"+name,type=Path,required=True)
    gpu = stages.add_parser("train")
    for name in ("pack","control","repo","raw","cache","out"):
        gpu.add_argument("--"+name,type=Path,required=True)
    gpu.add_argument("--pack-sha",required=True)
    stages.add_parser("self-test")
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    check(platform.system()=="Linux" and Path("/content").is_dir() and
          bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION")),
          "Real-data pilot is restricted to hosted Colab")
    args.out.mkdir(parents=True,exist_ok=True)
    (pack if args.stage=="pack" else train)(args)


if __name__ == "__main__":
    main()
