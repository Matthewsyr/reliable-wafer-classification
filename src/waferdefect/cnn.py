"""Reproducible, CPU-friendly PyTorch wafer-map classifiers.

This module has no dataset download, split selection, or final-test evaluation.
Inputs are already aligned development arrays; callers own source-row manifests.
All class weights are calculated from train_idx only. Validation chooses an
epoch by unweighted nine-class Macro-F1, with exact ties retaining the first epoch.

Official API references checked during implementation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate
https://docs.pytorch.org/docs/stable/notes/randomness.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

Deterministic settings target reruns on the same software/device. PyTorch does
not guarantee identical numerics across releases, platforms, or CPU versus GPU.
"""

from pathlib import Path
import hashlib
import json
import os
import platform
import random
import time

# Must be set before the first CUDA BLAS operation; do not silently relax it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
CHANNELS = {"small": [16, 32, 64], "medium": [32, 64, 128]}
PREPROCESSING = {
    "size": 64, "channels": ["valid_die_coverage", "failed_die_coverage"],
    "padding": "zero pad to max(H,W) square, centred; extra odd cell on bottom/right",
    "resize": "CPU float32 torch.nn.functional.interpolate mode=area",
    "normalization": "none; mask coverage values remain in [0,1]",
    "augmentation": "none",
}


def preprocess_map(wafer_map, size=64):
    """Return float32 [2,size,size] mask coverage, preserving aspect ratio.

    Channel 0 is valid (1 or 2), channel 1 is failed (2). Background is zero
    in both. Discrete area interpolation averages each channel independently;
    it retains nonzero contributions from thin failures during downsampling but
    does not promise exact conservation of physical die counts or defect shape.
    No resizing is part of the autograd graph, and no class label is consulted.
    """
    a = np.asarray(wafer_map)
    if a.ndim != 2 or a.size == 0 or a.dtype.kind not in "iuf":
        raise ValueError("Expected a nonempty 2-D numeric categorical map")
    if np.any((a != 0) & (a != 1) & (a != 2)) or not np.any(a != 0):
        raise ValueError("Map must contain 0/1/2 and at least one valid die")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    h, w = a.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    masks = np.zeros((2, side, side), dtype=np.float32)
    masks[0, top:top+h, left:left+w] = a != 0
    masks[1, top:top+h, left:left+w] = a == 2
    with torch.no_grad():
        out = F.interpolate(torch.from_numpy(masks)[None], size=(size, size), mode="area")[0].numpy()
    if not np.isfinite(out).all() or np.any(out < 0) or np.any(out > 1) or np.any(out[1] > out[0]):
        raise RuntimeError("Preprocessing violated mask coverage invariants")
    if np.any(a == 2) and not np.any(out[1] > 0):
        raise RuntimeError("Resizing lost every failed-die contribution")
    return np.ascontiguousarray(out, dtype=np.float32)


class WaferCNN(nn.Module):
    """Three single-convolution blocks, followed by a spatial 4x4 classifier."""
    def __init__(self, model_size="small", n_classes=9):
        super().__init__()
        if model_size not in CHANNELS or n_classes != len(CLASSES):
            raise ValueError("Expected small/medium model and the fixed nine classes")
        self.model_size = model_size
        layers, previous = [], 2
        for channels in CHANNELS[model_size]:
            layers.extend([
                nn.Conv2d(previous, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            ])
            previous = channels
        # Fixed 64 -> 32 -> 16 -> 8 -> 4. This matches adaptive 8->4 bins,
        # while avoiding AdaptiveAvgPool2d backward determinism restrictions.
        layers.extend([nn.AvgPool2d(2), nn.Flatten()])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous * 4 * 4, n_classes)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (2, 64, 64):
            raise ValueError("CNN expects [batch,2,64,64]")
        return self.classifier(self.features(x))  # logits; no softmax before CE


def build_model(model_size="small", n_classes=9):
    return WaferCNN(model_size=model_size, n_classes=n_classes)


def seed_everything(seed, num_threads=2):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _device(value):
    dev = torch.device(value)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("This reproducibility protocol supports cpu or cuda")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    return dev


def _indices(indices, length, name):
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    a = np.asarray(indices)
    if a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0:
        raise ValueError(f"{name} must be a nonempty 1-D integer array")
    a = a.astype(np.int64, copy=False)
    if a.min() < 0 or a.max() >= length or len(np.unique(a)) != len(a):
        raise ValueError(f"{name} has invalid or duplicate indices")
    return a


def _labels(y, length):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    a = np.asarray(y)
    if a.shape != (length,) or a.dtype.kind not in "iu" or np.any((a < 0) | (a >= 9)):
        raise ValueError("y must be an integer vector with labels 0..8")
    return a.astype(np.int64, copy=False)


def _check_x(x):
    if not isinstance(x, (np.ndarray, torch.Tensor)) or x.ndim != 4 or tuple(x.shape[1:]) != (2, 64, 64):
        raise ValueError("x must be a numpy array/memmap or CPU tensor [N,2,64,64]")
    if isinstance(x, torch.Tensor):
        if x.device.type != "cpu" or x.dtype != torch.float32 or x.requires_grad:
            raise ValueError("Store x as a CPU float32 tensor without gradients")
    elif x.dtype != np.float32:
        raise ValueError("Store x as float32; preprocessing is not performed in training")


def _take(x, indices):
    # Advanced indexing copies only the requested mini-batch, never all of x.
    if isinstance(x, torch.Tensor):
        return x.index_select(0, torch.from_numpy(np.asarray(indices, dtype=np.int64)))
    return torch.from_numpy(np.ascontiguousarray(x[indices]))


def _batches(indices, batch_size):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(indices), batch_size):
        yield indices[start:start+batch_size]


def _check_coverage(x, indices, batch_size):
    for part in _batches(indices, batch_size):
        a = _take(x, part).numpy()
        if (not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1)
                or np.any(a[:, 1] > a[:, 0] + 1e-6)
                or np.any(a[:, 0].sum(axis=(1, 2)) <= 0)):
            raise ValueError("x violates finite valid/failure coverage constraints")


def class_weights(y, train_idx, weighting="none"):
    """Training-only class weights; mean weight over nine classes equals one.

    sqrt_inverse = sqrt(1 / training_class_count), divided by its class mean.
    Weighted CE also divides each batch's weighted loss by its sum of weights.
    No sampler reweighting or validation frequencies are used.
    """
    y = _labels(y, len(y))
    train_idx = _indices(train_idx, len(y), "train_idx")
    counts = np.bincount(y[train_idx], minlength=9)
    if np.any(counts == 0):
        raise ValueError("Every class must be represented in training")
    if weighting == "none":
        weights = np.ones(9, dtype=np.float64)
    elif weighting == "sqrt_inverse":
        weights = np.sqrt(1.0 / counts)
        weights /= weights.mean()
    else:
        raise ValueError("weighting must be none or sqrt_inverse")
    return weights.astype(np.float32), counts


def classification_metrics(y_true, y_pred):
    """Unweighted fixed-nine-class metrics; JSON-serializable, zero_division=0."""
    truth, pred = _labels(y_true, len(y_true)), _labels(y_pred, len(y_true))
    if len(truth) == 0:
        raise ValueError("Cannot evaluate empty predictions")
    cm = np.bincount(9 * truth + pred, minlength=81).reshape(9, 9)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(9, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(9, dtype=float), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(9, dtype=float), where=support + predicted > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "per_class": [{"label": c, "precision": float(precision[k]), "recall": float(recall[k]),
                           "f1": float(f1[k]), "support": int(support[k])} for k, c in enumerate(CLASSES)],
            "confusion_matrix": cm.tolist()}


def predict_batches(model, x, indices=None, *, batch_size=256, device="cpu"):
    """Return int64 class indices in exactly the requested row order."""
    _check_x(x)
    ids = np.arange(len(x), dtype=np.int64) if indices is None else _indices(indices, len(x), "indices")
    if len(ids) == 0:
        return np.empty(0, dtype=np.int64)
    dev = _device(device)
    model.to(dev).eval()
    result = []
    with torch.inference_mode():
        for part in _batches(ids, batch_size):
            logits = model(_take(x, part).to(dev))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite inference logits")
            result.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(result).astype(np.int64, copy=False)


def _evaluate(model, x, y, indices, weights, batch_size, dev):
    model.eval()
    predictions, numerator, denominator = [], 0.0, 0.0
    with torch.inference_mode():
        for part in _batches(indices, batch_size):
            target = torch.from_numpy(y[part]).to(dev)
            logits = model(_take(x, part).to(dev))
            loss_sum = F.cross_entropy(logits, target, weight=weights, reduction="sum")
            numerator += float(loss_sum.item())
            denominator += float(weights[target].sum().item())
            predictions.append(logits.argmax(1).cpu().numpy())
    if not np.isfinite(numerator):
        raise RuntimeError("Nonfinite validation loss")
    pred = np.concatenate(predictions).astype(np.int64, copy=False)
    return pred, numerator / denominator, classification_metrics(y[indices], pred)


def _epoch(model, optimizer, x, y, indices, weights, batch_size, dev, rng):
    model.train()
    numerator, denominator = 0.0, 0.0
    order = rng.permutation(indices)
    for part in _batches(order, batch_size):
        inputs, target = _take(x, part).to(dev), torch.from_numpy(y[part]).to(dev)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = F.cross_entropy(model(inputs), target, weight=weights, reduction="sum")
        weight_sum = weights[target].sum()
        loss = loss_sum / weight_sum
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        numerator += float(loss_sum.detach().item())
        denominator += float(weight_sum.item())
    return numerator / denominator


def _json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_tree(v) for v in obj)
    return obj


def _save_checkpoint(path, model, optimizer, config, epoch, metrics, history, rng):
    payload = {
        "format_version": 1, "architecture": "WaferCNN_singleconv_v1",
        "model_size": model.model_size, "channels": CHANNELS[model.model_size],
        "class_order": CLASSES, "preprocessing": PREPROCESSING,
        "model_state_dict": _cpu_tree(model.state_dict()),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "config": config, "epoch": int(epoch), "metrics": metrics, "history": history,
        "rng_state": {"torch_cpu": torch.get_rng_state().cpu(),
                      "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                      "numpy_batch_shuffle": rng.bit_generator.state},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "platform": platform.platform()},
    }
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    """Load only a trusted project checkpoint; returns (model, payload)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (payload.get("format_version") != 1 or payload.get("architecture") != "WaferCNN_singleconv_v1"
            or payload.get("class_order") != CLASSES or payload.get("preprocessing") != PREPROCESSING):
        raise ValueError("Checkpoint architecture/classes/preprocessing mismatch")
    model = build_model(payload["model_size"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(_device(device)).eval()
    return model, payload


def _setup(x, y, train_idx, *, seed, model_size, weighting, epochs, batch_size, device, output_dir, val_idx=None):
    _check_x(x)
    y = _labels(y, len(x))
    ti = _indices(train_idx, len(x), "train_idx")
    vi = None if val_idx is None else _indices(val_idx, len(x), "val_idx")
    if vi is not None and np.intersect1d(ti, vi).size:
        raise ValueError("Training and validation rows overlap")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    dev = _device(device)
    seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output_dir must be empty for each independent fit")
    _json(output / "run_status.json", {"status": "incomplete", "final_test_evaluated": False})
    _check_coverage(x, ti if vi is None else np.concatenate([ti, vi]), batch_size)
    weights_np, counts = class_weights(y, ti, weighting)
    model = build_model(model_size).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False)
    config = {
        "seed": seed, "model_size": model_size, "channels": CHANNELS[model_size],
        "weighting": weighting, "class_counts_train": counts.tolist(), "class_weights": weights_np.tolist(),
        "epochs_budget": epochs, "batch_size": batch_size, "device": str(dev),
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "foreach": False,
        "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False,
        "num_threads": torch.get_num_threads(), "batch_loader": "single-process, indexed minibatches",
        "train_indices_sha256": hashlib.sha256(ti.astype("<i8").tobytes()).hexdigest(),
        "val_indices_sha256": None if vi is None else hashlib.sha256(vi.astype("<i8").tobytes()).hexdigest(),
        "train_rows": len(ti), "validation_rows": 0 if vi is None else len(vi),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selection_rule": "fixed epochs; no validation" if vi is None else "maximum validation nine-class Macro-F1; exact ties keep earliest epoch",
        "preprocessing": PREPROCESSING, "class_order": CLASSES,
        "loss_reporting": "weighted CE sum / sum of training-derived class weights; metrics are unweighted",
        "final_test_evaluated": False,
    }
    _json(output / "config.json", config)
    return y, ti, vi, dev, output, model, optimizer, torch.tensor(weights_np, device=dev), config, np.random.default_rng(seed)


def train_cnn(x, y, train_idx, val_idx, *, seed, model_size, weighting="none", epochs=8,
              batch_size=256, device="cpu", output_dir):
    """Train from scratch and return best-validation predictions and evidence.

    predictions/train_predictions are NumPy arrays in the corresponding input
    index order; save them separately. Other returned values are JSON-compatible.
    fit_seconds includes training and per-epoch validation/model-selection work.
    Train Macro-F1 is measured once using the restored best model in eval mode.
    The default budget is eight epochs, fixed before any real CNN fit using a
    synthetic CPU throughput check. Explicit epochs remain fully supported.
    """
    y, ti, vi, dev, out, model, optimizer, weights, config, rng = _setup(
        x, y, train_idx, seed=seed, model_size=model_size, weighting=weighting, epochs=epochs,
        batch_size=batch_size, device=device, output_dir=output_dir, val_idx=val_idx)
    history, best_f1, best_epoch = [], -1.0, None
    checkpoint_path = out / "best.pt"
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _epoch(model, optimizer, x, y, ti, weights, batch_size, dev, rng)
        _, val_loss, metrics = _evaluate(model, x, y, vi, weights, batch_size, dev)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
               "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            _save_checkpoint(checkpoint_path, model, optimizer, config, epoch, metrics, list(history), rng)
        _json(out / "history.json", history)
        print(f"epoch={epoch:02d}/{epochs} size={model_size} weighting={weighting} seed={seed} "
              f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_macro_f1={metrics['macro_f1']:.5f} "
              f"best_epoch={best_epoch} seconds={row['epoch_seconds']:.1f}", flush=True)
    fit_seconds = time.perf_counter() - start
    # Loading the serialized state prevents the common mutable-best-state bug.
    model, payload = load_checkpoint(checkpoint_path, device=dev)
    start = time.perf_counter()
    pred = predict_batches(model, x, vi, batch_size=batch_size, device=dev)
    predict_seconds = time.perf_counter() - start
    metrics = classification_metrics(y[vi], pred)
    if metrics != payload["metrics"]:
        raise RuntimeError("Restored checkpoint metrics differ from saved best validation metrics")
    start = time.perf_counter()
    train_pred = predict_batches(model, x, ti, batch_size=batch_size, device=dev)
    train_predict_seconds = time.perf_counter() - start
    train_metrics = classification_metrics(y[ti], train_pred)
    result = {"best_epoch": best_epoch, "history": history, "fit_seconds": fit_seconds,
              "predict_seconds": predict_seconds, "train_predict_seconds": train_predict_seconds,
              "parameter_count": config["parameter_count"], "checkpoint_path": str(checkpoint_path),
              "metrics": metrics, "train_metrics": train_metrics,
              "train_macro_f1": train_metrics["macro_f1"], "val_macro_f1": metrics["macro_f1"], "config": config}
    _json(out / "result.json", result)
    _json(out / "run_status.json", {"status": "completed", "best_epoch": best_epoch,
                                   "epochs_completed": epochs, "final_test_evaluated": False})
    result.update(predictions=pred, train_predictions=train_pred)
    return result


def final_train(x, y, train_idx, *, seed, model_size, weighting="none", epochs,
                batch_size=256, device="cpu", output_dir):
    """Fresh initialization; train exactly frozen epochs without any validation.

    The returned model is an nn.Module and must be excluded from JSON outputs.
    This function never receives final-test data or chooses an epoch using it.
    """
    y, ti, _, dev, out, model, optimizer, weights, config, rng = _setup(
        x, y, train_idx, seed=seed, model_size=model_size, weighting=weighting, epochs=epochs,
        batch_size=batch_size, device=device, output_dir=output_dir)
    history, checkpoint_path = [], out / "final.pt"
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        loss = _epoch(model, optimizer, x, y, ti, weights, batch_size, dev, rng)
        history.append({"epoch": epoch, "train_loss": loss, "epoch_seconds": time.perf_counter() - epoch_start})
        _json(out / "history.json", history)
        print(f"final_train epoch={epoch:02d}/{epochs} size={model_size} weighting={weighting} "
              f"seed={seed} train_loss={loss:.5f} seconds={history[-1]['epoch_seconds']:.1f}", flush=True)
    _save_checkpoint(checkpoint_path, model, optimizer, config, epochs, None, history, rng)
    fit_seconds = time.perf_counter() - start
    model.eval()
    result = {"history": history, "fit_seconds": fit_seconds, "parameter_count": config["parameter_count"],
              "checkpoint_path": str(checkpoint_path), "config": config, "epochs_completed": epochs}
    _json(out / "result.json", result)
    _json(out / "run_status.json", {"status": "completed", "epochs_completed": epochs, "final_test_evaluated": False})
    result["model"] = model
    return result


def benchmark_cpu(batch_size=64, batches=3, model_size="small"):
    """Synthetic CPU throughput check; includes one untimed training warmup.

    Extrapolate train_rows / training_samples_per_second per training epoch and
    val_rows / prediction_samples_per_second for each validation pass. Actual
    memmap I/O, checkpointing and hardware load can change full-run throughput.
    """
    seed_everything(917)
    model = build_model(model_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False)
    x = torch.rand(batch_size, 2, 64, 64)
    x[:, 1] *= x[:, 0]
    y = torch.arange(batch_size, dtype=torch.long) % 9
    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    model.train()
    step()
    start = time.perf_counter()
    for _ in range(batches):
        step()
    train_seconds = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        model(x)
        start = time.perf_counter()
        for _ in range(batches):
            model(x)
        prediction_seconds = time.perf_counter() - start
    return {"synthetic_only": True, "model_size": model_size, "batch_size": batch_size, "batches": batches,
            "num_threads": torch.get_num_threads(), "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_seconds": train_seconds, "prediction_seconds": prediction_seconds,
            "training_samples_per_second": batch_size * batches / train_seconds,
            "prediction_samples_per_second": batch_size * batches / prediction_seconds,
            "scope": "Synthetic tensor benchmark only; no real-data accuracy or training-time guarantee"}


def synthetic_smoke_test(output_dir):
    """Exercise preprocessing, leakage guards, training/reload and fixed refit.

    This is a software check, not a real-data experiment. Call explicitly in a
    fresh directory. Two small fits verify same-seed repeatability on CPU.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Synthetic smoke output directory must be empty")
    _json(out / "SYNTHETIC_ONLY.json", {"synthetic_only": True, "real_data_trained": False})
    normal = preprocess_map(np.ones((4, 12), dtype=np.uint8))
    assert normal.shape == (2, 64, 64) and normal.dtype == np.float32
    assert normal[1].sum() == 0 and normal[0, 0].sum() == 0
    thin = np.ones((128, 128), dtype=np.uint8)
    thin[:, 19] = 2
    reduced = preprocess_map(thin)
    assert np.any((reduced[1] > 0) & (reduced[1] < 1))
    assert reduced[1].sum() > 0 and np.all(reduced[1] <= reduced[0])
    for bad in [np.zeros((3, 3)), np.array([[1, np.nan]]), np.array([[3]]), np.ones(3)]:
        try:
            preprocess_map(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed map accepted")
    maps, labels = [], []
    for k in range(9):
        for replicate in range(6):
            a = np.ones((24, 32), dtype=np.uint8)
            row, col = 2 + (k // 3) * 7, 2 + (k % 3) * 9
            a[row:row+3, col:col+3] = 2
            maps.append(preprocess_map(a))
            labels.append(k)
    x, y = np.stack(maps), np.asarray(labels, dtype=np.int64)
    ti = np.array([i for i in range(len(y)) if i % 6 < 4], dtype=np.int64)
    vi = np.array([i for i in range(len(y)) if i % 6 >= 4], dtype=np.int64)
    w1, _ = class_weights(y, ti, "sqrt_inverse")
    changed = y.copy(); changed[vi] = 0
    w2, _ = class_weights(changed, ti, "sqrt_inverse")
    assert np.array_equal(w1, w2), "Validation labels affected class weights"
    params = dict(seed=17, model_size="small", weighting="sqrt_inverse", epochs=2, batch_size=18, device="cpu")
    a = train_cnn(x, y, ti, vi, output_dir=out / "repeat_a", **params)
    b = train_cnn(x, y, ti, vi, output_dir=out / "repeat_b", **params)
    assert a["best_epoch"] == b["best_epoch"] and np.array_equal(a["predictions"], b["predictions"])
    ma, pa = load_checkpoint(a["checkpoint_path"])
    _, pb = load_checkpoint(b["checkpoint_path"])
    assert all(torch.equal(pa["model_state_dict"][k], pb["model_state_dict"][k]) for k in pa["model_state_dict"])
    assert np.array_equal(predict_batches(ma, x, vi, batch_size=18), a["predictions"])
    with torch.inference_mode():
        assert build_model("medium").eval()(torch.from_numpy(x[:2])).shape == (2, 9)
    try:
        train_cnn(x, y, ti, ti, output_dir=out / "overlap_rejected", **params)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlapping train/validation accepted")
    final = final_train(x, y, np.arange(len(y)), seed=17, model_size="small", epochs=1,
                        batch_size=18, device="cpu", output_dir=out / "fixed_refit")
    _, pf = load_checkpoint(final["checkpoint_path"])
    assert pf["epoch"] == 1 and pf["metrics"] is None and pf["config"]["validation_rows"] == 0
    result = {"status": "passed", "synthetic_only": True, "real_data_trained": False,
              "same_seed_weights_identical": True, "checkpoint_prediction_roundtrip": True,
              "training_only_class_weights": True, "overlap_rejected": True,
              "fixed_epoch_refit_without_validation": True,
              "parameter_counts": {name: sum(p.numel() for p in build_model(name).parameters()) for name in CHANNELS}}
    _json(out / "smoke_checks.json", result)
    return result
