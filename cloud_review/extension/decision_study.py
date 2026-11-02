"""Hosted, development-only WM811K classification and inspection policy experiment.

python decision_study.py --archive ARCHIVE --out NEW_DIRECTORY [--new TRAINING_OUTPUT]
python decision_study.py --self-test
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np


CLASSES = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none")
SCRATCH, NORMAL = 7, 8
RF, RF77, CNN = OLD = ("rf_original71", "rf_linearity77", "cnn_bn_natural_7ep")
NEW = ("cnn_ce", "cnn_weighted", "cnn_sampled", "cnn_rot180", "cnn_focal")
ENSEMBLE = "ensemble_rf77_cnn"
SEEDS = (42, 123, 2026)
SCENARIOS = {"uniform_1000": 1000, "scratch_3000": 3000}
REVIEW_FP = 5
SPLIT_SALT = "wm811k-decision-component-20261101-v1|"
LEGACY_BETA = 1.9147552052282766
QUEUE_SEED = 20261101
PIN = "c86c2c3d77762d6d8083545bbb3377b09c13eb09d7119c17426b933602d0a0a7"
HEAD = ["source_row", "truth", *CLASSES]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=convert), encoding="utf8")


def read_table(path):
    with gzip.open(path, "rt", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def checked_file(base, spec):
    name = spec["file"]
    require(Path(name).name == name, "Manifest must contain simple filenames")
    path = base / name
    require(path.stat().st_size == spec["bytes"] and digest(path) == spec["sha256"], f"Input changed: {name}")
    return path


def inputs(archive, new):
    manifest = archive / "manifest.json"
    require(digest(manifest) == PIN, "Original score manifest does not match frozen public archive")
    m = json.loads(manifest.read_text())
    _, split = read_table(checked_file(archive, m["split_reference"]))
    require(len(split) == m["split_reference"]["rows"] and len({r["source_row"] for r in split}) == len(split),
            "Duplicate/missing source identities")
    for r in split:
        role = r["role"]
        require(role in ("train", "tune", "audit"), f"Unexpected input role {role}")
        if role == "tune":
            cal = hashlib.sha256((SPLIT_SALT + r["component"]).encode()).digest()[0] < 128
            r["partition"] = "calibration" if cal else "selection"
        else:
            r["partition"] = role
    for field in ("lot", "component", "map_sha256"):
        assignment = {}
        for r in split:
            require(bool(r[field]) and assignment.setdefault(r[field], r["partition"]) == r["partition"],
                    f"Cross-partition overlap or null {field}")

    records = {}
    def register(base, manifest, rows, models):
        require(tuple(manifest["class_order"]) == CLASSES and tuple(manifest["probability_columns"]) == CLASSES,
                "Class order mismatch")
        expected = {(k, s, role) for k in models for s in SEEDS for role in ("tune", "audit")}
        require(len(rows) == len(expected) and {(r["model"], r["seed"], r["role"]) for r in rows} == expected,
                "Incomplete probability archive")
        for r in rows:
            records[r["model"], r["seed"], r["role"]] = (base, r)

    register(archive, m, m["probability_files"], OLD)
    if new:
        base = new / "probabilities"
        m = json.loads((base / "manifest.json").read_text())
        require(m["complete"] is True, "New training is incomplete")
        register(base, m, m["files"], NEW)
    return records, split


def scores(records, refs, role):
    ids = [r["source_row"] for r in refs]
    truth = [r["truth"] for r in refs]
    result = {}
    for (model, seed, record_role), (base, spec) in records.items():
        if record_role != role:
            continue
        fields, rows = read_table(checked_file(base, spec))
        by_id = {r["source_row"]: r for r in rows}
        require(fields == HEAD and len(by_id) == len(ids) == len(rows) == spec["rows"] and by_id.keys() == set(ids),
                f"Score schema or IDs changed: {spec['file']}")
        aligned = [by_id[i] for i in ids]
        require(truth == [r["truth"] for r in aligned], "Truth/identity mismatch")
        p = np.array([[float(r[c]) for c in CLASSES] for r in aligned])
        require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                and np.allclose(p.sum(axis=1), 1, atol=2e-6, rtol=0), "Invalid probabilities")
        result[model, seed] = p
    for seed in SEEDS:
        result[ENSEMBLE, seed] = (result[RF77, seed] + result[CNN, seed]) / 2
    return result


def softmax(z):
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def calibrated(p, beta):
    return softmax(np.log(np.maximum(p, 1e-12)) * beta)


def fit_beta(p, y):
    logp = np.log(np.maximum(p, 1e-12))
    observed = logp[np.arange(len(y)), y].mean()
    def derivative(beta):
        return (softmax(logp * beta) * logp).sum(axis=1).mean() - observed
    lo, hi = .125, 8.
    if derivative(lo) >= 0:
        return lo
    if derivative(hi) <= 0:
        return hi
    for _ in range(40):
        middle = (lo + hi) / 2
        if derivative(middle) > 0:
            hi = middle
        else:
            lo = middle
    return (lo + hi) / 2


def averages(cm):
    n, pred = cm.sum(axis=1), cm.sum(axis=0)
    def div(a, b):
        return np.divide(a, b, out=np.zeros(9), where=b != 0)
    d = np.diag(cm)
    return n, pred, div(d, pred), div(d, n), div(2*d, n+pred)


def confusion(y, pred):
    return np.bincount(y * 9 + pred, minlength=81).reshape(9, 9)


def default_guess(p):
    return np.argmax(p, axis=1)


def scratch_guess(p, threshold):
    if threshold is None:
        return default_guess(p)
    other = p.copy()
    other[:, SCRATCH] = -1
    return np.where(p[:, SCRATCH] >= threshold, SCRATCH, other.argmax(axis=1))


def boundaries(score):
    order = np.argsort(-score, kind="stable")
    v = score[order]
    ends = np.flatnonzero(np.r_[v[:-1] != v[1:], True]) + 1
    return order, np.r_[0, ends], np.r_[np.nextafter(v[0], np.inf), v[ends-1]]


def prefix(values, order, counts):
    return np.r_[0, np.cumsum(values[order])][counts]


def average_precision_score(hits, score):
    # Non-interpolated one-vs-rest AP. Tied scores are evaluated as one threshold.
    order, counts, _ = boundaries(score)
    cumulative = prefix(hits, order, counts)
    return (np.diff(cumulative) * cumulative[1:] / counts[1:]).sum() / cumulative[-1]


def scratch_rule(y, p):
    identity = confusion(y, default_guess(p))
    baseline = (int(identity[SCRATCH, SCRATCH]), -int(identity[NORMAL, :NORMAL].sum()), -int(identity[NORMAL, SCRATCH]))
    other = p.copy()
    other[:, SCRATCH] = -1
    other_normal = other.argmax(axis=1) == NORMAL
    is_normal = y == NORMAL
    order, counts, th = boundaries(p[:, SCRATCH])
    tp = prefix(y == SCRATCH, order, counts)
    sfp = prefix(is_normal, order, counts)
    afp = (is_normal & ~other_normal).sum() + prefix(is_normal & other_normal, order, counts)
    ok = np.flatnonzero((tp >= baseline[0]) & (afp <= -baseline[1]) & (sfp <= -baseline[2]))
    if len(ok):
        i = ok[np.lexsort((-th[ok], sfp[ok], afp[ok], -tp[ok]))[0]]
        if (tp[i], -afp[i], -sfp[i]) > baseline:
            return float(th[i])
    return None


def severity(y, scratch_loss):
    return np.where(y == NORMAL, 0, np.where(y == SCRATCH, scratch_loss, 1000))


def benefit(p, scratch_loss):
    return (1000 + REVIEW_FP) * (1 - p[:, NORMAL]) - REVIEW_FP + (scratch_loss - 1000) * p[:, SCRATCH]


def thresholds(y, p):
    defect = y != NORMAL
    order, count, th = boundaries(1-p[:, NORMAL])
    tp = prefix(defect, order, count)
    f1 = 2 * tp / np.maximum(count + defect.sum(), 1)
    cap = np.flatnonzero(count <= len(y) // 5)
    result = dict(f1=float(th[np.argmax(f1)]), capacity=float(th[cap[np.argmax(tp[cap])]]), scratch=scratch_rule(y, p))
    for scenario, loss in SCENARIOS.items():
        order, count, th = boundaries(benefit(p, loss))
        penalties = severity(y, loss)
        avoided = prefix(penalties - REVIEW_FP * ~defect, order, count)
        result[scenario] = float(th[np.argmax(avoided)])
    return result


def hashkeys(values, domain):
    key = hashlib.sha256(f"{QUEUE_SEED}:{domain}".encode()).digest()
    return np.array([int.from_bytes(hashlib.blake2b(v.encode(), key=key, digest_size=8).digest(), "big")
                     for v in values], dtype=np.uint64)


class Queue:
    """Shared, label-blind ordering and slots across every ranking policy and model."""
    def __init__(self, refs, gate):
        self.gate = gate
        subset = np.flatnonzero(gate)
        ids = np.array([refs[i]["source_row"] for i in subset], dtype=str)
        lots = np.array([refs[i]["lot"] for i in subset], dtype=str)
        groups, group_id = np.unique(lots, return_inverse=True)
        rank = np.lexsort((groups, hashkeys(groups, "lot-order")))
        priority = np.empty_like(rank)
        priority[rank] = np.arange(len(rank))
        order = np.lexsort((ids, hashkeys(ids, "within-lot-order"), priority[group_id]))
        self.positions = subset[order]
        self.ids = ids[order]
        self.tie = hashkeys(self.ids, "common-tie-break")
        self.random = (hashkeys(self.ids, "random-policy") >> np.uint64(1)).astype(np.int64)
        self.chunks = [(i, min(i + 1000, len(ids))) for i in range(0, len(ids), 1000)]

    def top(self, score):
        mask = np.zeros(len(self.gate), dtype=bool)
        for a, b in self.chunks:
            rank = np.lexsort((self.ids[a:b], self.tie[a:b], -score[a:b]))[:(b-a)//5] + a
            mask[self.positions[rank]] = True
        return mask


def assessment(y, reviewed, loss):
    defect = y != NORMAL
    fp = int((reviewed & ~defect).sum())
    missed = ~reviewed & defect
    tp = int((reviewed & defect).sum())
    n_review = int(reviewed.sum())
    net_cost = int(severity(y[missed], loss).sum() + fp * REVIEW_FP)
    return dict(arrivals=len(y), normal_n=int((~defect).sum()), defect_n=int(defect.sum()),
                reviewed=n_review, tp=tp, fp=fp, fn=int(missed.sum()), tn=int((~reviewed & ~defect).sum()),
                precision=tp/n_review if n_review else None, recall=tp/defect.sum() if defect.any() else None,
                f1=2*tp/max(1, n_review + defect.sum()), normal_fpr=fp/max(1, (~defect).sum()),
                scratch_captured=int((reviewed & (y == SCRATCH)).sum()),
                cost_dollars=net_cost, cost_per_1000_arrivals=1000*net_cost/len(y))


def classification(tables, key, y, p, guess):
    cm = confusion(y, guess)
    support, count, precision, recall, f1 = averages(cm)
    aps = []
    for j, name in enumerate(CLASSES):
        ap = float(average_precision_score(y == j, p[:, j])) if support[j] else None
        aps.append(ap)
        tables["per_class"].append(dict(**key, label=name, support=support[j], predictions=count[j],
                                        precision=precision[j], recall=recall[j], f1=f1[j], average_precision=ap))
        tables["confusion"].append(dict(**key, actual=name, **dict(zip(CLASSES, cm[j]))))
    row = dict(**key, n=len(y), accuracy=np.trace(cm)/len(y), macro_f1=f1.mean(),
               macro_ap=np.mean([ap for ap in aps if ap is not None]),
               defect_ap=float(average_precision_score(y != NORMAL, 1-p[:, NORMAL])) if np.any(y != NORMAL) else None,
               scratch_ap=aps[SCRATCH], scratch_precision=precision[SCRATCH], scratch_recall=recall[SCRATCH],
               scratch_tp=cm[SCRATCH, SCRATCH], scratch_n=support[SCRATCH], normal_n=support[NORMAL],
               normal_scratch_fp=cm[NORMAL, SCRATCH], normal_any_fp=cm[NORMAL, :NORMAL].sum(),
               normal_any_fpr=1-recall[NORMAL] if support[NORMAL] else None)
    tables["classification"].append(row)
    return row


def calibration(tables, key, y, p):
    log = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-300))
    row = dict(**key, multiclass_nll=log.mean(), multiclass_brier=np.mean(np.sum((p-np.eye(9)[y])**2, axis=1)))
    for name, score, hits in (("defect", 1-p[:, NORMAL], y != NORMAL), ("scratch", p[:, SCRATCH], y == SCRATCH),
                              ("top_label", p.max(axis=1), p.argmax(axis=1) == y)):
        bins = np.clip((score*10).astype(int), 0, 9)
        ece = 0
        for b in range(10):
            m = bins == b
            pred = score[m].mean() if m.any() else None
            obs = hits[m].mean() if m.any() else None
            tables["reliability"].append(dict(**key, target=name, bin=b, lower=b/10, upper=(b+1)/10,
                                              n=m.sum(), mean_probability=pred, observed_fraction=obs))
            if m.any():
                ece += abs(pred-obs)*m.mean()
        row[name+"_ece"] = ece
        row[name+"_brier"] = np.mean((score-hits)**2)
    tables["calibration"].append(row)


def record_routing(tables, key, y, reviewed, loss, gate):
    row = assessment(y, reviewed, loss)
    candidate = assessment(y[gate], reviewed[gate], loss) if gate.any() else None
    row.update(candidates=int(gate.sum()), upstream_fn=int((~gate & (y != NORMAL)).sum()),
               candidate_unreviewed_fn=candidate["fn"] if candidate else 0,
               cost_per_1000_candidates=candidate["cost_per_1000_arrivals"] if candidate else None)
    tables["routing"].append({**key, **row})
    for j, name in enumerate(CLASSES):
        m = y == j
        tables["inspection_by_class"].append(dict(**key, label=name, total=m.sum(), candidates=(m & gate).sum(),
                                                  reviewed=(m & reviewed).sum(), not_reviewed=(m & ~reviewed).sum()))
    return row


def evaluate_phase(raw, refs, configs, phase, tables, require_legacy=False, selected_model=None):
    y = np.array([CLASSES.index(r["truth"]) for r in refs])
    gate = 1-calibrated(raw[RF,42], LEGACY_BETA)[:, NORMAL] >= .5
    queue = Queue(refs, gate)
    if require_legacy:
        require(len(refs) == 11591 and gate.sum() == 3642, "Legacy audit gate mismatch")
    all_rows = np.ones(len(y), dtype=bool)
    queue_ids = np.full(len(y), -1)
    for qid, (a, b) in enumerate(queue.chunks):
        queue_ids[queue.positions[a:b]] = qid
        tables["queue_inventory"].append(dict(phase=phase, queue=qid, candidates=b-a, slots=(b-a)//5,
                                              lots=len({refs[i]["lot"] for i in queue.positions[a:b]})))
    metrics, traces = {}, {}
    for (model, seed), r in raw.items():
        key = dict(phase=phase, model=model, seed=seed)
        config = configs[model, seed]
        p = calibrated(r, config["beta"])
        for name, ps in (("raw", r), ("calibrated", p)):
            calibration(tables, dict(**key, probabilities=name), y, ps)
        base = default_guess(p)
        scratch = scratch_guess(p, config["scratch"])
        argres = classification(tables, dict(**key, rule="argmax"), y, p, base)
        scrres = classification(tables, dict(**key, rule="scratch_joint"), y, p, scratch)
        tables["scratch_guard"].append(dict(**key, threshold=config["scratch"],
                scratch_tp_delta=scrres["scratch_tp"]-argres["scratch_tp"],
                normal_any_fp_delta=scrres["normal_any_fp"]-argres["normal_any_fp"],
                normal_scratch_fp_delta=scrres["normal_scratch_fp"]-argres["normal_scratch_fp"],
                all_three_hold=bool(scrres["scratch_tp"] >= argres["scratch_tp"] and
                    scrres["normal_any_fp"] <= argres["normal_any_fp"] and scrres["normal_scratch_fp"] <= argres["normal_scratch_fp"])))
        pd = 1-p[:, NORMAL]
        rnd = queue.top(queue.random)
        any_top = queue.top(pd[queue.positions])
        for sc, loss in SCENARIOS.items():
            value = benefit(p, loss)
            # Under equal loss the two score orderings are algebraically identical.
            loss_top = any_top if loss == 1000 else queue.top(value[queue.positions])
            selections = {"argmax": base != NORMAL, "scratch_joint": scratch != NORMAL, "f1": pd >= config["f1"],
                          "cap_threshold": pd >= config["capacity"], "net_value_zero": value > 0,
                          "min_cost": value >= config[sc], "queue_random": rnd, "queue_probability": any_top,
                          "queue_expected_value": loss_top}
            for policy, selected in selections.items():
                queued = policy.startswith("queue_")
                result = record_routing(tables, dict(**key, scenario=sc, policy=policy), y, selected, loss,
                                        gate if queued else all_rows)
                if policy == "queue_expected_value" and sc == "scratch_3000":
                    metrics[model, seed] = (argres["macro_f1"], result["cost_per_1000_arrivals"])
                    traces[model, seed] = severity(y, loss)*(~selected) + REVIEW_FP*((y == NORMAL) & selected)
            if phase == "audit_exploratory":
                for i in np.flatnonzero(rnd | any_top | loss_top):
                    tables["review_membership"].append(dict(**key, scenario=sc, source_row=refs[i]["source_row"],
                        random=int(rnd[i]), probability=int(any_top[i]), expected_value=int(loss_top[i])))
                if model == selected_model and sc == "scratch_3000":
                    for i, ref in enumerate(refs):
                        tables["risk_scores"].append(dict(**key, source_row=ref["source_row"],
                            historical_gate=int(gate[i]), queue=queue_ids[i], probability_defect=pd[i],
                            probability_scratch=p[i,SCRATCH], expected_missed_loss=value[i]+REVIEW_FP*p[i,NORMAL],
                            expected_net_benefit=value[i], random=int(rnd[i]), probability=int(any_top[i]),
                            expected_value=int(loss_top[i])))
            for t in np.linspace(0, 1, 101):
                a = assessment(y, pd >= t, loss)
                tables["threshold_curves"].append(dict(**key, scenario=sc, threshold=t, **a))
    return metrics, traces


def select_models(metrics, with_new):
    families = sorted({model for model, _ in metrics})
    score = {name: np.mean([metrics[name, seed] for seed in SEEDS], axis=0) for name in families}
    # CE is an eligible no-change fallback; identical mean values prefer CE, then the declared arm order.
    chosen = max(NEW, key=lambda name: score[name][0]) if with_new else CNN
    return dict(downstream_model=chosen, lowest_queue_cost_model=min(families, key=lambda name: score[name][1]),
                mean_selection_scores={m: dict(macro_f1=float(v[0]), queue_cost_per_1000=float(v[1])) for m,v in score.items()})


def paired_intervals(traces, refs, tables, selected):
    groups, inv = np.unique([r["component"] for r in refs], return_inverse=True)
    if len(groups) < 2:
        return
    sizes = np.bincount(inv)
    by_model = {m: np.mean([traces[m,s] for s in SEEDS], axis=0) for m, _ in traces}
    pairs = [(RF, RF77), (RF, ENSEMBLE), (RF, selected)]
    if "cnn_ce" in by_model:
        pairs += [("cnn_ce", m) for m in NEW[1:]]
    rng = np.random.default_rng(20261101)
    pairs = list(dict.fromkeys(pairs))
    differences = np.array([np.bincount(inv, weights=by_model[a]-by_model[b]) for a,b in pairs])
    draws = []
    for _ in range(500):
        sample = rng.integers(0, len(groups), len(groups))
        draws.append(1000 * differences[:, sample].sum(axis=1)/sizes[sample].sum())
    for pair, delta, ci in zip(pairs, differences, np.quantile(draws, [.025,.975], axis=0).T):
        tables["paired_cost"].append(dict(comparator=pair[0], candidate=pair[1], scenario="scratch_3000",
                policy="queue_expected_value", savings_per_1000_arrivals=1000*delta.sum()/len(refs),
                lo_95=ci[0], hi_95=ci[1], clusters=len(groups), reps=500))


def write_tables(out, tables):
    for name, rows in tables.items():
        if not rows:
            continue
        if name in ("review_membership", "risk_scores"):
            f = gzip.open(out/(name+".csv.gz"), "wt", encoding="utf8", newline="")
        else:
            f = (out/(name+".csv")).open("w", encoding="utf8", newline="")
        with f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def charts(out, tables, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in tables["threshold_curves"] if r["phase"] == "selection" and r["model"] == model and r["scenario"] == "scratch_3000"]
    fig, axes = plt.subplots(2, 1, figsize=(8,6), sharex=True, constrained_layout=True)
    for ax, columns in zip(axes, (("precision","recall"), ("cost_per_1000_arrivals",))):
        for col in columns:
            # Individual seed curves keep undefined precision absent when no reviews are routed.
            for seed in SEEDS:
                run = [r for r in rows if r["seed"] == seed]
                ax.plot([r["threshold"] for r in run], [np.nan if r[col] is None else r[col] for r in run],
                        color={"precision":"#087f5b","recall":"#2056af","cost_per_1000_arrivals":"#a34421"}[col],
                        alpha=.65, label=col if seed == SEEDS[0] else None)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set(ylabel="Fraction", ylim=(-.02,1.02), title=f"{model} | policy-selection groups | three seeds")
    axes[1].set(xlabel="Threshold on calibrated P(any defect)", ylabel="Simulated USD / 1,000 arrivals")
    fig.savefig(out/"threshold_precision_recall_cost.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.2,4.3), constrained_layout=True)
    for ax, target in zip(axes, ("defect","scratch")):
        ax.plot([0,1],[0,1], ls=":",color="gray")
        for kind, color in (("raw","#b15c24"),("calibrated","#164c8d")):
            for seed in SEEDS:
                series = [r for r in tables["reliability"] if r["phase"] == "audit_exploratory"
                          and r["model"] == model and r["seed"] == seed and r["target"] == target
                          and r["probabilities"] == kind and r["n"]]
                ax.plot([r["mean_probability"] for r in series],[r["observed_fraction"] for r in series],
                        ".-",color=color,alpha=.6,label=kind if seed == SEEDS[0] else None)
        ax.set(title=target, xlabel="Mean probability", ylabel="Observed fraction", xlim=(0,1),ylim=(0,1))
        ax.legend()
    fig.suptitle(f"{model} | previously seen development audit")
    fig.savefig(out/"calibration_reliability.png",dpi=150)
    plt.close(fig)


def fit(tune_scores, tune_refs):
    y = np.array([CLASSES.index(r["truth"]) for r in tune_refs])
    cal = np.array([r["partition"] == "calibration" for r in tune_refs])
    require(cal.any() and not cal.all(), "Need disjoint calibration and selection components")
    config = {}
    for k, p in tune_scores.items():
        beta = fit_beta(p[cal], y[cal])
        config[k] = dict(beta=beta, temperature=1/beta, **thresholds(y[~cal], calibrated(p[~cal], beta)))
    return config, {k:v[~cal] for k,v in tune_scores.items()}, [r for r,mask in zip(tune_refs,cal) if not mask]


def stages(tables, selected, with_new):
    for row in tables["routing"]:
        if row["phase"] != "audit_exploratory":
            continue
        lines = [
            ("1_RF71_archived_argmax", RF, "argmax"),
            ("1b_CNN_matched_CE_control", "cnn_ce" if with_new else CNN, "argmax"),
            ("2_validation_selected_training_arm" if with_new else "2_archived_CNN_proxy_no_new_training",
             selected, "argmax"),
            ("3_same_fit_F1_threshold", selected, "f1"),
            ("4_same_fit_minimum_cost", selected, "min_cost"),
            ("5_same_fit_fixed_gate_and_slots", selected, "queue_expected_value")]
        for stage, model, policy in lines:
            if (row["model"],row["policy"]) == (model, policy):
                tables["five_stages"].append(dict(stage=stage, **row))


def run(archive, out, new):
    require((Path("/content").is_dir() and any(k.startswith("COLAB_") for k in os.environ))
            or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.environ.get("GITHUB_ACTIONS") == "true",
            "Real-data execution requires a hosted Colab, Kaggle, or GitHub Actions runtime")
    records, split = inputs(archive,new)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"run_status.json", {"status":"running"})
    tables = defaultdict(list)
    tune_refs = [r for r in split if r["role"] == "tune"]
    tune_scores = scores(records,tune_refs,"tune")
    config, subset_scores, subset_refs = fit(tune_scores,tune_refs)
    metrics, _ = evaluate_phase(subset_scores,subset_refs,config,"selection",tables)
    del tune_scores, subset_scores
    chosen = select_models(metrics,bool(new))
    selection = chosen["downstream_model"]
    freeze = dict(scope="Existing development audit has been seen; no untouched final test in this experiment.",
        split_method="SHA256(utf8(salt + component)).digest()[0] < 128 gets calibration; all other tune components select policies.",
        component_salt=SPLIT_SALT, legacy_gate_beta=LEGACY_BETA, legacy_gate_source="Completed Colab cost/tune_calibration_freeze.json, RF71 seed 42 (historical full tune).",
        gate_rule="1-P(none) >= .5 after legacy temperature; fixed for every model/seed and policy.",
        fit_method="Each probability matrix gets one NLL-minimizing inverse-temperature bounded [0.125,8], floor 1e-12; ensemble averages raw RF77/CNN first.",
        model_choice="Max mean selection nine-class macro-F1 across matched seeds in the five new CNN arms; CE eligible, first declared arm breaks ties. Archive-only uses the historical CNN for downstream illustration.",
        cost_choice="Min observed selection cost on calibrated expected net benefit, ties fewer reviews. The separate analytic comparator routes strictly positive net benefit.",
        capacity_choice="F1 uses all distinct P(any defect) cutoffs, ties fewer reviews. Capacity threshold maximizes true positives under floor(selection n/5), ties fewer reviews; operational queue enforces its own batch quotas.",
        scratch_choice="Scratch priority versus argmax of remaining classes. Maximize true Scratch hits, then minimize total normal false alarms, then normal-to-Scratch; all three must match or improve same model/seed selection argmax. Identity if no improvement.",
        simulation="Perfect human inspection; $5 charged only for inspected normals. All missed defects cost $1000 except Scratch costs $3000 in the differential scenario. Synthetic queues contain 1000 flagged wafers each; final fractional queue floors 20%.",
        queue_seed=QUEUE_SEED, archive_manifest_sha256=PIN,
        new_manifest_sha256=digest(new/"probabilities"/"manifest.json") if new else None,
        source_sha256=digest(Path(__file__)), **chosen,
        params=[dict(model=k[0],seed=k[1],**v) for k,v in config.items()])
    write_json(out/"frozen_validation_policy.json",freeze)
    print(f"Validation choices frozen: {selection}; now reading previously seen development audit.",flush=True)
    for r in split:
        if r["role"] == "tune":
            tables["validation_partition"].append({k:r[k] for k in ("source_row","component","lot","partition")})
    for phase in ("train", "calibration", "selection", "audit"):
        group = [r for r in split if r["partition"] == phase]
        for name in CLASSES:
            count = sum(r["truth"] == name for r in group)
            tables["split_support"].append(dict(phase=phase, label=name, rows=count, total=len(group),
                fraction=count/len(group), components=len({r["component"] for r in group}), lots=len({r["lot"] for r in group})))
    audit_refs = [r for r in split if r["role"] == "audit"]
    _, traces = evaluate_phase(scores(records,audit_refs,"audit"),audit_refs,config,"audit_exploratory",tables,
                               require_legacy=True,selected_model=selection)
    paired_intervals(traces,audit_refs,tables,selection)
    stages(tables,selection,bool(new))
    write_tables(out,tables)
    charts(out,tables,selection)
    outputs = {p.name:digest(p) for p in sorted(out.iterdir()) if p.name != "run_status.json"}
    write_json(out/"run_status.json",dict(status="complete", models=len(config)//3, fits_scored=len(config),
               simulation="existing development only", training_extension=bool(new), validation_policy_sha256=outputs["frozen_validation_policy.json"],
               sha256=outputs))
    print(f"Complete: {len(config)} model/seed pairs, {len(outputs)} outputs in {out}", flush=True)


def self_test():
    # Entirely generated identifiers, labels, and probabilities. No external files.
    rng = np.random.default_rng(117)
    y = np.arange(450) % 9
    truth = [CLASSES[i] for i in y]
    refs = [dict(source_row=str(i),truth=c,lot=str(i//10),component=str(i//10),
                 partition="calibration" if i//10 % 2 == 0 else "selection") for i,c in enumerate(truth)]
    raw = {(m,s):softmax(rng.normal(size=(len(y),9))+1.4*np.eye(9)[y]) for m in (*OLD, *NEW) for s in SEEDS}
    for seed in SEEDS:
        raw[ENSEMBLE,seed] = (raw[RF77,seed]+raw[CNN,seed])/2
    conf, subs, rows = fit(raw,refs)
    tab = defaultdict(list)
    mm,_ = evaluate_phase(subs,rows,conf,"selection",tab)
    chosen = select_models(mm,True)["downstream_model"]
    _,traces = evaluate_phase(raw,refs,conf,"audit_exploratory",tab,selected_model=chosen)
    stages(tab,chosen,True)
    paired_intervals(traces,refs,tab,chosen)
    require(all(r["all_three_hold"] for r in tab["scratch_guard"] if r["phase"] == "selection"),"joint constraint")
    for r in tab["routing"]:
        require(r["fn"]+r["tp"] == r["defect_n"] and r["fp"]+r["tn"] == r["normal_n"],"routing identity")
        require(r["fn"] == r["upstream_fn"]+r["candidate_unreviewed_fn"], "upstream reconciliation")
        require(r["cost_dollars"] >= 1000*r["fn"] + 5*r["fp"], "cost lower bound")
    require(len(tab["five_stages"]) == 6*2*3 and len(tab["risk_scores"])==3*len(y),"stages")
    # Hand-worked ties of two positives among the first three followed by one negative.
    ap = average_precision_score(np.array([1,0,1,0]), np.array([.9,.9,.5,.2]))
    require(abs(ap-(.5+2/3)/2)<1e-12, "tied AP")
    sample = np.array([8,7,0,8])
    a = assessment(sample, np.array([1,0,1,0], dtype=bool),3000)
    require(a["cost_dollars"] == 3005 and a["tp"]==1 and a["fp"]==1 and a["fn"]==1, "hand-worked cost")
    print("Synthetic calibration, policy selection, class metrics, paired queues, cost, stages, and CI checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--new",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.archive and args.out:
        run(args.archive,args.out,args.new)
    else:
        parser.error("Provide --archive and --out, or --self-test")


if __name__ == "__main__":
    main()
