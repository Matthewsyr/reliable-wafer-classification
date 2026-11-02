"""Paired, aggregate-only evaluation for study 09. Run on the hosted runtime."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


SEEDS = (42, 123, 2026)
BASE = "baseline"
ARMS = (BASE, "real_repeat", "synthetic")
STUDENT = "student"
S, N = 7, 8
SIZES = {"train": 31457, "calibration": 5637, "selection": 5670, "audit": 11591}
STUDY_SHA = "8b05f0bdcc13a20ac032e58eecd7862cba085cd56be9ec121d54cebcd40a8c67"
DECISION_SHA = "7a89039475d80a9417407cd4d1c63ddeebda85ea9217518439448185c962164c"
RF77_BETA = 2.036757049926612
CALIBRATION_04_SHA = "f95fde21e70b8bf0fc582c8953bedf8825994c45c47032a7debd3a90a25b4a54"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path, digest):
    require(sha(path) == digest, f"Frozen reference code changed: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def save_csv(path, rows):
    require(rows, f"Empty output: {path.name}")
    with path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def registry(study, training, pseudo):
    base = training / "probabilities"
    src = base / "manifest.json"
    m = json.loads(src.read_text())
    require(m.get("complete", True) is True and tuple(m["class_order"]) == study.CLASSES
            and tuple(m["probability_columns"]) == study.CLASSES, "Incomplete or incompatible training manifest")
    gate = json.loads(pseudo.read_text()) if pseudo is not None else None
    enabled = gate is not None and gate["pseudo_train"] is True
    if enabled:
        require(gate["rows"] > 0 and gate["qualified_bands"] and len(gate["student_pseudo_sha256"]) == 64
                and len(gate["qualification_sha256"]) == 64, "Student gate missing qualifying evidence")
    arms = (*ARMS, STUDENT) if enabled else ARMS
    items = {(r["model"], r["seed"], r["role"]): (base, r) for r in m["files"]}
    expected = {(a, s, role) for a in arms for s in SEEDS for role in ("tune", "audit")}
    require(set(items) == expected and len(items) == len(m["files"]), "Missing/extra/duplicate prediction files")
    for arm in arms:
        for seed in SEEDS:
            rec = json.loads((training / "fits" / f"{arm}_s{seed}" / "run.json").read_text())
            require(rec["model"] == arm and rec["seed"] == seed and len(rec["files"]) == 2
                    and all(f == items[arm, seed, f["role"]][1] for f in rec["files"])
                    and {r["role"] for r in rec["files"]} == {"tune", "audit"}
                    and rec["binding"]["protocol_sha256"] == m["protocol_sha256"], "Unbound prediction receipt")
            if arm == STUDENT:
                require(rec["binding"]["gate_sha256"] == sha(pseudo), "Student trained under different pseudo gate")
    return items, arms, sha(src)


def aligned(study, refs, file_spec):
    base, spec = file_spec
    header, rows = study.read_table(study.checked_file(base, spec))
    require(header == study.HEAD and len(rows) == len(refs) == spec["rows"], "Wrong prediction schema/count")
    lookup = {r["source_row"]: r for r in rows}
    require(len(lookup) == len(rows) and lookup.keys() == {r["source_row"] for r in refs},
            "Prediction IDs differ from frozen sources")
    ordered = [lookup[r["source_row"]] for r in refs]
    require(all(a["truth"] == b["truth"] for a, b in zip(ordered, refs)), "Prediction truth differs")
    p = np.array([[float(r[c]) for c in study.CLASSES] for r in ordered])
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            and np.allclose(p.sum(axis=1), 1, atol=2e-6, rtol=0), "Invalid probabilities")
    return p


def summarize(study, phase, arm, seed, truth, p, per_class, matrices):
    cm = study.confusion(truth, study.default_guess(p))
    support, predicted, precision, recall, f1 = study.averages(cm)
    key = dict(phase=phase, arm=arm, seed=seed)
    ap = []
    for j, name in enumerate(study.CLASSES):
        value = float(study.average_precision_score(truth == j, p[:, j])) if support[j] else None
        ap.append(value)
        per_class.append(dict(**key, label=name, support=int(support[j]), predicted=int(predicted[j]),
                              tp=int(cm[j, j]), precision=float(precision[j]), recall=float(recall[j]),
                              f1=float(f1[j]), average_precision=value))
        matrices.append({**key, "truth": name, **dict(zip(study.CLASSES, map(int, cm[j])))})
    nnormal = int(support[N])
    any_fp, scratch_fp = int(cm[N, :N].sum()), int(cm[N, S])
    return dict(**key, rows=len(truth), scratch_n=int(support[S]), normal_n=nnormal,
                scratch_tp=int(cm[S, S]), scratch_pred=int(predicted[S]), scratch_precision=float(precision[S]),
                scratch_recall=float(recall[S]), scratch_ap=ap[S], macro_f1=float(f1.mean()),
                normal_any_fp=any_fp, normal_scratch_fp=scratch_fp, normal_other_fp=any_fp-scratch_fp,
                normal_any_fpr=any_fp/nnormal, normal_scratch_fpr=scratch_fp/nnormal,
                normal_any_fp_per_1000_normals=1000*any_fp/nnormal,
                normal_scratch_fp_per_1000_normals=1000*scratch_fp/nnormal)


def compare(left, ref):
    """Counts are candidate minus reference; guard uses exact counts per seed."""
    fields = ("scratch_tp", "normal_any_fp", "normal_scratch_fp", "normal_other_fp",
              "scratch_precision", "scratch_recall", "normal_any_fpr", "normal_scratch_fpr", "macro_f1")
    diff = {f"delta_{key}": left[key] - ref[key] for key in fields}
    dt, da, ds = (diff[f"delta_{k}"] for k in ("scratch_tp", "normal_any_fp", "normal_scratch_fp"))
    if dt < 0 or da > 0 or ds > 0:
        outcome = "failed"
    elif dt > 0 or da < 0:
        outcome = "improved"
    elif ds < 0:
        outcome = "fp_transfer_only"
    else:
        outcome = "tied"
    return dict(phase=left["phase"], arm=left["arm"], reference=ref["arm"], seed=left["seed"], **diff,
                joint_guard=outcome)


def grade(pairs):
    vals = [r["joint_guard"] for r in pairs]
    require(len(vals) == len(SEEDS), "Incomplete seed comparison")
    for label in ("failed", "improved", "fp_transfer_only"):
        if label in vals:
            return label
    return "tied"


def phase_tables(study, phase, refs, inputs, arms, tables):
    truth = np.array([study.CLASSES.index(r["truth"]) for r in refs])
    require(np.any(truth == S) and np.any(truth == N), "Missing primary denominators")
    summ = {}
    for (arm, seed), p in inputs.items():
        row = summarize(study, phase, arm, seed, truth, p, tables["per_class"], tables["confusion"])
        summ[arm, seed] = row
        tables["classification"].append(row)
    pairs = [(arm, BASE) for arm in arms if arm != BASE] + [("synthetic", "real_repeat")]
    grades = {}
    for arm, ref in pairs:
        by_seed = [compare(summ[arm, seed], summ[ref, seed]) for seed in SEEDS]
        tables["paired"].extend(by_seed)
        grades[f"{arm}_vs_{ref}"] = grade(by_seed)
    return grades


def costs(study, decision, refs, records, scores, betas, passed):
    ids = np.array([r["source_row"] for r in refs])
    lots = np.array([r["lot"] for r in refs])
    truth = np.array([study.CLASSES.index(r["truth"]) for r in refs])
    gate_p = aligned(study, refs, records[study.RF, 42, "audit"])
    gate = 1 - study.calibrated(gate_p, study.LEGACY_BETA)[:, N] >= 0.5
    require(int(gate.sum()) == 3642, "Mismatch against historical fixed candidate gate")
    order, chunks, queues, _, _, tie = decision._plans(ids[gate], lots[gate], decision.SEED)
    require([q["capacity"] for q in queues] == [200, 200, 200, 128], "Changed queue capacity")
    use = {(arm, seed): study.calibrated(scores[arm, seed], betas[f"{arm}_s{seed}"])
           for arm in (BASE, *passed) for seed in SEEDS}
    use["rf77_operational", 42] = study.calibrated(aligned(study, refs, records[study.RF77, 42, "audit"]), RF77_BETA)
    rows = []
    for (arm, seed), p in use.items():
        chosen = decision._top(1-p[gate][order, N], ids[gate][order], tie, chunks)
        selected = np.zeros(len(refs), dtype=bool)
        selected[np.flatnonzero(gate)[order][chosen]] = True
        require(int(selected.sum()) == 728 and np.all(gate[selected]), "Review selection invalid")
        for name in ("equal", "scratch_5x"):
            sc = decision.SCENARIOS[name]
            loss, effect = np.array(sc.missed_loss)[truth], np.array(sc.effect)[truth]
            total = float(loss.sum() - (loss*effect)[selected].sum() + sc.review_fee*selected.sum())
            rows.append(dict(arm=arm, seed=seed, scenario=name, arrivals=len(refs), candidates=int(gate.sum()),
                             reviewed=int(selected.sum()), scratch_outside_gate=int(((truth == S) & ~gate).sum()),
                             scratch_inside_gate_unreviewed=int(((truth == S) & gate & ~selected).sum()),
                             reviewed_scratch=int(((truth == S) & selected).sum()),
                             reviewed_other_defect=int(((truth < S) & selected).sum()),
                             reviewed_normal=int(((truth == N) & selected).sum()), total_simulated_cost=total,
                             cost_per_1000_arrivals=total*1000/len(refs)))
    return rows


def run(args, out):
    study = module("study09_frozen_eval", args.repo / "cloud_review/extension/decision_study.py", STUDY_SHA)
    decision = module("study09_frozen_policy", args.repo / "cloud_review/roadmap_07/decision.py", DECISION_SHA)
    old, split = study.inputs(args.repo / "results/probabilities", None)
    require(dict(Counter(r["partition"] for r in split)) == SIZES, "Split changed")
    records, arms, manifest_hash = registry(study, args.training, args.pseudo_decision)
    tables = {k: [] for k in ("classification", "paired", "per_class", "confusion")}
    tune = [r for r in split if r["role"] == "tune"]
    calibration = np.array([r["partition"] == "calibration" for r in tune])
    cal_y = np.array([study.CLASSES.index(r["truth"]) for r in tune])[calibration]
    tune_p = {(arm, seed): aligned(study, tune, records[arm, seed, "tune"]) for arm in arms for seed in SEEDS}
    betas = {f"{arm}_s{seed}": study.fit_beta(p[calibration], cal_y) for (arm, seed), p in tune_p.items()}
    grades = phase_tables(study, "selection", [r for r in tune if r["partition"] == "selection"],
                          {k: p[~calibration] for k, p in tune_p.items()}, arms, tables)
    passed = [a for a in ARMS[1:] if grades[f"{a}_vs_{BASE}"] == "improved"]
    freeze = dict(seeds=SEEDS, role_counts=SIZES, selection_outcome=grades, cost_eligible=passed,
                  decision="argmax of uncalibrated nine-class probabilities; deterministic first class on exact tie",
                  guard="For each paired seed require Scratch TP >= A, normal-any FP <= A and normal-Scratch FP <= A. "
                        "At least one strict improvement in Scratch TP or normal-any FP over the three seeds required "
                        "for cost replay. Redirection of normal FPs to another defect alone is not improvement. "
                        "Synthetic versus real-repeat separately tests substitution at equal sample budget.",
                  student_selection="descriptive only; both tune subpartitions used to qualify RF77 pseudo-labels",
                  cost_scenarios=["equal", "scratch_5x"], fixed_alerts=3642, fixed_slots=728,
                  cost_policy="RF71 s42 gate (>=0.5); fixed queue; rank by calibrated probability of any defect",
                  cnn_inverse_temperatures_from_calibration_only=betas, operational_rf77_s42_beta=RF77_BETA,
                  historical_04_temperature_receipt_sha256=CALIBRATION_04_SHA,
                  training_manifest_sha256=manifest_hash,
                  pseudo_decision_sha256=sha(args.pseudo_decision) if args.pseudo_decision else None)
    save_json(out / "selection_freeze.json", freeze)
    # Audit predictions and audit metrics are opened only after all candidate eligibility is frozen.
    audit = [r for r in split if r["role"] == "audit"]
    audit_p = {(arm, seed): aligned(study, audit, records[arm, seed, "audit"]) for arm in arms for seed in SEEDS}
    audit_grade = phase_tables(study, "audit_exploratory", audit, audit_p, arms, tables)
    if passed:
        tables["cost"] = costs(study, decision, audit, old, audit_p, betas, passed)
    for name, rows in tables.items():
        save_csv(out / f"{name}.csv", rows)
    render(out, tables, grades, audit_grade, passed, arms)
    save_json(out / "run_receipt.json", dict(complete=True, script_sha256=sha(Path(__file__)),
              source_unit="unique labeled source_row; nine-class argmax", official_test_used=False,
              frozen_development_split_sha256="e467d05e96c0f477554969048e4c265a3686b0232add057170c2007b1ec27c3a",
              training_manifest_sha256=manifest_hash, archive_manifest_sha256=study.PIN,
              selection=grades, audit_exploratory=audit_grade, cost_eligible=passed,
              files={f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()}))


def render(out, tables, grades, audit_grade, passed, arms):
    names = {BASE: "A. Baseline CNN", "real_repeat": "B. Repeated real Scratch",
             "synthetic": "C. Synthetic Scratch", STUDENT: "D. Student CNN"}
    words = {"failed": "At least one seed failed", "improved": "All seeds passed with a gain",
             "tied": "All seeds tied on all three counts", "fp_transfer_only": "False alarms shifted to another defect"}
    lines = ["# Scratch training on the development set", "",
             "Each row counts distinct labeled wafers. Scratch recall uses all true Scratch wafers in the partition. "
             "Normal false alarms include all eight defect classes. Each seed was scored on the same wafers.",
             "", "| Partition | Method | Seed | Scratch hits / actual | Scratch recall | Scratch precision | "
             "Normal false alarms / normals | Normal called Scratch | Macro-F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in tables["classification"]:
        phase = "Selection" if r["phase"] == "selection" else "Historical audit"
        lines.append(f'| {phase} | {names[r["arm"]]} | {r["seed"]} | {r["scratch_tp"]}/{r["scratch_n"]} | '
                     f'{r["scratch_recall"]:.1%} | {r["scratch_precision"]:.1%} | '
                     f'{r["normal_any_fp"]}/{r["normal_n"]} ({r["normal_any_fpr"]:.2%}) | '
                     f'{r["normal_scratch_fp"]} | {r["macro_f1"]:.2%} |')
    lines += ["", "## Paired comparisons", "", "| Comparison | Selection | Historical audit |", "|---|---|---|"]
    for pair in grades:
        left, right = pair.split("_vs_")
        caveat = " (descriptive only)" if left == STUDENT else ""
        lines.append(f"| {names[left]} vs. {names[right]} | {words[grades[pair]]}{caveat} | {words[audit_grade[pair]]} |")
    lines += ["",
              "The comparisons pair seeds 42, 123 and 2026. Each seed must retain its true Scratch hits, "
              "with no increase in total normal false alarms or normals called Scratch. A cost replay also requires "
              "at least one additional true Scratch hit or one fewer total normal false alarm in at least one seed. "
              "Changing only the defect category assigned to a normal wafer earns no gain.",
              "",
              "C versus B compares the full synthesis recipe at the same number of replacement training slots. "
              "Geometry and background constraints can change the donor wafer selected for C.",
              "",
              "These audit lots had been inspected during earlier development. This comparison is exploratory. "
              "The original frozen test remained outside the study."]
    if STUDENT in arms:
        lines.append("The selection lots contributed to teacher admission. The student's selection results are "
                     "descriptive. An independent set of labeled lots is required to evaluate generalization.")
    if passed:
        lines.append("Qualifying arms entered a cost replay on the same 3,642 alerts with 728 review slots. "
                     "The two hypothetical scenarios in `cost.csv` include losses before screening, losses among "
                     "unreviewed alerts and the review fees across all arrivals.")
    else:
        lines.append("Neither B nor C qualified for the cost replay. The existing review policy was retained.")
    (out / "REPORT.md").write_bytes(("\n".join(lines)+"\n").encode("utf8"))


def self_test(repo):
    study = module("study09_synthetic_test", repo / "cloud_review/extension/decision_study.py", STUDY_SHA)
    y = np.array([*range(9), S, N, N])
    a = np.eye(9)[y].copy()
    a[-1] = np.eye(9)[S]
    b = a.copy()
    b[-1] = np.eye(9)[0]  # Same total FP, fewer normal->Scratch, more normal->other.
    def row(arm, p):
        return summarize(study, "unit", arm, 42, y, p, [], [])
    d = compare(row("synthetic", b), row(BASE, a))
    require(d["joint_guard"] == "fp_transfer_only" and d["delta_normal_any_fp"] == 0 and d["delta_normal_scratch_fp"] == -1
            and d["delta_normal_other_fp"] == 1, "Normal FP transfer was obscured")
    b[-2] = np.eye(9)[0]
    r = compare(row("synthetic", b), row(BASE, a))
    require(r["joint_guard"] == "failed", "Normal total FP increase did not block")
    tie = compare(row(BASE, a), row(BASE, a))
    require(grade([tie]*3) == "tied" and grade([tie, tie, r]) == "failed", "All-seed gate broken")
    print("PASS synthetic evaluator checks", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--training", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--pseudo-decision", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test(a.repo)
    else:
        require(os.name == "posix" and Path("/content").is_dir()
                and (os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_BACKEND_VERSION")),
                "Real-row evaluation requires the hosted Colab runtime")
        require(a.training is not None and a.out is not None, "Supply --training and --out")
        require(not a.out.exists(), f"Choose a fresh evaluation output: {a.out}")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
            subdir = Path(tmp) / "evaluation"
            subdir.mkdir()
            run(a, subdir)
            subdir.rename(a.out)
        print(f"PASS {a.out / 'run_receipt.json'}", flush=True)
