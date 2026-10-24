#!/usr/bin/env python3
"""Recompute the published development decisions, with Python's standard library.

Run from any directory: python /path/to/portfolio/scripts/recompute_decisions.py
No model training, test data, third-party package or local workspace is needed.
Frozen thresholds are verified, never replaced or optimized on audit.
"""
from pathlib import Path
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
import sys

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
SCRATCH, NORMAL = 7, 8


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def metrics(truth, predicted):
    cm = [[0] * 9 for _ in CLASSES]
    for actual, guess in zip(truth, predicted):
        cm[actual][guess] += 1
    support = [sum(row) for row in cm]
    count = [sum(row[j] for row in cm) for j in range(9)]
    diagonal = [cm[j][j] for j in range(9)]
    precision = [diagonal[j] / count[j] if count[j] else 0.0 for j in range(9)]
    recall = [diagonal[j] / support[j] if support[j] else 0.0 for j in range(9)]
    f1 = [2 * diagonal[j] / (support[j] + count[j]) if support[j] + count[j] else 0.0 for j in range(9)]
    normal_fp = support[NORMAL] - diagonal[NORMAL]
    return dict(rows=len(truth), accuracy=sum(diagonal) / len(truth), macro_f1=sum(f1) / 9,
                scratch_precision=precision[SCRATCH], scratch_recall=recall[SCRATCH],
                scratch_f1=f1[SCRATCH], scratch_support=support[SCRATCH], scratch_tp=diagonal[SCRATCH],
                normal_count=support[NORMAL], normal_any_fp=normal_fp, normal_scratch_fp=cm[NORMAL][SCRATCH],
                normal_any_fp_per_1000=1000 * normal_fp / support[NORMAL],
                normal_scratch_fp_per_1000=1000 * cm[NORMAL][SCRATCH] / support[NORMAL],
                confusion_matrix=cm, per_class={c: dict(precision=precision[j], recall=recall[j], f1=f1[j],
                    support=support[j], predicted_count=count[j]) for j, c in enumerate(CLASSES)})


def score_summary(truth, scores, select_threshold):
    """Exact tied-score AP and the declared tune-only maximum-TP P>=0.8 rule."""
    ranked = sorted(zip(scores, truth), reverse=True)
    support = truth.count(SCRATCH)
    tp = 0
    ap = 0.0
    previous_tp = 0
    candidate = None
    feasible = 0
    distinct = 0
    i = 0
    while i < len(ranked):
        threshold = ranked[i][0]
        j = i
        while j < len(ranked) and ranked[j][0] == threshold:
            tp += ranked[j][1] == SCRATCH
            j += 1
        precision = tp / j
        ap += ((tp - previous_tp) / support) * precision
        previous_tp = tp
        distinct += 1
        if select_threshold and 5 * tp >= 4 * j:  # Nonempty by construction; no all-negative pseudo-rule.
            feasible += 1
            key = tp, Fraction(tp, j), threshold
            if candidate is None or key > candidate:
                candidate = key
        i = j
    return ap, (None if candidate is None else candidate[2]), distinct, feasible


def operating(truth, scores, threshold):
    positive = [score >= threshold for score in scores]
    tp = sum(p and y == SCRATCH for p, y in zip(positive, truth))
    fp = sum(p and y != SCRATCH for p, y in zip(positive, truth))
    support = truth.count(SCRATCH)
    fn, tn = support - tp, len(truth) - support - fp
    normal_count = truth.count(NORMAL)
    normal_fp = sum(p and y == NORMAL for p, y in zip(positive, truth))
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, predicted_positive=tp + fp,
                scratch_support=support, precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / support, f1=2 * tp / (2 * tp + fp + fn), normal_count=normal_count,
                normal_to_scratch_fp=normal_fp, normal_fp_per_1000=1000 * normal_fp / normal_count,
                precision80_feasible=bool(tp + fp > 0 and 5 * tp >= 4 * (tp + fp)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'probabilities')
    parser.add_argument('--output', type=Path, help='Optional verification receipt; reference data is never modified.')
    args = parser.parse_args()
    base = args.data_dir.resolve()
    manifest = load(base / 'manifest.json')
    checks = 0
    failures = []

    def check(name, condition, detail=None):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(dict(check=name, detail=detail))

    def equal(name, actual, expected):
        if isinstance(actual, dict):
            for key, value in actual.items():
                if key not in expected:
                    check(name + '.' + key, False, 'Missing reference key')
                else:
                    equal(name + '.' + key, value, expected[key])
        elif isinstance(actual, list):
            check(name + '.length', len(actual) == len(expected))
            for i, (value, wanted) in enumerate(zip(actual, expected)):
                equal(name + f'[{i}]', value, wanted)
        elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
            check(name, expected is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                  dict(computed=actual, saved=expected))
        else:
            check(name, actual == expected, dict(computed=actual, saved=expected))

    check('fixed_class_order', manifest['class_order'] == CLASSES)
    check('fixed_probability_record_count', len(manifest['probability_files']) == 18)
    for record in [manifest['split_reference'], manifest['decision_reference'], *manifest['probability_files']]:
        path = base / record['file']
        check('hash:' + record['file'], sha(path) == record['sha256'])
        check('file_size:' + record['file'], path.stat().st_size == record['bytes'])
    split = rows(base / manifest['split_reference']['file'])
    check('split_rows', len(split) == manifest['split_reference']['rows'] == 54355)
    check('source_identity_unique', len({r['source_row'] for r in split}) == len(split))
    for field in ['lot', 'component', 'map_sha256']:
        assigned = {}
        for row in split:
            assigned.setdefault(row[field], set()).add(row['role'])
        check('no_cross_role_' + field, all(len(roles) == 1 for roles in assigned.values()))
    expected_rows = {role: [r for r in split if r['role'] == role] for role in ['tune', 'audit']}
    reference = load(base / manifest['decision_reference']['file'])
    selected = {(r['model'], r['seed']): r for r in reference['selected_thresholds']}
    check('frozen_selection_count', len(selected) == 9)
    expected = {(r['model'], r['seed'], r['role']): r for r in reference['evaluated_records']}
    check('evaluation_count', len(expected) == 18)
    decisions = 0
    for record in manifest['probability_files']:
        key = record['model'], record['seed'], record['role']
        label = '|'.join(map(str, key))
        source = rows(base / record['file'])
        wanted_rows = expected_rows[record['role']]
        check('row_count:' + label, len(source) == record['rows'] == len(wanted_rows))
        check('row_order_and_truth:' + label,
              [(r['source_row'], r['truth']) for r in source] == [(r['source_row'], r['truth']) for r in wanted_rows])
        probabilities = [[float(r[c]) for c in CLASSES] for r in source]
        truth = [CLASSES.index(r['truth']) for r in source]
        check('probability_domain:' + label, all(all(math.isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p) - 1) < 1e-10 for p in probabilities))
        scores = [p[SCRATCH] for p in probabilities]
        ap, selected_threshold, candidates, feasible = score_summary(truth, scores, record['role'] == 'tune')
        result = expected[key]
        equal('average_precision:' + label, ap, result['average_precision'])
        choice = selected[key[:2]]
        # This is retrospective verification of tune optimum, never an audit selection.
        if record['role'] == 'tune':
            check('frozen_tune_threshold:' + label, selected_threshold == choice['threshold'],
                  dict(computed=selected_threshold, saved=choice['threshold']))
            equal('candidate_count:' + label, candidates, choice['tune_unique_candidate_count'])
            equal('feasible_count:' + label, feasible, choice['feasible_count'])
            equal('selection_status:' + label, 'selected' if selected_threshold is not None else 'no_rule_selected', choice['status'])
        raw = [max(range(9), key=p.__getitem__) for p in probabilities]
        before = metrics(truth, raw)
        equal('argmax:' + label, before, result['argmax'])
        decisions += 1
        threshold = choice['threshold']
        if threshold is None:
            check('no_rule_selected:' + label, result['scratch_priority'] is None)
            continue
        binary = operating(truth, scores, threshold)
        if record['role'] == 'tune':
            equal('frozen_tune_operating_point:' + label, binary, choice['tune_operating_point'])
        else:
            equal('frozen_audit_precision:' + label, binary['precision'], result['threshold_precision'])
            equal('frozen_audit_recall:' + label, binary['recall'], result['threshold_recall'])
        other = [j for j in range(9) if j != SCRATCH]
        predicted = [SCRATCH if p[SCRATCH] >= threshold else max(other, key=p.__getitem__) for p in probabilities]
        after = metrics(truth, predicted)
        equal('scratch_priority:' + label, after, result['scratch_priority'])
        decisions += 1
        delta = dict(scratch_tp_delta=after['scratch_tp'] - before['scratch_tp'],
                     normal_any_fp_delta=after['normal_any_fp'] - before['normal_any_fp'],
                     normal_scratch_fp_delta=after['normal_scratch_fp'] - before['normal_scratch_fp'],
                     scratch_tp_not_decreased=after['scratch_tp'] >= before['scratch_tp'],
                     normal_any_fp_not_increased=after['normal_any_fp'] <= before['normal_any_fp'],
                     normal_scratch_fp_not_increased=after['normal_scratch_fp'] <= before['normal_scratch_fp'],
                     normal_scratch_fp_recovered_to_none=sum(y == NORMAL and a == SCRATCH and b == NORMAL for y, a, b in zip(truth, raw, predicted)),
                     normal_scratch_fp_transferred_to_other_defect=sum(y == NORMAL and a == SCRATCH and b not in (SCRATCH, NORMAL) for y, a, b in zip(truth, raw, predicted)),
                     normal_previously_correct_now_defect=sum(y == NORMAL and a == NORMAL and b != NORMAL for y, a, b in zip(truth, raw, predicted)))
        delta['joint_noninferiority_point_pass'] = all(delta[name] for name in ['scratch_tp_not_decreased', 'normal_any_fp_not_increased', 'normal_scratch_fp_not_increased'])
        equal('decision_tradeoff:' + label, delta, result['priority_vs_argmax'])
    check('nine_class_decisions_count', decisions == 36)
    output = dict(status='PASS' if not failures else 'FAIL', checks_total=checks, checks_failed=len(failures), failures=failures,
                  probability_records_verified=18, frozen_tune_thresholds_verified=9, nine_class_decisions_verified=decisions,
                  dependencies='Python standard library only', threshold_policy='Verified frozen tune-only choices; audit choices never selected or modified.',
                  independent_confirmation=False, official_test_scored=False,
                  manifest_sha256=sha(base / 'manifest.json'), verifier_sha256=sha(Path(__file__)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
