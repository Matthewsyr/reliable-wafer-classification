"""WM-811K inspection decision protocol v1. All costs are simulated loss units.

This file has no training, network or I/O on import. The caller must provide untouched
held-out rows and probabilities from a model/calibrator fitted on other lots. The
caller, not this evaluator, enforces training/calibration/test disjointness.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

CLASS_ORDER = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full",
               "Random", "Scratch", "none")
SEED, QUEUE_SIZE, MAX_REVIEW = 20261101, 1000, 200
POLICIES = ("no_review", "random_fixed", "any_defect_fixed",
            "benefit_fixed", "benefit_positive_cap")
POOL_KINDS = ("all_labeled_simulation", "simulated_flagged_candidates",
              "documented_flagged_candidates")


@dataclass(frozen=True)
class Scenario:
    name: str
    missed_loss: tuple[float, ...]
    effect: tuple[float, ...]  # conditional fraction of the loss avoided by review
    review_fee: float
    review_harm: tuple[float, ...]  # additional cost conditional on reviewed true class


def _case(name, high=None):
    loss = tuple(0.0 if x == "none" else 5.0 if x == high else 1.0 for x in CLASS_ORDER)
    return Scenario(name, loss, (0.9,) * 8 + (0.0,), 0.1, (0.0,) * 9)


# Fixed illustrative stress scenarios, not measured process or financial parameters.
SCENARIOS = {s.name: s for s in (_case("equal"), _case("scratch_5x", "Scratch"),
                                _case("near_full_5x", "Near-full"))}
BENCHMARKS = (("random_fixed", "no_review"), ("any_defect_fixed", "random_fixed"),
              ("benefit_fixed", "any_defect_fixed"),
              ("benefit_positive_cap", "benefit_fixed"),
              ("benefit_positive_cap", "any_defect_fixed"),
              ("benefit_positive_cap", "random_fixed"),
              ("benefit_positive_cap", "no_review"))


def _names(values, field):
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1 or not len(arr):
        raise ValueError(f"{field}: nonempty one-dimensional sequence required")
    if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
           or str(v).strip().lower() in {"", "nan", "<na>", "nat"} for v in arr):
        raise ValueError(f"{field}: empty or missing identifier")
    return arr.astype(str)


def _labels(y):
    y = np.asarray(y)
    if y.dtype.kind in "iu":
        if (y < 0).any() or (y >= len(CLASS_ORDER)).any():
            raise ValueError("invalid class index")
        return y.astype(np.int64)
    lookup = {v: i for i, v in enumerate(CLASS_ORDER)}
    try:
        return np.asarray([lookup[str(x)] for x in y], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown label {exc}") from exc


def _keys(values, seed, domain):
    key = hashlib.sha256(f"{seed}:{domain}".encode()).digest()
    return np.fromiter((int.from_bytes(hashlib.blake2b(v.encode("utf-8"), key=key,
                        digest_size=8).digest(), "big") for v in values),
                       dtype=np.uint64, count=len(values))


def _digest(ids, lots):
    h = hashlib.sha256()
    # Length-prefix to avoid ambiguities across fields or separators.
    for a, b in zip(ids, lots):
        for x in (a, b):
            v = x.encode("utf-8")
            h.update(len(v).to_bytes(8, "big"))
            h.update(v)
    return h.hexdigest()


def _plans(ids, lots, seed):
    groups, group_idx = np.unique(lots, return_inverse=True)
    g_rank = np.lexsort((groups, _keys(groups, seed, "lot-order")))
    g_order = np.empty_like(g_rank)
    g_order[g_rank] = np.arange(len(groups))
    tie = _keys(ids, seed, "common-tie-break")
    order = np.lexsort((ids, _keys(ids, seed, "within-lot-order"), g_order[group_idx]))
    ids, lots = ids[order], lots[order]
    n = len(ids)
    chunks = [(i, min(i + QUEUE_SIZE, n)) for i in range(0, n, QUEUE_SIZE)]
    queue_meta = [dict(queue=j, rows=b-a, capacity=(b-a)*MAX_REVIEW//QUEUE_SIZE,
                       lots=len(np.unique(lots[a:b])),
                       prior_queue_shares_boundary_lot=bool(a and lots[a-1] == lots[a]))
                  for j, (a,b) in enumerate(chunks)]
    return order, chunks, queue_meta, group_idx[order], groups, tie[order]


def _top(score, ids, tie, chunks, positive=False):
    result = np.zeros(len(ids), dtype=bool)
    for a,b in chunks:
        k = (b-a)*MAX_REVIEW//QUEUE_SIZE
        rank = np.lexsort((ids[a:b], tie[a:b], -score[a:b]))[:k] + a
        if positive:
            rank = rank[score[rank] > 0.0]
        result[rank] = True
    return result


def _cluster_intervals(group_counts, differences, reps, seed):
    """Percentile CI of average paired savings/1000 for fixed decisions in resampled lots.

    This directly resamples whole lots, normalizes by sampled
    rows and never pretends the rankings are rerun. It conditions on fitted models,
    inferred candidates and simulated queues; it is not a CI for factory savings.
    """
    group_counts = np.asarray(group_counts, dtype=float)
    n = len(group_counts)
    if reps == 0 or n < 2:
        return [(None, None)] * differences.shape[1]
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, differences.shape[1]))
    # Each draw reuses the same sampled lot multiplicities across strategies.
    # Bound transient memory even if a pool has a very large number of lots.
    batch = min(32, max(1, 300_000 // n))
    for off in range(0, reps, batch):
        k = min(batch, reps - off)
        take = rng.integers(0, n, size=(k, n), dtype=np.int64)
        n_drawn = group_counts[take].sum(axis=1)
        for j in range(differences.shape[1]):
            draws[off:off+k, j] = 1000.0 * differences[:, j][take].sum(axis=1) / n_drawn
    limits = np.quantile(draws, [0.025, 0.975], axis=0)
    return [(float(lo), float(hi)) for lo,hi in zip(*limits)]


def _ratio(a, b):
    return float(a/b) if b else None


def evaluate(source_row, lot, y_true, probabilities, *, model="RF", scenario_names=None,
             seed=SEED, bootstrap_reps=500, pool_kind="all_labeled_simulation"):
    """Use the same held-out rows, candidate identities and seed for every model.

    source_row is a stable unique wafer-row identifier. lot is its true production lot.
    y_true: nine class names or integer indexes; probabilities: aligned n x 9 in
    CLASS_ORDER, never hard predicted labels. A true upstream flag must be fixed by the
    caller; no labels or predictions are used to select candidates here. Quotas/fixed
    policies never inspect beyond 20%; the last partial queue uses floor(0.2 * n).
    """
    if pool_kind not in POOL_KINDS:
        raise ValueError(f"pool_kind must be one of {POOL_KINDS}")
    if not isinstance(bootstrap_reps, (int, np.integer)) or bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be an integer >=0")
    ids, lots = _names(source_row, "source_row"), _names(lot, "lot")
    raw_y = np.asarray(y_true)
    if raw_y.ndim != 1 or len(raw_y) != len(ids) or len(lots) != len(ids):
        raise ValueError("held-out identities and y_true must be aligned")
    y = _labels(raw_y)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(ids)) != len(ids) or p.shape != (len(ids), len(CLASS_ORDER)):
        raise ValueError("duplicate source_row or probability shape mismatch")
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            and np.allclose(p.sum(axis=1), 1, rtol=0, atol=2e-5)):
        raise ValueError("invalid probability distribution: supply calibrated n x 9 matrix")
    sc = list(SCENARIOS) if scenario_names is None else list(scenario_names)
    if not sc or len(set(sc)) != len(sc) or not set(sc).issubset(SCENARIOS):
        raise ValueError("invalid or duplicate scenario names")
    order, chunks, queues, gi, groups, tie = _plans(ids, lots, seed)
    ids, lots, y, p = ids[order], lots[order], y[order], p[order]
    N, G = len(ids), len(groups)
    lot_n = np.bincount(gi, minlength=G)
    # Generate the random comparator from wafer identities, independent of label,
    # scores and queue shuffle. Using a uint64 descending sort avoids float ties.
    random_score = _keys(ids, seed, "random-policy") >> np.uint64(1)
    random_mask = _top(random_score.astype(np.int64), ids, tie, chunks)
    any_mask = _top(1-p[:,-1], ids, tie, chunks)
    id_order = np.argsort(ids)
    result = dict(protocol=dict(version="wm811k-triage-v1.0", class_order=CLASS_ORDER,
                       scenario_parameters=[asdict(SCENARIOS[x]) for x in sc],
                       queue_size=QUEUE_SIZE, max_reviews=MAX_REVIEW,
                       seed=int(seed), bootstrap_reps=int(bootstrap_reps), rows=N,
                       lots=G, model=str(model), pool_kind=pool_kind,
                       candidate_order_sha256=_digest(ids, lots),
                       ordered_labels_sha256=hashlib.sha256(y.astype("uint8").tobytes()).hexdigest(),
                       source_label_sha256=_digest(ids[id_order], y[id_order].astype(str)),
                       queue_rule="Hash-shuffle lots; hash-shuffle rows within each lot; concatenate to virtual queues of 1000. A boundary can split a lot. The last queue has floor(n/5) slots.",
                       uncertainty="Paired percentile 95% interval resamples lots; observed decisions and queues held fixed. It omits refit, queue assembly and cost-assumption uncertainty.",
                       units="Hypothetical relative loss units. Realized labels; assumed intervention effects, not observed cash or process intervention outcomes.",
                       input_precondition="Caller supplies disjoint train/calibration/assessment lots, common held-out pool, true source_row alignment, and class columns in documented order.",
                       numpy_version=np.__version__),
                  queues=queues, summary=[], per_class=[], paired=[], lot_costs=[],
                  selected_rows={})
    supports = np.bincount(y, minlength=9)
    for sname in sc:
        scenario = SCENARIOS[sname]
        L, e, harm = map(np.asarray, (scenario.missed_loss, scenario.effect, scenario.review_harm))
        # With equal losses & efficacy the ranking must agree exactly with
        # P(any defect), even for quantized probabilities and floating-point ties.
        value = ((1-p[:,-1])*float(L[0]*e[0])-scenario.review_fee if sname == "equal"
                 else p @ (L*e-harm) - scenario.review_fee)
        costs_by_policy = {}
        masks = {"no_review": np.zeros(N, dtype=bool), "random_fixed": random_mask,
                 "any_defect_fixed": any_mask,
                 "benefit_fixed": (any_mask if sname=="equal" else _top(value, ids, tie, chunks)),
                 "benefit_positive_cap": (any_mask & (value>0.0) if sname=="equal"
                                          else _top(value, ids, tie, chunks, positive=True))}
        result["selected_rows"][sname] = {}
        for policy, mask in masks.items():
            key = dict(model=str(model), pool_kind=pool_kind, scenario=sname, policy=policy)
            count = int(mask.sum())
            chosen = np.bincount(y[mask], minlength=9)
            avoided = mask * (L[y]*e[y])
            residual = L[y] - avoided
            cost = residual + mask*(scenario.review_fee + harm[y])
            lc = np.bincount(gi, weights=cost, minlength=G)
            costs_by_policy[policy] = lc
            total = float(cost.sum())
            base = float(L[y].sum())
            saved = float(base-total)
            result["summary"].append(dict(**key, rows=N, lots=G, reviewed=count,
                capacity=sum(q["capacity"] for q in queues), reviewed_per_1000=1000*count/N,
                true_normal_reviewed=int(chosen[-1]), true_defect_reviewed=count-int(chosen[-1]),
                true_defect_not_reviewed=int((supports-chosen)[:-1].sum()),
                reviewed_defect_fraction=_ratio(count-int(chosen[-1]), count),
                defect_coverage=_ratio(count-int(chosen[-1]), int(supports[:-1].sum())),
                missed_loss_without_inspection=base, gross_avoided_loss=float(avoided.sum()),
                avoided_share_of_no_review_loss=_ratio(float(avoided.sum()), base),
                residual_loss=float(residual.sum()), review_fees=float(count*scenario.review_fee),
                review_harm=float(harm[y[mask]].sum()), total_cost=total,
                cost_per_1000=1000*total/N, savings_vs_no_review_per_1000=1000*saved/N))
            for i, label in enumerate(CLASS_ORDER):
                result["per_class"].append(dict(**key, label=label, support=int(supports[i]),
                   reviewed=int(chosen[i]), not_reviewed=int(supports[i]-chosen[i]),
                   review_coverage=_ratio(int(chosen[i]), int(supports[i])),
                   assumed_gross_avoided_loss=float(chosen[i]*L[i]*e[i]),
                   remaining_loss=float(supports[i]*L[i] - chosen[i]*L[i]*e[i])))
            for group, nr, co in zip(groups, lot_n, lc):
                result["lot_costs"].append(dict(**key, lot=str(group), rows=int(nr),
                                                total_cost=float(co)))
            result["selected_rows"][sname][policy] = ids[mask].tolist()
        diffs = np.column_stack([costs_by_policy[b]-costs_by_policy[a] for a,b in BENCHMARKS])
        intervals = _cluster_intervals(lot_n, diffs, bootstrap_reps, seed)
        for j,(a,b) in enumerate(BENCHMARKS):
            result["paired"].append(dict(model=str(model), pool_kind=pool_kind, scenario=sname,
               intervention=a, comparator=b, paired_savings_per_1000=float(1000*diffs[:,j].sum()/N),
               ci_low=intervals[j][0], ci_high=intervals[j][1], bootstrap_reps=int(bootstrap_reps)))
    return result


def compare_models(new_result, base_result, *, policy="benefit_positive_cap", bootstrap_reps=500,
                   seed=SEED):
    """Paired difference, same scenario/policy/identities including opportunities displaced."""
    a, b = new_result["protocol"], base_result["protocol"]
    match = ("pool_kind","candidate_order_sha256","ordered_labels_sha256","scenario_parameters","seed")
    if any(a[k] != b[k] for k in match):
        raise ValueError("pool identities/order, scenarios and queue seed must match")
    rows = []
    for s in a["scenario_parameters"]:
        sname = s["name"]
        def extract(r):
            return {z["lot"]:(z["rows"],z["total_cost"]) for z in r["lot_costs"]
                    if z["policy"]==policy and z["scenario"]==sname}
        x, y = extract(new_result), extract(base_result)
        if not x or set(x) != set(y):
            raise ValueError("policy missing or lot membership mismatch")
        groups = sorted(x)
        ns = np.array([x[g][0] for g in groups])
        if any(x[g][0]!=y[g][0] for g in groups):
            raise ValueError("lot counts differ")
        dif = np.array([y[g][1]-x[g][1] for g in groups])[:, None]
        lo,hi = _cluster_intervals(ns, dif, bootstrap_reps, seed)[0]
        new_ids = set(new_result["selected_rows"][sname][policy])
        old_ids = set(base_result["selected_rows"][sname][policy])
        rows.append(dict(new_model=a["model"], base_model=b["model"], scenario=sname,
            policy=policy, paired_savings_per_1000=float(1000*dif.sum()/ns.sum()),
            ci_low=lo, ci_high=hi, entered=len(new_ids-old_ids), displaced=len(old_ids-new_ids)))
    return rows


def swap_by_class(new_result, base_result, source_row, y_true, *, policy="benefit_fixed"):
    """Decompose actual marginal changes in review lists by held-out true class.

    Use fixed-capacity comparisons to measure which classes really lose slots.
    Under the optional-cap policy there may be unequal numbers entering/leaving.
    """
    p, q = new_result["protocol"], base_result["protocol"]
    for k in ("pool_kind","candidate_order_sha256","ordered_labels_sha256","scenario_parameters","seed"):
        if p[k] != q[k]:
            raise ValueError("comparison requires identical candidates, truth, scenarios and queues")
    ids, yy = _names(source_row, "source_row"), _labels(y_true)
    if yy.shape != ids.shape or len(ids) != len(np.unique(ids)) or len(ids) != p["rows"]:
        raise ValueError("bad wafer-to-truth mapping")
    j = np.argsort(ids)
    if _digest(ids[j], yy[j].astype(str)) != p["source_label_sha256"]:
        raise ValueError("wafer-to-truth mapping does not match frozen evidence")
    truth = dict(zip(ids, yy.tolist()))
    output = []
    for s in p["scenario_parameters"]:
        key = s["name"]
        new_ids = set(new_result["selected_rows"][key][policy])
        old_ids = set(base_result["selected_rows"][key][policy])
        entered = np.bincount(np.asarray([truth[i] for i in new_ids-old_ids], dtype=int), minlength=9)
        displaced = np.bincount(np.asarray([truth[i] for i in old_ids-new_ids], dtype=int), minlength=9)
        for k,c in enumerate(CLASS_ORDER):
            en, dis = int(entered[k]), int(displaced[k])
            net_per_review = (s["missed_loss"][k]*s["effect"][k]-s["review_harm"][k]
                              -s["review_fee"])
            output.append(dict(new_model=p["model"], baseline_model=q["model"], scenario=key,
                       policy=policy,label=c,entered=en,displaced=dis,net_reviewed=en-dis,
                       net_savings_units=float((en-dis)*net_per_review)))
    return output


def write_results(evidence, outdir):
    """Use a fresh output directory to prevent overwrites. Returns its path."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refuse nonempty output directory: {out}")
    (out/"protocol.json").write_text(json.dumps(evidence["protocol"], ensure_ascii=False,
            allow_nan=False, indent=2), encoding="utf-8")
    for name in ("queues","summary","per_class","paired","lot_costs"):
        rows = evidence[name]
        path = out/(name+(".csv.gz" if name=="lot_costs" else ".csv"))
        writer = gzip.open if name=="lot_costs" else open
        with writer(path,"wt",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with gzip.open(out/"selected_rows.json.gz","wt",encoding="utf-8") as f:
        json.dump(evidence["selected_rows"],f,ensure_ascii=False)
    return out
