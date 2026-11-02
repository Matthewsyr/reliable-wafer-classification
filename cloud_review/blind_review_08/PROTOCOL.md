# Prospective evaluation on new production lots

**Status.** The two policies and adjudication rules are frozen. The workflow passed synthetic acceptance. Enrollment awaits consecutive production arrivals, an independent labeling budget and blinded readers. No production cohort or expert outcomes have been enrolled. [Frozen configuration](frozen_policy.json) · [Historical replay](../roadmap_07/REPORT.md)

## Freeze decisions before the first arrival

| Item | Specification |
|---|---|
| Upstream alert | RF71 seed 42, inverse temperature 1.9147552052282766, any-defect probability ≥0.5. |
| Ranking model | RF77 seed 42, inverse temperature 2.036757049926612. Floor class probabilities at 1e-12 before calibration. Use the registered weight hashes, class order and 36 archived angle constants. |
| Study window | Thirty consecutive 24-hour periods starting at preregistration's future timestamp. Form eligible queues in actual UTC arrival order with stable wafer ID as tie break. A full queue of 1,000 alerts permits 200 reviews. A tail with n alerts permits floor(n/5). |
| Baseline A | Rank by calibrated any-defect probability and fill the queue's capacity. |
| Challenger B | Rank by predicted avoidable loss minus the fee. Select only strictly positive net benefit, up to the same capacity. The common tie-breaking hash uses seed 20261101, followed by wafer ID. |
| Primary endpoint | B minus A in simulated total cost per 1,000 **all eligible arrivals**. Negative differences favor B. Settle once after the window closes and labels and adjudication are complete. |

Normal, ordinary-defect and Scratch missed losses are 0, 1 and 5 respectively. Each proposed review costs 0.1 and prevents an assumed 90% of loss. The sensitivity analysis reprices **the same selected lists** at equal losses. [Queue and tie-breaking implementation](../roadmap_07/decision.py)

## Required inputs and roles

1. Preregister one facility, line and product scope, a future start time, a manifest owner, independent MES count, labeling budget and reviewers. Deposit the registration SHA-256 independently before enrollment. Hold the model, gate, sampling rules and scope fixed throughout the study.
2. Provide a private, complete `arrivals.csv` with columns `wafer_id,lot,arrival_utc,map_file`. Include unalerted arrivals and one row per wafer. Require explicit time zones, source lot, and native 2-D numeric `.npy` maps (0 outside wafer, 1 pass, 2 fail). Paths are relative to the map directory. Reconcile all arrivals to the independent MES total. Resolve missing or incomplete lots, maps, times or identities before issuing cases.
3. Supply a registry of all previous training, pretraining, validation and review exposure. The existing registry protects all 811,457 WM-811K maps. Add other exposed sources and resolve cross-system physical-wafer and lot aliases, time provenance and map encoding before enrollment.
4. Assign two independent blinded readers and a third adjudicator familiar with the nine labels. Record reader IDs, timestamps, source hashes and disagreements. Resolve ambiguous cases with expert material before primary settlement.
5. The research labeling budget covers the union of both shadow policies plus samples of unselected alerts and unalerted arrivals, with two reads per case and arbitration. It is separate from operational review capacity and simulated inspection fees. Register the maximum number of verification cases with `--budget`. Exceeding it stops packet release. Preregister any revised sampling design in a new future window.

## Isolation, sampling and masking

Hash the native shape as little-endian int64 plus contiguous uint8 pixels. Link new wafers by source lot and identical map hash. Quarantine the whole connected component if any member matches a prior wafer ID, lot or map, including multi-hop links. Report all arrivals, quarantined records and the eligible denominator. Common normal maps can trigger conservative exclusions. Hidden equipment and process relationships remain unmeasured.

Within each eligible lot, include all three selected strata (both, A only, B only) with probability 1. Sample 10% without replacement, rounded up, separately for unalerted and alerted-but-selected-by-neither strata. Use at least two per nonempty stratum, or the entire stratum if smaller. The private ledger records each inclusion probability `pi = sampled / eligible`. A preregistered private random key controls sampling independently of labels.

Give each reader only anonymized `blind/` cases and canonically encoded maps. Withhold lots, paths, scores, alerts, selection, ranking and other opinions. The custodian retains `private_ledger.csv` and `enrollment.json` and independently timestamps the enrollment digest before labels arrive. The adjudicator resolves disputed anonymized cases. Retain digests of all released and returned material.

## Settlement

For each policy report losses from unalerted defects, alerted unreviewed defects, the 10% residual loss on reviewed defects, and 0.1 per selected review. Break upstream and downstream misses into Scratch and all other defects. Estimate missing class outcomes with Horvitz–Thompson weights 1/pi, count all review fees directly, and divide by all eligible arrivals. Estimated missed counts may be fractional.

Report absolute costs, stratified verification standard errors and the paired B−A interval. Labeling the entire selected union makes the in-cohort difference independent of unsampled wafers with identical dispositions. Resample lot/image-linked components 2,000 times for a paired 95% interval, conditional on the frozen queue and models. At least 30 independent components are required for a directional conclusion. Plan precision before recruitment. Support B if the interval is wholly below zero and A if wholly above. Otherwise report uncertainty. These cost and intervention assumptions require separate operational measurement.

## Commands

Run from the repository root with Python 3.10+, NumPy, joblib and scikit-learn 1.6.1. Registry building also needs pandas and psutil. Use a hosted CPU for the already checksum-bound historical pickle. Run future proprietary data and models only in the site's approved environment. Use the stage 07 frozen forest weights in `inputs/`. All output directories must be new. Set the variables from the registered study.

```sh
CODE=cloud_review/blind_review_08
python "$CODE/build_registry.py" --raw "$WM811K_PKL" --out prior08
# For other known exposures, add --extra other_exposures.csv above.
python "$CODE/prospective.py" register --registry prior08 --site "$SITE_LINE_SCOPE" \
  --start "$FUTURE_START_WITH_OFFSET" --budget "$AUDIT_CASE_BUDGET" --out registration08

# After the window closes and the count reconciles, supply the preregistered signed digest.
python "$CODE/prospective.py" enroll --plan registration08/registration.json \
  --plan-sha "$SIGNED_PLAN_SHA" --registry prior08 --arrivals arrivals.csv \
  --maps "$ORIGINAL_MAPS" --models "$FROZEN_MODELS" --mes-count "$SIGNED_MES_COUNT" --out enrollment08

# Both readers return case_id,label. Adjudicate every disagreement.
python "$CODE/prospective.py" settle --enrollment enrollment08 --enrollment-sha "$SIGNED_ENROLLMENT_SHA" \
  --reader1 reader1.csv --reader2 reader2.csv --adjudication resolved.csv --out results08
python "$CODE/synthetic_check.py"
# Optionally exercise the real frozen forests using generated maps.
python "$CODE/synthetic_check.py" --models "$FROZEN_MODELS"
```

The adjudication file may be omitted if both readers agree on every case. Keep the keys, maps, ledgers, returned labels and checkpoints private. The executed historical registry receipt contains 811,457 rows and zero added external records. If new data have been seen, extend the registry and freeze its new receipt before enrollment. [Historical registry receipt](wm811k_registry_receipt.json)

The [historical synthetic acceptance receipt](synthetic_acceptance.json) records the original source hashes. For byte-for-byte historical verification, use the [archived source revision](https://github.com/Matthewsyr/reliable-wafer-classification/tree/c216b86bd3573bf35326c38aae542f3ce0b99663/cloud_review/blind_review_08). The current implementation updates the protocol filename in its docstring.
