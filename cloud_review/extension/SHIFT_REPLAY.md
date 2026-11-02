# Rebuild the unlabeled pool and the historical size breakdown

## Rebuild the isolated pool

From the repository root in a hosted Colab or Kaggle CPU runtime, obtain `LSWMD.pkl` from `qingyi/wm811k-wafer-map/versions/1`. Set `RAW` to its path and `OUT` to your output directory. The scanner uses NumPy, pandas and psutil. It requires at least 8 GiB of available memory, over 2 GiB of temporary disk space and, by default, a 25-minute time limit.

```sh
python cloud_review/extension/shift_followup.py --stage pool \
  --previous cloud_review/results/legacy_04 \
  --reference data-reference/development_reference.csv.gz \
  --indexer cloud_review/extension/shift_audit_v1.py \
  --raw "$RAW" --out "$OUT"
```

The [frozen indexer](shift_audit_v1.py) has SHA-256 `786692eaea13c1b0a6d52d821e443e00f546735aa353a24d034caec94638b7e2`. It checks the raw archive before unpickling and hashes native dimensions with pixels. The [archived summary](../results/legacy_04/shift/shift_summary.json) also stores available memory and package versions, so a fresh run can produce a different receipt hash.

Check `$OUT/pool/pool_summary.json` against the historical 34,610 connected components, 8,848 protected components and 434,498 unique eligible maps. Compare decompressed pool records across runs, since gzip timestamps can differ. The full list contains source rows, lots, components, dimensions and image hashes and stays in the operator's storage. [Historical pool statistics](../results/shift/pool_summary.json) · [Original receipt](../results/shift/pool_receipt.json)

## Reproduce the historical cost strata

Exact selected-list replay requires the private row-level outputs of the stage-04 run `20261102T042635Z_bf1507`. Point `PREVIOUS` to the run directory containing:

- `cost/protocol.json` and `cost/tune_calibration_freeze.json` for the relative costs and frozen RF71 seed-42 alert rule.
- `cost/gate.csv` and `cost/summary.csv` for truth counts, review totals and losses.
- `cost/evidence/simulated_flagged_candidates/s{42,123,2026}/{RF71,RF77}/selected_rows.json.gz` for the six selected lists. The replay uses source rows from the `scratch_5x` scenario and `benefit_fixed` ranking.

```sh
python cloud_review/extension/shift_followup.py --stage cost \
  --previous "$PREVIOUS" --reference data-reference/development_reference.csv.gz \
  --anchor-probs results/probabilities/rf_original71_s42_audit.csv.gz --out "$OUT"
python cloud_review/extension/shift_followup.py --stage decision --out "$OUT"
```

The decision stage verifies the cost and pool receipts from the same output directory. The public aggregate tables support independent checks of size coverage, false-alarm rates and review counts. Reconstructing individual review assignments requires the saved lists. All performance and cost figures here describe historical development lots. [Counts by size](../results/shift/audit_cost_strata.csv) · [Historical cost receipt](../results/shift/cost_receipt.json) · [Size coverage and missed loss](../results/shift/pool_support_and_cost.csv)
