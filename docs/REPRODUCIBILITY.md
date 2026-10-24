# Reproducibility guide

There are three separate tasks: recomputing committed metrics, running the portable pipeline, and reconstructing historical experiments. The first is fast and needs only Python. Full training requires the external dataset. Re-evaluating an already exposed test does not make it independent again.

## 1. Replay the actual decisions without training

From the repository root:

```bash
python -I scripts/recompute_decisions.py
```

This standard-library-only script reads the committed 18 probability tables, checks their SHA256 values, independently selects each threshold from tune, recomputes Average Precision and 36 complete nine-class decisions, and checks paired false-alarm transfers. It passed 5,846 checks both in the project and after copying only the script and its data into a separate clean directory. It does not need NumPy, raw maps, checkpoints or access to the original workspace. Receipts and source-to-export hashes are in `results/probabilities/`.

The larger historical/final-round audits separately contain 14,317 PR/geometry checks and 260 additional geometry-summary checks. These numbers are different verification scopes, not independent scientific replications.

## 2. Install the portable package

Use a fresh Python 3.10 or 3.11 environment. Activate it using the command appropriate to your shell.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/verify_provenance.py
```

For CNN runs, install a PyTorch build compatible with the machine. A CPU-only example is:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

The package supports PyTorch 2.6–2.7. Local engineering validation actually used an existing **CPU PyTorch 2.6** environment after a different GPU environment stalled on import; it does not establish a fresh installation test of every supported version. CI is configured for CPU PyTorch 2.7.1. Historical final CNN training used **Python 3.13.15 / torch 2.11.0+cpu** on Linux; the later matched CNN predictions came from the recorded repair environment. The new RF ablation used Python 3.10.11, NumPy 2.1.3, scikit-learn 1.6.1 and joblib 1.6.0 on Windows. These are deliberately separate provenance records. Fixed seeds do not guarantee bit identity across these environments.

## 3. Acquire and verify data

Download [WM-811K from its recorded Kaggle provider](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map), subject to that provider's access terms. Use the original `LSWMD.pkl`, not a repackaged CSV or resized image export. Its audited SHA256 is:

```text
1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65
```

Place it at `data/raw/LSWMD.pkl` or substitute your own path in the commands below. Do not commit it or account credentials. The CLI refuses a checksum mismatch. The shipped [development reference](../data-reference/README.md) combines actual wafer identity, verified raw offsets, map hashes and the fixed exploratory split; it contains no raw map arrays. Its `source_row` is the original positional row identity, while `(lot, wafer_index)` is the wafer metadata identity. Map-content hashes are a separate concept.

The preferred indexed reader verifies the full source checksum, reads only selected uint8 payloads by offset and verifies each map hash. It does not execute pickle opcodes. The optional whole-file `pandas.read_pickle` fallback is allowed only after the same known checksum passes; it can require substantial memory and was not used for final delivery. Treat arbitrary pickle/joblib files as executable content, not generic safe data.

## 4. Prepare the fixed development experiment

This command uses the same compact file for metadata, raw offsets and preserved roles. It prepares all 54,355 development wafers and never reads final-test map payloads.

```bash
python -m waferdefect prepare --raw-pickle data/raw/LSWMD.pkl --metadata data-reference/development_reference.csv.gz --raw-index data-reference/development_reference.csv.gz --existing-split data-reference/development_reference.csv.gz --representation both --include-linearity --output data/prepared
```

The CNN array alone is approximately 1.78 GB. Allow several GB of local working space and memory for staging and fitting. Use `--representation rf` if only repeating the geometric RF comparison. Do not use `--limit` for scientific scores. Preparation emits validated metadata, split diagnostics, numeric arrays and their hashes. Existing nonempty output directories are rejected to avoid accidental mixing of runs.

Rows with missing labels are excluded from supervised fitting and counted; the string `none` remains the normal class. Invalid maps or identity violations are recorded/rejected by the audit rather than silently recoded. Exact-content duplicates and conflicting labels remain recorded; connected-component splitting keeps linked development rows together. The supplied roles reproduce the previously used 31,457 / 11,307 / 11,591 split. Creating a new grouped split is supported for engineering use but is a different experiment and should not inherit the published scores.

## 5. Repeat a matched RF comparison

```bash
python -m waferdefect train --data data/prepared --config configs/rf_matched_baseline.json --output runs/rf71_s42
python -m waferdefect train --data data/prepared --config configs/rf_matched_linearity.json --output runs/rf77_s42
python -m waferdefect evaluate --data data/prepared --run runs/rf71_s42 --role tune --output runs/rf71_s42_tune
python -m waferdefect evaluate --data data/prepared --run runs/rf77_s42 --role tune --output runs/rf77_s42_tune
python -m waferdefect evaluate --data data/prepared --run runs/rf71_s42 --role audit --output runs/rf71_s42_audit
python -m waferdefect evaluate --data data/prepared --run runs/rf77_s42 --role audit --output runs/rf77_s42_audit
```

The matched configs use the actual 200-tree, minimum-leaf-2, balanced-subsample RF and four workers. Repeat with copies of each config whose `seed` is 123 and 2026, and use new output directories. Do not select a favorable seed or alter the split. The lighter `rf_baseline.json`/`rf_linearity.json` defaults use two workers; they preserve the statistical configuration but are not the recorded cost configuration.

The portable package reuses the original feature implementation byte-for-byte and the final geometry helper with tracked source hashes. It adds train-only imputation and explicit data/artifact checks. The final-round scientific fits were made with the frozen original runner; the compact package was separately checked with 18 critical tests and a 90-map real-data run, **not** by repeating six more full fits. Metric replay exactly checks the shipped predictions; a new full fit is a reproduction attempt whose outputs must be compared, not assumed identical.

## 6. CNN and small real-data engineering run

```bash
python -m waferdefect train --data data/prepared --config configs/cnn_bn_fixed7.json --device cpu --output runs/cnn_s42
python -m waferdefect evaluate --data data/prepared --run runs/cnn_s42 --role tune --device cpu --output runs/cnn_s42_tune
```

`cnn_bn_fixed7.json` specifies the original medium BatchNorm architecture and seven fixed epochs; `cnn_medium.json` demonstrates the historical eight-epoch validation-selection procedure. Neither creates a new official final-test result. The final-round CNN comparison reused existing repair-screen logits after verifying exact training-row identity; retraining here is not claimed to reproduce GPU logits bit-for-bit.

For a small engineering check with the actual public maps:

```bash
python scripts/smoke_from_index.py --raw data/raw/LSWMD.pkl --metadata data-reference/development_reference.csv.gz --index data-reference/development_reference.csv.gz --split data-reference/development_reference.csv.gz --device cpu --output runs/real_smoke
```

This deterministically selects six training, two tune and two audit rows per class (90 rows total), then executes preparation, RF training/evaluation and one CNN epoch/evaluation. The local run passed. Its tiny-sample accuracy is intentionally not a scientific result; only pipeline execution is claimed.

## 7. Freeze a decision and understand test guards

For a newly trained fit, the following illustrates a tune-selected Scratch rule, explicitly acknowledging that the original source test has been exposed:

```bash
python -m waferdefect freeze --data data/prepared --run runs/rf77_s42 --precision-floor 0.8 --test-source-sha256 1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65 --acknowledge-previous-test-exposure --output runs/rf77_s42_freeze.json
```

The freeze binds model, configuration, training-data manifest and threshold to hashes. A threshold is chosen only on rows not used to fit that model. No feasible nonempty rule means no freeze is written. The original committed thresholds live in `results/protocol_pr/threshold_selection.json`; recomputing a threshold for a new fit must never overwrite that historical record. In this study the threshold rule was rejected as a repair despite meeting the tune precision floor.

Final-test preparation/evaluation requires an earlier freeze; evaluation also requires `--once` and creates an exclusive receipt. The CLI records evaluation of the exposed original source as a historical reassessment, not independent confirmation. These controls prevent accidental reruns, not deliberate deletion or external editing by a user. No new final-test run is part of this guide's standard workflow. Preserve original final scores; obtain genuinely new untouched data for new confirmatory claims.

## 8. Historical artifacts and weights

The [evaluation history](EVALUATION_HISTORY.md) links the exact saved Kaggle notebook versions. Their saved outputs contain historical experiment records/checkpoints; availability requires the user's provider access. No new hosted weight download or release is claimed. If a historical checkpoint cannot be obtained, recreate it using its recorded source, environment, configuration, development indices and seed; label the result a rerun rather than replacing the original metric table.

The source snapshots under `scripts/legacy/` retain the historical runners for inspection. They expect their original Kaggle/notebook input layout and are not the portable CLI. The original feature/CNN modules are tracked by `src/waferdefect/provenance.json`. The local work archive also retains the six actual RF models; they are excluded from the lightweight GitHub tree. Model weight SHA256, tree counts, file sizes and timing are recorded in `results/geometry/runtime_cost_summary.json`.

`results/historical/` preserves actual original tables, not fabricated examples. `results/geometry/` and `results/protocol_pr/` preserve the bounded exploratory round. `results/probabilities/` makes its decision calculations portable without redistributing raw images or large models. `results/engineering/` records implementation verification. Paths embedded in older provenance documents describe the original execution environment, not prerequisites for the portable commands.

## 9. Before publication

```bash
python scripts/check_publish.py
python scripts/verify_provenance.py
python -I scripts/recompute_decisions.py
```

The publication check excludes raw/model binaries, credential filenames and files larger than 10 MiB. It does not establish absence of every possible secret; the delivery also performs a content-pattern scan and reviews the staged file list. Tests, provenance and metric checks complement one another and do not certify operational reliability.

To regenerate the saved historical CNN training curves, install the optional `matplotlib` package and run `python scripts/plot_training_history.py`. It reads the committed epoch CSVs and performs no training.
