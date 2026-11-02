# Reproduce the development Scratch study

Use the scripts from this directory with a separate checkout of the frozen historical dependencies. The original run used one hosted Colab T4. Obtain `LSWMD.pkl` from Kaggle dataset version `qingyi/wm811k-wafer-map/versions/1`. Adapt the paths to your session.

```bash
git clone https://github.com/Matthewsyr/reliable-wafer-classification.git /content/frozen-wafer
git -C /content/frozen-wafer checkout --detach 985c3117a20ea2abdfa9812f5a46d873cd089a47
REPO=/content/frozen-wafer
RAW=/path/to/LSWMD.pkl
RUN=/content/wafer09
mkdir -p "$RUN"
```

The frozen checkout supplies the rasterizer, CNN, development group assignments and RF77 probabilities. Scripts check SHA-256 digests and decode only the indexed development maps. Training, threshold design, selection and historical development audit contain 31,457, 5,637, 5,670 and 11,591 labeled wafers. Each lot and each component linked by exact image hashes stays within a single partition.

## Matched training

```bash
python -u train09.py --repo "$REPO" --raw "$RAW" --cache "$RUN/cache" --out "$RUN/training" \
  --arms baseline real_repeat synthetic --seeds 42 123 2026
python -u pseudo09.py qualify --repo "$REPO" --out "$RUN/rf_gate"
```

The three arms share the medium CNN, optimizer, initialization and natural training order within a seed. Each runs seven epochs of 31,457 exposures and 246 updates. The repeat arm replaces 80 normal slots with genuine training scratches. The synthetic arm uses the same slots and transplants observed trace dies onto size-compatible normal training maps. It preserves the wafer boundary and existing failures. Geometric filters can require a different donor from the one drawn for the repeat arm.

## Teacher qualification

Use the raw votes of RF77 seed 42. Count normal training wafers with the same ordered dimensions and group sizes into support bands of zero, 1–24 and at least 25. On design lots, take only Scratch argmax predictions. Test cutoffs of 0.50, 0.60, 0.70, 0.80, 0.90 and 0.95. Choose the eligible cutoff with greatest coverage in each band, taking the higher cutoff on a tie. Require at least 12 candidates from five lots, 90% observed precision and a one-sided 95% Wilson lower bound of 75%. On confirmation lots, evaluate the frozen cutoff once and additionally require at least 80% precision after removing any single contributing lot.

Only a nonempty `qualified_bands` result requires the RF77 weights, the isolated 50,000-map unlabeled sample and the source registry. See `python pseudo09.py select --help` for the inputs. Verify the teacher against frozen reference scores, then select at most 80 admitted maps by hash, capped at two per lot. If the resulting `pseudo_train` flag is true, repeat the training command with `--arms student --pseudo "$RUN/rf_gate"`. Pseudo-labels receive a cross-entropy weight of 0.25. The historical gate admitted zero maps.

## Paired evaluation and audit

```bash
python -u evaluate09.py --repo "$REPO" --training "$RUN/training" \
  --pseudo-decision "$RUN/rf_gate/select_receipt.json" --out "$RUN/evaluation"
python -u audit09.py --repo "$REPO" --training "$RUN/training" \
  --eval "$RUN/evaluation" --out "$RUN/audit09"
python -u transitions09.py --repo "$REPO" --training "$RUN/training" \
  --eval "$RUN/evaluation" --out "$RUN/transitions"
```

Apply the three selection criteria to nine-class argmax predictions for each seed. Only an arm that meets every criterion across the three seeds and improves either Scratch hits or all normal false alarms advances to the cost simulation. Report the fixed comparisons on the previously examined audit lots. Estimate uncertainty by resampling whole lot/image components within a seed. Verify the baseline confusion matrices, paired replacement slots and original die provenance for the saved examples.

The [English evaluation report](results/evaluation/REPORT.md) uses the published aggregate counts. The historical [run receipt](results/evaluation/run_receipt.json) and dependent audit receipts preserve original artifact and code hashes. To verify the historical report or recover the exact evaluator bytes, use the original publication commit `8ab683fc1dfd6d27a61871eb054ec2249a8a4e21`. The English renderer generates a new report and a new receipt on a fresh run. The simulation inputs and quantitative results are unchanged.

```bash
git show 8ab683fc1dfd6d27a61871eb054ec2249a8a4e21:cloud_review/scratch_09/results/evaluation/REPORT.zh-CN.md | sha256sum
# d48a7cd4ac624cf149ffb1e60dc04516346ecb161e90b6975de93095b9451248
```
