# Model Card — Reliable Wafer Defect Classification

## Purpose and status

This repository is a research and engineering case study in generalization, rare-class detection and decision trade-offs on WM-811K. It is suitable for reproducing controlled experiments and inspecting failure modes. It is **not validated for manufacturing disposition, automatic wafer rejection, process diagnosis or production deployment**.

There is no newly promoted production model. Historical RF/CNN checkpoints preserve the original frozen experiment. Later repair and geometry results are development-only exploration. See [evaluation history](EVALUATION_HISTORY.md) and the [project report](../reports/Project_Report.md).

## Models and input contract

| Item | Historical RF baseline | Historical medium CNN |
|---|---|---|
| Input | 71 spatial features from a categorical wafer map | 2×64×64 valid-die / failed-die coverage channels |
| Estimator | 200-tree RandomForestClassifier; unlimited depth; minimum leaf size 2; sqrt feature sampling; balanced_subsample | Three Conv–BatchNorm–ReLU–MaxPool blocks, average pooling, 9-logit head |
| Trainable CNN parameters | Not applicable | 111,625 |
| Final fit | All original development rows; seed 42 | All original development rows; seed 42; 7 epochs |
| Decision | Nine-class argmax | Nine-class argmax |

The fixed class order is `Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none`. Unlabeled rows are not assigned to `none`. The original grid values 0/1/2 describe outside-wafer/passing/failing die. Padding and resizing operate on separate binary masks; no fitted input-normalization transform is used; the CNN itself learns BatchNorm running statistics.

The final geometric ablation appends six label-free line descriptors to the original 71 features. One is a deterministic transformation of existing covariance moments; five describe narrow-strip organization. This is a separate experimental feature mode, not a replacement historical checkpoint.

## Data and evaluation

WM-811K contains about 811,457 wafer maps; the historical audit found 172,950 labeled rows. The original development/test split contains 54,355/118,595 labeled rows with zero overlapping production lots. Test normal support is 110,701; test Scratch support is 693. Fourteen exact-map hash groups contain conflicting labels, and 3,158 test rows have map content identical to development content. Original identities and test rows were preserved; a predeclared sensitivity result was reported separately.

Historical three-seed lot-validation Macro-F1 was **0.852818 ± 0.008787** for RF and **0.847272 ± 0.012804** for CNN (sample SD, not a confidence interval). The original frozen all-wafer test gave:

| Metric | RF | CNN |
|---|---:|---:|
| Macro-F1 | 0.448672 | 0.451551 |
| Accuracy | 0.697146 | 0.477878 |

Full per-class precision/recall/F1/support and confusion matrices are included in `results/historical/`. These measurements do not establish statistically significant CNN superiority. The test was later inspected for diagnosis, so subsequent choices cannot be evaluated on it as if it were untouched.

New development experiments use a fixed lot/exact-map connected-component partition of 31,457 training, 11,307 tuning and 11,591 audit rows. The audit has already been inspected; its use is explicitly exploratory. The three model seeds share the same held-out wafers. The audit's 80 Scratch examples and 7 Near-full examples limit precision and subgroup conclusions.

## Thresholds and use of scores

Model scores are not calibrated operational probabilities. Scratch one-vs-rest decisions and their precision–recall curves are defined separately from nine-class argmax. The 80% precision floor is a hypothetical analytical constraint. Thresholds are selected only on tune and frozen for their associated fitted model. They are not automatically transferable to a newly refitted model or a different prevalence.

Scratch-priority rules are also evaluated as complete nine-class classifiers: if the Scratch score reaches the threshold, predict Scratch; otherwise choose the largest of the other eight scores. Report normal→any-defect errors as well as normal→Scratch errors. No-positive predictions do not count as a feasible precision-qualified classifier.

## Final exploratory findings

The fixed RF geometry block raised audit mean Scratch recall from 32.08% to 42.08% and reduced normal total errors from 12.35 to 7.17 per 1,000. This is a previously observed development audit, not a new independent test. All three paired seeds improve the joint point estimates. The 80%-precision Scratch-priority rule is not adopted: relative to the geometric RF argmax it loses 2/6/4 Scratch hits and adds 2/1/3 normal total errors across seeds. See the source-bound report for uncertainty and full nine-class metrics.

## Known failure modes

- Strong development-to-original-test degradation for both model families.
- Underrepresented normal-wafer geometries and size distributions are associated with false Scratch predictions; this association is not a causal attribution percentage.
- Sparse normal failures, fragmented scratches, local clusters and label ambiguity can create overlapping image evidence.
- Area resizing can remove thin/small defects; the valid-mask geometry can encode product/lot characteristics.
- BatchNorm running-statistic sensitivity and refit variation can change the decision boundary. Development repair candidates failed their joint recall/normal-error requirements.
- No reliable process-tool metadata, adjudicated ambiguous labels, manufacturing cost function or confirmed chronology is available.

## Reproducibility and artifacts

Code, seeds, configuration, split manifests, saved measurement tables, training histories, source hashes and validation records are included. Raw data and large model binaries are excluded from Git. [Reproduction instructions](REPRODUCIBILITY.md) identify the data source, checksum, historical Kaggle versions and weight-recovery/retraining options. Fixed seeds do not ensure bit-identical results across CPU/GPU, library versions and operating systems.

AI assistance and the difference between computed evidence, inspection notes and hypotheses are documented in [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
