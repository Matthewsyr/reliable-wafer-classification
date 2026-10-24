# Evaluation history and limits

## Historical frozen experiment

The original stage-2 baseline contained 30 development experiments. Stage 3 reused its exact validation partitions for an 18-run CNN matrix. CNN selection used mean lot-validation Macro-F1 across three seeds; epoch selection was restricted to the predefined eight-epoch budget. The selected model was refitted for seven epochs, the median selected development epoch, on all 54,355 development wafers. The RF configuration was preserved from stage 2.

`selection_freeze.json` was saved at **2026-10-22 06:48:42 UTC**, before first original-test processing at **07:42:49 UTC**. Scoring completed at **07:51:38 UTC**. These UTC values are provenance timestamps, not local report-generation dates.

Both frozen models were evaluated on the unchanged 118,595-row original labeled test, comprising 4,953 lots with zero shared development lots. This is the historical final evaluation; it is not repeated for newly selected models. The primary all-wafer results were RF Macro-F1 0.448672 and CNN Macro-F1 0.451551. The predeclared nonidentical-map sensitivity analysis retained 115,437 rows and gave 0.444347 and 0.448847 respectively. The small numerical CNN advantage is not a demonstrated statistically significant win.

## What happened after the test

The original-test predictions were inspected to understand normal→Scratch errors, size coverage and sensitivity to BatchNorm statistics. The original test is therefore no longer untouched for later research decisions. Diagnostic rescoring and sample interventions are recorded as diagnostics, not replacement official scores.

Two subsequent development repair rounds used a fixed lot/exact-map-component train/tune/audit split. Both produced useful failure evidence but did not satisfy their registered repair objectives. By the 2026-10-24 engineering and geometry round, that audit had already been inspected; calling it a newly independent test would be incorrect.

## Final delivery round

The final round adds one bounded geometric-feature ablation, development-only Scratch precision–recall decision analysis, reusable code and tests, and documentation. Its protocols are frozen before the new scores and threshold decisions. Original final scores remain historical evidence; no new model is promoted by optimizing that original test. New comparisons use the preserved development split and are explicitly exploratory.

This closes the requested analysis and engineering work without manufacturing a new independent dataset. A future confirmatory evaluation requires new data whose labels and outcomes have not influenced these decisions.

## Source versions

- [Stage 1 data audit, version 360906152](https://www.kaggle.com/code/yueransong29/wm-811k-01-data-audit-and-lot-leakage-checks?scriptVersionId=360906152)
- [Stage 2 baseline, version 360929617](https://www.kaggle.com/code/yueransong29/wm-811k-02-ml-baseline-and-lot-validation?scriptVersionId=360929617)
- [Stage 3 CNN/final test, version 360951977](https://www.kaggle.com/code/yueransong29/wm-811k-03-cnn-training-and-lot-test?scriptVersionId=360951977)

Direct web retrieval of these version pages was unavailable during the final delivery audit. Work continued from the preserved local source, manifests, provenance, checkpoints and hash-verified results; no missing remote execution was claimed.
