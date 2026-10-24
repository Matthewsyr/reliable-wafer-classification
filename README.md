# Reliable Wafer Defect Classification
### Generalization, Rare-Class Detection, and Decision Trade-offs

A reproducible WM-811K case study about what breaks when a strong validation result meets a shifted population—and how to investigate improvements without hiding false alarms or reusing an exposed test as independent evidence.

**Status:** bounded experiments completed; research prototype. Historical frozen test and later development exploration are reported separately. No production deployment claim.

## Key findings

- **Generalization failure:** RF/CNN lot-validation Macro-F1 of 85.28%/84.73% fell to 44.87%/45.16% on the original frozen test.
- **Useful exploratory direction:** a fixed geometry feature block improved mean Scratch recall from 32.08% to 42.08% while lowering normal total false alarms from 12.35 to 7.17 per 1,000 on an already-observed development audit. All three model seeds improved both point estimates.
- **A threshold is not automatically a repair:** the illustrative 80%-precision rule reduced RF77 audit Scratch hits and increased normal total errors relative to argmax. It was not adopted.

![Generalization gap](reports/figures/generalization_gap.png)

## Matched exploratory ablation

Same fixed train/tune/audit rows; 3 seeds; six new RF fits and three reused, lineage-checked CNN fits. Audit support: 80 Scratch and 7,991 normal wafers. Means across seeds below do not create additional independent examples.

| Model / nine-class argmax | Macro-F1 | Scratch precision | Scratch recall | Scratch AP | Normal→any / 1,000 | Normal→Scratch / 1,000 |
|---|---|---|---|---|---|---|
| RF · 71 original features | 83.88% | 50.27% | 32.08% | 0.3186 | 12.35 | 0.584 |
| RF · 71 + 6 geometry features | 87.22% | 90.14% | 42.08% | 0.5599 | 7.17 | 0.125 |
| CNN · matched training rows | 82.39% | 20.10% | 49.58% | 0.3004 | 33.12 | 20.231 |

The original 71 features already include covariance. PCA anisotropy is a re-expression of that information; the experiment tests a six-feature block including narrow-strip organization. It cannot attribute the gain to PCA alone. A fresh independent test of the new candidate is unavailable.

## Read the project

- [English project report](reports/Project_Report.md) — protocols, final-test degradation, ablation, PR decisions and limitations.
- [中文技术总结](docs/TECHNICAL_SUMMARY.zh-CN.md) — 用中文解释结果与取舍。
- [Error analysis and real Scratch examples](docs/ERROR_ANALYSIS.md).
- [Model Card](docs/MODEL_CARD.md), [evaluation history](docs/EVALUATION_HISTORY.md), [experiment journal](docs/EXPERIMENT_JOURNAL.md).
- [Reproducibility](docs/REPRODUCIBILITY.md), [AI assistance](docs/AI_ASSISTANCE.md), [interview preparation](Technical%20Interview%20Preparation.md), [resume bullets](docs/RESUME_BULLETS.md).

## Quick start

Use Python 3.10 or 3.11 and a fresh environment. The following installs the RF pipeline and tests; CNN dependencies are optional.

```bash
python -m venv .venv
# Activate .venv using your shell's normal command.
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/verify_provenance.py
python scripts/recompute_decisions.py
```

Metric replay uses the committed compressed prediction tables and needs no raw dataset or model training. See the [reproduction guide](docs/REPRODUCIBILITY.md) for verified dataset acquisition, the fixed split, CPU/GPU setup and full training commands. The original raw data and checkpoint binaries are intentionally external; weights can be recreated or recovered from the recorded historical Kaggle outputs.

## Repository map

```text
src/waferdefect/       Data, split, RF/CNN, geometry, evaluation and freeze CLI
configs/              Reusable bounded experiment configurations
tests/                Split, label, preprocessing, metric and protocol checks
scripts/              Integrity checks, metric replay and engineering smoke run
data-reference/       Compact metadata, fixed split and original-data checksum
results/              Actual historical / exploratory scores, freezes and checks
reports/              English report and source-backed figures
docs/                 Error analysis, Model Card, Chinese summary and explanation
```

## Data and sources

[WM-811K dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map). The study preserves the original labeled development/test designation and documents exact-map overlap and label conflicts. [Recorded Kaggle versions and verification history](docs/EVALUATION_HISTORY.md). Obtain data under the provider's terms; this repository does not redistribute the full dataset.

The existing official test was examined after its first frozen evaluation. Subsequent ideas and development audit results are exploratory; they must not be represented as a second pristine final test. AI contributed to implementation, execution and documentation; this is disclosed and does not imply completed personal human review.
