# Reliable Wafer Defect Classification

A WM-811K case study in detecting rare scratches and choosing which wafers receive a limited number of human inspections. The original CNN lost nearly 40 Macro-F1 points on the frozen test. This project traces that failure from wafer dimensions through model training to the inspection queue.

[中文技术总结](docs/TECHNICAL_SUMMARY.zh-CN.md) · [Interview guide](Technical%20Interview%20Preparation.md) · [Reproduce the results](docs/REPRODUCIBILITY.md) · [English technical report](reports/Project_Report.md)

**Project status.** The historical study is complete. The review baseline for a future blind evaluation screens wafers with RF71, then ranks alerts with RF77 by the calibrated probability of any defect. At most 200 wafers per 1,000 **alerts** go to inspection. Production performance and savings still require independently labeled new lots. [Frozen review protocol](cloud_review/blind_review_08/PROTOCOL.zh-CN.md)

## Why performance dropped

| Original model | Lot-held-out validation Macro-F1 | Frozen test Macro-F1 |
|---|---:|---:|
| Random forest | 85.28% | 44.87% |
| CNN | 84.73% | 45.16% |

Normal wafers made up 67.57% of labeled development wafers and 93.34% of labeled test wafers. Among the CNN’s normal wafers misclassified as Scratch, 97.31% had dimensions represented by fewer than 25 development normals. This identifies where errors concentrated. The dataset cannot separate the effects of wafer dimensions, product mix and batch conditions. [Error analysis and wafer examples](docs/ERROR_ANALYSIS.md)

![Validation and original frozen-test Macro-F1](reports/figures/generalization_gap.png)

## Which changes earned their place?

The comparisons below use the **historical development audit**. Each seed saw the same 80 true scratches and 7,991 normal wafers. These lots had already been inspected during research.

| Experiment | What happened on those lots | Decision |
|---|---|---|
| Add six geometry features to the random forest | Mean Scratch recall rose from 32.08% to 42.08%. Total false alarms fell from 12.35 to 7.17 per 1,000 normal wafers. | Retain the 77-feature forest (RF77) for review ranking. |
| Reweight, resample or pretrain the CNN | Weighted sampling reached 74.17% Scratch recall, while 26.91% of normals triggered an alarm. In one masked-pretraining run, Scratch hits rose from 38 to 45 and normal false alarms rose from 264 to 328. | Keep standard cross-entropy as the CNN control. |
| Repeat real scratches or transplant Scratch patterns onto normal maps | Synthetic training cut normal-to-Scratch errors from 1.90% to 1.15%, but total normal false alarms rose from 3.03% to 3.29%. Neither augmentation passed the joint development-selection rule across all three seeds. | Retain the control. No size group supplied enough evidence to admit RF77 pseudo-labels for student training. |

[Geometry ablation](reports/Project_Report.md) · [Imbalance and calibration](cloud_review/results/full/SUMMARY.md) · [Paired pretraining pilot](cloud_review/results/ssl_paired/pilot_summary.json) · [Matched Scratch experiments](cloud_review/scratch_09/README.md) · [Teacher precision by size](cloud_review/scratch_09/RF77.zh-CN.md)

**What a lower Scratch false-alarm rate can hide.** In one audit run (seed 2026), the baseline CNN called 224 normal wafers Scratch. After synthetic training, 165 of those same wafers were called localized defects (Loc), 32 were correctly called normal and 27 were still called Scratch. The new model also lost 17 previously detected true scratches. [Paired wafer transitions](cloud_review/scratch_09/results/paired_transitions.json)

## Reproduce one result

From the repository root, run the following with Python 3.10 or 3.11. It uses the standard library and the 18 archived development prediction tables to recompute the geometry study’s decisions, thresholds and false-alarm transfers.

```bash
python -I scripts/recompute_decisions.py
```

For full training, obtain the original WM-811K wafer maps and verify the recorded checksum. The guides cover the split, model inputs, training environment and output checks. [Data and training instructions](docs/REPRODUCIBILITY.md) · [Scratch experiment reproduction](cloud_review/scratch_09/REPRODUCE.zh-CN.md)

## Inspection costs and remaining evidence

The historical simulations fix the alert pool and inspection budget, then use true labels to compare rankings. Costs are scenario assumptions. The proposed blind study also samples unflagged wafers and measures cost per 1,000 **arrivals**, so defects missed by the upstream screen count against the system. [Historical queue replay](cloud_review/roadmap_07/REPORT.zh-CN.md) · [Wafer-level review exchanges](cloud_review/exchange_08/HISTORICAL_EXCHANGE.zh-CN.md) · [Prospective protocol](cloud_review/blind_review_08/PROTOCOL.zh-CN.md)

[Model card](docs/MODEL_CARD.md) · [Evaluation history](docs/EVALUATION_HISTORY.md) · [Experiment journal](docs/EXPERIMENT_JOURNAL.md) · [AI assistance and verification](docs/AI_ASSISTANCE.md) · [Dataset source](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
