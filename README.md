# Wafer Defect Classification and Review Prioritization

This project uses WM-811K wafer maps to classify defect patterns and prioritize wafers for human inspection. A sharp drop in test performance prompted a closer question: **can a model detect more real scratches without sending more normal wafers to review?**

## 1. Establish a baseline

WM-811K contains 811,457 wafer maps. Of these, 172,950 are labeled as normal or as one of eight defect types. We preserved the original development and test assignments and used production lots to separate training from validation. A random forest using 71 spatial features (RF71) reached **85.28% validation Macro-F1**. A CNN trained on maps of valid and failed dies reached **84.73%**.

## 2. Investigate the test failure

On the frozen test set, Macro-F1 fell to **44.87% for RF71 and 45.16% for the CNN**. Among normal wafers the CNN misclassified as Scratch, **97.31%** had dimensions represented by fewer than 25 normal wafers in development.

Subsequent experiments returned to development data. Some repairs reduced normal-to-Scratch errors but missed real scratches or moved false alarms into another defect class. We therefore tracked three outcomes for each training seed: true Scratch recall, the fraction of normal wafers called Scratch, and the fraction called **any defect**.

## 3. Compare model changes

Adding six features that describe the linear arrangement of failed dies produced RF77. The table reports three-seed means on the same historical development audit, which contained 80 true scratches and 7,991 normal wafers.

| Model | Scratch recall | False alarms per 1,000 normal wafers |
|---|---:|---:|
| RF71 | 32.08% | 12.35 |
| RF77 | **42.08%** | **7.17** |

CNN experiments exposed a different trade-off. Weighted sampling reached 74.17% Scratch recall but flagged 26.91% of normal wafers. In one matched run, self-supervised pretraining on 50,000 isolated unlabeled maps found seven additional scratches and raised the normal false-alarm count by 64. Standard cross-entropy remained the CNN control.

## 4. Evaluate a limited review queue

With at most **200 inspections per 1,000 alerts**, we compared random selection, ranking by defect probability, and ranking by expected avoidable loss. Historical labels determined which defects each policy caught. Simulated cost per 1,000 **arriving wafers** included defects missed before they entered the queue. We retained RF71 screening followed by RF77 defect-probability ranking as the baseline for a future blind evaluation. All cost assumptions are hypothetical.

## 5. Test synthetic scratches and pseudo-labels

Nine matched CNN fits compared the original training distribution, repeated sampling of real scratches, and synthetic scratches. On the historical audit, synthetic training reduced normal-to-Scratch errors from **1.90% to 1.15%**, yet total normal false alarms rose from **3.03% to 3.29%**. In one seed, 165 of the 224 normal wafers originally called Scratch were reclassified as localized defects. RF77 pseudo-labels also fell short of the prespecified evidence threshold in all three size groups, so student CNN training was skipped.

The historical study ends with RF77 as the review baseline. Replacing it requires independently labeled new lots evaluated under the same inspection budget.

## Reproduce the geometry comparison

From the repository root, replay the archived predictions to verify classification metrics, thresholds, and false-alarm transfers:

```bash
python -I scripts/recompute_decisions.py
```
