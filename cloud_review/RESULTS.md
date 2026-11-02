# Review costs and unlabeled wafer coverage

Hosted Colab run `20261102T042635Z_bf1507` calibrated probabilities on 11,307 development wafers and replayed review decisions on 11,591 previously examined audit wafers. Snowflake independently recomputed costs from frozen outcomes. Costs in this study use hypothetical relative loss units.

## Review decisions on the same alerts

The seed-42 RF71 screen admitted wafers with calibrated defect probability of at least 0.5. Three models ranked the same 3,642 alerts from 903 lots, including 3,484 true defects and 158 normals. Fixed lot ordering produced queues of 1,000, 1,000, 1,000 and 642 alerts with 200, 200, 200 and 128 slots. The screen admitted 62 of the 80 true scratches, leaving 18 beyond the reach of every ranking.

A missed ordinary defect cost 1 unit and a normal wafer had zero defect loss. Each inspection cost 0.1 units and was assumed to avert 90% of the inspected defect's loss. Separate rankings used equal defect losses, fivefold Scratch loss and fivefold Near-full loss. Reference labels determined the simulated costs.

| Fivefold Scratch loss, all slots filled | Seed 42 | Seed 123 | Seed 2026 |
|---|---:|---:|---:|
| RF71 cost per 1,000 alerts | 833.169 | 830.203 | 831.439 |
| RF77 cost per 1,000 alerts | 814.388 | 813.893 | 815.376 |
| RF77 saving per 1,000 alerts | 18.781 | 16.310 | 16.063 |
| Saving per 1,000 arrivals, including upstream misses | 5.901 | 5.125 | 5.047 |
| Additional true scratches reviewed by RF77, net | 3 | 2 | 2 |
| Fewer normal wafers reviewed by RF77 | 64 | 58 | 57 |

RF77 also found more defects in the other classes. Conditional 95% lot-bootstrap intervals for savings per 1,000 alerts were [1.225, 39.414], [1.255, 32.729] and [0.111, 31.598]. The seeds share the same wafers. With rankings recomputed for equal losses or fivefold Near-full loss, the forests had identical point estimates of cost. Production arrival dates, realized costs and human intervention outcomes were unavailable.

## Snowflake verification and numerical reproducibility

Worksheet 01 reproduced the three paired comparisons from reference-class counts. Worksheet 02 held the lists selected at fivefold Scratch loss fixed and repriced them at multipliers 0, 1, 2, 5 and 10. All 15 results matched. Changing the ranking itself requires wafer-level scores. [SQL definitions and inputs](sql/README.md)

The private seed-42 export preserved the four queues, lots, reference labels, model probabilities, original floating-point dot products and frozen lists for all 3,642 alerts. Snowflake ranked each queue using `ROW_NUMBER()`. With the original dot-product scores and tie ordering, each model selected exactly its 728 archived wafers. Switching to RF77 brought 339 wafers into review and displaced 339. Costs per 1,000 alerts were 833.168588688 for RF71 and 814.387699066 for RF77, a difference of 18.780889621.

Reconstructing the benefit algebraically from two probabilities introduced rounding differences of at most `8.88e-16`. This changed 62 RF77 membership flags near ties, while both versions selected 56 true scratches and had identical cost in this run. A separate first SQL attempt matched membership but failed the `1e-7` cost tolerance after taking a fixed-point mean before multiplying by 1,000. Summing costs first and dividing the total in double precision passed the original tolerance. The archived runs retain both outcomes.

## Wafer sizes absent from the normal training class

Exactly 3,181 audit wafers had native height and width absent among training normals. They included 643 normals, 53 scratches and 2,485 other defects. The screen admitted **152/643 normals (23.64%)** in this group and **6/7,348 (0.08%)** at other dimensions. The group accounted for 96.2% of the 158 false alerts. These counts measure any-defect alerts. [Audit strata](results/shift/audit_cost_strata.csv)

| Fivefold Scratch loss within this size group, seeds 42 / 123 / 2026 | RF71 | RF77 |
|---|---:|---:|
| Normals reviewed (152 alerted) | 82 / 75 / 76 | 19 / 18 / 20 |
| Scratches reviewed (50 of 53 alerted) | 46 / 47 / 47 | 47 / 47 / 46 |
| Alerted defects left unreviewed | 1,988 / 1,997 / 1,994 | 2,003 / 2,001 / 2,005 |
| Missed loss for these unreviewed defects | 2,004 / 2,009 / 2,006 | 2,015 / 2,013 / 2,021 |
| Total simulated cost in group | 2,164.6 / 2,167.5 / 2,165.2 | 2,166.7 / 2,165.0 / 2,172.0 |

RF77 freed 63, 57 and 56 slots previously spent on normals in this group. The net change in scratches reviewed was +1, 0 and −1. Three true scratches never passed the screen. The group alone contained 2,523 alerted defects, so even allocating all 728 slots to it would leave at least 1,795 unreviewed. Lower overall cost can accompany higher loss in a size group. [Cost receipt](results/shift/cost_receipt.json)

## Isolating unlabeled maps

The audit scanned 811,457 verified source maps, comprising 638,507 unlabeled, 54,355 labeled development and 118,595 labeled original-test wafers. Linking records transitively by lot or exact pixel hash produced 34,610 components. Excluding the 8,848 components touching a labeled wafer removed 153,525 unlabeled records. Among the remaining 484,982 records, retaining the earliest source row for each exact map yielded **434,498 unique maps in 25,762 components**. Direct-overlap exclusion alone left 465,745 maps. Transitive isolation removed another 31,247. The pool receipt binds the private row manifest to its hash. [Pool summary](results/shift/pool_summary.json) · [Manifest receipt](results/shift/pool_receipt.json)

The isolated pool contains matching dimensions for **2,455 of the 3,181 audit wafers (77.18%)** in the zero-normal-coverage group. Dimensions for the other 726 were absent. An earlier 78,154/78,158 figure described original-test normals and the directly isolated pool. Matching dimensions establish image availability, while defect labels remain unknown. This stage measured coverage and cost. [Size coverage and loss](results/shift/pool_support_and_cost.csv) · [Study decision](results/shift/decision.json) · [Frozen predictions and provenance](https://github.com/Matthewsyr/reliable-wafer-classification/tree/aaa667c0421ece13e3a71fa616fb10287b4453ff/results/probabilities)
