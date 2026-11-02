# Unlabeled coverage and masked pretraining

WM-811K contains 638,507 unlabeled maps, 54,355 labeled development maps and 118,595 labeled original-test maps. The unlabeled pool spans 41,608 lots and 533,150 distinct native-shape and pixel hashes. Of the unlabeled wafers, 66,666 share a development lot and 5,230 share a test lot. Exact map matches affect 1,816 and 5,666 unlabeled wafers respectively. Conditions can overlap.

## Failed-die fraction

Divide the number of failed dies by the number of valid dies. A normal label describes the whole-wafer pattern. A small fraction or zero failed dies does not establish a label for an unlabeled wafer. Zero failures occur on 58,327 unlabeled maps.

| Population | Wafers | 25th percentile | Median | 75th percentile |
|---|---:|---:|---:|---:|
| Unlabeled | 638,507 | 1.40% | 5.25% | 13.79% |
| Labeled development | 54,355 | 10.89% | 15.44% | 19.72% |
| Previously inspected original test | 118,595 | 6.09% | 9.08% | 12.37% |

The stage 04 cloud run `20261102T042635Z_bf1507` produced the quantiles and counts. Total variation distances between binned fractions were 0.4532 (unlabeled versus development) and 0.3780 (unlabeled versus original test). [Shift evidence](../results/legacy_04/shift/shift_summary.json) · [Reproduction code](../extension/shift_audit_v1.py)

## Which sizes are covered?

Exactly 34 native sizes occurred among original-test normals and no development normals. These sizes account for 78,158/110,701 (70.60%) test normals. After direct lot or image exclusions, the remaining unlabeled pool covered the sizes of 78,154 of these test normals.

Using only the 31,457 training wafers as the reference for normal sizes gives a different cohort. The earlier 11,591-wafer development audit contains 3,181 wafers whose sizes lacked a training normal. For seeds 42, 123 and 2026, RF71 assigned 617/601/605 of 728 slots to that cohort, including 82/75/76 normals. RF77 assigned 539/540/538 slots, including 19/18/20 normals.

At precisely the 34 target sizes, the development audit contains 209 true defects, 206 of them alerted. Under fivefold Scratch loss and full-capacity benefit ranking, RF71 reviewed 64/59/63 of those wafers and RF77 reviewed 67/80/74. Every model and seed reviewed two true scratches and zero normals at these sizes. The zero follows from the target definition, which excludes all development normals at these sizes. [Original-test queue replay](REPORT.md)

## Isolated unlabeled pool and pilot

Exclude all 8,848 components connected to labeled wafers by lot or exact map. This leaves 484,982 eligible unlabeled records and, after within-pool deduplication, 434,498 distinct candidates across 25,762 components. Multi-hop exclusions removed 31,247 more distinct images than direct overlap exclusions alone. Remaining sizes cover 2,455 of the 3,181 audit wafers lacking training-normal size support (77.18%).

The pilot used 50,000 prespecified isolated unlabeled maps for one masked-density epoch with 391 updates. A seed-42 control used random initialization, the same architecture, labeled rows and seven supervised epochs. On development selection, pretraining found six more scratches and raised normal false alarms by 35. On the historical audit, Scratch hits rose from 38/80 to 45/80, total normal false alarms from 264/7,991 to 328/7,991, and normal-to-Scratch errors from 205 to 208. The pilot failed the joint criteria. [Pool summary](../results/shift/pool_summary.json) · [Paired pilot](../results/ssl_paired/pilot_summary.json) · [Pilot code](../extension/ssl_pilot.py)
