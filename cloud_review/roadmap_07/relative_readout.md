# Inspection ranking in a fixed development alert pool

This retrospective simulation uses the original labels of the development audit, which had been inspected during research. Lots and exact-image components were separated across training, tuning and audit.

After calibration on tuning batches, RF71 seed 42 screened 3,642 of 11,591 arrivals at an any-defect probability of at least 0.5. The 903 contributing lots formed three full 1,000-alert queues and a final queue of 642. Their capacities were 200, 200, 200 and 128, for 728 total inspections. Every model used the same alerts, order and capacity.

An unreviewed defect incurs a loss of 1, or 5 for Scratch or Near-full in the corresponding scenario. Normal loss is 0. Each review costs 0.1 and is assumed to prevent 90% of the defect loss. Reviewed wafers cost 0.1 + 0.1L. Expected net benefit is 0.9 times probability-weighted loss minus 0.1. Outcomes use observed labels. Costs are hypothetical relative units.

## Simulated cost per 1,000 alerted candidates

Cells list seeds 42, 123 and 2026 in order. Identical rounded results are shown once. Lower cost is preferable. The random reference is the same frozen draw.

| Scenario | Ranker | Random, fill capacity | Any-defect probability, fill | Net benefit, fill | Positive net benefit, capped |
|---|---|---:|---:|---:|---:|
| Equal losses | RF71 | 803.624 | 796.705 | 796.705 | 796.705 |
| Equal losses | RF77 | 803.624 | 796.705 | 796.705 | 796.705 |
| Equal losses | CNN | 803.624 | 796.705 | 796.705 | 796.705 |
| Scratch 5x | RF71 | 856.892 | 864.800 | 833.169 / 830.203 / 831.439 | 833.169 / 830.203 / 831.439 |
| Scratch 5x | RF77 | 856.892 | 864.800 | 814.388 / 813.893 / 815.376 | 814.388 / 813.893 / 815.376 |
| Scratch 5x | CNN | 856.892 | 864.800 | 825.261 / 831.439 / 825.508 | 825.261 / 831.439 / 825.508 |
| Near-full 5x | RF71 | 808.347 | 801.428 | 797.474 | 797.474 |
| Near-full 5x | RF77 | 808.347 | 800.439 / 801.428 / 800.439 | 797.474 | 797.474 |
| Near-full 5x | CNN | 808.347 | 797.474 / 797.474 / 800.439 | 797.474 | 797.474 |

All queues contained enough positive-benefit candidates to fill all 728 slots. The [full matrix](relative_policy_matrix.csv) gives 108 combinations with number reviewed, normals reviewed, unreviewed defects, loss outside the alert pool and whole-stream cost per 1,000 arrivals.

## Which wafers changed places with RF77?

The table holds full-capacity benefit ranking and fivefold Scratch loss fixed. RF71 and RF77 share an upstream miss cost. Including these misses changes the denominator from 3,642 alerts to 11,591 arrivals. Intervals resample paired lots conditional on the models, queues and selected lists.

| Seed | Entered / displaced | Scratch entered / displaced | Normal entered / displaced | Savings per 1,000 alerts [95% interval] | Savings per 1,000 arrivals |
|---:|---:|---:|---:|---:|---:|
| 42 | 339 / 339 | 3 / 0 | 3 / 67 | 18.781 [1.225, 39.414] | 5.901 |
| 123 | 340 / 340 | 3 / 1 | 2 / 60 | 16.310 [1.255, 32.729] | 5.125 |
| 2026 | 322 / 322 | 3 / 1 | 2 / 59 | 16.063 [0.111, 31.598] | 5.047 |

Under equal losses or fivefold Near-full loss the realized RF77 versus RF71 cost differences were zero to numerical precision. Each seed uses the same cohort. [Per-seed entries, displacements and uncertainty](relative_geometry_swaps.csv)
