# Which wafers displaced other reviews?

At seed 42, the cost-weighted list added a net 3,054 normal wafers at native dimensions with zero normal training examples. This accounts for 91.1% of its net increase in normal reviews. At these dimensions, 7,730 wafers selected exclusively by the cost-weighted policy had a mean predicted Scratch probability of 24.10%, or about 1,863 expected scratches in aggregate. The labels contained 74 scratches (0.96%).

Baseline A ranks by calibrated RF77 probability of any defect. B ranks the same RF77 candidates by predicted positive net benefit, with ordinary defect loss 1, Scratch loss 5, review fee 0.1 and avoidable loss fraction 90%. RF71 alerted on 49,229 of 118,595 previously examined test arrivals. The common capacity of up to 200 per 1,000 alerts yielded 9,845 reviews. At seed 42, 1,627 were common to both lists and 8,218 were exclusive to each. The exchange comparison uses only the two exclusive groups.

## Results by normal training-size coverage

Coverage is the count of exact ordered native height/width matches among the 20,780 normals in 31,457 supervised training records. Each row's means and fractions use its exclusive wafers as denominator.

| Matching training normals | Exclusive list | Wafers | True normals | True scratches | Mean predicted Scratch | Observed Scratch |
|---|---|---:|---:|---:|---:|---:|
| 0 | A only | 7,680 | 4,455 | 65 | 0.89% | 0.85% |
| 0 | B only | 7,730 | 7,509 | 74 | 24.10% | 0.96% |
| 1–24 | A only | 285 | 150 | 8 | 0.81% | 2.81% |
| 1–24 | B only | 66 | 54 | 4 | 13.36% | 6.06% |
| ≥25 | A only | 253 | 17 | 1 | 0.33% | 0.40% |
| ≥25 | B only | 422 | 411 | 5 | 21.61% | 1.18% |

The net changes in normal reviews by size band are +3,054, −96 and +394, totaling +3,352. Several exclusive cells contain few scratches. This training-only coverage definition differs from the earlier 34 target shapes defined across **all development normals**.

## Which defects lost inspections?

B displaced 3,522 other defects and added 161, a net reduction of 3,361 other defects reviewed. Edge-Loc, Edge-Ring and Loc accounted for 81.5% of that reduction.

| True class | A only, displaced | B only, entered | Net B − A |
|---|---:|---:|---:|
| Edge-Loc | 1,351 | 40 | −1,311 |
| Edge-Ring | 866 | 1 | −865 |
| Loc | 652 | 88 | −564 |
| Random | 208 | 3 | −205 |
| Center | 226 | 29 | −197 |
| Donut | 132 | 0 | −132 |
| Near-full | 87 | 0 | −87 |
| Scratch | 74 | 83 | +9 |
| Normal | 4,622 | 7,974 | +3,352 |

## Whole-stream cost

| Seed 42 historical ledger | A | B |
|---|---:|---:|
| Loss outside the upstream alert pool | 3,079 | 3,079 |
| Loss on unreviewed alerts | 2,947 | 6,263 |
| Residual loss on reviewed defects | 464.0 | 132.4 |
| Review fees | 984.5 | 984.5 |
| Simulated total cost | 7,474.5 | 10,458.9 |
| **Cost per 1,000 arrivals** | **63.025** | **88.190** |
| Unreviewed scratches, inside and outside the gate | 542 | 533 |
| Unreviewed other defects, inside and outside the gate | 3,316 | 6,677 |

The shared upstream gate missed 297 scratches and 1,594 other defects. At seed 42 the model predicted that B would save 37.948 units per 1,000 arrivals, but label-based settlement increased cost by 25.165 per 1,000. The total increment is 0.9 × (3,361 − 5 × 9) = 2,984.4 relative units.

| Seed | Net normal / Scratch / other defects reviewed | Predicted B − A cost per 1,000 arrivals | Realized B − A per 1,000 | Break-even Scratch loss multiplier on fixed lists |
|---|---|---:|---:|---:|
| 42 | +3,352 / +9 / −3,361 | −37.948 | +25.165 | 373.44 |
| 123 | +2,706 / +31 / −2,737 | −29.883 | +19.594 | 88.29 |
| 2026 | +2,336 / +40 / −2,376 | −26.704 | +16.513 | 59.40 |

The three seeds share the same arrivals. At zero training-normal support, the other seeds' mean B-only Scratch probabilities were 22.02% and 22.14%, compared with actual frequencies of 1.13% and 1.31%. This discrepancy concerns the selected competitors for scarce slots. It cannot identify calibration across the full stream or isolate the effect of dimensions. Break-even multipliers reprice the same two lists. Changing the ranking itself requires reselecting and settling the resulting lists. Factory costs and intervention success remain unknown.

## Verification and evidence

Snowflake recomputed 76, 71 and 72 groups for the three seeds. All 219 groups and 24 fields (5,256 cells) agreed with the hosted Python reference. Counts matched exactly, and the largest absolute numeric difference was below 5e-10. Each seed's 50 queues respected capacity. The two implementations verify the same records, labels and frozen decisions.

[Snowflake result](results/snowflake_exchange.csv) · [Verification receipt](results/snowflake_verification.json) · [SQL reproduction](SQL_REPRODUCE.md) · [Size and class cells](results/exchange_sql_cells.csv) · [Size calibration](results/exchange_size_calibration.csv) · [Reliability bins](results/reliability_cells.csv) · [Whole-stream ledger](results/arm_cost_components.csv) · [Fixed-list sensitivities](results/exchange_seed_summary.csv) · [Prospective study](../blind_review_08/PROTOCOL.md)
