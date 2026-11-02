# Which wafers received the inspection slots?

On the original test, 78,158 of 110,701 normal wafers (70.60%) had native dimensions absent from development normals. Call these dimensions **T**. With fivefold Scratch loss and RF77 seed 42, the positive-net-benefit policy allocated **8,631 of 9,845 available inspections (87.7%)** to normal wafers in T. It used 9,845 slots in total.

### Allocation to target sizes and realized cost

Cells list seeds 42, 123 and 2026 in order. The policy reviews positive-net-benefit alerts in descending order. Scratch counts include all dimensions. Costs use observed labels and hypothetical relative units.

| Population | Ranker | Reviewed normals in T / total capacity | True scratches reviewed | Cost per 1,000 arrivals | Cost per 1,000 alerts |
|---|---|---|---|---|---|
| Full original test | RF71 | 8,928 / 8,797 / 8,774 (each out of 9,845) | 58 / 59 / 72 | 94.800 / 94.618 / 94.238 | 165.833 / 165.394 / 164.480 |
| Full original test | RF77 | 8,631 / 7,969 / 7,412 (each out of 9,845) | 160 / 177 / 203 | 88.190 / 82.901 / 77.968 | 149.910 / 137.167 / 125.284 |
| Full original test | CNN | 8,419 / 5,490 / 8,863 (each out of 9,845) | 170 / 226 / 177 | 87.575 / 60.422 / 87.720 | 148.429 / 83.016 / 148.776 |
| Exclude development-linked components | RF71 | 7,506 / 7,486 / 7,474 (each out of 8,877) | 64 / 71 / 83 | 94.495 / 94.316 / 93.882 | 165.932 / 165.506 / 164.472 |
| Exclude development-linked components | RF77 | 6,698 / 5,802 / 5,503 (each out of 8,877) | 168 / 168 / 188 | 80.551 / 73.038 / 69.647 | 132.739 / 114.854 / 106.784 |
| Exclude development-linked components | CNN | 7,275 / 4,783 / 7,739 (each out of 8,877) | 153 / 192 / 165 | 87.740 / 60.456 / 87.306 | 149.852 / 84.905 / 148.818 |

## Shared alert pool and capacity

RF71 seed 42 supplied the common upstream screen with a development-calibrated probability of any defect of at least 0.5. Frozen lot and wafer hashes determine the simulated order. Each queue of 1,000 alerts permits up to 200 inspections. A partial queue of n alerts permits floor(n/5). Queues are rebuilt separately for each population. These historical labels were previously examined.

| Population | Labeled arrivals | Lots | Alerts | Available slots | Upstream missed defects | Normal arrivals in T |
|---|---:|---:|---:|---:|---:|---:|
| Full original test | 118,595 | 4,953 | 49,229 | 9,845 | 1,891 | 78,158 |
| Exclude exact development duplicates | 115,437 | 4,953 | 49,150 | 9,830 | 1,867 | 78,158 |
| Exclude development-linked components | 105,655 | 4,315 | 44,385 | 8,877 | 1,579 | 72,160 |

An unreviewed wafer incurs its class loss L. A reviewed wafer incurs 0.1 + 0.1L. Base defect loss is 1 and normal loss is 0. Scenarios assign equal losses, fivefold Scratch loss or fivefold Near-full loss. Each assumes a review fee of 0.1 and a 90% avoidable loss fraction. Whole-stream cost includes upstream misses.

### Compare rankings at the same capacity

RF77 is fixed below and a missed Scratch costs five times an ordinary missed defect. The random policy uses one frozen sample. Defect probability is one minus the normal probability. Net benefit weights probabilities by class loss and subtracts the review fee.

| Population | Ranking | Slots used | True scratches reviewed | Cost per 1,000 arrivals |
|---|---|---|---|---|
| Full original test | Fixed random sample | 9,845 / 9,845 / 9,845 | 91 / 91 / 91 | 86.490 / 86.490 / 86.490 |
| Full original test | Any-defect probability | 9,845 / 9,845 / 9,845 | 151 / 146 / 163 | 63.025 / 63.306 / 61.455 |
| Full original test | Positive net benefit | 9,845 / 9,845 / 9,845 | 160 / 177 / 203 | 88.190 / 82.901 / 77.968 |
| Exclude development-linked components | Fixed random sample | 8,877 / 8,877 / 8,877 | 84 / 84 / 84 | 87.042 / 87.042 / 87.042 |
| Exclude development-linked components | Any-defect probability | 8,877 / 8,877 / 8,877 | 123 / 118 / 133 | 63.898 / 64.128 / 62.254 |
| Exclude development-linked components | Positive net benefit | 8,877 / 8,877 / 8,877 | 168 / 168 / 188 | 80.551 / 73.038 / 69.647 |

**Ranking with RF77 held fixed.** Any-defect ranking had lower realized cost in both populations for every seed. Positive-net-benefit ranking increased cost per 1,000 arrivals by 25.165 / 19.594 / 16.513 units in the full population and 16.653 / 8.910 / 7.394 after removing development-linked components.

Both lists reviewed 9,845 wafers in the full population at seed 42. Switching to expected net benefit increased scratches reviewed from 151 to 160 (+9), increased normal reviews from 5,809 to 9,161 (+3,352) and reduced reviews of other true defects from 3,885 to 524 (-3,361). Review fees were equal. The net substitution increased simulated loss by 2,984.4 units under the frozen weights.

## Effect of the six RF77 geometry features

Hold full-capacity net-benefit ranking fixed and change the model from RF71 to RF77. Positive savings per 1,000 arrivals favor RF77. Direction is called consistent only if all three point estimates are positive in both populations.

| Scenario | Full population savings (42 / 123 / 2026) | Component-isolated savings (42 / 123 / 2026) | Positive for both populations and all seeds |
|---|---|---|---|
| Equal losses | 7.247 / 7.445 / 8.067 | 7.539 / 8.007 / 8.672 | Yes |
| Scratch 5x | 6.610 / 11.717 / 16.271 | 13.944 / 21.279 / 24.235 | Yes |
| Near-full 5x | 7.247 / 7.437 / 8.067 | 7.539 / 8.007 / 8.672 | Yes |

Entries and displacements below compare RF77 with RF71 under fivefold Scratch loss. Intervals for savings per 1,000 alerted candidates resample paired lots conditional on the models, queues and selected lists.

| Population | Seed | Entered / displaced | Net scratches reviewed | Net normals reviewed | Savings per 1,000 alerts [95% interval] |
|---|---:|---:|---:|---:|---:|
| Full original test | 42 | 3,859 / 3,859 | +102 | -463 | 15.924 [13.814, 18.152] |
| Full original test | 123 | 4,150 / 4,150 | +118 | -1072 | 28.227 [25.708, 31.001] |
| Full original test | 2026 | 4,677 / 4,677 | +131 | -1620 | 39.196 [36.103, 42.644] |
| Exclude development-linked components | 42 | 4,550 / 4,550 | +104 | -1221 | 33.194 [29.921, 36.172] |
| Exclude development-linked components | 123 | 5,000 / 5,000 | +97 | -2110 | 50.652 [46.621, 54.829] |
| Exclude development-linked components | 2026 | 5,361 / 5,361 | +105 | -2425 | 57.688 [53.314, 61.943] |

## Development audit and the unlabeled-map pilot

On the earlier development audit, switching to RF77 under fivefold Scratch loss saved 5.901 / 5.125 / 5.047 units per 1,000 arrivals. All seeds used the same 11,591 development audit wafers. Population differences limit comparisons with original test costs.

The 638,507 unlabeled wafers supplied information about dimensions, lot overlap and duplicate maps. Removing all components linked to labeled lots or maps yielded 434,498 unique candidate maps. A paired single-seed masked-pretraining pilot used 50,000 isolated unlabeled maps and one extra epoch before the same supervised training. Audit Scratch hits rose from 38 to 45, while total normal false alarms rose from 264 to 328. The joint criteria failed. [Unlabeled coverage and failed-die distributions](shift_and_ssl.md)

A prospective evaluation still needs independent lots, actual alert and arrival timing, realized review effectiveness and manufacturing losses.
