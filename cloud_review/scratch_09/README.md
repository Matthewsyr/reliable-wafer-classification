# Scratch augmentation and RF77 pseudo-labels

We compared a standard cross-entropy CNN (A), repeated sampling of genuine training scratches (B), and synthetic scratches transplanted onto genuine training normals (C). B and C replaced the same 80 normal slots per epoch. All three arms used the same order, initialization, optimizer and number of updates within each seed. All evaluation labels were real.

## Development selection

There were 60 true scratches and 3,999 normal wafers among 5,670 labeled wafers. Percentages are means across seeds. Counts in parentheses are individual results for seeds 42, 123 and 2026, respectively.

| Training arm | Scratch recall (hits) | All normal false alarms (count) | Normals called Scratch (count) |
|---|---:|---:|---:|
| A. Standard CNN | 37.22% (27, 11, 29) | 2.32% (88, 22, 168) | 1.61% (85, 18, 90) |
| B. Repeat real scratches | 41.67% (36, 8, 31) | 5.22% (134, 38, 454) | 2.37% (120, 4, 160) |
| C. Synthetic scratches | 40.56% (37, 19, 17) | 2.44% (133, 18, 142) | 0.83% (91, 2, 6) |

The prespecified rule required every seed to preserve Scratch hits while increasing neither normal false alarms nor normals called Scratch. A gain in Scratch hits or a reduction in all normal false alarms was required to advance to cost simulation. B passed for zero seeds and C for one. With seed 123, C gained eight hits and avoided four false alarms relative to A. With seed 2026, C avoided 26 false alarms but lost 12 hits. The standard CNN remains the training control and RF77 remains the inspection baseline. [Paired comparisons](results/evaluation/paired.csv)

## Historical development audit

The audit covered 11,591 wafers, including 80 true scratches and 7,991 normals. These lots had been examined in previous studies. The original test was excluded.

| Training arm | Scratch recall (hits) | All normal false alarms (count) | Normals called Scratch (count) |
|---|---:|---:|---:|
| A. Standard CNN | 43.75% (38, 22, 45) | 3.03% (264, 96, 366) | 1.90% (205, 27, 224) |
| B. Repeat real scratches | 44.17% (45, 13, 48) | 5.56% (292, 171, 869) | 2.03% (223, 4, 259) |
| C. Synthetic scratches | 42.08% (48, 25, 28) | 3.29% (328, 115, 345) | 1.15% (232, 15, 29) |

In seed 2026, A called 224 normal wafers Scratch. C called 165 of those wafers Loc, correctly classified 32 as normal and kept 27 as Scratch. Two other normal wafers acquired a Scratch prediction. Total normal false alarms fell from 366 to 345, but C lost 17 previous Scratch hits and gained none. [Paired transitions by seed](results/paired_transitions.json)

In seed 42, C raised Scratch recall by 12.50 percentage points (paired 95% interval +3.70 to +21.92) and raised the normal false-alarm rate by 0.80 points (+0.45 to +1.21). Intervals resample whole components linked by lots or identical maps, conditional on the trained seed. [Component-level intervals](results/audit09/paired_component_intervals.csv)

## Reproducibility and teacher qualification

All 486 cells in the six baseline confusion matrices matched the archived cross-entropy result. All 1,680 B/C replacement slots matched. Synthetic maps preserved at least 90% of the proposed trace. Donor and recipient had exactly matching native dimensions in 504 cases, and 12 stored examples passed die-by-die checks. C used the same donor as B in 1,085 slots. The C–B comparison therefore includes geometric donor selection and background matching. [Augmentation checks](results/audit09/synthesis_epoch_qa.csv)

No size band qualified for teacher-generated Scratch labels. The student CNN was skipped. [RF77 precision by size](RF77.md) · [Reproduction guide](REPRODUCE.md) · [Per-seed classification results](results/evaluation/REPORT.md)
