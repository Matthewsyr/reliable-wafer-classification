# Scratch training on the development set

Each row counts distinct labeled wafers. Scratch recall uses all true Scratch wafers in the partition. Normal false alarms include all eight defect classes. Each seed was scored on the same wafers.

| Partition | Method | Seed | Scratch hits / actual | Scratch recall | Scratch precision | Normal false alarms / normals | Normal called Scratch | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Selection | A. Baseline CNN | 42 | 27/60 | 45.0% | 20.3% | 88/3999 (2.20%) | 85 | 85.24% |
| Selection | A. Baseline CNN | 123 | 11/60 | 18.3% | 28.9% | 22/3999 (0.55%) | 18 | 83.85% |
| Selection | A. Baseline CNN | 2026 | 29/60 | 48.3% | 21.8% | 168/3999 (4.20%) | 90 | 83.89% |
| Selection | B. Repeated real Scratch | 42 | 36/60 | 60.0% | 20.0% | 134/3999 (3.35%) | 120 | 84.60% |
| Selection | B. Repeated real Scratch | 123 | 8/60 | 13.3% | 57.1% | 38/3999 (0.95%) | 4 | 83.23% |
| Selection | B. Repeated real Scratch | 2026 | 31/60 | 51.7% | 15.2% | 454/3999 (11.35%) | 160 | 79.94% |
| Selection | C. Synthetic Scratch | 42 | 37/60 | 61.7% | 25.5% | 133/3999 (3.33%) | 91 | 86.25% |
| Selection | C. Synthetic Scratch | 123 | 19/60 | 31.7% | 70.4% | 18/3999 (0.45%) | 2 | 85.76% |
| Selection | C. Synthetic Scratch | 2026 | 17/60 | 28.3% | 68.0% | 142/3999 (3.55%) | 6 | 84.12% |
| Historical audit | A. Baseline CNN | 42 | 38/80 | 47.5% | 12.5% | 264/7991 (3.30%) | 205 | 83.41% |
| Historical audit | A. Baseline CNN | 123 | 22/80 | 27.5% | 29.3% | 96/7991 (1.20%) | 27 | 84.39% |
| Historical audit | A. Baseline CNN | 2026 | 45/80 | 56.2% | 14.8% | 366/7991 (4.58%) | 224 | 82.47% |
| Historical audit | B. Repeated real Scratch | 42 | 45/80 | 56.2% | 13.1% | 292/7991 (3.65%) | 223 | 81.48% |
| Historical audit | B. Repeated real Scratch | 123 | 13/80 | 16.2% | 38.2% | 171/7991 (2.14%) | 4 | 81.84% |
| Historical audit | B. Repeated real Scratch | 2026 | 48/80 | 60.0% | 14.5% | 869/7991 (10.87%) | 259 | 79.25% |
| Historical audit | C. Synthetic Scratch | 42 | 48/80 | 60.0% | 14.4% | 328/7991 (4.10%) | 232 | 84.08% |
| Historical audit | C. Synthetic Scratch | 123 | 25/80 | 31.2% | 41.0% | 115/7991 (1.44%) | 15 | 83.36% |
| Historical audit | C. Synthetic Scratch | 2026 | 28/80 | 35.0% | 43.1% | 345/7991 (4.32%) | 29 | 83.04% |

## Paired comparisons

| Comparison | Selection | Historical audit |
|---|---|---|
| B. Repeated real Scratch vs. A. Baseline CNN | At least one seed failed | At least one seed failed |
| C. Synthetic Scratch vs. A. Baseline CNN | At least one seed failed | At least one seed failed |
| C. Synthetic Scratch vs. B. Repeated real Scratch | At least one seed failed | At least one seed failed |

The comparisons pair seeds 42, 123 and 2026. Each seed must retain its true Scratch hits, with no increase in total normal false alarms or normals called Scratch. A cost replay also requires at least one additional true Scratch hit or one fewer total normal false alarm in at least one seed. Changing only the defect category assigned to a normal wafer earns no gain.

C versus B compares the full synthesis recipe at the same number of replacement training slots. Geometry and background constraints can change the donor wafer selected for C.

These audit lots had been inspected during earlier development. This comparison is exploratory. The original frozen test remained outside the study.
Neither B nor C qualified for the cost replay. The existing review policy was retained.
