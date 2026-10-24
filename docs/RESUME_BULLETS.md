# Resume bullets — ML / Applied AI Engineering, Summer 2027

Use these only after personally reviewing the repository and being able to explain the underlying decisions. AI assistance is disclosed in AI_ASSISTANCE.md; automated work does not imply personal manual execution.

- Built a reproducible WM-811K classification pipeline with lot-aware validation, data-integrity checks, and RF/CNN baselines; documented a Macro-F1 drop from 85.28%/84.73% in validation to 44.87%/45.16% on the frozen original test.
- Evaluated a fixed geometric-feature ablation across three seeds, improving mean Scratch recall from 32.08% to 42.08% while reducing normal-wafer total false alarms from 12.35 to 7.17 per 1,000 on an explicitly exploratory development audit.
- Engineered threshold freezing, full nine-class error accounting, and reproducible metric replay; independently verified 18 prediction records and 36 decision outputs with 14,317 checks, and rejected a precision-constrained rule that worsened recall and normal false alarms.
