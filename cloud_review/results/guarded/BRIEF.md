# CNN selection with joint Scratch and false-alarm constraints

Each of the three seeds must preserve the matched CE Scratch hit count and increase neither total false alarms on normal wafers nor normal-to-Scratch errors. Among eligible arms, selection uses mean nine-class Macro-F1 across seeds.

Eligible arms: cnn_ce. The selected arm was cnn_ce (84.33% Macro-F1). The unrestricted comparison selected cnn_ce.

CE is the retained control. Its paired differences against itself are zero.

| Audit seed | Scratch hit change | Normal false-alarm change | Normal-to-Scratch change | All conditions held |
|---|---:|---:|---:|---|
| 42 | +0 | +0 | +0 | Yes |
| 123 | +0 | +0 | +0 | Yes |
| 2026 | +0 | +0 | +0 | Yes |

All three seeds saw the same 80 scratches and 7,991 normal wafers. See paired_guard_deltas.csv for every arm. Independent validation requires new lots.
