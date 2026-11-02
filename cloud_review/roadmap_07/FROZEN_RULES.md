# Frozen rules for the historical inspection replay

These rules were recorded before examining stage 07 ranking, cost or size-occupancy results. The original test labels had already informed error analysis. All weights, temperatures, gates, seeds and cost scenarios were fixed for the replay.

## Populations and denominators

Define T as ordered native (height, width) pairs observed among original-test normals but absent among all 36,730 normals in the 54,355-wafer labeled development set. Of 110,701 original-test normals, 78,158 (70.60%) fall in T. Define T before filtering and use it for retrospective stratification in all three populations.

| Population | Inclusion |
|---|---|
| Original test | All 118,595 labeled test wafers in 4,953 lots. |
| Exclude direct development duplicates | Drop 3,158 test wafers sharing exact native shape and pixel hash with a development map. |
| Exclude development-linked components | Across all 811,457 wafers, connect wafers by original lot or identical map hash and remove every component touching a development lot. |

Rebuild alert queues independently for each nested population. Within-test repeats remain recorded arrivals. Component exclusion controls observable multi-hop overlap through batches and maps, including unlabeled maps. Equipment, time and process relationships are unobserved.

## Models, review and cost

Use the frozen RF71, RF77 (six additional geometry features) and seven-epoch naturally sampled CNN for seeds 42, 123 and 2026. Reuse the nine development-tuned stage 04 temperatures. Floor each nine-class raw probability at 1e-12 before the temperature transform.

RF71 seed 42 supplies a common upstream gate at calibrated any-defect probability ≥0.5 and inverse temperature 1.9147552052282766. Filter the population, then order alerted candidates by the frozen stage 04 lot and wafer hashes (seed 20261101). Each 1,000 **alerts** permit up to 200 reviews. The final queue of n alerts permits floor(n/5). Ordering simulates arrivals because WM-811K lacks timestamps.

Compare no reviews, fixed random selection, any-defect ranking, full-capacity net-benefit ranking, and reviewing positive net benefit up to capacity. The missed loss L is zero for normal and one for an ordinary defect. Scenarios give equal defect losses, fivefold Scratch loss, or fivefold Near-full loss. Each review costs 0.1 and is assumed to prevent 90% of the loss. Predicted net benefit is `0.9 × Σ p(c)L(c) − 0.1`. Use observed labels for settlement. An unreviewed wafer costs L and a reviewed wafer 0.1 + 0.1L. All units are hypothetical.

Report total target wafers, target normals and target scratches across policies and seeds. Divide reviewed counts by available capacity for slot occupancy and by actual reviews for review composition. Arrival and candidate shares use the corresponding population totals. Leave undefined rates blank.

The primary comparison uses total cost per 1,000 **arrivals**, including upstream misses. Also report per-1,000-alert cost. Define model savings as RF71 cost minus RF77 cost at the same seed and scenario. Report entries and displacements by true class. The 500 paired lot bootstrap replicates condition on the fixed model, candidate queues and decisions. All seeds replay the same arrivals.

## Frozen inputs

Historical reference checkout: `92d2002c2b76e2db2fec492bda72ea66605d5e35`.

| Input | SHA-256 |
|---|---|
| Private stage 07 input archive | `54a2a6269136af87939eb553e2805f18f8443bb2a91d9814066bf9a5dec4b421` |
| Original Kaggle v1 pickle | `1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65` |
| Protected test manifest / development reference | `e47a81e1d2d162961dea3f40819fa9edb9fea38b98b8a9e2eabfd8673c8c650e` / `4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879` |
| Full lot/component mapping | `ae0587ed311b19d702d5e61f200aaeae4e8e49acd47cc557cbe15f78d71600d9` |
| Stage 04 cost protocol / decision source | `9e8411c52e0e11357547353a21621d956c049439daaafbf4a1b6daf3896753af` / `7a89039475d80a9417407cd4d1c63ddeebda85ea9217518439448185c962164c` |
| Frozen tuning temperatures / original probability manifest | `f95fde21e70b8bf0fc582c8953bedf8825994c45c47032a7debd3a90a25b4a54` / `c86c2c3d77762d6d8083545bbb3377b09c13eb09d7119c17426b933602d0a0a7` |

| Seed | RF71 weights | RF77 weights | CNN weights |
|---:|---|---|---|
| 42 | `27cb744cf7a5b798d4b55df77d056a93b2034cd874f19a4939fb3df09d36687f` | `09edefdb9ebbbc847a2817d10e826b1d999251b8d388a2010b472ac67167ffa2` | `3c9fbb2c2b58d20d3986b80186a692474137ef40694f98aceaac58ef5277179e` |
| 123 | `0fe40e3569964a8393794faa68b87170d7c0e1fa7b3f9ce4bce7e6e3478121dc` | `a13e0110999f488ea084a5a5d5e640049fb3818532f4f02a042830d185b082f6` | `6a81fa4f6b101c005329e7f54234ae03ec4a5c75acc3569fd66ff51487ac359d` |
| 2026 | `d7c26fdb07ef4038329f49f6339819871cdbe663ed26a6a4ead5c104a62fa605` | `b1f0100ff502aece24d516981d0ed41dc31176ec3395005b049170f376d0db95` | `7c419dd523715339f822b5c46349b0b6822550089f67c0a316a9c9904525b3dc` |
