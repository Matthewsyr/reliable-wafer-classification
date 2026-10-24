# Technical Interview Preparation

This is a guide to explaining the actual repository, including negative results. Read the linked reports and run the tests before presenting the work as your own understanding. AI assistance is disclosed in `docs/AI_ASSISTANCE.md`.

## A defensible 90-second introduction

“I investigated reliability rather than just accuracy on WM-811K. The historical experiment compared random forests with a compact 111,625-parameter CNN, using production-lot-aware development splits and a frozen original test evaluation. Development Macro-F1 was around 85% for both families, but the original-test Macro-F1 fell to about 45%. I preserved that failure instead of replacing it with a favorable validation score. Subsequent development-only work investigated normal-wafer size coverage, retraining instability, explicit line geometry, and the precision–recall cost of Scratch decisions. The repository contains executable pipelines, frozen protocols, real results, tests, and a record of what cannot be established from this dataset.”

The final geometry block raised exploratory audit mean Scratch recall from 32.08% to 42.08% and reduced normal total errors from 12.35 to 7.17 per 1,000. The illustrative 80%-precision rule was rejected because it lost Scratch hits and increased normal total errors relative to that model’s argmax. Explain both findings together and retain the already-observed-audit qualification.

## 1. Walk through the data flow

1. Acquire the original dataset from the cited provider and verify the recorded SHA256. A pickle can execute code: deserialize only a trusted, verified source. The raw dataset is not committed to Git.
2. Preserve source-row identity, lot, original train/test designation, label, shape, and map-content hash. A wafer identity is not the same thing as its image hash.
3. Interpret the discrete grid as outside-wafer, valid passing die, and failing die. Check invalid values and missing metadata explicitly.
4. Keep the original labeled test designation unchanged. Historical development contains 54,355 labeled rows; the historical test contains 118,595. Unlabeled rows do not become normal examples.
5. For exploratory work, connect rows sharing either a lot or an exact-map hash; assign whole connected components to train/tune/audit. The preserved development partition has 31,457/11,307/11,591 rows. The audit was already inspected before the final engineering round.
6. RF consumes the original 71 engineered features, or those same features plus the fixed line-geometry block. CNN consumes separate valid-die and failed-die masks, centered square padding, and area resizing to 64×64. Fractional values after resizing represent coverage, not a new die category.
7. Fit models and any learned preprocessing only on training rows. Select model/threshold decisions on tune. Bind configuration, input manifests, weights, seeds, and selection rules to saved hashes.
8. Evaluate the frozen decision on the reserved partition. Save per-class support, confusion matrices, probabilities, runtime and protocol scope. Historical original-test results and later exploratory results remain separate.

Be able to point to the actual CLI and unit test implementing every step. A diagram is not proof that the code follows it.

## 2. Why RF and a small CNN?

RF is an interpretable, inexpensive baseline for spatial summaries such as radial density, local bins and covariance. It needs no feature standardization for split ordering. A small CNN tests whether learned spatial filters add useful structure without introducing an external pretrained domain or a large architecture search. The historical medium model has 111,625 trainable parameters. Its failure on the original test is evidence against assuming that a more expressive image model automatically solves distribution shift.

This is not an architecture novelty claim. The useful engineering contribution is controlled comparison, data boundaries, decision analysis and reproducibility. A Transformer would require a new justification and more compute; it would not repair a flawed evaluation protocol.

## 3. Why does Scratch precision conflict with recall?

Scratch is rare, while normal maps can contain sparse failed dies. Increasing Scratch class weight makes errors on training Scratch rows more costly, shifting the fitted boundary. Lowering a Scratch decision threshold predicts more positives: this may recover true scratches and also add normal or other-defect false positives. Precision is TP/(TP+FP); recall is TP/(TP+FN). Neither denominator is overall dataset size.

Precision also depends on prevalence. A validation threshold meeting 80% precision does not guarantee 80% precision when defect prevalence, normal-size coverage, score calibration or conditional appearance changes. The 80% floor in this project is an experimental assumption, not a factory-approved cost model.

The one-vs-rest normal false-positive rate counts only true normal wafers predicted Scratch. Nine-class normal total false-positive rate counts true normals predicted as **any** of the eight defects. These are different quantities. A Scratch-priority rule must therefore be evaluated as a complete nine-class decision, not just a binary PR curve.

## 4. What exactly is a geometric-feature ablation?

Hold training rows, model hyperparameters, class weights, seeds and evaluation rows fixed. Change only the feature block. Original features already contain coordinate variances and covariance, so PCA anisotropy is a deterministic re-expression of existing information. Its explicit form may help a finite axis-aligned tree, but it is not new observational information. Fixed narrow-strip support and continuity features test additional spatial organization.

Inspect paired differences across the same validation wafers and seeds. Report no improvement or inconsistent improvement when that is the result. A positive average alone is insufficient when one seed worsens, class support is small, or the added features change other-class errors. No finite set of checks proves absence of all leakage or guarantees future robustness.

## 5. Lot-wise versus random split

A random wafer split estimates performance on a mixture in which training and validation can share manufacturing lots. A lot split tests transfer to other lots from the available population. It does not establish chronological future-time generalization because reliable time/process metadata are unavailable.

An exact-map hash crossing a split can occur naturally, especially for normal grids. It is not automatically a duplicate wafer identity or incorrect copying. Fourteen content-hash groups had conflicting labels. The historical experiment documented them without changing the original test; its predeclared sensitivity analysis excluded test maps identical to development and was reported alongside the primary all-wafer metric. Later component grouping is stricter but evaluates a different development construction, so its scores cannot be substituted for the historical headline.

## 6. How were leakage and chance checked?

- Assert row-identity separation and the required lot/hash/component separation.
- Keep test labels out of training, epoch choice, feature definition, threshold choice and cleaning decisions.
- Reuse fixed manifests; do not search for a favorable split.
- Record input, code, configuration and weight hashes and retain failure logs.
- Use matched seeds and paired predictions; distinguish model-seed variation from different data splits.
- Cluster uncertainty calculations by lot/hash connected component. Three model seeds evaluated on the same 80 Scratch wafers do not create 240 independent cases.
- Independently recompute metrics and selection from saved probabilities. Synthetic unit-test passes establish implementation behavior, not real model quality.

The original test was first scored after a saved freeze, then later inspected for diagnosis. That history cannot be undone. New geometry and threshold analyses are development-only exploration, not a second pristine final test. A new independent dataset would be needed for new confirmatory claims.

## 7. Explain the failed repairs without hiding them

The first repair round reduced some normal→Scratch predictions but lost true Scratch recall and sometimes increased normal→other-defect errors. The second used GroupNorm, D4 augmentation, parameter averaging, Scratch weighting, and fixed logit offsets. Its selected rule passed development screening but failed joint requirements after refitting. On the same 80 Scratch / 7,991 normal audit rows, the three refit comparisons were:

| Seed | Scratch TP, baseline→candidate | Normal total FP, baseline→candidate |
|---|---:|---:|
| 42 | 51→50 | 61→27 |
| 123 | 39→52 | 9→34 |
| 2026 | 67→45 | 1,633→29 |

The candidate's normal false-positive counts were more stable, but it did not preserve both objectives per seed. In seed 123, four old normal→Scratch errors moved to other defects and none of those six errors became normal. This is why total normal errors are an explicit guardrail.

These bundled changes do not isolate a causal contribution of GroupNorm, augmentation or averaging individually. The source-bound reports state that limit.

## 8. What did AI do, and what should I personally verify?

AI assisted with code writing, orchestration, experiment execution, error-image review and report drafting. Automated checks include source/manifest hashes, split assertions, unit tests, metric recomputation and result-source binding. Separate AI reviewers improve error detection but do not count as independent human validation or new experimental data.

Before an interview, personally run the quickstart; trace one wafer through preprocessing; derive one confusion-matrix row; explain a PR threshold by hand; inspect the chosen success/miss/low-confidence Scratch examples; and explain one rejected repair. Do not claim these personal review steps were completed merely because the agent produced this guide.

## 9. What remains unresolved?

No process/tool metadata or confirmed event chronology is available to identify a physical root cause. Conflicting labels cannot be adjudicated from image geometry alone. Exact-size undercoverage and BatchNorm sensitivity are evidence about failure mechanisms, not a decomposition of causal responsibility. Rare-class sample sizes limit certainty. Public-dataset transfer is not production deployment validation. A verified new lot/time/size-family dataset and a real cost policy are needed before an operational threshold can be justified.

## Practice questions

Explain why high accuracy can coexist with unusable defect recall. Explain why PR-AUC and the selected operating point answer different questions. Explain why you would reject a threshold with no predicted positives. Show why a post-test threshold change invalidates a pristine-test claim. Explain how feature extraction handles zero/one failed die. Distinguish deterministic repeatability on one software stack from numerical identity across CPU/GPU and library versions.
