# Portable decision recomputation

Run `python scripts/recompute_decisions.py` from the repository root. The script also runs from any other directory when addressed by its path. It requires only Python 3.10 or later and the standard library; no raw dataset, model checkpoint, GPU or scientific package is needed.

The 18 compressed CSV files contain original dataset row identifiers, true labels and the fixed nine probability columns for three model variants, three seeds and two development roles. Every score was exported with 17 significant digits and checked for exact float64 roundtrip. The full development split reference preserves lot, image-hash and connected-component separation. `manifest.json` records file sizes and SHA-256 hashes. `decision_reference.json` contains the frozen tune choices and published results, without local filesystem paths.

The verifier checks data identity and row alignment, rederives the nine tune threshold optima under the declared precision rule, and recomputes AP, all 36 nine-class decisions and normal-error transitions. It does not choose any threshold on audit, train a model or modify the frozen reference. `verification_receipt.json` records the local isolated-Python run; `cleanroom_verification.json` records a second run using only this bundle and the script in a fresh temporary directory.

These are previously examined development predictions. Precision ≥ 80% is a hypothetical point-estimate constraint. Recomputability does not turn the reused audit into independent confirmation and does not establish deployment readiness. The official test is not included or rescored.
