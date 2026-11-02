# Reproduce the historical inspection replay

The original test labels had been inspected before this retrospective replay. All nine classifiers, temperatures, the gate, capacity, tie breaks and loss scenarios were frozen. [Rules and hashes](FROZEN_RULES.md) · [Cost receipt](../results/shift/cost_receipt.json) · [Historical reference checkout](https://github.com/Matthewsyr/reliable-wafer-classification/tree/92d2002c2b76e2db2fec492bda72ea66605d5e35)

Use a hosted Colab CPU with at least 12 GB RAM, NumPy, pandas, PyTorch, joblib and scikit-learn 1.6.1. The original dataset is `LSWMD.pkl` from Kaggle `qingyi/wm811k-wafer-map/versions/1`. An exact model replay also needs the original private `WM811K_07_inputs.zip` with these members:

| Member | Source |
|---|---|
| `inputs/rf_{original71,linearity77}_s{42,123,2026}.joblib` | Six geometry-ablation forest checkpoints |
| `inputs/cnn_s{42,123,2026}.pt` | Historical CNN checkpoints, first two seeds from the repair screen's `bn_natural` and seed 2026 from its second-round baseline |
| `inputs/protected_final_test_manifest.csv.gz` | Stage 3 test manifest |
| `inputs/tune_calibration_freeze.json`, `previous/cost/{protocol,tune_calibration_freeze}.json` | Stage 04 temperature and cost definitions |
| `inputs/lot_components.csv.gz` | Full lot/component registry from the unlabeled-pool audit |

Copy the archive and the development-only feature caches to `/content/`. The SHA-256 digests of `development_features.npz` and `linearity_features.npy` are `61c98d4dc415687e20386cd180b48a3ab3f87099e54b31defbae6606329d745b` and `93039e3cbeb64fb3ebacc730e150edf4e3ee980fefd7fb09857fee261c6c8865` respectively.

Run the following notebook cell on hosted Colab. Keep the stage 07 scripts separate from the pinned historical checkout supplying the model dependencies.

```bash
%%bash
set -eu
cd /content
git clone -q https://github.com/Matthewsyr/reliable-wafer-classification.git project
CODE=/content/project/cloud_review/roadmap_07
REF=/content/wm07_ref
git -C project worktree add --detach "$REF" 92d2002c2b76e2db2fec492bda72ea66605d5e35
python -m pip -q install scikit-learn==1.6.1
echo '54a2a6269136af87939eb553e2805f18f8443bb2a91d9814066bf9a5dec4b421  WM811K_07_inputs.zip' | sha256sum -c -
unzip -q WM811K_07_inputs.zip -d private07
python - <<'PY'
import kagglehub
from pathlib import Path
p = Path(kagglehub.dataset_download('qingyi/wm811k-wafer-map/versions/1',path='LSWMD.pkl'))
Path('/content/wm07_raw_path').write_text(str(p if p.is_file() else p/'LSWMD.pkl'))
PY
RAW=$(cat /content/wm07_raw_path)
python -u "$CODE/diagnose_parity.py" --repo "$REF" --raw "$RAW" --inputs /content/private07/inputs \
  --geometry /content/linearity_features.npy --base /content/development_features.npz \
  --windows-lut "$CODE/windows_strip_trig.json" --extra 1024 --out /content/parity_report.json
python - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('/content/parity_report.json').read_bytes())
assert p['n']==1090 and p['base']=={} and p['windows_angles']=={}
PY
python -u "$CODE/score_portable.py" --repo "$REF" --raw "$RAW" --inputs /content/private07/inputs \
  --windows-lut "$CODE/windows_strip_trig.json" --parity-report /content/parity_report.json \
  --out /content/wm07_scores
python -u "$CODE/replay_target.py" --decision "$CODE/decision.py" --repo "$REF" \
  --input /content/wm07_scores --cost /content/private07/previous/cost \
  --lot-components /content/private07/inputs/lot_components.csv.gz --out /content/wm07_replay
```

A pre-scoring parity check covers 1,090 development wafers. A difference in the last bit of certain trigonometric constants between the training environment and cloud NumPy changed strip bins and tree paths. The first run stopped before test scoring. Restoring the original 36 sine/cosine pairs reproduced all 71 base and six geometry features on the development parity set. Across three RF77 seeds, the maximum probability difference was 2.22e-16 under the original tolerance. [Constants](windows_strip_trig.json) · [Portability fix](strip_portability.py) · [Scoring entry point](score_portable.py)

The scoring and replay commands require fresh output directories. The first verifies data, checkpoints and identities, checks all nine models on archived development predictions, and emits private metadata, gate/candidate scores and a receipt. The second settles lists against actual labels and writes a run receipt. Preserve private artifacts in persistent storage and release the runtime when finished.

## Check published results

To inspect the public evidence without original checkpoints, run the following from the repository root:

```bash
CODE=cloud_review/roadmap_07
python "$CODE/check_replay_aggregates.py" --results "$CODE/results"
python "$CODE/render_closeout.py" --run "$CODE/results" --support "$CODE" --repo . --out /tmp/wafer-replay-report.md
```

From the stage 07 directory, model-independent fixtures are available via `python score_replay.py --self-test` and `python replay_target.py --decision decision.py --self-test`. Separately trained weights represent a new run. To rebuild the stage 04 development summary from the private aggregates and verify the original replay on Colab:

```bash
python "$CODE/build_relative_readout.py" --cost /content/private07/previous/cost --out /content/dev04_summary
python "$CODE/check_replay_aggregates.py" --results /content/wm07_replay
python "$CODE/render_closeout.py" --run /content/wm07_replay --support "$CODE" --repo "$REF" --out /content/REPORT.md
```
