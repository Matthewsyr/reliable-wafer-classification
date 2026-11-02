# Reproduce the development experiments

The [training notebook](WM811K_05_Imbalance_Study.ipynb) compares five CNN training methods. The decision study reuses their scores to measure Scratch recall, false alarms on normal wafers, and simulated review costs under a common inspection budget. The audit lots had already been examined during earlier research.

## 1. Prepare the hosted training run

Run the notebook on a hosted Colab GPU with Python, NumPy, PyTorch, Git and KaggleHub. It fetches `LSWMD.pkl` from `qingyi/wm811k-wafer-map/versions/1` and checks out the historical data and model definitions at commit `aaa667c0421ece13e3a71fa616fb10287b4453ff`. The notebook verifies the source file, reference index and CNN source by SHA-256 before reading the 54,355 development maps. It checks each indexed map against its individual hash. The local float32 raster cache takes about 1.78 GB.

The default training output is `/content/wm811k_training`. To retain epoch checkpoints across sessions, mount your own Drive and create the notebook's optional persistence directory before running the training cells:

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
Path("/content/drive/MyDrive/Colab Notebooks/WM811K_Extension_20261101").mkdir(
    parents=True, exist_ok=True
)
```

## 2. Train the five matched arms

The fixed development roles contain 31,457 training, 11,307 tuning and 11,591 audit wafers. Wafers connected by lot or exact image hash stay in the same role. Seeds 42, 123 and 2026 use the medium CNN, batch size 128, seven epochs, and AdamW with learning rate 0.001 and weight decay 0.0001. Each epoch draws 31,457 training examples and completes 246 optimization steps.

| Arm | Training change relative to the matched control |
|---|---|
| `cnn_ce` | Natural sampling and unweighted cross-entropy |
| `cnn_weighted` | Loss weights proportional to inverse square root of training class counts |
| `cnn_sampled` | Weighted sampling with replacement using the same per-class weights |
| `cnn_rot180` | Rotate both map channels by 180° with probability 0.5 during training |
| `cnn_focal` | Focal loss with gamma 2 and alpha 1 |

The preprocessing pads each centered map to a square and area-resamples valid-die and failed-die coverage to 64 × 64. Rotation preserves both channels and the wafer boundary. The experiment assumes the class survives a 180° change in readout orientation. Equipment-coordinate diagnostics would need a separate test of that assumption. Validation and audit maps retain their native orientation.

Run the notebook cells in sequence. To prepare the cache or resume individual fits, use the notebook's `repo` and `raw` variables:

```python
import subprocess, sys
from pathlib import Path

ext = Path("/content/drive/MyDrive/Colab Notebooks/WM811K_Extension_20261101")
out = ext / "training" if ext.is_dir() else Path("/content/wm811k_training")
cmd = [sys.executable, "-u", "/content/train_imbalance.py",
       "--repo", str(repo), "--raw", str(raw), "--out", str(out),
       "--cache", "/content/wm811k_xcache", "--device", "cuda"]
subprocess.run(cmd + ["--prepare-only"], check=True)
subprocess.run(cmd + ["--arms", "cnn_ce", "--seeds", "42"], check=True)
subprocess.run(cmd, check=True)
```

Each epoch saves the model, optimizer and sampler state. Use one writer per output directory. A changed runtime or protocol requires a fresh directory. A complete run reports 15 fits in `run_status.json` and `complete: true` in `probabilities/manifest.json`. Each fit emits two per-wafer files with the source row, true label and nine probabilities.

## 3. Freeze decisions and build the public evidence

Copy the current scripts in this directory into `/content/wm811k_extension`. The notebook supplies the pinned reference checkout and the training output. The decision stage uses NumPy and Matplotlib. The evidence publisher uses the standard library. Choose new output directories for each run.

```python
extension = Path("/content/wm811k_extension")
decision = out.parent / "decision_full_v1"
public = out.parent / "decision_public_v1"
subprocess.run([sys.executable, str(extension / "decision_study.py"),
                "--archive", str(repo / "results/probabilities"),
                "--new", str(out), "--out", str(decision)], check=True)
subprocess.run([sys.executable, str(extension / "render_evidence.py"),
                "--run", str(decision), "--out", str(public)], check=True)
```

The tuning partition is split by component. One subset calibrates temperatures; the other selects thresholds and ranks arms by three-seed mean Macro-F1. The study writes the selected model and decisions to `frozen_validation_policy.json` before reading audit probabilities. Omitting `--new` replays only the older archived models.

A supplementary selection checks the three paired conditions on each seed: Scratch hits must hold or rise; total normal false alarms and normal-to-Scratch errors must hold or fall. To reproduce this selection from the public aggregate tables alone, run from the repository root:

```bash
python cloud_review/extension/guarded_selection.py --evidence cloud_review/results/full --out guarded_replay
```

The capacity study fixes the upstream RF71 seed-42 alert gate at calibrated any-defect probability ≥ 0.5. Rankings share a cap of 200 reviews per 1,000 alerts, rounded down for the last partial queue. Missed defects outside the alert pool count in cost per 1,000 arrivals. These scenarios assume perfect human inspection, $5 per inspected normal wafer, $1,000 per missed ordinary defect, and either $1,000 or $3,000 per missed Scratch.

## 4. Inspect the results

[Training, calibration and routing results](../results/full/SUMMARY.md) · [Joint-constraint comparison](../results/guarded/BRIEF.md) · [Unlabeled pool and historical cost replay](SHIFT_REPLAY.md)

The public files contain aggregate counts, curves, figures, frozen policy parameters and hashes. Raw maps, checkpoints and per-wafer inspection lists are external. Each seed uses the same audit wafers.

The original [aggregate manifest](https://github.com/Matthewsyr/reliable-wafer-classification/blob/c216b86bd3573bf35326c38aae542f3ce0b99663/cloud_review/results/full/manifest.json) records the renderer and files used for the historical release. The current distribution retains the English summary and every quantitative artifact byte for byte. Its manifest records the historical manifest's digest. The unchanged [selection freeze](../results/guarded/selection_freeze.json) and [guarded summary](../results/guarded/guarded_summary.json) refer to that historical manifest and selection script. Running the English publisher and selector produces new provenance hashes and the same numerical outcomes.

## 5. Optional one-seed pretraining pilot

Generate the isolated pool with the [pool replay](SHIFT_REPLAY.md) before running the pilot. It excludes all components connected to any labeled split by lot or exact image. On a hosted CPU, the packer selects 50,000 of the 434,498 unique eligible maps in deterministic hash order. It checks each map and exports offsets, native dimensions and hashes. Transfer `seed42_offsets.csv.gz` and `receipt.json` to the GPU runtime and verify the receipt digest against the packer's output. Keep `ssl_pilot.py` beside `train_imbalance.py` and `ssl_evaluate.py` beside `decision_study.py`.

```bash
timeout 12m python -u ssl_pilot.py pack --pool POOL_STAGE --raw LSWMD.pkl --out PACK
timeout 50m python -u ssl_pilot.py train --pack PACK --pack-sha CPU_RECEIPT_SHA \
  --control TRAINING_ROOT --repo PINNED_REPO --raw LSWMD.pkl --cache CLOUD_CACHE --out SSL_RUN
timeout 8m python -u ssl_evaluate.py --archive PINNED_REPO/results/probabilities \
  --control TRAINING_ROOT --pilot SSL_RUN --out SSL_COMPARISON
```

The pilot masks failed-die density in patches on an 8 × 8 grid while retaining valid-die coverage. It performs up to 391 additional updates in one pretraining pass, then repeats the seven supervised epochs used by the seed-42 CE control. Evaluation compares both initializations on the same wafers, alert gate and capacity. The primary comparison uses nine-class argmax and reports Scratch hits and both normal false-alarm counts. [Paired pilot results](../results/ssl_paired/pilot_summary.json)
