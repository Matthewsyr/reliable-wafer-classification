"""Render actual recorded historical CNN curves; never train or select a model.

Optional dependency: matplotlib. Run from any working directory.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / 'results/historical/cnn_and_final/epoch_history.csv').open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
with (ROOT / 'results/historical/cnn_and_final/final_cnn_training_history.csv').open(newline='', encoding='utf-8') as handle:
    final = list(csv.DictReader(handle))
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), layout='constrained')
for seed, color in [('42', '#146b8c'), ('123', '#bf6c25'), ('2026', '#188268')]:
    selected = [r for r in rows if r['protocol'] == 'lot' and r['model'] == 'cnn_medium_unweighted' and float(r['fraction']) == 1 and r['seed'] == seed]
    selected.sort(key=lambda r: int(r['epoch']))
    assert len(selected) == 8
    epochs = [int(r['epoch']) for r in selected]
    axes[0].plot(epochs, [float(r['train_loss']) for r in selected], marker='o', color=color, label=f'Seed {seed}')
    axes[1].plot(epochs, [float(r['val_macro_f1'])*100 for r in selected], marker='o', color=color)
axes[2].plot([int(r['epoch']) for r in final], [float(r['train_loss']) for r in final], marker='o', color='#146b8c')
axes[0].set(title='Development training loss', ylabel='Cross-entropy loss')
axes[1].set(title='Development lot-validation score', ylabel='Nine-class Macro-F1 (%)')
axes[2].set(title='Frozen 7-epoch final fit', ylabel='Training cross-entropy loss')
axes[0].legend(frameon=False)
for axis in axes:
    axis.set_xlabel('Epoch')
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.15)
fig.suptitle('Historical medium CNN: recorded curves, not a new training run', fontsize=13)
out = ROOT / 'reports/figures/cnn_training_history.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180)
print(out.name)
