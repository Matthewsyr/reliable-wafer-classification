# %% Configuration and reproducibility
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm.auto import tqdm

from wafer_features import FEATURE_NAMES, extract_features, self_test

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
SEEDS = [42, 123, 2026]
EXPECTED_SHA256 = '1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'
OUT = Path(os.environ.get('WM811K_OUTPUT', '/kaggle/working/wm811k_baseline'))
OUT.mkdir(parents=True, exist_ok=True)
assert not any(OUT.iterdir()), 'Output directory is not empty. Set WM811K_OUTPUT to a new directory for a new run.'
CONFIG = {
    'stage': 'development_validation_only', 'final_test_evaluated': False,
    'class_order': CLASSES, 'seeds': SEEDS,
    'group_validation': 'GroupShuffleSplit, test_size=0.2 of lots, first and only split per seed',
    'random_validation': 'stratified wafer split, same validation row count as paired group split',
    'rf': {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
    'weights': [None, 'balanced_subsample'], 'depth_ablation': [6, 12, None],
    'training_fractions': [0.1, 0.3, 1.0],
    'selection_rule': 'mean group-validation macro-F1 across all three seeds; tie by model name',
    'final_test_policy': 'No test features or test predictions. Freeze RF and CNN before separate final evaluation.',
    'features': list(FEATURE_NAMES),
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

save_json('run_status.json', {'status':'incomplete', 'started_utc':datetime.now(timezone.utc).isoformat()})
if '__file__' in globals():
    CONFIG['source_sha256'] = {p.name:sha256_file(p) for p in [Path(__file__), Path(__file__).with_name('wafer_features.py')]}
save_json('protocol.json', CONFIG)
ENVIRONMENT = {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'joblib', 'tqdm', 'ipython']}
ENVIRONMENT.update(python=platform.python_version(), platform=platform.platform())
save_json('environment.json', ENVIRONMENT)
(OUT / 'requirements-runtime.txt').write_text('\n'.join(f'{p}=={v}' for p, v in ENVIRONMENT.items() if p not in ['python', 'platform']) + '\n', encoding='utf-8')
print('Feature contract:', self_test(), flush=True)
print('Protocol fixed before training. Final test is not evaluated.', flush=True)

# %% Read trusted, hash-pinned data and enforce identity checks
paths = [Path(os.environ['WM811K_DATA'])] if 'WM811K_DATA' in os.environ else list(Path('/kaggle/input').rglob('LSWMD.pkl'))
assert len(paths) == 1, f'Expected one LSWMD.pkl, found {len(paths)}'
DATA_PATH = paths[0]
data_sha256 = sha256_file(DATA_PATH)
assert data_sha256 == EXPECTED_SHA256, 'Input differs from audited dataset; review before running.'
t0 = time.perf_counter()
df = pd.read_pickle(DATA_PATH)  # Only the known Kaggle pickle with the verified hash.
df = df.reset_index(drop=True)  # Explicit positional IDs: no index-aligned joins.
print(f'Loaded {len(df):,} rows in {time.perf_counter()-t0:.1f}s', flush=True)

def scalar_text(value):
    while isinstance(value, (list, tuple, np.ndarray)):
        a = np.asarray(value, dtype=object)
        if a.size == 0:
            return None
        if a.size != 1:
            raise ValueError('Unexpected multi-value metadata field')
        value = a.reshape(-1)[0]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value).strip() or None

split_col = next(c for c in ['trianTestLabel', 'trainTestLabel'] if c in df.columns)
canonical = {c.lower(): c for c in CLASSES}
raw_labels = df['failureType'].map(scalar_text)
assert all(v is None or v.lower() in canonical for v in raw_labels), 'Unknown class'
meta = pd.DataFrame({
    'source_row': np.arange(len(df), dtype=np.int64),
    'label': [canonical[v.lower()] if v is not None else None for v in raw_labels],
    'lot': df['lotName'].map(scalar_text),
    'wafer_index': pd.to_numeric(df['waferIndex'], errors='raise'),
    'source_split': df[split_col].map(scalar_text).str.lower().replace({'training':'train', 'testing':'test'}),
})
meta = meta.loc[meta['label'].notna()].reset_index(drop=True)
assert len(meta) == 172950
assert meta['source_row'].is_unique
assert meta['lot'].notna().all() and meta['source_split'].isin(['train', 'test']).all()
assert meta['wafer_index'].notna().all() and (meta['wafer_index'] > 0).all() and (meta['wafer_index'] % 1 == 0).all()
assert not meta.duplicated(['lot', 'wafer_index']).any()
original_train = meta['source_split'].eq('train')
assert original_train.sum() == 54355 and (~original_train).sum() == 118595
assert not (set(meta.loc[original_train, 'lot']) & set(meta.loc[~original_train, 'lot']))

# Pixel hashes are an integrity audit, not features, and never used to tune the model.
hashes = []
for row in tqdm(meta['source_row'], desc='Labeled map integrity and duplicate audit'):
    a = np.asarray(df.iloc[int(row)]['waferMap'])
    assert a.ndim == 2 and a.size > 0 and a.dtype.kind in 'iuf'
    assert not np.any((a != 0) & (a != 1) & (a != 2))
    n_valid = int(np.count_nonzero(a))
    assert n_valid > 0 and float(df.iloc[int(row)]['dieSize']) == n_valid
    h = hashlib.sha256(np.asarray(a.shape, dtype='<i8').tobytes())
    h.update(np.ascontiguousarray(a, dtype=np.uint8).tobytes())
    hashes.append(h.hexdigest())
meta['map_sha256'] = hashes
duplicate_summary = meta.groupby('map_sha256').agg(rows=('source_row','size'), labels=('label','nunique'), lots=('lot','nunique'), splits=('source_split','nunique'))
duplicate_summary = duplicate_summary.loc[duplicate_summary['rows'] > 1]
duplicate_summary.to_csv(OUT / 'duplicate_map_groups.csv')
meta.loc[meta['map_sha256'].isin(duplicate_summary.index)].to_csv(OUT / 'duplicate_map_rows.csv.gz', index=False)
meta.loc[~original_train].to_csv(OUT / 'protected_final_test_manifest.csv.gz', index=False)
dev = meta.loc[original_train].reset_index(drop=True)
dev_maps = [df.iloc[int(i)]['waferMap'] for i in dev['source_row']]
audit = {
    'labeled_rows': len(meta), 'development_rows': len(dev), 'protected_final_test_rows': int((~original_train).sum()),
    'development_lots': int(dev['lot'].nunique()), 'original_train_test_shared_lots': 0,
    'duplicate_map_groups': len(duplicate_summary),
    'duplicate_groups_with_conflicting_labels': int((duplicate_summary['labels'] > 1).sum()),
    'duplicate_groups_across_original_train_test': int((duplicate_summary['splits'] > 1).sum()),
    'duplicate_policy': 'Report without silently removing naturally identical maps, including normal maps.',
}
save_json('integrity_audit.json', audit)
print(json.dumps(audit, indent=2), flush=True)
del df, meta, raw_labels, hashes, original_train
gc.collect()

# %% Development-only spatial features; no fitted preprocessing
t0 = time.perf_counter()
X = np.vstack([extract_features(w) for w in tqdm(dev_maps, desc='Extracting 71 spatial features')])
del dev_maps
gc.collect()
y = dev['label'].map({c: i for i,c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
assert X.shape == (len(dev), len(FEATURE_NAMES)) and np.isfinite(X).all()
np.savez_compressed(OUT / 'development_features.npz', X=X, y=y, source_row=dev['source_row'].to_numpy(), feature_names=np.asarray(FEATURE_NAMES))
dev.to_csv(OUT / 'development_metadata.csv.gz', index=False)
print(f'Features {X.shape}; elapsed {time.perf_counter()-t0:.1f}s; test features were not constructed.', flush=True)

# %% Fixed paired validation splits and nested training samples
splits, split_rows, support_rows, manifests = {}, [], [], []
for seed in SEEDS:
    gt, gv = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups=dev['lot']))
    rt, rv = train_test_split(np.arange(len(dev)), test_size=len(gv), random_state=seed, stratify=y)
    for protocol, train_idx, val_idx in [('lot', gt, gv), ('random', rt, rv)]:
        assert not set(train_idx) & set(val_idx)
        assert len(train_idx) + len(val_idx) == len(dev)
        assert set(y[train_idx]) == set(y[val_idx]) == set(range(len(CLASSES))), 'Missing class: do not select a different seed after seeing results.'
        shared_lots = set(dev.iloc[train_idx]['lot']) & set(dev.iloc[val_idx]['lot'])
        if protocol == 'lot':
            assert len(shared_lots) == 0
        shared_hashes = set(dev.iloc[train_idx]['map_sha256']) & set(dev.iloc[val_idx]['map_sha256'])
        splits[protocol, seed] = (train_idx, val_idx)
        split_rows.append({'protocol':protocol, 'seed':seed, 'train_rows':len(train_idx), 'val_rows':len(val_idx),
            'train_lots':int(dev.iloc[train_idx]['lot'].nunique()), 'val_lots':int(dev.iloc[val_idx]['lot'].nunique()),
            'shared_lots':len(shared_lots), 'shared_image_hashes':len(shared_hashes),
            'val_rows_with_train_identical_map':int(dev.iloc[val_idx]['map_sha256'].isin(shared_hashes).sum())})
        manifest = dev[['source_row','lot','wafer_index','label','map_sha256']].copy()
        manifest['protocol'], manifest['seed'], manifest['role'] = protocol, seed, 'train'
        manifest.loc[val_idx, 'role'] = 'validation'
        manifests.append(manifest)
        for role, idx in [('train', train_idx), ('validation', val_idx)]:
            for cls in CLASSES:
                rows = dev.iloc[idx].loc[dev.iloc[idx]['label'].eq(cls)]
                support_rows.append({'protocol':protocol, 'seed':seed, 'role':role, 'label':cls,
                    'wafers':len(rows), 'lots':int(rows['lot'].nunique()), 'share':len(rows)/len(idx)})
pd.concat(manifests, ignore_index=True).to_csv(OUT / 'split_manifests.csv.gz', index=False)
pd.DataFrame(split_rows).to_csv(OUT / 'split_diagnostics.csv', index=False)
pd.DataFrame(support_rows).to_csv(OUT / 'split_class_support.csv', index=False)
display(pd.DataFrame(split_rows))
del manifests

# All ablations use identical validation rows for a given protocol/seed.
def nested_subset(train_idx, fraction, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for k in range(len(CLASSES)):
        order = rng.permutation(train_idx[y[train_idx] == k])
        chunks.append(order[:max(1, int(np.floor(len(order)*fraction)))])
    return np.sort(np.concatenate(chunks))

sample_manifests = []
for seed in SEEDS:
    ti, _ = splits['lot',seed]
    prior = set()
    for fraction in [0.1,0.3,1.0]:
        subset = nested_subset(ti,fraction,seed)
        assert prior.issubset(set(subset))
        prior = set(subset)
        sm = dev.iloc[subset][['source_row','lot','label']].copy()
        sm['seed'], sm['fraction'] = seed, fraction
        sample_manifests.append(sm)
pd.concat(sample_manifests, ignore_index=True).to_csv(OUT / 'learning_curve_manifests.csv.gz', index=False)
del sample_manifests

# %% Experiments: full fixed matrix, checkpoint after each fit
MODEL_SPECS = {
    'dummy_most_frequent': {'kind':'dummy'},
    'rf_unweighted_d12': {'kind':'rf', 'class_weight':None, 'max_depth':12},
    'rf_weighted_d12': {'kind':'rf', 'class_weight':'balanced_subsample', 'max_depth':12},
    'rf_weighted_d6': {'kind':'rf', 'class_weight':'balanced_subsample', 'max_depth':6},
    'rf_weighted_full': {'kind':'rf', 'class_weight':'balanced_subsample', 'max_depth':None},
}
CONFIG['model_specs'] = MODEL_SPECS
CONFIG['rf']['n_jobs'] = 2
save_json('protocol.json', CONFIG)
scores, per_class, predictions, cm_rows, learning_support = [], [], [], [], []

def make_model(name, seed):
    spec = MODEL_SPECS[name]
    if spec['kind'] == 'dummy':
        return DummyClassifier(strategy='most_frequent')
    return RandomForestClassifier(n_estimators=200, max_depth=spec['max_depth'], min_samples_leaf=2,
        max_features='sqrt', class_weight=spec['class_weight'], n_jobs=2, random_state=seed)

def checkpoint():
    pd.DataFrame(scores).to_csv(OUT / 'validation_scores.csv', index=False)
    pd.DataFrame(per_class).to_csv(OUT / 'validation_per_class.csv', index=False)
    pd.DataFrame(cm_rows).to_csv(OUT / 'confusion_matrices.csv', index=False)
    pd.DataFrame(learning_support).to_csv(OUT / 'experiment_training_support.csv', index=False)

def run_experiment(protocol, seed, model_name, fraction=1.0):
    train_idx, val_idx = splits[protocol, seed]
    if fraction < 1:
        train_idx = nested_subset(train_idx, fraction, seed)
    model = make_model(model_name, seed)
    started = time.perf_counter()
    model.fit(X[train_idx], y[train_idx])
    fit_seconds = time.perf_counter()-started
    started = time.perf_counter()
    pred = model.predict(X[val_idx])
    predict_seconds = time.perf_counter()-started
    train_pred = model.predict(X[train_idx])
    identity = {'protocol':protocol, 'seed':seed, 'model':model_name, 'fraction':fraction}
    score = {**identity, 'train_rows':len(train_idx), 'train_lots':int(dev.iloc[train_idx]['lot'].nunique()),
        'val_rows':len(val_idx), 'accuracy':accuracy_score(y[val_idx],pred),
        'macro_f1':f1_score(y[val_idx],pred,labels=np.arange(9),average='macro',zero_division=0),
        'train_macro_f1':f1_score(y[train_idx],train_pred,labels=np.arange(9),average='macro',zero_division=0),
        'fit_seconds':fit_seconds, 'predict_seconds':predict_seconds}
    scores.append(score)
    precision, recall, f1, support = precision_recall_fscore_support(y[val_idx],pred,labels=np.arange(9),zero_division=0)
    for k, cls in enumerate(CLASSES):
        per_class.append({**identity, 'label':cls, 'precision':precision[k], 'recall':recall[k], 'f1':f1[k],
            'support':int(support[k]), 'val_lots':int(dev.iloc[val_idx].loc[y[val_idx]==k,'lot'].nunique())})
        learning_support.append({**identity, 'label':cls, 'train_wafers':int(np.sum(y[train_idx]==k)),
            'train_lots':int(dev.iloc[train_idx].loc[y[train_idx]==k,'lot'].nunique())})
    cm = confusion_matrix(y[val_idx],pred,labels=np.arange(9))
    for i, true_class in enumerate(CLASSES):
        for j, predicted_class in enumerate(CLASSES):
            cm_rows.append({**identity,'true_label':true_class,'predicted_label':predicted_class,'count':int(cm[i,j])})
    pv = dev.iloc[val_idx][['source_row','lot','label']].copy()
    for key, value in identity.items():
        pv[key] = value
    pv['predicted_label'] = [CLASSES[k] for k in pred]
    pv['identical_map_in_train'] = dev.iloc[val_idx]['map_sha256'].isin(set(dev.iloc[train_idx]['map_sha256'])).to_numpy()
    predictions.append(pv)
    if protocol == 'lot' and seed == SEEDS[0] and fraction == 1 and model_name.startswith('rf_'):
        joblib.dump({'model':model, 'feature_names':FEATURE_NAMES, 'class_order':CLASSES,
            'scope':'seed42 group-training subset; not final full-development model'}, OUT / f'{model_name}_validation.joblib', compress=3)
    checkpoint()
    print(f'{protocol:6s} seed={seed:4d} {model_name:22s} fraction={fraction:.1f} macroF1={score["macro_f1"]:.4f} accuracy={score["accuracy"]:.4f} fit={fit_seconds:.1f}s', flush=True)

for seed in SEEDS:
    for protocol in ['lot','random']:
        for model_name in ['dummy_most_frequent','rf_unweighted_d12','rf_weighted_d12']:
            run_experiment(protocol,seed,model_name)
    for model_name in ['rf_weighted_d6','rf_weighted_full']:
        run_experiment('lot',seed,model_name)
    for fraction in [0.1,0.3]:
        run_experiment('lot',seed,'rf_weighted_d12',fraction)
assert len(scores) == 30
pd.concat(predictions, ignore_index=True).to_csv(OUT / 'validation_predictions.csv.gz', index=False)

# %% Report validation evidence, including caveats and a provisional RF choice
results = pd.DataFrame(scores)
summary = results.groupby(['protocol','model','fraction'], dropna=False).agg(
    macro_f1_mean=('macro_f1','mean'), macro_f1_sd=('macro_f1','std'),
    accuracy_mean=('accuracy','mean'), train_macro_f1_mean=('train_macro_f1','mean'),
    fit_seconds_mean=('fit_seconds','mean')).reset_index()
summary.to_csv(OUT / 'validation_summary.csv', index=False)
display(summary.round(4))
eligible = summary.loc[summary['protocol'].eq('lot') & summary['fraction'].eq(1) & summary['model'].str.startswith('rf_')]
selected = eligible.sort_values(['macro_f1_mean','model'],ascending=[False,True]).iloc[0]['model']
selection = {'status':'provisional RF configuration based on development validation', 'model':selected,
    'specification':MODEL_SPECS[selected], 'selection_rule':CONFIG['selection_rule'],
    'final_test_evaluated':False, 'cnn_not_trained':True,
    'selection_bias_note':'These validation data select the RF; their scores are not independent final estimates.'}
save_json('provisional_rf_selection.json',selection)
pc = pd.DataFrame(per_class)
recalls = pc.loc[pc['protocol'].eq('lot') & pc['model'].eq(selected) & pc['fraction'].eq(1)]
recalls.pivot(index='label',columns='seed',values='recall').reindex(CLASSES).to_csv(OUT / 'selected_rf_recall_by_seed.csv')
display(Markdown('### Provisional RF: ' + selected + ' — group-validation recalls'))
display(recalls[['seed','label','recall','support','val_lots']])

fig, axes = plt.subplots(1, 3, figsize=(17,5))
base_names = ['dummy_most_frequent','rf_unweighted_d12','rf_weighted_d12']
for offset, protocol in [(-.18,'random'),(.18,'lot')]:
    z = summary.set_index(['protocol','model','fraction']).loc[[(protocol,n,1.0) for n in base_names]]
    axes[0].bar(np.arange(3)+offset,z['macro_f1_mean'],width=.36,yerr=z['macro_f1_sd'],capsize=3,label=protocol)
axes[0].set_xticks(np.arange(3),['Dummy','RF unweighted','RF weighted'],rotation=15)
axes[0].set_title('Validation split comparison (3 seeds)')
axes[0].set_ylabel('Macro-F1'); axes[0].set_ylim(0,1); axes[0].legend()
z = summary.loc[summary['protocol'].eq('lot') & summary['model'].eq('rf_weighted_d12')].sort_values('fraction')
axes[1].errorbar(z['fraction']*100,z['macro_f1_mean'],yerr=z['macro_f1_sd'],marker='o',capsize=3)
axes[1].set_title('Nested wafer learning curve'); axes[1].set_xlabel('Training wafers retained (%)'); axes[1].set_ylabel('Macro-F1')
for seed in SEEDS:
    z = recalls.loc[recalls['seed'].eq(seed)].set_index('label').reindex(CLASSES)
    axes[2].plot(np.arange(9),z['recall'],marker='.',label=str(seed))
axes[2].set_xticks(np.arange(9),CLASSES,rotation=55,ha='right'); axes[2].set_ylim(0,1.05)
axes[2].set_title('Selected RF: per-class validation recall'); axes[2].legend(title='Seed')
fig.tight_layout(); fig.savefig(OUT / 'baseline_validation_results.png',dpi=160,bbox_inches='tight'); plt.show()

notes = '''# WM-811K: traditional ML development baseline

This run evaluates development validation only. The original 118,595-wafer test set remains unscored.
All supervised feature extraction and fitting use the original 54,355 training wafers.

## Protocol
- Three predeclared seeds: 42, 123, 2026. GroupShuffleSplit reserves 20% of lots; it does not guarantee class stratification.
- Paired random validation is stratified by class and matches the group split validation row count.
- Identical models see the same splits. Nested 10%/30%/100% training samples use a fixed validation set.
- Primary metric: nine-class unweighted Macro-F1. Per-class precision/recall/F1, wafer support and lot support are exported.
- Error bars are descriptive standard deviations across three overlapping holdouts, not confidence intervals.
- The provisional RF is selected only by mean group-validation Macro-F1. CNN has not been trained.

## Interpretation limits
- Random versus lot score differences mix batch similarity, class composition, and validation sample difficulty. They are not a causal estimate of leakage.
- The class-weight and depth ablations keep the other RF settings fixed. Learning curves vary wafer count, not independently controlled lot diversity.
- Identical map hashes are reported, not silently removed. All-pass normal wafers can naturally have identical maps; identity duplication and map duplication are different.
- Features exclude lot ID, wafer index and source split. Valid-mask geometry may still encode product or batch differences.
- Features describe existing die test outcomes. They cannot identify a defective machine or prove RF/plasma causation.
- The public data have no validated chronological timestamps, so batch isolation does not establish future-time performance.
- Fixed input SHA256, split manifests, code and package versions enable reruns. Different runtimes may still differ numerically.
- Freeze feature/preprocessing choices and RF/CNN selection rules before a separate final evaluation. Never use test scores to select another model.

## Files
`validation_summary.csv` summarizes results; `validation_per_class.csv` provides class metrics/support.
`split_manifests.csv.gz` and `learning_curve_manifests.csv.gz` are exact source-row lists for reuse by CNN.
`validation_predictions.csv.gz` supports error analysis, including flags for validation maps identical to training maps.
`development_features.npz` contains only development features; checkpoints are group-training subset models, not deployment models.
'''
(OUT / 'EXPERIMENT_NOTES.md').write_text(notes,encoding='utf-8')
save_json('run_status.json', {'status':'completed','experiment_count':len(results),'final_test_evaluated':False})
provenance = {'input_path':str(DATA_PATH),'input_bytes':DATA_PATH.stat().st_size,'input_sha256':data_sha256,
    'completed_utc':datetime.now(timezone.utc).isoformat(),'environment':ENVIRONMENT,'experiment_count':len(results),
    'final_test_evaluated':False,'provisional_rf':selected,
    'artifact_sha256':{p.name:sha256_file(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'run_provenance.json'}}
save_json('run_provenance.json',provenance)
print('COMPLETED: 30 development experiments; final test remains unscored.',flush=True)
