"""WM-811K stage 3: fixed-manifest CNN validation, then one frozen final evaluation.

Execute beside wafer_cnn.py and the unchanged stage-2 wafer_features.py.
Only the hash-pinned public pickle is unpickled. No API or network is used.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from tqdm.auto import tqdm

from wafer_features import FEATURE_NAMES, extract_features
from wafer_cnn import PREPROCESSING, CHANNELS, preprocess_map, train_cnn, final_train, predict_batches, synthetic_smoke_test


CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
SEEDS = [42, 123, 2026]
LABEL_IDS = np.arange(len(CLASSES))
DATA_SHA256 = '1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'
BASELINE_PROVENANCE_SHA256 = 'f32c0bd04982766fdc48892e487a50455ba53333c3eff381698af94125d3a28f'
FEATURE_SOURCE_SHA256 = '0dc546f82fe12a1ec054db26692597ae727c8195e24c428eb0b9c0afceb3e0f4'
OUT = Path(os.environ.get('WM811K_CNN_OUTPUT', '/kaggle/working/wm811k_cnn'))
RUN_STARTED = False
CNN_SPECS = {
    'cnn_small_unweighted': {'model_size': 'small', 'weighting': 'none'},
    'cnn_small_sqrt_inverse': {'model_size': 'small', 'weighting': 'sqrt_inverse'},
    'cnn_medium_unweighted': {'model_size': 'medium', 'weighting': 'none'},
}
CONFIG = {
    'stage': 'cnn_validation_and_frozen_final_test',
    'class_order': CLASSES, 'seeds': SEEDS,
    'input_sha256': DATA_SHA256,
    'baseline_run_provenance_sha256': BASELINE_PROVENANCE_SHA256,
    'image_size': 64,
    'preprocessing': 'Separate binary valid-die and failed-die masks; centered zero square padding; area resize to 64x64; channels [valid,failed], float32. No fitted normalization or augmentation.',
    'epochs': 8, 'batch_size': 256,
    'budget_rationale': {
        'revised_utc': '2026-10-22T00:24:59.9369062Z',
        'previous_epochs': 12, 'fixed_epochs': 8,
        'decision_state': 'before any real-data CNN fitting or validation scores',
        'scope': 'All devices and configurations use eight epochs; the fixed 18-run matrix is unchanged. No score-dependent budget adaptation.',
        'evidence': {
            'source': 'Kaggle notebook version 360950931, synthetic CPU benchmark',
            'kaggle_version': 360950931, 'synthetic_only': True, 'batch_size': 256,
            'small': {'training_samples_per_second': 291.41045, 'prediction_samples_per_second': 675.0552},
            'medium': {'training_samples_per_second': 124.16624, 'prediction_samples_per_second': 285.0802},
        },
        'reason': 'The synthetic CPU rates imply approximately 9.5 hours for the original 12-epoch validation matrix, before final fitting and additional I/O. Reduce the fixed initial compute budget before observing real-data CNN results; benchmark extrapolations are not runtime guarantees.',
    },
    'optimizer': {'name': 'AdamW', 'lr': 0.001, 'weight_decay': 0.0001},
    'model_specs': CNN_SPECS,
    'architecture': {'name': 'WaferCNN_singleconv_v1', 'channels': CHANNELS,
                     'classifier': 'three Conv2d/BatchNorm/ReLU/MaxPool blocks, AvgPool2d(2), linear nine-class logits'},
    'preprocessing_contract': PREPROCESSING,
    'selection_rule': 'Mean lot-validation Macro-F1 across seeds 42/123/2026 at fraction=1; descending score then ascending model name. Random validation and learning curves are ineligible.',
    'best_epoch_rule': 'Earliest epoch attaining maximum validation Macro-F1, without changing the eight-epoch budget.',
    'final_cnn_epochs_rule': 'Integer median of the three selected configuration best epochs; reinitialize with seed 42 and fit all development wafers without final-test early stopping.',
    'rf_policy': 'Use the existing stage-2 selected RF configuration without reselection, refit all development features with seed 42.',
    'final_subsets': ['all', 'nonidentical_to_development'],
    'final_subset_rule': 'The second subset excludes test maps whose precomputed map_sha256 appears anywhere in full development; it is a predeclared sensitivity analysis, not a replacement primary endpoint.',
    'primary_metric': 'Fixed nine-class unweighted Macro-F1 with zero_division=0; per-class recall is uninterpretable when support is zero.',
    'statistical_note': 'SD over three overlapping development holdouts is descriptive, not a confidence interval. Random/lot differences mix group similarity, composition, and sample difficulty.',
    'test_policy': 'Write and hash selection_freeze.json before final-test preprocessing, prediction, or scoring. Score both frozen models; do not use scores to choose or revise models.',
    'cache_policy': 'Temporary memory-mapped preprocessed images are not deliverables and are removed after use.',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_unique(filename, env_name=None):
    if env_name and env_name in os.environ:
        result = [Path(os.environ[env_name])]
    else:
        result = list(Path('/kaggle/input').rglob(filename))
    require(len(result) == 1 and result[0].is_file(), f'Expected one {filename}; found {result}')
    return result[0]


def read_csv(path):
    # Keep the literal normal class "none" and lot identifiers intact.
    return pd.read_csv(path, keep_default_na=False)


def verify_baseline():
    if 'WM811K_BASELINE' in os.environ:
        root = Path(os.environ['WM811K_BASELINE'])
    else:
        root = find_unique('development_features.npz').parent
    provenance_path = root / 'run_provenance.json'
    require(sha256_file(provenance_path) == BASELINE_PROVENANCE_SHA256, 'Stage-2 provenance differs from the completed verified run.')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    require(provenance['input_sha256'] == DATA_SHA256 and provenance['experiment_count'] == 30,
            'Unexpected baseline input or experiment count.')
    require(provenance['final_test_evaluated'] is False, 'Baseline already scored final test.')
    verified = {}
    for name, expected in sorted(provenance['artifact_sha256'].items()):
        path = root / name
        require(path.is_file() and sha256_file(path) == expected, f'Baseline artifact integrity failure: {name}')
        verified[name] = expected
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    require(protocol['class_order'] == CLASSES and protocol['seeds'] == SEEDS, 'Baseline class order/seeds differ.')
    require(protocol['features'] == list(FEATURE_NAMES), 'Traditional feature order changed.')
    feature_source = Path(__file__).with_name('wafer_features.py')
    require(sha256_file(feature_source) == FEATURE_SOURCE_SHA256, 'The unchanged stage-2 feature extractor is required.')
    selection = json.loads((root / 'provisional_rf_selection.json').read_text(encoding='utf-8'))
    require(selection['final_test_evaluated'] is False and selection['model'] == provenance['provisional_rf'], 'RF selection evidence disagrees.')
    summary = read_csv(root / 'validation_summary.csv')
    eligible = summary.loc[summary['protocol'].eq('lot') & summary['fraction'].eq(1) & summary['model'].str.startswith('rf_')]
    selected = eligible.sort_values(['macro_f1_mean', 'model'], ascending=[False, True]).iloc[0]['model']
    require(selected == selection['model'], 'RF selection does not reproduce from baseline summary.')
    require(selection['specification'] == protocol['model_specs'][selected], 'RF specification changed.')
    save_json('baseline_integrity.json', {'verified_utc': utc_now(), 'root': str(root),
        'run_provenance_sha256': BASELINE_PROVENANCE_SHA256, 'verified_artifact_sha256': verified})
    return root, protocol, selection


def load_manifests(baseline):
    dev = read_csv(baseline / 'development_metadata.csv.gz')
    test = read_csv(baseline / 'protected_final_test_manifest.csv.gz')
    split_manifest = read_csv(baseline / 'split_manifests.csv.gz')
    sample_manifest = read_csv(baseline / 'learning_curve_manifests.csv.gz')
    require(len(dev) == 54355 and len(test) == 118595, 'Unexpected development/test sizes.')
    require(dev['source_row'].is_unique and test['source_row'].is_unique, 'Repeated source_row.')
    require(dev['source_split'].eq('train').all() and test['source_split'].eq('test').all(), 'Source split differs.')
    require(not set(dev['source_row']) & set(test['source_row']), 'Development/test rows overlap.')
    require(not set(dev['lot']) & set(test['lot']), 'Development/test lots overlap.')
    identical_test_rows = int(test['map_sha256'].isin(set(dev['map_sha256'])).sum())
    require(identical_test_rows == 3158, 'Fixed-data identical-map sensitivity support changed.')
    require(set(dev['label']) == set(test['label']) == set(CLASSES), 'Class coverage differs.')
    require(not dev.duplicated(['lot', 'wafer_index']).any() and not test.duplicated(['lot', 'wafer_index']).any(), 'Repeated wafer identity.')
    indexer = pd.Series(np.arange(len(dev), dtype=np.int64), index=dev['source_row'])
    expected_keys = {(p, s) for p in ['lot', 'random'] for s in SEEDS}
    require(set(split_manifest.groupby(['protocol', 'seed']).groups) == expected_keys, 'Unexpected split manifest groups.')
    splits = {}
    for (protocol, seed), rows in split_manifest.groupby(['protocol', 'seed'], sort=False):
        require(len(rows) == len(dev) and rows['source_row'].is_unique, 'Split is not a development partition.')
        require(set(rows['source_row']) == set(dev['source_row']), 'Split source rows differ.')
        require(set(rows['role']) == {'train', 'validation'}, 'Unexpected role in split manifest.')
        aligned = rows.set_index('source_row').loc[dev['source_row']]
        for field in ['lot', 'label', 'map_sha256', 'wafer_index']:
            require(np.array_equal(aligned[field].to_numpy(), dev[field].to_numpy()), f'Split metadata differs: {field}')
        train = np.flatnonzero(aligned['role'].eq('train').to_numpy())
        val = np.flatnonzero(aligned['role'].eq('validation').to_numpy())
        require(set(dev.iloc[train]['label']) == set(dev.iloc[val]['label']) == set(CLASSES), 'Class absent from a manifest split.')
        if protocol == 'lot':
            require(not set(dev.iloc[train]['lot']) & set(dev.iloc[val]['lot']), 'Lot split leaks groups.')
        splits[protocol, int(seed)] = (train, val)
    learning = {}
    require(set(sample_manifest.groupby(['seed', 'fraction']).groups) == {(s, f) for s in SEEDS for f in [.1, .3, 1.]}, 'Unexpected learning manifest groups.')
    for seed in SEEDS:
        previous = set()
        for fraction in [.1, .3, 1.]:
            rows = sample_manifest.loc[sample_manifest['seed'].eq(seed) & sample_manifest['fraction'].eq(fraction)]
            require(rows['source_row'].is_unique and set(rows['source_row']).issubset(set(dev['source_row'])), 'Invalid learning source rows.')
            positions = indexer.loc[rows['source_row']].to_numpy(dtype=np.int64)
            require(set(positions).issubset(set(splits['lot', seed][0])), 'Learning subset escapes training partition.')
            require(previous.issubset(set(positions)), 'Learning samples are not nested.')
            previous = set(positions)
            for field in ['lot', 'label']:
                require(np.array_equal(rows[field].to_numpy(), dev.iloc[positions][field].to_numpy()), 'Learning metadata mismatch.')
            if fraction == 1.:
                require(set(positions) == set(splits['lot', seed][0]), 'Full learning subset differs from training partition.')
            learning[seed, fraction] = positions
    with np.load(baseline / 'development_features.npz', allow_pickle=False) as archive:
        features = archive['X'].copy()
        y = archive['y'].copy()
        require(np.array_equal(archive['source_row'], dev['source_row'].to_numpy()), 'Feature/source-row alignment changed.')
        require(archive['feature_names'].tolist() == list(FEATURE_NAMES), 'Feature names changed.')
    require(features.shape == (len(dev), len(FEATURE_NAMES)) and np.isfinite(features).all(), 'Invalid baseline features.')
    require(np.array_equal(y, dev['label'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()), 'Feature target alignment changed.')
    # Exact copies preserve compressed-file hashes and supply all split evidence in this run.
    for name in ['development_metadata.csv.gz', 'protected_final_test_manifest.csv.gz',
                 'split_manifests.csv.gz', 'learning_curve_manifests.csv.gz']:
        shutil.copyfile(baseline / name, OUT / name)
    for name in ['validation_summary.csv', 'provisional_rf_selection.json', 'protocol.json', 'run_provenance.json']:
        shutil.copyfile(baseline / name, OUT / ('baseline_' + name))
    return dev, test, features, y, splits, learning


def experiment_matrix():
    matrix = []
    for seed in SEEDS:
        for protocol in ['lot', 'random']:
            matrix.append({'protocol': protocol, 'seed': seed, 'model': 'cnn_small_unweighted', 'fraction': 1.})
        matrix.append({'protocol': 'lot', 'seed': seed, 'model': 'cnn_small_sqrt_inverse', 'fraction': 1.})
        matrix.append({'protocol': 'lot', 'seed': seed, 'model': 'cnn_medium_unweighted', 'fraction': 1.})
        for fraction in [.1, .3]:
            matrix.append({'protocol': 'lot', 'seed': seed, 'model': 'cnn_small_unweighted', 'fraction': fraction})
    require(len(matrix) == 18, 'CNN experiment matrix changed.')
    return matrix


def identity_name(identity):
    return f"{identity['protocol']}_s{identity['seed']}_{identity['model']}_f{identity['fraction']:.1f}"


def build_cache(maps, rows, path, description):
    cache = np.memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 2, 64, 64))
    for position, row in enumerate(tqdm(rows, desc=description)):
        cache[position] = preprocess_map(maps[int(row)], size=64)
    cache.flush()
    return cache


def close_cache(cache, path):
    cache.flush()
    cache._mmap.close()
    Path(path).unlink()


def metric_rows(y_true, prediction, metadata, identity):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, prediction, labels=LABEL_IDS, zero_division=0)
    per_class = []
    for i, cls in enumerate(CLASSES):
        per_class.append({**identity, 'label': cls, 'precision': float(precision[i]), 'recall': float(recall[i]),
            'f1': float(f1[i]), 'support': int(support[i]), 'lots': int(metadata.loc[np.asarray(y_true) == i, 'lot'].nunique()),
            'recall_defined': bool(support[i] > 0)})
    cm = confusion_matrix(y_true, prediction, labels=LABEL_IDS)
    confusion = [{**identity, 'true_label': a, 'predicted_label': b, 'count': int(cm[i, j])}
                 for i, a in enumerate(CLASSES) for j, b in enumerate(CLASSES)]
    return per_class, confusion


def validate_cnns(cache, y, dev, splits, learning, device):
    scores, class_rows, cm_rows, predictions, histories, support_rows = [], [], [], [], [], []
    for identity in experiment_matrix():
        protocol, seed, model, fraction = (identity[k] for k in ['protocol', 'seed', 'model', 'fraction'])
        train_idx, val_idx = splits[protocol, seed]
        if fraction < 1:
            train_idx = learning[seed, fraction]
        run_dir = OUT / 'validation_checkpoints' / identity_name(identity)
        result = train_cnn(cache, y, train_idx, val_idx, seed=seed, **CNN_SPECS[model],
            epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'], device=device, output_dir=run_dir)
        prediction = np.asarray(result['predictions'], dtype=np.int64)
        train_prediction = np.asarray(result['train_predictions'], dtype=np.int64)
        require(prediction.shape == (len(val_idx),) and train_prediction.shape == (len(train_idx),), 'CNN prediction length differs.')
        require(np.isin(prediction, LABEL_IDS).all() and np.isin(train_prediction, LABEL_IDS).all(), 'Invalid CNN class prediction.')
        history = result['history']
        require(len(history) == CONFIG['epochs'], 'CNN run did not finish its fixed budget.')
        expected_epoch = min(history, key=lambda h: (-h['val_macro_f1'], h['epoch']))['epoch']
        require(int(result['best_epoch']) == int(expected_epoch), 'Best epoch violates predeclared tie rule.')
        scores.append({**identity, 'train_rows': len(train_idx), 'train_lots': int(dev.iloc[train_idx]['lot'].nunique()),
            'val_rows': len(val_idx), 'accuracy': float(accuracy_score(y[val_idx], prediction)),
            'macro_f1': float(f1_score(y[val_idx], prediction, labels=LABEL_IDS, average='macro', zero_division=0)),
            'train_macro_f1': float(f1_score(y[train_idx], train_prediction, labels=LABEL_IDS, average='macro', zero_division=0)),
            'fit_seconds': float(result['fit_seconds']), 'predict_seconds': float(result['predict_seconds']),
            'train_predict_seconds': float(result['train_predict_seconds']), 'best_epoch': int(result['best_epoch']),
            'parameter_count': int(result['parameter_count'])})
        pc, cm = metric_rows(y[val_idx], prediction, dev.iloc[val_idx], identity)
        for item in pc:
            item['val_lots'] = item.pop('lots')
        class_rows.extend(pc)
        cm_rows.extend(cm)
        for cls in CLASSES:
            z = dev.iloc[train_idx].loc[dev.iloc[train_idx]['label'].eq(cls)]
            support_rows.append({**identity, 'label': cls, 'train_wafers': len(z), 'train_lots': int(z['lot'].nunique())})
        pv = dev.iloc[val_idx][['source_row', 'lot', 'label']].copy()
        for key, value in identity.items():
            pv[key] = value
        pv['predicted_label'] = [CLASSES[i] for i in prediction]
        pv['identical_map_in_train'] = dev.iloc[val_idx]['map_sha256'].isin(set(dev.iloc[train_idx]['map_sha256'])).to_numpy()
        predictions.append(pv)
        histories.extend([{**identity, **epoch} for epoch in history])
        pd.DataFrame(scores).to_csv(OUT / 'validation_scores.csv', index=False)
        pd.DataFrame(class_rows).to_csv(OUT / 'validation_per_class.csv', index=False)
        pd.DataFrame(cm_rows).to_csv(OUT / 'confusion_matrices.csv', index=False)
        pd.DataFrame(histories).to_csv(OUT / 'epoch_history.csv', index=False)
        pd.DataFrame(support_rows).to_csv(OUT / 'experiment_training_support.csv', index=False)
        pd.concat(predictions, ignore_index=True).to_csv(OUT / 'validation_predictions.csv.gz', index=False)
        print(f"{identity_name(identity)} Macro-F1={scores[-1]['macro_f1']:.4f} best_epoch={result['best_epoch']}", flush=True)
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    require(len(scores) == 18, 'Incomplete validation matrix.')
    results = pd.DataFrame(scores)
    summary = results.groupby(['protocol', 'model', 'fraction'], dropna=False).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), train_macro_f1_mean=('train_macro_f1', 'mean'),
        fit_seconds_mean=('fit_seconds', 'mean')).reset_index()
    summary.to_csv(OUT / 'validation_summary.csv', index=False)
    return results, summary


def freeze_selection(results, summary, rf_selection, baseline_protocol, device):
    eligible = summary.loc[summary['protocol'].eq('lot') & summary['fraction'].eq(1.)]
    require(set(eligible['model']) == set(CNN_SPECS), 'CNN candidate eligibility changed.')
    selected = eligible.sort_values(['macro_f1_mean', 'model'], ascending=[False, True]).iloc[0]['model']
    selected_rows = results.loc[results['protocol'].eq('lot') & results['fraction'].eq(1.) & results['model'].eq(selected)].sort_values('seed')
    require(selected_rows['seed'].tolist() == SEEDS, 'CNN selection is missing a seed.')
    epochs = int(np.median(selected_rows['best_epoch'].to_numpy()))
    require(1 <= epochs <= CONFIG['epochs'], 'Invalid frozen final training budget.')
    rf_parameters = dict(baseline_protocol['rf'])
    rf_parameters.update(max_depth=rf_selection['specification']['max_depth'],
        class_weight=rf_selection['specification']['class_weight'], random_state=42, n_jobs=2)
    manifest_names = ['development_metadata.csv.gz', 'protected_final_test_manifest.csv.gz', 'split_manifests.csv.gz', 'learning_curve_manifests.csv.gz']
    frozen = {'frozen_utc': utc_now(), 'final_test_evaluated': False, 'protocol': CONFIG,
        'cnn': {'model': selected, **CNN_SPECS[selected], 'seed': 42, 'epochs': epochs,
                'selection_best_epochs': selected_rows[['seed', 'best_epoch']].to_dict(orient='records'),
                'training_rows': 54355, 'batch_size': CONFIG['batch_size'], 'device': device},
        'rf': {'model': rf_selection['model'], 'specification': rf_selection['specification'],
               'parameters': rf_parameters, 'seed': 42, 'training_rows': 54355},
        'source_sha256': {p.name: sha256_file(p) for p in sorted((OUT / 'source').glob('*.py'))},
        'manifest_sha256': {name: sha256_file(OUT / name) for name in manifest_names},
        'selection_evidence_sha256': {name: sha256_file(OUT / name) for name in ['validation_scores.csv', 'validation_summary.csv', 'epoch_history.csv', 'baseline_provisional_rf_selection.json']}}
    save_json('selection_freeze.json', frozen)
    freeze_hash = sha256_file(OUT / 'selection_freeze.json')
    (OUT / 'selection_freeze.sha256').write_text(freeze_hash + '  selection_freeze.json\n', encoding='utf-8')
    save_json('run_status.json', {'status': 'selection_frozen', 'validation_experiment_count': 18,
        'final_test_evaluated': False, 'freeze_sha256': freeze_hash, 'updated_utc': utc_now()})
    print(f'FROZEN before final-test processing: {frozen["cnn"]}; RF={rf_selection["model"]}', flush=True)
    return frozen, freeze_hash


def final_evaluation(maps, dev_cache, dev, test, features, y, frozen, freeze_hash, cache_dir, device):
    require(sha256_file(OUT / 'selection_freeze.json') == freeze_hash, 'Frozen configuration changed.')
    cnn = frozen['cnn']
    final_cnn = final_train(dev_cache, y, np.arange(len(dev)), seed=42,
        model_size=cnn['model_size'], weighting=cnn['weighting'], epochs=cnn['epochs'],
        batch_size=CONFIG['batch_size'], device=device, output_dir=OUT / 'final_models' / cnn['model'])
    pd.DataFrame(final_cnn['history']).to_csv(OUT / 'final_cnn_training_history.csv', index=False)
    rf_spec = frozen['rf']['specification']
    rf_params = dict(frozen['rf']['parameters'])
    rf_params['max_depth'] = rf_spec['max_depth']
    rf_params['class_weight'] = rf_spec['class_weight']
    rf_params['random_state'] = 42
    rf = RandomForestClassifier(**rf_params)
    started = time.perf_counter()
    rf.fit(features, y)
    rf_fit_seconds = time.perf_counter() - started
    joblib.dump({'model': rf, 'feature_names': FEATURE_NAMES, 'class_order': CLASSES,
                 'scope': 'all 54355 development wafers; frozen final model', 'freeze_sha256': freeze_hash},
                OUT / 'final_models' / 'selected_rf.joblib', compress=3)
    # Both models are now frozen and fitted. Test features are first constructed here.
    require(sha256_file(OUT / 'selection_freeze.json') == freeze_hash, 'Frozen configuration changed before test.')
    first_test_processing_utc = utc_now()
    save_json('final_test_start.json', {'first_test_processing_utc': first_test_processing_utc,
        'freeze_sha256': freeze_hash,
        'final_cnn_checkpoint_sha256': sha256_file(final_cnn['checkpoint_path']),
        'final_rf_checkpoint_sha256': sha256_file(OUT / 'final_models' / 'selected_rf.joblib'),
        'models': [frozen['rf']['model'], cnn['model']], 'final_test_scored': False})
    test_y = test['label'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    started = time.perf_counter()
    test_features = np.vstack([extract_features(maps[int(row)]) for row in tqdm(test['source_row'], desc='Frozen RF: final test features')])
    rf_feature_seconds = time.perf_counter() - started
    require(test_features.shape == (len(test), len(FEATURE_NAMES)) and np.isfinite(test_features).all(), 'Invalid final RF features.')
    started = time.perf_counter()
    rf_prediction = rf.predict(test_features)
    rf_predict_seconds = time.perf_counter() - started
    del test_features, rf
    gc.collect()
    test_path = cache_dir / 'test_images.float32'
    started = time.perf_counter()
    test_cache = build_cache(maps, test['source_row'].to_numpy(), test_path, 'Frozen CNN: final test preprocessing')
    cnn_preprocess_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cnn_prediction = predict_batches(final_cnn['model'], test_cache, batch_size=CONFIG['batch_size'], device=device)
    cnn_predict_seconds = time.perf_counter() - started
    close_cache(test_cache, test_path)
    identical = test['map_sha256'].isin(set(dev['map_sha256'])).to_numpy()
    final_scores, final_classes, final_confusion, final_predictions = [], [], [], []
    for model, prediction, fit_seconds, predict_seconds, preprocess_seconds in [
        (frozen['rf']['model'], rf_prediction, rf_fit_seconds, rf_predict_seconds, rf_feature_seconds),
        (cnn['model'], cnn_prediction, float(final_cnn['fit_seconds']), cnn_predict_seconds, cnn_preprocess_seconds),
    ]:
        prediction = np.asarray(prediction, dtype=np.int64)
        require(prediction.shape == (len(test),) and np.isin(prediction, LABEL_IDS).all(), 'Invalid final predictions.')
        pv = test[['source_row', 'lot', 'label']].copy()
        pv['predicted_label'] = [CLASSES[i] for i in prediction]
        pv['model'], pv['seed'], pv['identical_map_in_development'] = model, 42, identical
        final_predictions.append(pv)
        for subset, keep in [('all', np.ones(len(test), dtype=bool)), ('nonidentical_to_development', ~identical)]:
            require(keep.any(), 'Predeclared sensitivity subset is empty.')
            identity = {'model': model, 'subset': subset, 'seed': 42}
            final_scores.append({**identity, 'rows': int(keep.sum()), 'lots': int(test.loc[keep, 'lot'].nunique()),
                'accuracy': float(accuracy_score(test_y[keep], prediction[keep])),
                'macro_f1': float(f1_score(test_y[keep], prediction[keep], labels=LABEL_IDS, average='macro', zero_division=0)),
                'fit_seconds': fit_seconds, 'full_test_predict_seconds': predict_seconds,
                'full_test_preprocess_seconds': preprocess_seconds, 'freeze_sha256': freeze_hash})
            pc, cm = metric_rows(test_y[keep], prediction[keep], test.loc[keep], identity)
            for item in pc:
                item['test_lots'] = item.pop('lots')
            final_classes.extend(pc)
            final_confusion.extend(cm)
    pd.DataFrame(final_scores).to_csv(OUT / 'final_test_scores.csv', index=False)
    pd.DataFrame(final_classes).to_csv(OUT / 'final_test_per_class.csv', index=False)
    pd.DataFrame(final_confusion).to_csv(OUT / 'final_test_confusion_matrices.csv', index=False)
    pd.concat(final_predictions, ignore_index=True).to_csv(OUT / 'final_test_predictions.csv.gz', index=False)
    save_json('final_test_audit.json', {'test_rows': len(test), 'test_lots': int(test['lot'].nunique()),
        'development_lots': int(dev['lot'].nunique()), 'shared_lots': 0,
        'test_rows_identical_to_development': int(identical.sum()),
        'sensitivity_rows': int((~identical).sum()), 'freeze_sha256': freeze_hash,
        'first_test_processing_utc': first_test_processing_utc, 'test_scoring_completed_utc': utc_now(),
        'interpretation': 'Identical map patterns can occur naturally, particularly normal maps. This sensitivity analysis does not replace the all-wafer primary evaluation.'})
    del final_cnn, test_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(final_scores)


def plot_results(summary, final_scores):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    eligible = summary.loc[summary['protocol'].eq('lot') & summary['fraction'].eq(1.)].sort_values('model')
    axes[0].bar(np.arange(len(eligible)), eligible['macro_f1_mean'], yerr=eligible['macro_f1_sd'], capsize=4)
    axes[0].set_xticks(np.arange(len(eligible)), eligible['model'], rotation=35, ha='right')
    axes[0].set_title('CNN configuration: lot validation'); axes[0].set_ylabel('Macro-F1'); axes[0].set_ylim(0, 1)
    curve = summary.loc[summary['protocol'].eq('lot') & summary['model'].eq('cnn_small_unweighted')].sort_values('fraction')
    axes[1].errorbar(curve['fraction'] * 100, curve['macro_f1_mean'], yerr=curve['macro_f1_sd'], marker='o', capsize=4)
    axes[1].set_title('CNN nested wafer learning curve'); axes[1].set_xlabel('Training wafers retained (%)'); axes[1].set_ylabel('Macro-F1')
    if not final_scores.empty:
        primary = final_scores.loc[final_scores['subset'].eq('all')]
        axes[2].bar(np.arange(len(primary)), primary['macro_f1'])
        axes[2].set_xticks(np.arange(len(primary)), primary['model'], rotation=35, ha='right')
        axes[2].set_title('Frozen models: independent-lot final test'); axes[2].set_ylabel('Macro-F1'); axes[2].set_ylim(0, 1)
    else:
        axes[2].text(.5, .5, 'Final test has not been evaluated', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_axis_off()
    fig.tight_layout(); fig.savefig(OUT / 'cnn_and_final_results.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def complete_provenance(environment, *, mode, final_test_evaluated, experiment_count, data_path=None, freeze_hash=None):
    save_json('run_status.json', {'status': 'completed', 'run_mode': mode,
        'synthetic_only': mode == 'smoke', 'validation_experiment_count': experiment_count,
        'final_model_count': 2 if final_test_evaluated else 0, 'final_test_evaluated': final_test_evaluated,
        'freeze_sha256': freeze_hash, 'completed_utc': utc_now()})
    provenance = {'input_path': None if data_path is None else str(data_path),
        'input_bytes': None if data_path is None else data_path.stat().st_size,
        'input_sha256': None if data_path is None else DATA_SHA256,
        'baseline_run_provenance_sha256': BASELINE_PROVENANCE_SHA256,
        'completed_utc': utc_now(), 'run_mode': mode, 'synthetic_only': mode == 'smoke',
        'validation_experiment_count': experiment_count, 'final_test_evaluated': final_test_evaluated,
        'freeze_sha256': freeze_hash, 'environment': environment,
        'artifact_sha256': {p.relative_to(OUT).as_posix(): sha256_file(p) for p in sorted(OUT.rglob('*'))
                           if p.is_file() and p.name != 'run_provenance.json'}}
    save_json('run_provenance.json', provenance)


def _main():
    global RUN_STARTED
    mode = os.environ.get('WM811K_RUN_MODE', 'full').lower()
    require(mode in {'smoke', 'validation', 'full'}, 'WM811K_RUN_MODE must be smoke, validation, or full.')
    OUT.mkdir(parents=True, exist_ok=True)
    require(not any(OUT.iterdir()), 'Output directory must be empty; set WM811K_CNN_OUTPUT for a new run.')
    save_json('run_status.json', {'status': 'incomplete', 'started_utc': utc_now(), 'run_mode': mode, 'final_test_evaluated': False})
    RUN_STARTED = True
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONFIG['device'] = device
    CONFIG['run_mode'] = mode
    CONFIG['torch_cpu_threads'] = 2
    CONFIG['experiment_matrix'] = experiment_matrix()
    (OUT / 'source').mkdir()
    for name in ['cnn_pipeline.py', 'wafer_cnn.py', 'wafer_features.py']:
        shutil.copyfile(Path(__file__).with_name(name), OUT / 'source' / name)
    CONFIG['source_sha256'] = {p.name: sha256_file(p) for p in sorted((OUT / 'source').glob('*.py'))}
    save_json('protocol.json', CONFIG)
    environment = {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'matplotlib', 'joblib', 'tqdm']}
    environment.update(python=platform.python_version(), platform=platform.platform(), device=device,
        gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    save_json('environment.json', environment)
    (OUT / 'requirements-runtime.txt').write_text('\n'.join(f'{name}=={environment[name]}' for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'matplotlib', 'joblib', 'tqdm']) + '\n', encoding='utf-8')
    if mode == 'smoke':
        print('SYNTHETIC SOFTWARE CHECK ONLY:', synthetic_smoke_test(OUT / 'synthetic_smoke'), flush=True)
        complete_provenance(environment, mode=mode, final_test_evaluated=False, experiment_count=0)
        return
    normal_check = preprocess_map(np.ones((4, 12), dtype=np.uint8))
    require(normal_check.shape == (2, 64, 64) and normal_check.dtype == np.float32 and not normal_check[1].any(),
            'Preprocessing contract check failed.')
    baseline, baseline_protocol, rf_selection = verify_baseline()
    dev, test, features, y, splits, learning = load_manifests(baseline)
    data_path = find_unique('LSWMD.pkl', 'WM811K_DATA')
    require(sha256_file(data_path) == DATA_SHA256, 'Input pickle differs from the audited dataset; refusing to unpickle.')
    raw = pd.read_pickle(data_path).reset_index(drop=True)
    require(len(raw) == 811457, 'Unexpected raw dataset size.')
    maps = raw['waferMap'].to_numpy()
    # Only development image rows are accessed before freezing. Test metadata are structural evidence.
    cache_parent = Path(os.environ.get('WM811K_CACHE', '/kaggle/temp'))
    cache_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='wm811k_cnn_', dir=cache_parent) as temp_dir:
        cache_dir = Path(temp_dir)
        dev_path = cache_dir / 'development_images.float32'
        dev_cache = build_cache(maps, dev['source_row'].to_numpy(), dev_path, 'Development CNN preprocessing')
        results, summary = validate_cnns(dev_cache, y, dev, splits, learning, device)
        frozen, freeze_hash = freeze_selection(results, summary, rf_selection, baseline_protocol, device)
        if mode == 'full':
            final_scores = final_evaluation(maps, dev_cache, dev, test, features, y, frozen, freeze_hash, cache_dir, device)
            save_json('run_status.json', {'status': 'final_test_scored', 'run_mode': mode,
                'validation_experiment_count': 18, 'final_test_evaluated': True,
                'freeze_sha256': freeze_hash, 'updated_utc': utc_now()})
        else:
            final_scores = pd.DataFrame()
        close_cache(dev_cache, dev_path)
        del dev_cache
    del raw, maps, features
    gc.collect()
    plot_results(summary, final_scores)
    (OUT / 'EXPERIMENT_NOTES.md').write_text(
        f'Run mode: {mode}. Final test evaluated: {mode == "full"}.\n\n' + '''# WM-811K: CNN and independent-lot final evaluation

Stage 2 supplied hash-verified development features, exact validation partitions and nested wafer subsets. This stage runs the fixed 18-experiment CNN matrix; it does not regenerate favorable splits.

CNN input comprises valid-die and failed-die channels, centered square padding and area resizing to 64x64. The original categorical map is not treated as a continuous grayscale photograph. Preprocessing has no fitted statistics; no augmentation, pretrained model or external data are used. Resizing can lose small defects. Valid-mask geometry may still encode product or batch differences.

Three full-data lot-validation configurations are eligible for CNN selection. Selection uses mean Macro-F1 across the same three stage-2 development seeds. Earliest maximum validation epoch is retained within a fixed eight-epoch budget. The initial budget was reduced from 12 to eight epochs, uniformly across all configurations and devices, before any real-data CNN fitting or validation scores, based only on the synthetic CPU benchmark in Kaggle notebook version 360950931; the 18-run matrix is unchanged. The timestamp and throughput measurements are recorded in protocol.json budget_rationale. Final training uses the selected three best epochs' median and all development wafers with seed 42. The RF configuration remains the existing stage-2 choice; both final models are refitted, rather than reusing a validation-subset checkpoint.

selection_freeze.json and its SHA256 are written before final-test feature construction, preprocessing, prediction or scoring. The final test consists of the original 118595 labeled wafers with no development lots. Both frozen models are reported without post-test selection. The all-wafer result is primary; excluding maps identical to any development map is a predeclared sensitivity analysis. Identical patterns are not automatically duplicate identities.

Primary metric is nine-class Macro-F1. Per-class recall, support and lot support are exported. Zero-support class recall is flagged undefined even though the metric convention uses zero_division=0. Accuracy is secondary because normal wafers dominate this test set. CNN class weighting uses training labels only. Imbalance, model-size and sample-size comparisons each keep the remaining relevant settings fixed.

Validation SDs across three overlapping holdouts are not confidence intervals. CNN validation scores are selected-epoch estimates and not independent final estimates. The random/lot gap is not a causal leakage estimate: class composition, batch characteristics and sample difficulty also differ. Nested learning curves change wafer counts and do not independently isolate lot diversity. Final metrics describe held-out lots in this public dataset, not verified future-time performance, physical fault causation or production deployment performance.

Raw and preprocessed image arrays are excluded from deliverables. Source, package versions, input and artifact hashes, exact source-row manifests, epoch histories, predictions, best validation checkpoints and both final models are retained. Reruns across devices/runtimes can differ numerically despite fixed seeds.
''', encoding='utf-8')
    require(sha256_file(OUT / 'selection_freeze.json') == freeze_hash, 'Freeze file was altered after scoring.')
    complete_provenance(environment, mode=mode, final_test_evaluated=mode == 'full',
        experiment_count=18, data_path=data_path, freeze_hash=freeze_hash)
    if mode == 'full':
        print('COMPLETED: 18 CNN validation runs and two frozen-model independent-lot final evaluations.', flush=True)
        print(final_scores[['model', 'subset', 'rows', 'macro_f1', 'accuracy']].to_string(index=False), flush=True)
    else:
        print('COMPLETED: 18 CNN validation runs; selection frozen; final test NOT evaluated.', flush=True)


def main():
    global RUN_STARTED
    RUN_STARTED = False
    try:
        _main()
    except Exception as error:
        if RUN_STARTED and OUT.is_dir() and (OUT / 'run_status.json').exists():
            previous = json.loads((OUT / 'run_status.json').read_text(encoding='utf-8'))
            previous.update(status='failed', error_type=type(error).__name__, error=str(error), failed_utc=utc_now())
            save_json('run_status.json', previous)
        raise


if __name__ == '__main__':
    main()
