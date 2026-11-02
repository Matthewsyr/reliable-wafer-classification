"""Hosted, inference-only retrospective replay of the already exposed original WM-811K test.

Inputs: six rf_{original71,linearity77}_sSEED.joblib, three cnn_sSEED.pt,
protected_final_test_manifest.csv.gz, tune_calibration_freeze.json. No model is fit.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd


PIN = '92d2002c2b76e2db2fec492bda72ea66605d5e35'
RAW = '1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'
TEST = 'e47a81e1d2d162961dea3f40819fa9edb9fea38b98b8a9e2eabfd8673c8c650e'
FREEZE = 'f95fde21e70b8bf0fc582c8953bedf8825994c45c47032a7debd3a90a25b4a54'
DEV = '4e6564b2b8cd090353f65b7389b9e80eba768f138faf9ef95e3e3130f8fcd879'
MANIFEST = 'c86c2c3d77762d6d8083545bbb3377b09c13eb09d7119c17426b933602d0a0a7'
SEEDS = (42, 123, 2026)
CLASSES = ('Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch','none')
RF = ('rf_original71', 'rf_linearity77')
CNN = 'cnn_bn_natural_7ep'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(4 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def checked(path, sha):
    check(digest(path) == sha, f'Input digest changed: {path.name}')
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def verify_map(a, ref):
    check(a.ndim == 2 and a.dtype == np.uint8 and min(a.shape) > 0, 'Unexpected pixel format')
    h = hashlib.sha256(np.asarray(a.shape, dtype='<i8').tobytes())
    h.update(np.ascontiguousarray(a).tobytes())
    check(h.hexdigest() == ref, 'Wafer image identity mismatch')


def softmax(z):
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def gate(p, beta):
    return 1 - softmax(np.log(np.maximum(p, 1e-12)) * beta)[:, -1] >= .5


def valid_prob(p, n):
    check(p.shape == (n, 9) and np.isfinite(p).all() and np.all(p >= 0) and
          np.allclose(p.sum(1), 1, rtol=0, atol=1e-10), 'Invalid probability vectors')
    return p


def load_models(repo, inp):
    import joblib
    import sklearn
    import torch
    from waferdefect import cnn

    check(sklearn.__version__ == '1.6.1', 'Install scikit-learn==1.6.1 before loading the pinned forests')
    forests = read_json(repo/'results/geometry/runtime_cost_summary.json')['training_records']
    cnns = read_json(repo/'results/geometry/cnn_reference_lineage.json')['references']
    rf_models, cnn_models, hashes = {}, {}, {}
    for r in forests:
        name, s = r['model'], r['seed']
        f = inp/f'{name}_s{s}.joblib'
        model = joblib.load(checked(f, r['weights_sha256']))
        check(np.array_equal(model.classes_, np.arange(9)) and model.n_features_in_ == r['features'],
              'Forest features or class order mismatch')
        model.n_jobs = 1  # Preserve tree summation order without worker oversubscription.
        rf_models[name, s] = model
        hashes[f.name] = r['weights_sha256']
    for r in cnns:
        s = r['seed']
        f = inp/f'cnn_s{s}.pt'
        model = cnn.build_model('medium')
        model.load_state_dict(torch.load(checked(f, r['weights_sha256']), map_location='cpu', weights_only=True))
        cnn_models[s] = model.eval()
        hashes[f.name] = r['weights_sha256']
    check(set(rf_models) == {(name,s) for name in RF for s in SEEDS} and set(cnn_models) == set(SEEDS),
          'Incomplete frozen model collection')
    return rf_models, cnn_models, hashes


def base_features(maps):
    from waferdefect.features import extract_features
    return np.stack([extract_features(a) for a in maps])


def plus_features(maps, base):
    from waferdefect.geometry import extract_linearity_features
    extra = np.stack([extract_linearity_features(a) for a in maps])
    return np.column_stack((base, extra))


def cnn_scores(models, maps):
    import torch
    from waferdefect.cnn import preprocess_map

    result = {s: np.empty((len(maps),9), dtype=np.float64) for s in SEEDS}
    with torch.inference_mode():
        for start in range(0, len(maps), 64):
            batch = maps[start:start+64]
            x = torch.from_numpy(np.stack([preprocess_map(a) for a in batch]))
            for s, model in models.items():
                logits = model(x).double().cpu().numpy()
                result[s][start:start+len(batch)] = softmax(logits)
    return result


def development_check(repo, archive, dev, frame, forests, cnns):
    # Prespecified calibration-free parity check, chosen solely by truth class and source order.
    sample = (dev[dev.role == 'tune'].sort_values('source_row').groupby('label',sort=False)
              .head(8).sort_values('source_row'))
    ids = sample.source_row.to_numpy()
    check(set(sample.label) == set(CLASSES), 'Missing class in fixed tune check')
    maps = frame.waferMap.iloc[ids].tolist()
    for a, ref in zip(maps, sample.map_sha256):
        verify_map(a, ref)
    x = base_features(maps)
    xx = {RF[0]: x, RF[1]: plus_features(maps,x)}
    cprobs = cnn_scores(cnns,maps)
    errors = {}
    for rec in archive['probability_files']:
        if rec['role'] != 'tune':
            continue
        name, s = rec['model'], rec['seed']
        f = repo/'results/probabilities'/rec['file']
        rows = pd.read_csv(checked(f,rec['sha256']), float_precision='round_trip').set_index('source_row').loc[ids]
        check(list(rows.columns) == ['truth',*CLASSES] and rows.truth.to_list() == sample.label.to_list(),
              'Development probability/identity alignment mismatch')
        p = cprobs[s] if name == CNN else forests[name,s].predict_proba(xx[name])
        ref = rows[list(CLASSES)].to_numpy()
        d = float(np.abs(p-ref).max())
        errors[f'{name}_s{s}'] = d
        check(d < (1e-4 if name == CNN else 1e-10) and np.array_equal(p.argmax(1),ref.argmax(1)),
              f'Frozen archive parity failed for {name} seed {s}: {d:g}')
    check(len(errors) == 9,'Missing archived reference scores')
    return dict(rows=len(ids), source_row_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                max_absolute_probability_difference=errors, argmax_identical=True)


def test_metadata(frame, ref, dev):
    from waferdefect.labels import canonical_label, scalar_text

    check(len(frame) == 811457 and len(ref) == 118595 and ref.source_row.is_unique and
          ref.source_split.eq('test').all() and ref.lot.nunique() == 4953 and
          len(dev) == 54355 and dev.source_row.is_unique and dev.lot.nunique() == 5809 and
          not set(ref.lot).intersection(dev.lot) and not set(ref.source_row).intersection(dev.source_row),
          'Protected development/test partitions do not match historical split')
    ids = ref.source_row.to_numpy()
    splitcol = 'trianTestLabel' if 'trianTestLabel' in frame else 'trainTestLabel'
    raw_test = frame.iloc[ids]
    check(all(canonical_label(raw) == expect for raw,expect in zip(raw_test.failureType,ref.label))
          and all(scalar_text(v).lower() in ('test','testing') for v in raw_test[splitcol])
          and all(scalar_text(raw) == expect for raw,expect in zip(raw_test.lotName,ref.lot))
          and np.array_equal(raw_test.waferIndex.to_numpy(dtype=np.int64),ref.wafer_index.to_numpy()),
          'Protected original test metadata do not match the verified raw source')
    maps = raw_test.waferMap.tolist()
    for a, refhash in zip(maps, ref.map_sha256):
        verify_map(a,refhash)
    dims = np.asarray([a.shape for a in maps])
    dup = ref.map_sha256.isin(set(dev.map_sha256)).to_numpy()
    check(int(dup.sum()) == 3158 and int(ref.label.eq('none').sum()) == 110701, 'Test support changed')
    meta = pd.DataFrame(dict(source_row=ids,truth=ref.label.to_numpy(),lot=ref.lot.to_numpy(),
                             height=dims[:,0],width=dims[:,1],duplicate_dev=dup.astype('int8')))
    return meta, maps


def score(maps, meta, models, cnns, beta, out):
    ids = meta.source_row.to_numpy()
    x = np.empty((len(maps),71),dtype=np.float32)
    for start in range(0,len(maps),4096):
        xb = base_features(maps[start:start+4096])
        x[start:start+len(xb)] = xb
    anchor = valid_prob(models[RF[0],42].predict_proba(x),len(x))
    rows = np.flatnonzero(gate(anchor,beta))
    np.savez_compressed(out/'anchor_rf71_s42.npz',source_row=ids,RF71=anchor)
    print(f'Fixed gate: {len(rows):,} of {len(maps):,}; computing candidate geometry',flush=True)
    subset = [maps[i] for i in rows]
    xx = np.empty((len(rows),77),dtype=np.float32)
    for start in range(0,len(rows),1024):
        batch = subset[start:start+1024]
        xx[start:start+len(batch)] = plus_features(batch,x[rows[start:start+len(batch)]])
        if start % 8192 == 0:
            print(f'Candidate geometry {start+len(batch):,}/{len(rows):,}',flush=True)
    print('Scoring three frozen CNNs, shared image preprocessing',flush=True)
    c = cnn_scores(cnns,subset)
    for s in SEEDS:
        p = anchor[rows] if s == 42 else models[RF[0],s].predict_proba(x[rows])
        np.savez_compressed(out/f'scores_s{s}.npz', source_row=ids[rows],
                            RF71=valid_prob(p,len(rows)), RF77=valid_prob(models[RF[1],s].predict_proba(xx),len(rows)),
                            CNN=valid_prob(c[s],len(rows)))
    return len(rows)


def run(a):
    import sklearn
    import torch
    from waferdefect.cnn import CLASSES as model_classes

    check(platform.system() == 'Linux' and Path('/content').is_dir() and
          any(k.startswith('COLAB_') for k in os.environ), 'Real wafer scoring requires hosted Google Colab')
    check(not a.out.exists(), 'Choose a fresh output directory')
    signal.signal(signal.SIGALRM, lambda *_: sys.exit('Stopped at 45-minute hosted replay limit'))
    signal.alarm(45*60)
    t0 = time.monotonic()
    head = subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
    check(head == PIN and tuple(model_classes) == CLASSES, 'Repository or model class contract changed')
    subprocess.run(['git','-C',str(a.repo),'diff','--quiet','HEAD','--','src/waferdefect','results/geometry','results/probabilities'],
                   check=True)
    paths = dict(raw=checked(a.raw,RAW),
                 protected=checked(a.inputs/'protected_final_test_manifest.csv.gz',TEST),
                 dev=checked(a.repo/'data-reference/development_reference.csv.gz',DEV),
                 freeze=checked(a.inputs/'tune_calibration_freeze.json',FREEZE),
                 archive=checked(a.repo/'results/probabilities/manifest.json',MANIFEST))
    freeze = read_json(paths['freeze'])
    beta = freeze['configurations']['rf_s42']['inverse_temperature']
    check(beta == 1.9147552052282766, 'Historical gate was modified')
    archive = read_json(paths['archive'])
    check(tuple(archive['class_order']) == tuple(archive['probability_columns']) == CLASSES, 'Class order changed')
    dev, ref = pd.read_csv(paths['dev']), pd.read_csv(paths['protected']).sort_values('source_row')
    torch.set_num_threads(min(2,os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    models, cnns, hashes = load_models(a.repo,a.inputs)
    # Load the monolithic original only after checking its complete raw hash. Never retain a full raster tensor.
    frame = pd.read_pickle(paths['raw'])
    parity = development_check(a.repo,archive,dev,frame,models,cnns)
    print('All nine frozen models reproduce the development check:',parity,flush=True)
    meta, maps = test_metadata(frame,ref,dev)
    del frame, dev, ref
    gc.collect()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as temp:
        stage = Path(temp)/'score'
        stage.mkdir()
        meta.to_csv(stage/'test_meta.csv.gz', index=False)
        n = score(maps,meta,models,cnns,beta,stage)
        hashes_out = {p.name:digest(p) for p in sorted(stage.iterdir())}
        receipt = dict(status='complete',scope='Retrospective original test; exposed for past diagnoses.',
                       raw_sha256=RAW, dev_sha256=DEV, test_manifest_sha256=TEST, archive_manifest_sha256=MANIFEST,
                       temperature_freeze_sha256=FREEZE, repo=PIN, checkpoint_sha256=hashes,
                       script_sha256=digest(Path(__file__)), parity=parity, class_order=CLASSES, rows=len(meta),
                       candidates=n, lots=int(meta.lot.nunique()), cross_split_lots=0, duplicate_dev=3158,
                       gate_beta=beta, threshold=.5, versions=dict(torch=str(torch.__version__),numpy=np.__version__,
                       sklearn=sklearn.__version__,pandas=pd.__version__), seconds=time.monotonic()-t0,outputs=hashes_out)
        (stage/'score_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
        stage.rename(a.out)
    print('Score replay complete:',n,'candidates,',len(meta),'test wafers',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('repo','raw','inputs','out'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--self-test',action='store_true')
    a = p.parse_args()
    if a.self_test:
        m = np.array([[0,1],[2,1]],dtype=np.uint8)
        h = hashlib.sha256(np.asarray(m.shape,dtype='<i8').tobytes()+m.tobytes()).hexdigest()
        verify_map(m,h)
        probs=np.eye(9)
        check(np.array_equal(gate(probs,1.9147552052282766),np.arange(9)!=8),'Synthetic gate')
        check(valid_prob(probs,9).shape==(9,9),'Synthetic probability contract')
        print('Synthetic hash, gate and probability contract passed.')
    else:
        check(all([a.repo,a.raw,a.inputs,a.out]),'Missing paths')
        sys.path.insert(0,str(a.repo/'src'))
        run(a)


if __name__ == '__main__':
    main()
