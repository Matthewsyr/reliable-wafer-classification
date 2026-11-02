"""Development-only feature and decision-path comparison. No original-test data are accessed."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import score_replay as s
from strip_portability import windows_extractor

GEOCACHE = '93039e3cbeb64fb3ebacc730e150edf4e3ee980fefd7fb09857fee261c6c8865'
BASECACHE = '61c98d4dc415687e20386cd180b48a3ab3f87099e54b31defbae6606329d745b'


def feature_delta(x, ref, names):
    abs_delta = np.abs(x.astype(np.float64)-ref)
    return {n: {'differing':int((abs_delta[:,j]!=0).sum()),'max_abs':float(abs_delta[:,j].max())}
            for j,n in enumerate(names) if abs_delta[:,j].max()>0}


def first_divergence(model, a, b):
    for i,t in enumerate(model.estimators_):
        if t.apply(a[None])[0] == t.apply(b[None])[0]:
            continue
        tr=t.tree_; n=0
        while tr.children_left[n] != tr.children_right[n]:
            f=int(tr.feature[n]); threshold=float(tr.threshold[n])
            ia=a[f]<=threshold; ib=b[f]<=threshold
            if ia!=ib:
                return dict(tree=i,node=int(n),feature=f,threshold=threshold,
                            saved=float(a[f]),fresh=float(b[f]))
            n=tr.children_left[n] if ia else tr.children_right[n]
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','raw','inputs','geometry','base','windows-lut','out'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--extra',type=int,default=0,help='Additional uniformly chosen tune rows, fixed seed 7')
    a=p.parse_args()
    sys.path.insert(0,str(a.repo/'src'))
    sys.path.insert(0,str(a.repo/'cloud_review/extension'))
    from train_imbalance import indexed_map
    from waferdefect import geometry
    from waferdefect.features import FEATURE_NAMES
    import os
    s.check(Path('/content').is_dir() and any(k.startswith('COLAB_') for k in os.environ),'Use hosted Colab')
    s.checked(a.raw,s.RAW)
    dr=s.checked(a.repo/'data-reference/development_reference.csv.gz',s.DEV)
    ref=pd.read_csv(dr)
    tune=ref[ref.role=='tune'].sort_values('source_row')
    rows=tune.groupby('label',sort=False).head(8)
    if a.extra:
        rows=pd.concat([rows,tune.sample(n=a.extra,random_state=7)]).drop_duplicates('source_row')
    rows=rows.sort_values('source_row')
    ids=rows.source_row.to_numpy()
    s.check(np.array_equal(np.sort(ref.development_index),np.arange(54355)), 'Invalid historical cache index')
    saved_g=np.load(s.checked(a.geometry,GEOCACHE),allow_pickle=False)[rows.development_index.to_numpy()]
    with np.load(s.checked(a.base,BASECACHE),allow_pickle=False) as z:
        pos=pd.Index(z['source_row']).get_indexer(ids)
        s.check(np.min(pos)>=0,'Invalid cache join')
        saved_b=z['X'][pos].astype(np.float32)
    with a.raw.open('rb') as h:
        maps=[indexed_map(h,r) for r in rows.to_dict('records')]
    b=s.base_features(maps)
    native=np.stack([geometry.extract_linearity_features(x) for x in maps])
    portable=windows_extractor(a.repo,a.windows_lut)
    fixed=np.stack([portable(x) for x in maps])
    model_dict,_,_=s.load_models(a.repo,a.inputs)
    archive=s.read_json(s.checked(a.repo/'results/probabilities/manifest.json',s.MANIFEST))
    geo_names=geometry.FEATURE_NAMES
    result=dict(n=len(rows),raw_sha256=s.RAW,dev_sha256=s.DEV,
                scope='Existing calibration/selection-development rows only',
                windows_cache_sha256=GEOCACHE,base_cache_sha256=BASECACHE,
                base=feature_delta(b,saved_b,FEATURE_NAMES),native=feature_delta(native,saved_g,geo_names),
                windows_angles=feature_delta(fixed,saved_g,geo_names),models={})
    original=np.column_stack((saved_b,saved_g))
    for rec in archive['probability_files']:
        name,seed=rec['model'],rec['seed']
        if rec['role']!='tune' or name not in s.RF:
            continue
        raw=pd.read_csv(s.checked(a.repo/'results/probabilities'/rec['file'],rec['sha256']),
                        float_precision='round_trip').set_index('source_row').loc[ids,list(s.CLASSES)].to_numpy()
        model=model_dict[name,seed]
        variants={'cached':saved_b,'fresh':b} if name==s.RF[0] else {
            'cached':original,'fresh_base_cached_geometry':np.column_stack((b,saved_g)),
            'cached_base_native_geometry':np.column_stack((saved_b,native)),
            'fresh':np.column_stack((b,native)),'fresh_base_windows_trig':np.column_stack((b,fixed))}
        results={}
        for variant,x in variants.items():
            probs=model.predict_proba(x)
            err=np.max(np.abs(probs-raw),axis=1)
            v=dict(max_error=float(err.max()),mismatching_rows=int((err>=1e-10).sum()),
                   changed_argmax=int((probs.argmax(1)!=raw.argmax(1)).sum()))
            if v['mismatching_rows'] and variant!='cached':
                ix=int(err.argmax())
                v.update(source_row=int(ids[ix]),first_branch=first_divergence(model,variants['cached'][ix],x[ix]))
            results[variant]=v
        result['models'][f'{name}_s{seed}']=results
    a.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=1),flush=True)


if __name__=='__main__':
    main()
