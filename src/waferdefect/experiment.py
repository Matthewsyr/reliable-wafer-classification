"""Portable RF/CNN fits, frozen decisions, and explicit one-time test receipts."""
from pathlib import Path
import platform
import shutil
import time
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from .data import load_prepared, ORIGINAL_DATA_SHA256
from .evaluation import classification_metrics, decide, scratch_pr, validate_probabilities
from .io import read_json, write_json, write_csv, sha256_file, now
from .labels import CLASSES


def environment():
    libraries = {}
    for name in ('numpy','pandas','scikit-learn','joblib','torch'):
        try: libraries[name] = version(name)
        except PackageNotFoundError: libraries[name] = None
    return {'python':platform.python_version(),'platform':platform.platform(),'libraries':libraries}


def build_rf(config):
    """Historical RF settings; any imputation is fitted in the train-only pipeline."""
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    params = config.get('rf',{})
    allowed = {'n_estimators','max_depth','min_samples_leaf','max_features','class_weight','n_jobs'}
    if set(params)-allowed: raise ValueError('Unknown RF parameter')
    defaults = dict(n_estimators=200,max_depth=None,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',n_jobs=2)
    defaults.update(params)
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                     ('classifier',RandomForestClassifier(random_state=int(config['seed']),**defaults))])


def _features(data, config):
    base = np.load(Path(data)/'features.npy',mmap_mode='r',allow_pickle=False)
    if config.get('feature_set','spatial71') == 'spatial71': return base
    if config['feature_set'] == 'spatial71+linearity':
        extra = np.load(Path(data)/'geometry.npy',mmap_mode='r',allow_pickle=False)
        if len(extra) != len(base): raise ValueError('Geometry row mismatch')
        return np.concatenate((base,extra),axis=1)
    raise ValueError('Unsupported feature set')


def train(data, config_path, output, device='cpu'):
    data, output = Path(data), Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Each independent run needs a new empty directory')
    output.mkdir(parents=True,exist_ok=True)
    manifest,records,split,y = load_prepared(data)
    config = read_json(config_path)
    if config.get('kind') not in {'rf','cnn'}: raise ValueError('kind must be rf or cnn')
    fit_roles = config.get('fit_roles',['train'])
    if not fit_roles or not set(fit_roles).issubset({'train','tune','validation'}): raise ValueError('Audit/final-test fitting is forbidden')
    train_ids = np.asarray([i for i,r in enumerate(split) if r['role'] in fit_roles],dtype=np.int64)
    validation_role = config.get('validation_role','tune')
    if validation_role not in {'tune','validation'}: raise ValueError('Model selection can use tune/validation only')
    val_ids = np.asarray([i for i,r in enumerate(split) if r['role']==validation_role],dtype=np.int64)
    fixed = bool(config.get('fixed_epochs',False))
    if not len(train_ids) or set(y[train_ids]) != set(range(9)): raise ValueError('Training requires all nine classes; do not silently reroll the split')
    if not fixed and (not len(val_ids) or set(train_ids)&set(val_ids)): raise ValueError('Independent development train/validation rows required')
    start = time.perf_counter()
    write_json(output/'config.json',config)
    write_json(output/'run.json',{'status':'incomplete','original_test_used':False})
    if config['kind']=='rf':
        import joblib
        x = _features(data,config)
        model = build_rf(config)
        model.fit(x[train_ids],y[train_ids])
        artifact = output/'model.joblib'
        joblib.dump(model,artifact)
        fit_details = {'training_class_counts':np.bincount(y[train_ids],minlength=9).tolist(),
                       'imputer_statistics':model.named_steps['imputer'].statistics_.tolist(),
                       'parameters':model.named_steps['classifier'].get_params()}
    else:
        from .cnn import train_cnn, final_train
        x = np.load(data/'images.npy',mmap_mode='r',allow_pickle=False)
        options = dict(seed=int(config['seed']),model_size=config.get('model_size','medium'),weighting=config.get('weighting','none'),
                       epochs=int(config.get('epochs',8)),batch_size=int(config.get('batch_size',256)),device=device,output_dir=output/'cnn_fit')
        fit_details = final_train(x,y,train_ids,**options) if fixed else train_cnn(x,y,train_ids,val_ids,**options)
        artifact = output/'model.pt'
        shutil.copyfile(fit_details['checkpoint_path'],artifact)
        fit_details = {k:v for k,v in fit_details.items() if k not in {'model','predictions','train_predictions'}}
    seconds = time.perf_counter()-start
    write_json(output/'fit_details.json',fit_details)
    train_sources = [records[i]['source_row'] for i in train_ids]
    write_json(output/'training_source_rows.json',train_sources)
    run = {'status':'completed','kind':config['kind'],'created_utc':now(),'fit_seconds':seconds,'training_rows':len(train_ids),
           'fit_roles':fit_roles,'class_order':list(CLASSES),'dataset_sha256':sha256_file(data/'dataset.json'),
           'source_sha256':manifest['source_sha256'],'config_sha256':sha256_file(output/'config.json'),
           'model_file':artifact.name,'model_sha256':sha256_file(artifact),'fit_details_sha256':sha256_file(output/'fit_details.json'),
           'training_source_rows_sha256':sha256_file(output/'training_source_rows.json'),
           'original_test_used':False,'synthetic_fixture':manifest['synthetic_fixture'],'limited_smoke':manifest.get('limited_smoke',False),'environment':environment()}
    write_json(output/'run.json',run)
    return run


def verify_run(run_dir):
    run_dir = Path(run_dir)
    run = read_json(run_dir/'run.json')
    if run.get('status')!='completed' or run['class_order']!=list(CLASSES): raise ValueError('Incomplete or incompatible model run')
    for filename,key in ((run['model_file'],'model_sha256'),('config.json','config_sha256'),('fit_details.json','fit_details_sha256'),('training_source_rows.json','training_source_rows_sha256')):
        if Path(filename).name != filename or sha256_file(run_dir/filename)!=run[key]: raise ValueError(f'Run artifact changed: {filename}')
    return run,read_json(run_dir/'config.json')


def probabilities(run_dir,data,indices,device='cpu',batch_size=256):
    run_dir, data = Path(run_dir),Path(data)
    run,config = verify_run(run_dir)
    if run['kind']=='rf':
        import joblib
        # Loading only the locally trained, hash-bound artifact. Do not use an
        # untrusted joblib file: pickle-based model formats execute Python.
        model = joblib.load(run_dir/run['model_file'])
        raw = model.predict_proba(_features(data,config)[indices])
        p = np.zeros((len(indices),9),dtype=np.float64)
        p[:,model.classes_.astype(int)] = raw
    else:
        import torch
        from .cnn import load_checkpoint
        model,_ = load_checkpoint(run_dir/run['model_file'],device=device)
        x = np.load(data/'images.npy',mmap_mode='r',allow_pickle=False)
        chunks=[]
        with torch.inference_mode():
            for start in range(0,len(indices),batch_size):
                part = indices[start:start+batch_size]
                z = model(torch.from_numpy(np.ascontiguousarray(x[part])).to(device))
                chunks.append(torch.softmax(z,dim=1).cpu().numpy())
        p = np.concatenate(chunks)
    return validate_probabilities(p)


def freeze(run_dir,data,output,*,test_source_sha256,precision_floor=None,validation_role='tune',
           acknowledge_previous_test_exposure=False,device='cpu'):
    """Freeze model, input, and optional validation-only decision rule before test."""
    output,run_dir,data = Path(output),Path(run_dir),Path(data)
    if output.exists(): raise ValueError('Never overwrite an existing decision freeze')
    if len(test_source_sha256)!=64 or any(c not in '0123456789abcdef' for c in test_source_sha256): raise ValueError('Expected lowercase source SHA-256')
    if test_source_sha256==ORIGINAL_DATA_SHA256 and not acknowledge_previous_test_exposure:
        raise ValueError('This project already observed the original test; acknowledge historical reassessment, not independent confirmation')
    run,config = verify_run(run_dir)
    manifest,records,split,y = load_prepared(data)
    if sha256_file(data/'dataset.json')!=run['dataset_sha256']: raise ValueError('Model and development dataset differ')
    decision={'kind':'argmax','scratch_threshold':None}
    threshold_evidence=None
    if precision_floor is not None:
        if validation_role not in {'tune','validation'} or validation_role in run['fit_roles']: raise ValueError('Threshold selection must use validation rows unseen in fitting')
        ids=np.asarray([i for i,r in enumerate(split) if r['role']==validation_role],dtype=np.int64)
        threshold_evidence=scratch_pr(y[ids],probabilities(run_dir,data,ids,device),precision_floor)
        if threshold_evidence['selected'] is None: raise ValueError('No validation threshold satisfies the declared precision floor; no freeze written')
        decision={'kind':'scratch_priority','scratch_threshold':threshold_evidence['selected']['threshold']}
    frozen={'schema_version':1,'created_utc':now(),'decision_frozen':True,'class_order':list(CLASSES),
            'run_sha256':sha256_file(run_dir/'run.json'),'model_sha256':run['model_sha256'],'config_sha256':run['config_sha256'],
            'development_dataset_sha256':run['dataset_sha256'],'test_source_sha256':test_source_sha256,
            'decision':decision,'validation_role':validation_role if precision_floor is not None else None,'threshold_evidence':threshold_evidence,
            'previous_test_exposure_acknowledged':bool(acknowledge_previous_test_exposure),
            'scope':'historical_reassessment_not_independent' if test_source_sha256==ORIGINAL_DATA_SHA256 else 'user_declared_test_no_independence_claim'}
    write_json(output,frozen)
    return frozen


def evaluate(run_dir,data,output,*,role='tune',partition='development',freeze_path=None,once=False,device='cpu'):
    run_dir,data,output = Path(run_dir),Path(data),Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Evaluation output must be new and empty')
    run,config = verify_run(run_dir)
    frozen=read_json(freeze_path) if freeze_path else None
    if frozen is not None and (not frozen.get('decision_frozen') or frozen['run_sha256']!=sha256_file(run_dir/'run.json') or frozen['model_sha256']!=run['model_sha256'] or frozen['config_sha256']!=run['config_sha256']):
        raise ValueError('Model/configuration differs from the decision freeze')
    if partition=='final-test':
        if not once or frozen is None: raise ValueError('Final-test evaluation requires --once and a prior --freeze')
        if not frozen.get('decision_frozen') or frozen['run_sha256']!=sha256_file(run_dir/'run.json') or frozen['model_sha256']!=run['model_sha256'] or frozen['config_sha256']!=run['config_sha256']:
            raise ValueError('Model/configuration changed after freeze')
    manifest,records,split,y = load_prepared(data,expected_partition=partition)
    if partition=='development' and sha256_file(data/'dataset.json')!=run['dataset_sha256']: raise ValueError('Evaluation development rows differ from model dataset')
    if partition=='final-test' and manifest['source_sha256']!=frozen['test_source_sha256']: raise ValueError('Test source differs from freeze')
    if partition=='final-test' and manifest['source_sha256']==ORIGINAL_DATA_SHA256 and not frozen.get('previous_test_exposure_acknowledged'):
        raise ValueError('Original-test reassessment acknowledgement is missing')
    if partition=='final-test':
        # Exclusive create before any model inference. A failed attempt remains
        # consumed; this is an audit guard, not a security boundary against deletion.
        receipt_path=run_dir/'final_test_once_receipt.json'
        receipt={'status':'started','created_utc':now(),'freeze_sha256':sha256_file(freeze_path),
                 'test_dataset_sha256':sha256_file(data/'dataset.json'),'scope':frozen['scope']}
        with receipt_path.open('x',encoding='utf-8') as handle:
            import json
            json.dump(receipt,handle,indent=2)
        ids=np.arange(len(y),dtype=np.int64)
    else:
        if role not in {'train','tune','validation','audit'}: raise ValueError('Unknown development role')
        ids=np.asarray([i for i,r in enumerate(split) if r['role']==role],dtype=np.int64)
    if not len(ids): raise ValueError('No evaluation rows')
    start=time.perf_counter()
    p=probabilities(run_dir,data,ids,device)
    threshold=frozen['decision']['scratch_threshold'] if frozen else None
    prediction=decide(p,threshold)
    metrics=classification_metrics(y[ids],prediction)
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/'probabilities.npy',p,allow_pickle=False)
    rows=[{'source_row':records[i]['source_row'],'label':records[i]['label'],'predicted_label':CLASSES[int(prediction[j])],
           'scratch_probability':float(p[j,7])} for j,i in enumerate(ids)]
    write_csv(output/'predictions.csv',rows)
    result={'partition':partition,'role':role if partition=='development' else 'final-test','metrics':metrics,
            'decision':frozen['decision'] if frozen else {'kind':'argmax','scratch_threshold':None},
            'run_sha256':sha256_file(run_dir/'run.json'),'dataset_sha256':sha256_file(data/'dataset.json'),
            'predict_seconds':time.perf_counter()-start,'class_order':list(CLASSES),
            'synthetic_fixture':manifest['synthetic_fixture'],'limited_smoke':manifest.get('limited_smoke',False),'scope':frozen['scope'] if partition=='final-test' else 'development_evaluation',
            'probabilities_sha256':sha256_file(output/'probabilities.npy'),'predictions_sha256':sha256_file(output/'predictions.csv')}
    write_json(output/'metrics.json',result)
    if partition=='final-test':
        receipt.update(status='completed',result_sha256=sha256_file(output/'metrics.json'))
        write_json(receipt_path,receipt)
    return result
