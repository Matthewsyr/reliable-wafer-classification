"""Run tiny REAL DEVELOPMENT CLI smoke checks from a verified raw-map index.

These outputs verify engineering execution only, never estimate model quality.
The original final-test maps are not read. Requires all supplied metadata rows
to be original-development rows, and an existing lot/hash-disjoint role split.
"""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import numpy as np
from waferdefect.data import ORIGINAL_DATA_SHA256, indexed_map
from waferdefect.io import read_csv,write_csv,write_json,read_json,sha256_file
from waferdefect.labels import CLASSES
from waferdefect.split import assert_disjoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--index',required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--split',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty smoke directory')
    out.mkdir(parents=True,exist_ok=True)
    raw_hash=sha256_file(args.raw)
    if raw_hash!=ORIGINAL_DATA_SHA256:raise ValueError('Wrong source file')
    metadata=read_csv(args.metadata)
    if any(r['source_split']!='train' for r in metadata):raise ValueError('Supply a development-only metadata manifest')
    role_by_id={r['source_row']:r['role'] for r in read_csv(args.split)}
    counts=Counter();chosen=[]
    targets={'train':6,'tune':2,'audit':2}
    for row in metadata:
        role=role_by_id[row['source_row']];key=(role,row['label'])
        if role in targets and counts[key]<targets[role]:chosen.append(row);counts[key]+=1
    if any(counts[role,label]!=n for role,n in targets.items() for label in CLASSES):raise ValueError('Insufficient development class/role coverage')
    assert_disjoint(chosen,[role_by_id[r['source_row']] for r in chosen])
    chosen_ids={r['source_row'] for r in chosen}
    index={r['source_row']:r for r in read_csv(args.index) if r['source_row'] in chosen_ids}
    maps={}
    with Path(args.raw).open('rb') as handle:
        for row in chosen:
            record=index[row['source_row']]
            if record['source_split']!='train' or record['map_sha256']!=row['map_sha256']:raise ValueError('Index does not match chosen development identity')
            maps[f"row_{row['source_row']}"]=indexed_map(record,handle)
    np.savez(out/'maps.npz',**maps)
    write_csv(out/'metadata.csv',chosen)
    write_csv(out/'split.csv',[{'source_row':r['source_row'],'role':role_by_id[r['source_row']]} for r in chosen])
    repository=Path(__file__).resolve().parents[1]
    commands=[['prepare','--maps-npz',str(out/'maps.npz'),'--metadata',str(out/'metadata.csv'),'--existing-split',str(out/'split.csv'),
               '--output',str(out/'prepared'),'--representation','both','--include-linearity','--smoke-run']]
    for kind in ('rf','cnn'):
        commands += [['train','--data',str(out/'prepared'),'--config',str(repository/'configs'/f'{kind}_smoke.json'),'--output',str(out/f'{kind}_run'),'--device',args.device],
                     ['evaluate','--run',str(out/f'{kind}_run'),'--data',str(out/'prepared'),'--output',str(out/f'{kind}_eval'),'--device',args.device]]
    for i,command in enumerate(commands):
        with (out/f'command_{i}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','waferdefect',*command],stdout=log,stderr=subprocess.STDOUT,check=True)
    evidence={'scope':'real_development_engineering_smoke_not_scientific_performance','rows':len(chosen),
              'class_role_counts':{f'{role}:{label}':count for (role,label),count in counts.items()},
              'original_source_sha256':raw_hash,'metadata_source_sha256':sha256_file(args.metadata),'split_source_sha256':sha256_file(args.split),
              'original_test_maps_read':False,'commands':commands,'all_commands_passed':True,
              'rf_fit_seconds':read_json(out/'rf_run/run.json')['fit_seconds'],'cnn_fit_seconds':read_json(out/'cnn_run/run.json')['fit_seconds'],
              'rf_evaluated_rows':read_json(out/'rf_eval/metrics.json')['metrics']['n'],
              'cnn_evaluated_rows':read_json(out/'cnn_eval/metrics.json')['metrics']['n']}
    write_json(out/'smoke_evidence.json',evidence)
    print(f'Real development CLI smoke passed: {len(chosen)} wafers; final test untouched.')


if __name__=='__main__':main()
