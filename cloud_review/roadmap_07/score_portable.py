"""Run frozen scoring with the original stored 36-angle constants after the development parity gate."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

import score_replay as original
from diagnose_parity import BASECACHE, GEOCACHE
from strip_portability import LUT, windows_extractor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','raw','inputs','out','windows-lut','parity-report'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args()
    report=json.loads(a.parity_report.read_bytes())
    original.check(report['n']>=60 and report['dev_sha256']==original.DEV and
                   report['raw_sha256']==original.RAW and report['windows_cache_sha256']==GEOCACHE
                   and report['base_cache_sha256']==BASECACHE,'Invalid development-only parity receipt')
    for seed in original.SEEDS:
        for model,variant in [(original.RF[0],'fresh'),(original.RF[1],'fresh_base_windows_trig')]:
            for k in ('cached',variant):
                v=report['models'][f'{model}_s{seed}'][k]
                original.check(v['max_error']<1e-10 and v['mismatching_rows']==0 and v['changed_argmax']==0,
                               f'Development parity gate failed for {model} seed {seed} {k}')
    extractor=windows_extractor(a.repo,a.windows_lut)
    original.plus_features=lambda maps,base: np.column_stack((base,np.stack([extractor(m) for m in maps])))
    sys.path.insert(0,str(a.repo/'src'))
    out=a.out
    original.check(not out.exists(),'Use a fresh or unoccupied output directory')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        a.out=Path(tmp)/'score'
        original.run(a)
        receipt_path=a.out/'score_receipt.json'
        r=json.loads(receipt_path.read_bytes())
        r['numerical_compatibility']={'scope':'Only the original 36 strip-search angle constants are pinned.',
                                      'launcher_sha256':original.digest(Path(__file__)), 'lut_sha256':LUT,
                                      'helper_sha256':original.digest(Path(__file__).with_name('strip_portability.py')),
                                      'diagnostic_sha256':original.digest(Path(__file__).with_name('diagnose_parity.py')),
                                      'development_report_sha256':original.digest(a.parity_report),
                                      'development_probe_rows':report['n']}
        receipt_path.write_text(json.dumps(r,indent=2),encoding='utf8')
        a.out.rename(out)


if __name__=='__main__':
    main()
