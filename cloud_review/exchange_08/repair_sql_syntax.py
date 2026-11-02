"""Repair the CASE clauses of the three previously packed 08 worksheets byte-for-byte."""
import argparse
import hashlib
import json
from pathlib import Path

from make_snowflake_exchange import LABELS

EXPECTED = {
    '08_private_exchange_s42.sql':'05525d0a799edafc2c8236ca136c8bbb2288e7f98983f3ded51ae653a04fca12',
    '08_private_exchange_s123.sql':'dc5b15c3031d9ed1be50f3652bfa669f6d9a27769ae9279389a3e2f0929df596',
    '08_private_exchange_s2026.sql':'b3616007489acd691ed08b2e6702073e248cb9fd8a1e6ceb1d735a5c9966eadc',
}


def repair(source,out):
    prefix=b'CASE f.value[3]::INTEGER '
    clauses=[f"WHEN {i} THEN '{s}'".encode() for i,s in enumerate(LABELS)]
    old=prefix+b', '.join(clauses)+b' END AS truth'
    new=prefix+b' '.join(clauses)+b' END AS truth'
    out.mkdir(parents=True,exist_ok=False)
    files={}
    for n,digest in EXPECTED.items():
        raw=(source/n).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest and raw.count(old)==1
        fixed=raw.replace(old,new,1)
        assert fixed.replace(new,old,1)==raw and len(fixed)==len(raw)-8
        p=out/(Path(n).stem+'_v2.sql')
        p.write_bytes(fixed)
        files[p.name]={'old_sha256':digest,'sha256':hashlib.sha256(fixed).hexdigest(),'bytes':len(fixed)}
    (out/'syntax_repair_receipt.json').write_text(json.dumps(files,indent=2),encoding='utf8')
    print(json.dumps(files))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();repair(a.source,a.out)
