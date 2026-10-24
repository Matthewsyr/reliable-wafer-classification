"""Merge existing development-only identity/index/split evidence, never maps.

Does not deserialize the source pickle. wafer_index is taken from the historical
metadata; valid_die_count is measured from maps by the historical scanner and
is deliberately not relabeled as the original source dieSize field.
"""
import argparse
import csv
import gzip
import io
from pathlib import Path
from waferdefect.io import read_csv, sha256_file, write_json
from waferdefect.split import assert_disjoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('metadata','index','split','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    metadata=read_csv(args.metadata); index={r['source_row']:r for r in read_csv(args.index)}
    split=read_csv(args.split); roles={r['source_row']:r for r in split}
    ids=[r['source_row'] for r in metadata]
    if len(ids)!=len(set(ids)) or set(ids)!=set(roles):raise ValueError('Identity coverage differs')
    rows=[]
    for row in metadata:
        sid=row['source_row']; raw=index[sid]; role=roles[sid]
        if row['source_split']!='train':raise ValueError('Only original development metadata allowed')
        for key in ('label','lot','source_split','map_sha256'):
            if row[key]!=raw[key] or row[key]!=role[key]:raise ValueError(f'Conflicting {key} for {sid}')
        # Identity from metadata; measured geometry/byte offsets from raw index;
        # final role and stress annotations from the frozen split manifest.
        rows.append({**raw,'wafer_index':row['wafer_index'],**{k:v for k,v in role.items() if k not in raw}})
    assert_disjoint(rows,[r['role'] for r in rows])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Never overwrite a published reference')
    with out.open('wb') as binary:
        with gzip.GzipFile(fileobj=binary,mode='wb',mtime=0,filename='') as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
                writer=csv.DictWriter(text,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out.with_name('development_reference_provenance.json'),{
        'rows':len(rows),'scope':'Original development metadata only; no wafer-map arrays or final-test rows',
        'sources':{key:{'artifact':Path(getattr(args,key)).name,'sha256':sha256_file(getattr(args,key))} for key in ('metadata','index','split')},
        'output_sha256':sha256_file(out),'wafer_index_origin':'Historical development_metadata.csv.gz, not synthesized',
        'valid_die_count_origin':'Measured map count from the historical raw-map scanner; original dieSize is not claimed',
        'lot_hash_cross_role_overlap':0,'duplicate_policy':'All distinct wafer identities retained; identical maps connected for splitting'})
    print(f'Wrote {len(rows)} verified development references: {out}')


if __name__=='__main__':main()
