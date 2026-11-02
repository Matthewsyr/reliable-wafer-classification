"""Create private, read-only Snowflake SELECTs and compare exported aggregate results.

Hosted: python make_snowflake_exchange.py build --run EXCHANGE_DIRECTORY --out FRESH_DIR
Anywhere: python make_snowflake_exchange.py compare --expected EXPECTED.csv --actual RESULT.csv [...]
          python make_snowflake_exchange.py self-test
"""
import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import zlib
from collections import Counter, defaultdict


SEEDS = (42, 123, 2026)
LABELS = ('Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch','none')
BOUNDS = ((.01,'0-.01'),(.05,'.01-.05'),(.1,'.05-.10'),(.25,'.10-.25'),
          (.5,'.25-.50'),(.75,'.50-.75'),(float('inf'),'.75-1'))
TEMPLATE = Path(__file__).with_name('08_exchange_query.sql')
KEYS = ('seed','grain','side','size_band','true_class','probability_bin')
COUNTS = ('n','scratch_n','normal_n','other_defect_n')
SCORES = ('predicted_scratch_n','predicted_defect_n','mean_p_scratch','observed_scratch_rate',
          'scratch_overprediction','realized_net_benefit','predicted_net_benefit')
CHECKS = ('union_rows','union_ids','a_reviews','b_reviews','common_reviews','queues_checked','queue_overages')
FIELDS = (*KEYS,*COUNTS,*SCORES,*CHECKS)
LIMIT = 990000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    op = gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8-sig',newline='') as f:
        yield from csv.DictReader(f)


def decode_row(r):
    seed, sid, q, support = [int(r[k]) for k in ('seed','source_row','queue','train_normal_count')]
    lab = LABELS.index(r['truth'])
    a,b = int(r['in_a']),int(r['in_b'])
    ps,pa = float(r['p_scratch']),float(r['p_any'])
    require(seed in SEEDS and min(sid,q,support)>=0 and a in (0,1) and b in (0,1)
            and a+b>=1 and 0<=ps<=1 and -1e-14<=pa<=1 and ps<=pa+1e-13, 'Invalid selected-union record')
    return seed, (sid,q,support,lab,ps,pa,a,b)


def collect(rows):
    ledgers=defaultdict(list)
    for row in rows:
        seed,vals=decode_row(row)
        ledgers[seed].append(vals)
    for seed,lst in ledgers.items():
        require(len({r[0] for r in lst})==len(lst),f'Duplicate identity in input seed {seed}')
    return ledgers


def statistics(ledgers):
    groups=defaultdict(list)
    audits={}
    for seed,lst in ledgers.items():
        aq,bq=Counter(),Counter()
        for sid,q,support,lab,ps,pa,a,b in lst:
            aq[q]+=a
            bq[q]+=b
            if a==b:
                continue
            side='A_only' if a else 'B_only'
            band='0' if support==0 else '1-24' if support<25 else '25+'
            pb=next(name for top,name in BOUNDS if ps<top)
            for grain,size,label,bin_ in [('total','all','all','all'),('size',band,'all','all'),
                                         ('class',band,LABELS[lab],'all'),('reliability',band,'all',pb)]:
                groups[seed,grain,side,size,label,bin_].append((lab,ps,pa))
        common=sum(a*b for *_,a,b in lst)
        over=sum(q<0 or q>49 or max(aq[q],bq[q])>(45 if q==49 else 200) for q in aq)
        audits[seed]=(len(lst),len({r[0] for r in lst}),sum(aq.values()),sum(bq.values()),common,len(aq),over)
    result=[]
    for k,values in sorted(groups.items()):
        n=len(values)
        sc=sum(lab==7 for lab,_,_ in values)
        normal=sum(lab==8 for lab,_,_ in values)
        sp=math.fsum(ps for _,ps,_ in values)
        ap=math.fsum(pa for _,_,pa in values)
        scores=(sp,ap,sp/n,sc/n,(sp-sc)/n,.9*(n-normal+4*sc)-.1*n,.9*(ap+4*sp)-.1*n)
        result.append(dict(zip(FIELDS,(*k,n,sc,normal,n-normal-sc,*scores,*audits[k[0]]))))
    return result,audits


def sql_text(parts):
    payload=',\n'.join(f"({seed},'{chunk}')" for seed,chunk in parts)
    labels=' '.join(f"WHEN {i} THEN '{lab}'" for i,lab in enumerate(LABELS))
    source=f"""    SELECT chunks.seed::INTEGER AS seed, f.value[0]::INTEGER AS source_row,
           f.value[1]::INTEGER AS queue, CASE f.value[3]::INTEGER {labels} END AS truth,
           f.value[2]::INTEGER AS train_normal_count, f.value[4]::DOUBLE AS p_scratch,
           f.value[5]::DOUBLE AS p_any, f.value[6]::INTEGER AS in_a, f.value[7]::INTEGER AS in_b
    FROM (SELECT column1 AS seed,column2 AS payload FROM VALUES
{payload}
         ) chunks, LATERAL FLATTEN(INPUT=>PARSE_JSON(
             DECOMPRESS_STRING(TO_BINARY(chunks.payload,'BASE64'),'ZLIB'))) f"""
    t=TEMPLATE.read_text(encoding='utf8')
    begin=t.index('/* BEGIN INPUT */')
    end=t.index('/* END INPUT */')+len('/* END INPUT */')
    return t[:begin]+source+t[end:]


def chunks(ledgers):
    parts=[]
    for seed,lst in sorted(ledgers.items()):
        for j in range(0,len(lst),400):
            raw=json.dumps(lst[j:j+400],separators=(',',':'),allow_nan=False).encode('utf8')
            b64=base64.b64encode(zlib.compress(raw,9)).decode('ascii')
            require(json.loads(zlib.decompress(base64.b64decode(b64)))==[list(x) for x in lst[j:j+400]],
                    'Private packed rows did not round-trip')
            parts.append((seed,b64))
    return parts


def verify_cloud_refs(stats,run):
    cells=list(load_csv(run/'public/exchange_sql_cells.csv'))
    expected={(int(r['seed']),r['side'],r['size_support'],r['truth']):r for r in cells}
    require(len(expected)==len(cells),'Duplicate aggregated class/size reference')
    reference_keys=set()
    for r in stats:
        if r['grain']!='class':
            continue
        k=(r['seed'],r['side'],r['size_band'],r['true_class'])
        ref=expected[k]
        require(int(ref['n'])==r['n'] and
                all(math.isclose(float(ref[c]),r[d],abs_tol=1e-8,rel_tol=1e-11) for c,d in
                    [('sum_p_scratch','predicted_scratch_n'),('sum_p_any','predicted_defect_n')]),
                'Selected-union rows disagree with independent exclusive-ID grouping')
        reference_keys.add(k)
    require(reference_keys=={k for k,v in expected.items() if int(v['n'])>0},'Class-size keys disagree')


def build(run,out):
    require(platform.system()=='Linux' and Path('/content').is_dir() and any(k.startswith('COLAB_') for k in os.environ),
            'Build wafer-level SQL only inside the hosted Colab runtime')
    data=run/'private/selected_union.csv.gz'
    ledgers=collect(load_csv(data))
    require(set(ledgers)==set(SEEDS),'Missing seeds')
    stats,audits=statistics(ledgers)
    for seed, audit in audits.items():
        require(audit[2:4]==(9845,9845) and audit[-2:]==(50,0), f'07 queue mismatch for {seed}')
    verify_cloud_refs(stats,run)
    out.mkdir(parents=True,exist_ok=False)
    exp=out/'expected_snowflake_exchange.csv'
    with exp.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(stats)
    pp=chunks(ledgers)
    sql=sql_text(pp)
    bundles={'all':sql} if len(sql.encode('utf8'))<=LIMIT else {
        f's{s}':sql_text([p for p in pp if p[0]==s]) for s in SEEDS}
    scripts={}
    for name,query in bundles.items():
        require(len(query.encode('utf8'))<=LIMIT,'A seed exceeds the maximum private worksheet size')
        path=out/f'08_private_exchange_{name}.sql'
        path.write_bytes(query.encode('utf8'))
        scripts[path.name]={'bytes':path.stat().st_size,'sha256':sha(path)}
    receipt={'status':'complete','input_union_sha256':sha(data),'independent_class_cells_sha256':sha(run/'public/exchange_sql_cells.csv'),
             'template_sha256':sha(TEMPLATE),'generator_sha256':sha(Path(__file__)),
             'expected_csv_sha256':sha(exp),'expected_groups':len(stats),'sql':scripts}
    (out/'snowflake_input_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt),flush=True)


def canonical_rows(paths):
    table={}
    for path in paths:
        for raw in load_csv(path):
            r={k.lower():v for k,v in raw.items()}
            require(set(r)==set(FIELDS),'Wrong Snowflake export schema')
            k=tuple(r[c] for c in KEYS)
            require(k not in table,'Duplicate output key across files')
            table[k]=r
    return table


def compare(expected,actual):
    ref=canonical_rows([expected]);got=canonical_rows(actual)
    require(ref and ref.keys()==got.keys(),'Missing or extra SQL result groups')
    biggest={k:0. for k in SCORES}
    for k,r in ref.items():
        for c in COUNTS+CHECKS:
            require(int(got[k][c])==int(r[c]),f'Count mismatch {k}, {c}')
        for c in SCORES:
            observed,wanted=float(got[k][c]),float(r[c])
            require(math.isclose(observed,wanted,rel_tol=1e-10,abs_tol=1e-8),f'Score mismatch {k}, {c}')
            biggest[c]=max(biggest[c],abs(observed-wanted))
    return {'status':'pass','groups':len(ref),'checked_fields':len(FIELDS),'max_abs_differences':biggest,
            'result_sha256':[sha(p) for p in actual],'expected_sha256':sha(expected)}


def self_test():
    ledgers={42:[(1,0,0,7,.76,.95,1,0),(2,0,0,8,.25,.83,0,1),(3,0,25,1,.22,.9,1,1)]}
    stats,audit=statistics(ledgers)
    assert audit[42]==(3,3,2,2,1,1,0)
    assert next(r for r in stats if r['side']=='B_only' and r['grain']=='reliability')['probability_bin']=='.25-.50'
    assert sum(r['n'] for r in stats if r['grain']=='total')==2
    assert sum(r['realized_net_benefit'] for r in stats if r['grain']=='total' and r['side']=='A_only')==4.4
    pp=chunks(ledgers);sql=sql_text(pp)
    assert len(pp)==1 and 'WM08_SELECTED_UNION\n' not in sql
    assert ", WHEN" not in sql and "WHEN 0 THEN 'Center' WHEN 1 THEN 'Donut'" in sql
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'expected.csv'
        with path.open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(stats)
        assert compare(path,[path])['groups']==8
    print('Synthetic union, exclusions, bins, net losses, packet and result checks passed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    b=sub.add_parser('build'); b.add_argument('--run',type=Path,required=True);b.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('compare');c.add_argument('--expected',type=Path,required=True);c.add_argument('--actual',nargs='+',type=Path,required=True)
    c.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command=='self-test':
        self_test()
    elif a.command=='build':
        build(a.run,a.out)
    else:
        result=compare(a.expected,a.actual)
        if a.receipt:
            a.receipt.write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result))


if __name__=='__main__':
    main()
