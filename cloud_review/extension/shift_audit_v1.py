"""Read-only WM-811K distribution audit for Colab or Kaggle. No classifier training."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import lru_cache
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import tempfile

SHA = '1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'
EXPECTED = {'development': 54355, 'test': 118595, 'unlabeled': 638507}
CLASSES = {'center','donut','edge-loc','edge-ring','loc','near-full','random','scratch','none'}
GROUPS = tuple(EXPECTED)


def require_cloud():
    colab = Path('/content').is_dir() and any(k.startswith('COLAB_') for k in os.environ)
    kaggle = Path('/kaggle/working').is_dir() and any(k.startswith('KAGGLE_') for k in os.environ)
    if platform.system() != 'Linux' or not (colab or kaggle):
        raise RuntimeError('Run on the hosted Colab or Kaggle runtime; local execution is disabled.')
    import psutil
    available = psutil.virtual_memory().available
    for base, cap, used in [('/sys/fs/cgroup', 'memory.max', 'memory.current'),
                            ('/sys/fs/cgroup/memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes')]:
        try:
            remaining = int((Path(base)/cap).read_text()) - int((Path(base)/used).read_text())
            available = min(available, remaining)
        except (ValueError, FileNotFoundError):
            pass
    if available < 8 * 2**30:
        raise RuntimeError(f'Only {available/2**30:.1f} GiB available; use a cloud CPU runtime with >=8 GiB free.')
    return ('Colab' if colab else 'Kaggle'), round(available/2**30, 2)


def save_json(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def scalar(x):
    import numpy as np
    import pandas as pd
    while isinstance(x, (tuple, list, np.ndarray)):
        a = np.asarray(x, dtype=object)
        if not a.size: return None
        if a.size != 1: raise ValueError('Unexpected multivalue label or lot')
        x = a.reshape(-1)[0]
    if x is None or pd.isna(x): return None
    if isinstance(x, bytes): x = x.decode('utf-8')
    return str(x).strip() or None


@lru_cache(maxsize=32)
def masks(h, w):
    import numpy as np
    y, x = np.ogrid[:h, :w]
    r2 = ((y-(h-1)/2) / max(h/2, .5))**2 + ((x-(w-1)/2) / max(w/2, .5))**2
    return r2 <= .25, r2 >= .75**2


def index_maps(raw, db):
    import numpy as np
    import pandas as pd
    # The published file is a monolithic pickle, not a chunked table. Verify BEFORE unpickling.
    if sha_file(raw) != SHA: raise ValueError('Raw source SHA-256 mismatch; refusing to unpickle')
    df = pd.read_pickle(raw)
    if len(df) != sum(EXPECTED.values()): raise ValueError('Unexpected number of raw wafers')
    split = 'trianTestLabel' if 'trianTestLabel' in df else 'trainTestLabel'
    # Reuse the existing Series, do not copy or normalize all 811,457 wafer arrays at once.
    cols = ['waferMap','dieSize','lotName','waferIndex',split,'failureType']
    src = zip(*(df[c].array for c in cols))
    db.execute('CREATE TABLE m (id INTEGER PRIMARY KEY, g TEXT, label TEXT, lot TEXT, widx REAL, src TEXT, h INT, w INT, dies REAL, fail REAL, center REAL, edge REAL, digest BLOB, good INT)')
    checks, buf = Counter(), []
    for i,(a,die,lot,idx,s,label) in enumerate(src):
        s, label = (scalar(s) or '').lower(), (scalar(label) or '').lower()
        s = {'training':'train','testing':'test'}.get(s,s)
        if label and (label not in CLASSES or s not in ('train','test')):
            raise ValueError(f'Unexpected labeled record at source row {i}')
        g = 'unlabeled' if not label else ('development' if s=='train' else 'test')
        lot, a = scalar(lot), np.asarray(a)
        if not lot: checks[f'{g}:missing_lot'] += 1
        die = float(die) if pd.notna(die) and np.isfinite(float(die)) else None
        idx = float(idx) if pd.notna(idx) and np.isfinite(float(idx)) else None
        h = w = fail = center = edge = digest = None
        good = 0
        if a.ndim==2 and a.size and a.dtype.kind in 'iuf':
            h,w = map(int, a.shape)
            valid, bad = a!=0, a==2
            n, nf = int(valid.sum()), int(bad.sum())
            codes_ok = int((a==1).sum()) + nf == n
            if codes_ok:
                hh = hashlib.sha256(np.asarray((h,w),dtype='<i8').tobytes())
                hh.update(np.ascontiguousarray(a,dtype=np.uint8).tobytes())
                digest = hh.digest()
            else: checks[f'{g}:unexpected_pixel'] += 1
            if die != n: checks[f'{g}:dieSize_mismatch'] += 1
            if n == 0: checks[f'{g}:no_valid_die'] += 1
            if n and codes_ok:
                fail = nf/n
                c, e = masks(h,w)
                nc, ne = int(np.count_nonzero(valid & c)), int(np.count_nonzero(valid & e))
                center = int(np.count_nonzero(bad & c))/nc if nc else None
                edge = int(np.count_nonzero(bad & e))/ne if ne else None
                good = int(die == n)
        else: checks[f'{g}:bad_shape_or_type'] += 1
        buf.append((i,g,label,lot,idx,s,h,w,die,fail,center,edge,digest,good))
        if len(buf)==4096:
            db.executemany('INSERT INTO m VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',buf); buf.clear()
        if (i+1)%100000==0: print(f'Indexed {i+1:,}/{len(df):,} wafers',flush=True)
    db.executemany('INSERT INTO m VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',buf); db.commit()
    del src, df; gc.collect()
    return dict(checks)


def emit(db, path, sql, args=()):
    cur = db.execute(sql,args)
    with path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow([x[0] for x in cur.description]); wr.writerows(cur)


def analyze(db, out, export):
    import numpy as np
    db.execute('CREATE INDEX grp ON m(g)')
    db.execute('CREATE INDEX lot ON m(lot)')
    db.execute('CREATE INDEX dig ON m(digest)')
    one=lambda s,p=(): db.execute(s,p).fetchone()[0]
    counts=dict(db.execute('SELECT g,COUNT(*) FROM m GROUP BY g'))
    if counts != EXPECTED: raise ValueError(f'Group counts differ from audited release: {counts}')
    if one("SELECT COUNT(*) FROM (SELECT lot FROM m WHERE g='development' INTERSECT SELECT lot FROM m WHERE g='test')"):
        raise ValueError('Original labeled development/test lots overlap')
    emit(db,out/'group_counts.csv',"""SELECT g,count(*) rows,count(DISTINCT lot) lots,count(DISTINCT digest) pixel_hashes,
           sum(good) valid_map_and_dieSize,sum(fail=0) all_pass_wafer_rows, count(fail) fail_ratio_denominator
           FROM m GROUP BY g""")
    emit(db,out/'shape_counts.csv','SELECT g,h height,w width,COUNT(*) wafers,COUNT(DISTINCT lot) lots FROM m GROUP BY g,h,w ORDER BY g,wafers DESC,h,w')
    emit(db,out/'unlabeled_split_field.csv',"SELECT src raw_split_field,COUNT(*) wafers FROM m WHERE g='unlabeled' GROUP BY src")
    sums = []
    for group in GROUPS:
        for col in ('h','w','dies','widx','fail','center','edge','wafers_per_lot'):
            if col=='wafers_per_lot':
                rows=db.execute('SELECT COUNT(*) FROM m WHERE g=? AND lot IS NOT NULL GROUP BY lot',(group,))
            else:
                rows=db.execute(f'SELECT {col} FROM m WHERE g=? AND {col} IS NOT NULL',(group,))
            arr=np.fromiter((v[0] for v in rows), dtype=float)
            q=np.quantile(arr,[0,.1,.25,.5,.75,.9,.99,1]) if len(arr) else [None]*8
            sums.append((group,col,len(arr),*q))
    with (out/'quantiles.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(('group','variable','n','min','p10','p25','median','p75','p90','p99','max'));w.writerows(sums)

    # Fixed bins and total-variation distance; size distance uses the exact ordered (h,w).
    def tv(c,d):
        a,b=sum(c.values()),sum(d.values())
        return .5*sum(abs(c.get(k,0)/a-d.get(k,0)/b) for k in c.keys()|d.keys()) if a and b else None
    dists,coverage = {},{}
    for key,cats,filt in [
        ('height_width','h,w','h IS NOT NULL AND w IS NOT NULL'),
        ('failed_fraction_bins', 'CASE WHEN fail=0 THEN 0 WHEN fail<=.01 THEN 1 WHEN fail<=.05 THEN 2 WHEN fail<=.10 THEN 3 WHEN fail<=.25 THEN 4 WHEN fail<=.50 THEN 5 ELSE 6 END','fail IS NOT NULL')]:
        dist={}
        for group in GROUPS:
            rows=db.execute(f'SELECT {cats},COUNT(*) FROM m WHERE g=? AND {filt} GROUP BY {cats}',(group,))
            dist[group]={tuple(r[:-1]):r[-1] for r in rows}
        dists[key]={f'{x}_vs_{y}': tv(dist[x],dist[y]) for x,y in [('unlabeled','development'),('unlabeled','test'),('development','test')]}
        if key=='height_width':
            for x,y in [('unlabeled','development'),('unlabeled','test'),('test','development'),('test','unlabeled')]:
                novel=[k for k in dist[x] if k not in dist[y]]
                n=sum(dist[x][k] for k in novel);total=sum(dist[x].values())
                coverage[f'{x}_shapes_not_in_{y}']={'rows':n,'denominator_valid_shapes':total,'row_share':n/total,'unique_new_shapes':len(novel)}
    normal=one("SELECT COUNT(*) FROM m WHERE g='test' AND label='none' AND h IS NOT NULL AND w IS NOT NULL")
    notseen=one("""SELECT COUNT(*) FROM m t WHERE t.g='test' AND t.label='none' AND t.h IS NOT NULL AND t.w IS NOT NULL
        AND (t.h,t.w) NOT IN (SELECT DISTINCT h,w FROM m WHERE g='development' AND label='none' AND h IS NOT NULL AND w IS NOT NULL)""")
    coverage['test_none_shapes_not_in_development_none']={'rows':notseen,'denominator_valid_shapes':normal,'row_share':notseen/normal}
    overlaps={}
    for other in ('development','test','any_labeled'):
        clause="g!='unlabeled'" if other=='any_labeled' else f"g='{other}'"
        fields={}
        for col in ('lot','digest'):
            available, n, uniq=db.execute(f"""SELECT count({col}),
              count(CASE WHEN {col} IN (SELECT {col} FROM m WHERE {clause} AND {col} IS NOT NULL) THEN 1 END),
              count(DISTINCT CASE WHEN {col} IN (SELECT {col} FROM m WHERE {clause} AND {col} IS NOT NULL) THEN {col} END)
              FROM m WHERE g='unlabeled'""").fetchone()
            fields[col]={'unlabeled_rows_with_this_field':available,'unlabeled_rows_matching_reference':n,'distinct_shared_values':uniq}
        overlaps[other]=fields
    db.execute("""CREATE TEMP TABLE eligible AS SELECT id,digest,lot FROM m WHERE g='unlabeled' AND good=1 AND src IN ('','train') AND lot IS NOT NULL AND digest IS NOT NULL
        AND lot NOT IN (SELECT DISTINCT lot FROM m WHERE g!='unlabeled' AND lot IS NOT NULL)
        AND digest NOT IN (SELECT DISTINCT digest FROM m WHERE g!='unlabeled' AND digest IS NOT NULL)""")
    db.execute('CREATE INDEX eligible_digest ON eligible(digest)')
    # Keep one deterministic source row per identical map in this conservative pool.
    db.execute('CREATE TEMP TABLE safe AS SELECT MIN(id) AS id FROM eligible GROUP BY digest')
    pool={'eligible_before_internal_hash_dedup':one('SELECT COUNT(*) FROM eligible'),
          'unique_map_rows':one('SELECT COUNT(*) FROM safe'),
          'unique_lots_after_dedup':one('SELECT COUNT(DISTINCT lot) FROM safe JOIN m USING(id)'),
          'exported_manifest':bool(export)}
    all_u=set(db.execute("SELECT DISTINCT h,w FROM m WHERE g='unlabeled' AND h IS NOT NULL AND w IS NOT NULL"))
    safe_u=set(db.execute('SELECT DISTINCT h,w FROM safe JOIN m USING(id)'))
    for scope in ('all','none'):
        filt='' if scope=='all' else " AND label='none'"
        known=set(db.execute("SELECT DISTINCT h,w FROM m WHERE g='development' AND h IS NOT NULL AND w IS NOT NULL"+filt))
        novel={(h,w):n for h,w,n in db.execute("SELECT h,w,COUNT(*) FROM m WHERE g='test' AND h IS NOT NULL AND w IS NOT NULL"+filt+" GROUP BY h,w") if (h,w) not in known}
        denom=sum(novel.values())
        covered=lambda pool:sum(n for hw,n in novel.items() if hw in pool)
        coverage[f'novel_test_{scope}_versus_development_{scope}_covered_by_unlabeled']={
            'test_rows_at_novel_dimensions':denom,'covered_by_any_unlabeled':covered(all_u),'covered_by_conservative_pool':covered(safe_u),
            'fraction_covered_by_any_unlabeled':covered(all_u)/denom if denom else None,
            'fraction_covered_by_conservative_pool':covered(safe_u)/denom if denom else None}
    if export:
        with gzip.open(out/'safe_pretraining_pool.csv.gz','wt',newline='',encoding='utf-8',compresslevel=5) as f:
            wr=csv.writer(f);wr.writerow(['source_row','lot','map_sha256'])
            wr.writerows(db.execute('SELECT id,lot,lower(hex(digest)) FROM safe JOIN m USING(id) ORDER BY id'))
    return {'counts':counts,'unlabeled_overlap_with':overlaps,'shape_coverage':coverage,'total_variation':dists,'conservative_pretraining_pool':pool}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='New, empty output directory')
    p.add_argument('--temp',type=Path,help='Cloud local temporary directory, with >=2 GiB free')
    p.add_argument('--export-safe-pool',action='store_true',help='Also save eligible source IDs, lots and hashes')
    a=p.parse_args()
    host, avail=require_cloud()
    if not a.raw.is_file(): raise FileNotFoundError(a.raw)
    tmp=a.temp or Path(tempfile.gettempdir())
    if shutil.disk_usage(tmp).free < 2*2**30: raise RuntimeError('Need >=2 GiB free cloud temporary storage.')
    a.out.mkdir(parents=True,exist_ok=True)
    if any(a.out.iterdir()): raise FileExistsError('Choose a new empty output directory')
    status=a.out/'run_status.json'
    started=datetime.now(timezone.utc).isoformat()
    save_json(status,{'status':'running','started_utc':started})
    try:
        with tempfile.TemporaryDirectory(dir=tmp) as tmpdir:
            db=sqlite3.connect(Path(tmpdir)/'metadata.sqlite')
            try:
                db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-49152')
                checks=index_maps(a.raw,db)
                summary=analyze(db,a.out,a.export_safe_pool)
            finally: db.close()
        from importlib.metadata import version
        summary.update(raw_sha256=SHA,source_sha256=sha_file(Path(__file__)),cloud=host,
                       initial_available_gib=avail,python=platform.python_version(),quality_flags=checks,
                       versions={name:version(name) for name in ('numpy','pandas','psutil')})
        save_json(a.out/'shift_summary.json',summary)
        save_json(status,{'status':'completed','started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary,indent=2), flush=True)
    except Exception as e:
        save_json(status,{'status':'failed','started_utc':started,'error':str(e)})
        raise


if __name__=='__main__': main()
