"""Audited, development-first dataset preparation.

The original pickle is only deserialized after its published SHA-256 matches.
For bounded-memory access, pass the verified raw-index CSV plus metadata; this
reads uint8 payloads at recorded offsets and checks each map hash, never pickle
opcodes. Synthetic examples use a numeric-only NPZ and are labeled as fixtures.
"""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import numpy as np
from .features import extract_features, FEATURE_NAMES
from .io import read_csv, read_json, write_json, write_csv, sha256_file, now
from .labels import CLASSES, scalar_text, canonical_label, encode_labels
from .split import grouped_split, align_external_split

ORIGINAL_DATA_SHA256 = '1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'


def map_hash(wafer_map):
    array = np.asarray(wafer_map)
    extract_features(array)  # Enforce the historical categorical-map contract.
    digest = hashlib.sha256(np.asarray(array.shape,dtype='<i8').tobytes())
    digest.update(np.ascontiguousarray(array,dtype=np.uint8).tobytes())
    return digest.hexdigest()


def indexed_map(record, handle):
    height, width = int(record['height']), int(record['width'])
    size, offset = int(record['raw_data_nbytes']), int(record['raw_data_offset'])
    if height < 1 or width < 1 or size != height*width or offset < 0 or record['order'] not in {'C','F'}:
        raise ValueError('Invalid raw-index geometry or offsets')
    handle.seek(offset)
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError('Truncated indexed map')
    array = np.frombuffer(payload,dtype=np.uint8).reshape((height,width),order=record['order'])
    if map_hash(array) != record['map_sha256']:
        raise ValueError('Raw indexed map checksum mismatch')
    return array


def source_records(raw_pickle=None, metadata=None, maps_npz=None, raw_index=None):
    """Yield metadata and lazy map loaders; no final-test map read by default."""
    if maps_npz:
        if raw_pickle or raw_index or not metadata:
            raise ValueError('NPZ input requires metadata and excludes raw-pickle input')
        with np.load(maps_npz,allow_pickle=False) as archive:
            for i, row in enumerate(read_csv(metadata)):
                key = row.get('npz_key',f'row_{row["source_row"]}')
                yield row, lambda key=key: archive[key]
    elif raw_pickle:
        if sha256_file(raw_pickle) != ORIGINAL_DATA_SHA256:
            raise ValueError('Raw pickle hash differs from the audited public WM-811K file; refusing deserialization')
        if raw_index:
            if not metadata:
                raise ValueError('Bounded raw-index access needs a metadata manifest')
            index_rows = read_csv(raw_index)
            if len(index_rows) != len({r['source_row'] for r in index_rows}):
                raise ValueError('Repeated source row in raw index')
            index = {r['source_row']:r for r in index_rows}
            with Path(raw_pickle).open('rb') as handle:
                for row in read_csv(metadata):
                    entry = index.get(row['source_row'])
                    if entry is None:
                        raise ValueError('Metadata source row absent from raw index')
                    for field in ('label','lot','source_split','map_sha256'):
                        if field in row and field in entry and row[field] != entry[field]:
                            raise ValueError(f'Metadata/raw-index disagreement: {field}')
                    yield row, lambda entry=entry: indexed_map(entry,handle)
        else:
            import pandas as pd
            frame = pd.read_pickle(raw_pickle).reset_index(drop=True)
            split_name = 'trianTestLabel' if 'trianTestLabel' in frame else 'trainTestLabel'
            for i, row in frame.iterrows():
                record = {'source_row':i,'label':row['failureType'],'lot':row['lotName'],
                          'wafer_index':row['waferIndex'],'source_split':row[split_name], 'die_size':row['dieSize']}
                yield record, lambda i=i: frame.iloc[i]['waferMap']
    else:
        raise ValueError('Provide a trusted raw pickle or a numeric NPZ plus metadata')


def prepare(output, *, raw_pickle=None, metadata=None, maps_npz=None, raw_index=None,
            representation='both', existing_split=None, seed=42, partition='development', freeze=None,
            synthetic_fixture=False, limit=None, include_linearity=False, smoke_run=False):
    output = Path(output)
    if not raw_pickle and not maps_npz:
        raise ValueError('Provide a trusted raw pickle or a numeric NPZ plus metadata')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty preparation directory')
    output.mkdir(parents=True,exist_ok=True)
    source_path = Path(raw_pickle or maps_npz)
    source_digest = sha256_file(source_path)
    if partition not in {'development','final-test'}:
        raise ValueError('Unsupported partition')
    if partition == 'final-test':
        if not freeze or limit is not None:
            raise ValueError('Final-test preparation requires a prior freeze and forbids row truncation')
        frozen = read_json(freeze)
        if frozen.get('test_source_sha256') != source_digest or frozen.get('decision_frozen') is not True:
            raise ValueError('Final-test source differs from the frozen input')
    if limit is not None and (limit < 1 or existing_split):
        raise ValueError('Positive smoke-test limit cannot be combined with a full external split')
    if representation not in {'rf','cnn','both'}:
        raise ValueError('Representation must be rf/cnn/both')
    # Stage selected maps as numeric NPZ members so raw-pickle references are not
    # held a second time in memory. This temporary archive is removed after use.
    import zipfile
    staging = output/'maps.staging.npz'
    records, issues, exclusions = [], [], Counter()
    seen_ids, seen_identity = set(), set()
    audited = 0
    try:
        with zipfile.ZipFile(staging,'w',compression=zipfile.ZIP_STORED) as archive:
            for raw, loader in source_records(raw_pickle,metadata,maps_npz,raw_index):
                audited += 1
                source_id = raw.get('source_row')
                try:
                    label = canonical_label(raw.get('label'))
                    if label is None:
                        exclusions['missing_label_supervised_only'] += 1
                        continue
                    source_split = (scalar_text(raw.get('source_split')) or '').lower()
                    source_split = {'training':'train','testing':'test'}.get(source_split,source_split)
                    if source_split not in {'train','test'}:
                        raise ValueError('Missing or unknown original train/test membership')
                    wanted = 'train' if partition == 'development' else 'test'
                    if source_split != wanted:
                        exclusions['outside_selected_original_partition'] += 1
                        continue
                    sid = int(source_id)
                    if sid < 0 or str(source_id).strip() not in {str(sid), str(float(sid))}:
                        raise ValueError('Invalid stable source-row identity')
                    lot = scalar_text(raw.get('lot'))
                    if lot is None:
                        raise ValueError('Missing lot identifier')
                    wafer = float(raw['wafer_index'])
                    if not np.isfinite(wafer) or wafer < 1 or wafer != int(wafer):
                        raise ValueError('Invalid wafer index')
                    identity = (lot,int(wafer))
                    if sid in seen_ids or identity in seen_identity:
                        raise ValueError('Repeated sample identity (source_row or lot+wafer_index); manual audit required')
                    array = np.asarray(loader())
                    digest = map_hash(array)
                    if raw.get('map_sha256') and digest != raw['map_sha256']:
                        raise ValueError('Existing per-map checksum differs')
                    valid = int(np.count_nonzero(array))
                    if raw.get('die_size') is not None and float(raw['die_size']) != valid:
                        raise ValueError('Valid-die count differs from source dieSize')
                    seen_ids.add(sid); seen_identity.add(identity)
                    record = {'source_row':sid,'label':label,'lot':lot,'wafer_index':int(wafer),
                              'source_split':source_split,'map_sha256':digest,'height':array.shape[0],
                              'width':array.shape[1],'valid_die_count':valid,'failed_die_count':int((array==2).sum())}
                    with archive.open(f'row_{sid}.npy','w') as handle:
                        np.lib.format.write_array(handle,np.ascontiguousarray(array,dtype=np.uint8),allow_pickle=False)
                    records.append(record)
                    if limit is not None and len(records) >= limit:
                        exclusions['stopped_after_declared_smoke_limit'] = True
                        break
                except (ValueError,KeyError,TypeError,OverflowError) as exc:
                    issues.append({'source_row':str(source_id),'reason':str(exc)})
        groups = defaultdict(list)
        for row in records:
            groups[row['map_sha256']].append(row)
        duplicates = [{'map_sha256':digest,'rows':len(group),'labels':sorted({r['label'] for r in group}),
                       'lots':len({r['lot'] for r in group}),'source_rows':[r['source_row'] for r in group]}
                      for digest,group in groups.items() if len(group)>1]
        audit = {'schema_version':1,'created_utc':now(),'source_sha256':source_digest,'partition':partition,
                 'audit_scope':'selected original partition; no claim about unprepared final-test maps',
                 'source_rows_inspected':audited,'retained_rows':len(records),'exclusions':dict(exclusions),
                 'invalid_rows':issues,'duplicate_map_groups':len(duplicates),
                 'conflicting_label_groups':sum(len(row['labels'])>1 for row in duplicates),
                 'duplicate_policy':'Retain distinct wafer identities, including identical maps and conflicting labels; connect lot/hash groups for development splitting.',
                 'invalid_policy':'Fail preparation with an explicit issue list; never silently drop invalid labeled maps.',
                 'synthetic_fixture':bool(synthetic_fixture),'limited_smoke':limit is not None or bool(smoke_run)}
        write_json(output/'preparation_audit.json',audit)
        write_json(output/'duplicate_groups.json',duplicates)
        if issues or not records:
            raise ValueError(f'Preparation rejected {len(issues)} invalid records; inspect preparation_audit.json')
        write_csv(output/'metadata.csv',records)
        if partition == 'development':
            split = align_external_split(records,read_csv(existing_split)) if existing_split else grouped_split(records,seed)
        else:
            split = [{'source_row':r['source_row'],'component':'original_test_preserved','role':'final-test'} for r in records]
        write_csv(output/'split.csv',split)
        targets = encode_labels([r['label'] for r in records])
        np.save(output/'targets.npy',targets,allow_pickle=False)
        features = np.lib.format.open_memmap(output/'features.npy',mode='w+',dtype=np.float32,shape=(len(records),len(FEATURE_NAMES))) if representation in {'rf','both'} else None
        images = np.lib.format.open_memmap(output/'images.npy',mode='w+',dtype=np.float32,shape=(len(records),2,64,64)) if representation in {'cnn','both'} else None
        geometry = None
        if include_linearity:
            from .geometry import extract_linearity_features, LINEARITY_FEATURE_NAMES
            geometry = np.lib.format.open_memmap(output/'geometry.npy',mode='w+',dtype=np.float32,shape=(len(records),len(LINEARITY_FEATURE_NAMES)))
        if images is not None:
            from .cnn import preprocess_map
        with np.load(staging,allow_pickle=False) as archive:
            for i,row in enumerate(records):
                array = archive[f'row_{row["source_row"]}']
                if features is not None: features[i] = extract_features(array)
                if images is not None: images[i] = preprocess_map(array)
                if geometry is not None: geometry[i] = extract_linearity_features(array)
        if features is not None: features.flush(); del features
        if images is not None: images.flush(); del images
        if geometry is not None: geometry.flush(); del geometry
        artifacts = ['metadata.csv','split.csv','targets.npy','preparation_audit.json','duplicate_groups.json']
        artifacts += [name for name in ('features.npy','images.npy','geometry.npy') if (output/name).exists()]
        manifest = {'schema_version':1,'partition':partition,'class_order':list(CLASSES),'rows':len(records),
                    'source_sha256':source_digest,'representation':representation,'feature_names':list(FEATURE_NAMES),
                    'geometry_feature_names':list(LINEARITY_FEATURE_NAMES) if include_linearity else [],
                    'split_seed':seed if not existing_split else None,'existing_split_sha256':sha256_file(existing_split) if existing_split else None,
                    'synthetic_fixture':bool(synthetic_fixture),'limited_smoke':limit is not None or bool(smoke_run),
                    'preprocessing':'Historical binary valid/failed masks, centered square pad, area 64x64',
                    'artifacts_sha256':{name:sha256_file(output/name) for name in artifacts}}
        write_json(output/'dataset.json',manifest)
        return manifest
    finally:
        if staging.exists(): staging.unlink()


def load_prepared(root, expected_partition='development', verify=True):
    root = Path(root)
    manifest = read_json(root/'dataset.json')
    if manifest['partition'] != expected_partition or manifest['class_order'] != list(CLASSES):
        raise ValueError('Prepared partition or class order mismatch')
    if verify:
        for filename,digest in manifest['artifacts_sha256'].items():
            if Path(filename).name != filename or sha256_file(root/filename) != digest:
                raise ValueError(f'Prepared artifact changed: {filename}')
    records, split = read_csv(root/'metadata.csv'), read_csv(root/'split.csv')
    if [r['source_row'] for r in records] != [r['source_row'] for r in split]:
        raise ValueError('Prepared metadata/split row alignment mismatch')
    y = np.load(root/'targets.npy',allow_pickle=False)
    if not np.array_equal(y,encode_labels([r['label'] for r in records])):
        raise ValueError('Prepared target order differs from metadata')
    if expected_partition=='development':
        from .split import assert_disjoint
        assert_disjoint(records,[r['role'] for r in split])
        if any(r['source_split']!='train' for r in records):
            raise ValueError('Original test row reached a development dataset')
    return manifest,records,split,y
