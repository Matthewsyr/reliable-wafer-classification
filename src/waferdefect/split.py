"""Split connected lot/content groups without silently discarding duplicates."""
from collections import defaultdict
import numpy as np


def connected_components(records):
    """Unite rows sharing lot OR exact map hash, including transitive links."""
    n = len(records)
    parent = list(range(n))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        aa, bb = root(a), root(b)
        if aa != bb:
            parent[max(aa, bb)] = min(aa, bb)

    for field in ("lot", "map_sha256"):
        first = {}
        for i, row in enumerate(records):
            value = row.get(field)
            if not value:
                raise ValueError(f"Missing grouping key {field}")
            if value in first:
                union(i, first[value])
            else:
                first[value] = i
    groups = defaultdict(list)
    for i in range(n):
        groups[root(i)].append(i)
    # Names depend on stable source-row identities, not input ordering.
    labels = [None] * n
    for rows in groups.values():
        name = f"component_{min(int(records[i]['source_row']) for i in rows)}"
        for i in rows:
            labels[i] = name
    return labels


def assert_disjoint(records, roles):
    if len(records) != len(roles) or len({str(r['source_row']) for r in records}) != len(records):
        raise ValueError("Role length or source-row uniqueness violated")
    collisions = {}
    for field in ("lot", "map_sha256"):
        seen = defaultdict(set)
        for row, role in zip(records, roles):
            seen[row[field]].add(role)
        collisions[field] = [key for key, values in seen.items() if len(values) > 1]
    if any(collisions.values()):
        raise ValueError(f"Split leaks lot or exact-map content: { {k: len(v) for k,v in collisions.items()} }")
    return {"lot_cross_role": 0, "map_hash_cross_role": 0, "source_rows_unique": True}


def grouped_split(records, seed=42, train_fraction=.6, tune_fraction=.2):
    if not 0 < train_fraction < 1 or not 0 < tune_fraction < 1 or train_fraction + tune_fraction >= 1:
        raise ValueError("Positive train/tune/audit proportions are required")
    components = connected_components(records)
    unique = np.asarray(sorted(set(components)))
    if len(unique) < 5:
        raise ValueError("At least five independent lot/hash components are needed")
    shuffled = np.random.default_rng(seed).permutation(unique)
    a, b = int(len(unique) * train_fraction), int(len(unique) * (train_fraction + tune_fraction))
    if not 0 < a < b < len(unique):
        raise ValueError("Fractions produce an empty split; declare another protocol")
    role = {component: "train" if i < a else "tune" if i < b else "audit" for i, component in enumerate(shuffled)}
    roles = [role[c] for c in components]
    assert_disjoint(records, roles)
    return [{"source_row": row["source_row"], "component": component, "role": r}
            for row, component, r in zip(records, components, roles)]


def align_external_split(records, manifest):
    if len(manifest) != len({str(row['source_row']) for row in manifest}):
        raise ValueError("External manifest has repeated source-row identifiers")
    by_id = {str(row['source_row']): row for row in manifest}
    if set(by_id) != {str(row['source_row']) for row in records}:
        raise ValueError("External manifest must cover exactly the prepared rows")
    components = connected_components(records)
    result = []
    for row, component in zip(records, components):
        saved = by_id[str(row['source_row'])]
        if saved['role'] not in {'train', 'tune', 'audit', 'validation'}:
            raise ValueError("Unsupported development role")
        for field in ('lot', 'map_sha256', 'label'):
            if field in saved and saved[field] != row[field]:
                raise ValueError(f"External split metadata mismatch: {field}")
        result.append({"source_row": row['source_row'], "component": component, "role": saved['role']})
    assert_disjoint(records, [row['role'] for row in result])
    return result

