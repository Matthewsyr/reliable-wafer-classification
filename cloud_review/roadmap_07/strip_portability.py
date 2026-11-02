"""Replay the Windows NumPy constants used by the pinned 36-angle strip search."""
import hashlib
import json
from pathlib import Path

GEOMETRY = '36228305e1cc824798bec07fb559d1f6bc4798339e776a981b0ecf23b8975b24'
LUT = 'efbcbb00f2e3f71a94fcd198b5a765d3fb233d61f56025aab8f943462f5ab121'


def windows_extractor(repo, lookup):
    src = (Path(repo)/'src/waferdefect/geometry.py').read_bytes()
    lut = Path(lookup).read_bytes()
    if hashlib.sha256(src).hexdigest() != GEOMETRY or hashlib.sha256(lut).hexdigest() != LUT:
        raise ValueError('Geometry implementation or angle-table provenance mismatch')
    angles = {degree:(float.fromhex(sine),float.fromhex(cosine))
              for degree,sine,cosine in json.loads(lut)['angle_sin_cos_hex']}
    if sorted(angles) != list(range(0,180,5)):
        raise ValueError('Unexpected angle grid')
    # Reuse the exact original algorithm with only the two environment-dependent
    # scalar trig calls substituted. The upstream module stays unmodified.
    lines = src.decode().replace('\r\n','\n')
    original = 'theta = np.deg2rad(angle)\n        sine, cosine = np.sin(theta), np.cos(theta)'
    if lines.count(original) != 1:
        raise ValueError('Original angle calculation changed')
    scope = {'_archived_angles':angles}
    exec(compile(lines.replace(original,'sine, cosine = _archived_angles[angle]'),'<pinned-strip-trig-replay>','exec'),scope)
    return scope['extract_linearity_features']
