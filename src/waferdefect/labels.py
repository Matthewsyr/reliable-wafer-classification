"""The historical nine-class order is an explicit serialization contract."""
import numpy as np

CLASSES = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none")
SCRATCH, NORMAL = 7, 8
_CANONICAL = {name.lower(): name for name in CLASSES}


def scalar_text(value):
    while isinstance(value, (tuple, list, np.ndarray)):
        array = np.asarray(value, dtype=object)
        if array.size == 0:
            return None
        if array.size != 1:
            raise ValueError("Metadata must be scalar or a singleton nested array")
        value = array.reshape(-1)[0]
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip() or None


def canonical_label(value):
    value = scalar_text(value)
    if value is None:
        return None  # Missing label, distinct from the literal class "none".
    if value.lower() not in _CANONICAL:
        raise ValueError(f"Unknown class: {value!r}")
    return _CANONICAL[value.lower()]


def encode_labels(values):
    result = [canonical_label(value) for value in values]
    if any(value is None for value in result):
        raise ValueError("Supervised encoding rejects missing labels")
    return np.asarray([CLASSES.index(value) for value in result], dtype=np.int64)


def decode_labels(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any((values < 0) | (values >= 9)):
        raise ValueError("Expected a one-dimensional integer vector in [0,8]")
    return [CLASSES[int(value)] for value in values]

