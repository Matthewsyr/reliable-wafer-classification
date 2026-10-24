"""Deterministic, untrained spatial features for categorical WM-811K wafer maps.

Input: a nonempty numeric H x W array, 0=background, 1=pass, 2=fail.
Output: 71 finite float32 values in FEATURE_NAMES order. No labels, lot IDs,
wafer indices, split information, or learned preprocessing enter this module.

Coordinates are pixel centres scaled separately by the map's width/height to
(-1, 1). Rates use valid dies as denominator; occupancy uses all grid cells.
Empty regions and absent-defect moments are encoded as zero. Regional support
features distinguish an empty region from a populated region with zero failures.
Raw H/W and die counts are deliberately excluded, but valid-mask geometry can
still proxy product or batch characteristics. This is not rotation invariant:
grid, angular and signed-coordinate features retain spatial orientation.
"""

from functools import lru_cache

import numpy as np


_GLOBAL_NAMES = (
    "defect_ratio", "valid_occupancy",
    "defect_centroid_x", "defect_centroid_y",
    "defect_variance_x", "defect_variance_y", "defect_covariance_xy",
    "defect_mean_radius", "defect_std_radius", "defect_centroid_radius",
    "valid_centroid_x", "valid_centroid_y",
    "valid_variance_x", "valid_variance_y", "valid_covariance_xy",
)
FEATURE_NAMES = (
    _GLOBAL_NAMES
    + tuple(f"grid_r{r}_c{c}_defect_rate" for r in range(4) for c in range(4))
    + tuple(f"grid_r{r}_c{c}_occupancy" for r in range(4) for c in range(4))
    + tuple(f"ring_{r}_defect_rate" for r in range(4))
    + tuple(f"ring_{r}_valid_share" for r in range(4))
    + tuple(f"sector_{s}_defect_rate" for s in range(8))
    + tuple(f"sector_{s}_valid_share" for s in range(8))
)


@lru_cache(maxsize=64)
def _geometry(height, width):
    """Cache only shape-based geometry, independent of every sample's values.

    Rings: radius [0,.25), [.25,.5), [.5,.75), [.75,infinity).
    Sectors: eight equal angular bins from -pi to pi; y increases downward.
    Grid bins divide the full rectangular map into 4 x 4 regions. For a map
    smaller than four cells in a dimension, some bins contain zero cells.
    """
    xs = (np.arange(width, dtype=np.float64) + 0.5) * 2.0 / width - 1.0
    ys = (np.arange(height, dtype=np.float64) + 0.5) * 2.0 / height - 1.0
    x, y = np.meshgrid(xs, ys)
    x, y = x.ravel(), y.ravel()
    radius = np.hypot(x, y)
    grid_x = np.minimum(((x + 1.0) * 2.0).astype(np.int64), 3)
    grid_y = np.minimum(((y + 1.0) * 2.0).astype(np.int64), 3)
    grid = grid_y * 4 + grid_x
    rings = np.searchsorted(np.array([0.25, 0.5, 0.75]), radius, side="right")
    angle = np.arctan2(y, x)
    sectors = np.minimum(((angle + np.pi) * (4.0 / np.pi)).astype(np.int64), 7)
    grid_cells = np.bincount(grid, minlength=16).astype(np.float64)
    result = (x, y, radius, grid, rings, sectors, grid_cells)
    for array in result:
        array.setflags(write=False)
    return result


def _safe_ratio(numerator, denominator):
    return np.divide(numerator, denominator,
                     out=np.zeros_like(numerator, dtype=np.float64),
                     where=denominator > 0)


def _moments(x, y, mask):
    """Population moments of selected die locations; zero for an empty set."""
    if not np.any(mask):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    xx, yy = x[mask], y[mask]
    mx, my = xx.mean(), yy.mean()
    dx, dy = xx - mx, yy - my
    return (mx, my, np.mean(dx * dx), np.mean(dy * dy), np.mean(dx * dy))


def extract_features(wafer_map):
    """Return the deterministic feature vector; reject malformed/empty maps.

    Integral-valued floating arrays are accepted; NaN, infinity, boolean,
    string/object and values outside {0,1,2} are rejected. A map containing
    only background is rejected because its defect rate is undefined.
    The caller should preserve source-row IDs separately when stacking vectors.
    """
    a = np.asarray(wafer_map)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("wafer_map must be a nonempty 2-D array")
    if a.dtype.kind not in "iuf" or np.any((a != 0) & (a != 1) & (a != 2)):
        raise ValueError("wafer_map must contain numeric categorical values 0, 1, 2")
    flat = a.ravel()
    valid, defect = flat != 0, flat == 2
    n_valid, n_defect = int(valid.sum()), int(defect.sum())
    if n_valid == 0:
        raise ValueError("wafer_map has no valid dies")
    x, y, radius, grid, rings, sectors, grid_cells = _geometry(*a.shape)
    dm, vm = _moments(x, y, defect), _moments(x, y, valid)
    dr = radius[defect]
    base = np.array([
        n_defect / n_valid, n_valid / a.size,
        *dm, dr.mean() if n_defect else 0.0,
        dr.std() if n_defect else 0.0, np.hypot(dm[0], dm[1]), *vm,
    ], dtype=np.float64)
    parts = [base]
    for region_ids, n_regions, denominator in (
        (grid, 16, grid_cells), (rings, 4, n_valid), (sectors, 8, n_valid)
    ):
        valid_counts = np.bincount(region_ids[valid], minlength=n_regions)
        defect_counts = np.bincount(region_ids[defect], minlength=n_regions)
        parts.extend((_safe_ratio(defect_counts, valid_counts),
                      _safe_ratio(valid_counts, denominator)))
    result = np.concatenate(parts).astype(np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("Feature contract violated")
    return result


def self_test():
    """Fast synthetic contract checks, callable in the training notebook."""
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert len(names) == len(FEATURE_NAMES) == 71
    normal = extract_features(np.ones((5, 7), dtype=np.uint8))
    assert normal.dtype == np.float32 and normal.shape == (71,)
    assert np.isfinite(normal).all()
    assert normal[names["defect_ratio"]] == 0
    assert normal[names["valid_occupancy"]] == 1
    assert all(normal[names[name]] == 0 for name in _GLOBAL_NAMES[2:10])

    # Background is not part of the failure-rate denominator.
    isolated = np.zeros((9, 9), dtype=np.uint8)
    isolated[4, 4] = 2
    f = extract_features(isolated)
    assert f[names["defect_ratio"]] == 1
    assert np.isclose(f[names["valid_occupancy"]], 1 / 81)
    assert f[names["defect_centroid_x"]] == f[names["defect_centroid_y"]] == 0
    assert f[names["ring_0_defect_rate"]] == 1
    assert f[names["ring_0_valid_share"]] == 1
    assert all(f[names[f"ring_{i}_defect_rate"]] == 0 for i in (1, 2, 3))

    # Rectangular maps use independent axes and preserve the sign of position.
    rectangular = np.ones((3, 9), dtype=np.uint8)
    rectangular[0, -1] = 2
    g = extract_features(rectangular)
    assert np.isclose(g[names["defect_centroid_x"]], 8 / 9)
    assert np.isclose(g[names["defect_centroid_y"]], -2 / 3)
    reflected = extract_features(rectangular[:, ::-1])
    assert np.isclose(g[names["defect_centroid_x"]], -reflected[names["defect_centroid_x"]])
    assert g[names["defect_ratio"]] == reflected[names["defect_ratio"]]
    assert np.isclose(sum(g[names[f"ring_{i}_valid_share"]] for i in range(4)), 1)
    assert np.isclose(sum(g[names[f"sector_{i}_valid_share"]] for i in range(8)), 1)

    # Tiny maps have empty grid regions; they must stay finite, with zero support.
    tiny = extract_features(np.array([[1, 2]], dtype=np.uint8))
    assert np.isfinite(tiny).all() and tiny[names["defect_ratio"]] == 0.5
    for bad in (np.zeros((2, 2)), np.array([[1, 3]]), np.array([[np.nan]]),
                np.array([1, 2]), np.empty((0, 2)), np.array([[True]])):
        try:
            extract_features(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed map was accepted")
    return {"features": len(FEATURE_NAMES), "synthetic_contract_checks": "passed"}


if __name__ == "__main__":
    print(self_test())
