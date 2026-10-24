"""Fixed, label-free linearity descriptors for categorical wafer maps.

The first descriptor is a nonlinear re-expression of covariance already in the
original 71 features, not new information. The other five describe a narrow
strip's multipoint support, background-adjusted concentration, span, continuity
and thickness; these are not determined by global first/second moments.
No parameters are fitted, no random proposals are used, and no labels enter.
"""
import numpy as np

FEATURE_NAMES = (
    'pca_anisotropy_existing_moments',
    'strip_failed_share',
    'strip_excess_failed_share',
    'strip_span_fraction',
    'strip_axial_occupancy',
    'strip_rms_halfwidth',
)
LINEARITY_FEATURE_NAMES = FEATURE_NAMES
ANGLE_DEGREES = tuple(range(0, 180, 5))
BIN_PHASES = (0.0, 0.5)
STRIP_WIDTH_DIES = 2.0
AXIAL_BIN_WIDTH_DIES = 2.0
MIN_FAILURES = 3
SPAN_QUANTILES = (0.05, 0.95)


def extract_linearity_features(wafer_map, return_details=False):
    """Return six finite float32 descriptors, with explicit degenerate cases.

    Coordinates for strip search are pixel centres in original cell units,
    centered on the map (same scale for both axes). For each fixed orientation
    theta and phase, q=-sin(theta)*x+cos(theta)*y; strips satisfy
    floor(q/2 + phase)=bin. Pick the strip containing the most failed dies.
    Exact ties retain the first angle, then first phase, then lowest bin.
    This is a coarse deterministic line-consensus search, not fitted RANSAC.

    PCA descriptor uses independently H/W-normalized coordinates, exactly as
    original defect_variance_x/y/covariance_xy; (lambda1-lambda2)/(lambda1+lambda2).
    Failed share = in-strip failures / all failures. Excess share subtracts
    in-strip valid dies / all valid dies (uniform-failure background reference).
    Span = 5th-to-95th percentile axial inlier span / full valid axial extent.
    Occupancy = occupied 2-die axial bins / bins between first and last inlier.
    RMS halfwidth = RMS orthogonal distance to inlier mean / strip halfwidth.
    PCA is zero for <2 failures; the five strip values are zero for <3 failures.
    Empty/degenerate spans give zero. No raw size, ID, or label is a feature.
    """
    a = np.asarray(wafer_map)
    if a.ndim != 2 or not a.size or a.dtype.kind not in 'iuf':
        raise ValueError('Expected nonempty numeric two-dimensional map')
    if np.any((a != 0) & (a != 1) & (a != 2)) or not np.any(a != 0):
        raise ValueError('Map must contain only 0/1/2 and at least one valid die')
    h, w = a.shape
    fy, fx = np.nonzero(a == 2)
    vy, vx = np.nonzero(a != 0)
    n = len(fx)
    result = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    details = {'failed_count':n, 'valid_count':len(vx), 'degenerate':n < MIN_FAILURES,
               'best_angle_degrees':None, 'strip_q_low':None, 'strip_q_high':None,
               'inlier_count':0, 'valid_strip_count':0}
    if n < 2:
        return (result, details) if return_details else result
    fx = fx.astype(np.float64) + 0.5 - w/2
    fy = fy.astype(np.float64) + 0.5 - h/2
    vx = vx.astype(np.float64) + 0.5 - w/2
    vy = vy.astype(np.float64) + 0.5 - h/2
    xx, yy = 2*fx/w, 2*fy/h
    dx, dy = xx-xx.mean(), yy-yy.mean()
    var_x, var_y, cov = np.mean(dx*dx), np.mean(dy*dy), np.mean(dx*dy)
    trace = var_x+var_y
    if trace > 0:
        result[0] = np.clip(np.sqrt((var_x-var_y)**2+4*cov*cov)/trace, 0, 1)
    if n < MIN_FAILURES:
        return (result, details) if return_details else result
    best_count = -1
    best = None
    for angle in ANGLE_DEGREES:
        theta = np.deg2rad(angle)
        sine, cosine = np.sin(theta), np.cos(theta)
        q = -sine*fx + cosine*fy
        for phase in BIN_PHASES:
            bins = np.floor(q/STRIP_WIDTH_DIES + phase).astype(np.int64)
            lo = int(bins.min())
            counts = np.bincount(bins-lo)
            index = int(counts.argmax())
            count = int(counts[index])
            if count > best_count:
                best_count = count
                best = (angle, sine, cosine, phase, index+lo, bins == index+lo)
    angle, sine, cosine, phase, bin_id, inlier = best
    valid_q = -sine*vx + cosine*vy
    valid_inlier = np.floor(valid_q/STRIP_WIDTH_DIES + phase).astype(np.int64) == bin_id
    failed_share = best_count/n
    result[1] = failed_share
    result[2] = failed_share - valid_inlier.mean()
    t = cosine*fx[inlier] + sine*fy[inlier]
    valid_t = cosine*vx + sine*vy
    extent = float(np.ptp(valid_t))
    if extent > 0:
        low, high = np.quantile(t, SPAN_QUANTILES, method='linear')
        result[3] = np.clip((high-low)/extent, 0, 1)
    axial_bins = np.floor(t/AXIAL_BIN_WIDTH_DIES).astype(np.int64)
    result[4] = len(np.unique(axial_bins))/(int(axial_bins.max()-axial_bins.min())+1)
    q = -sine*fx[inlier] + cosine*fy[inlier]
    result[5] = np.sqrt(np.mean((q-q.mean())**2))/(STRIP_WIDTH_DIES/2)
    details.update(best_angle_degrees=angle,
                   strip_q_low=(bin_id-phase)*STRIP_WIDTH_DIES,
                   strip_q_high=(bin_id+1-phase)*STRIP_WIDTH_DIES,
                   inlier_count=best_count,valid_strip_count=int(valid_inlier.sum()))
    if result.shape != (6,) or not np.isfinite(result).all():
        raise RuntimeError('Linearity feature contract violated')
    return (result, details) if return_details else result
