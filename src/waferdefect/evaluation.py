"""Fixed-nine-class metrics and explicit Scratch decision semantics."""
import numpy as np
from .labels import CLASSES, NORMAL, SCRATCH


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 9 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError("Expected finite normalized [N,9] probabilities in fixed class order")
    return p


def decide(probabilities, scratch_threshold=None):
    """Default argmax; optional Scratch-priority, else best non-Scratch class.

    The priority rule is a different nine-class system and must be re-evaluated
    in full. A rejected Scratch argmax falls back to the best OTHER class.
    """
    p = validate_probabilities(probabilities)
    if scratch_threshold is None:
        return p.argmax(1).astype(np.int64)
    if not 0 <= scratch_threshold <= 1:
        raise ValueError("Scratch probability threshold must be in [0,1]")
    other = p.copy()
    other[:, SCRATCH] = -np.inf
    return np.where(p[:, SCRATCH] >= scratch_threshold, SCRATCH, other.argmax(1)).astype(np.int64)


def classification_metrics(truth, predicted):
    y, p = np.asarray(truth), np.asarray(predicted)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or y.dtype.kind not in 'iu' or p.dtype.kind not in 'iu' or np.any((y<0)|(y>8)|(p<0)|(p>8)):
        raise ValueError("Equal nonempty integer class-id vectors in [0,8] required")
    cm = np.bincount(9*y+p, minlength=81).reshape(9,9)
    support, positives, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp,positives,out=np.zeros(9,dtype=float),where=positives>0)
    recall = np.divide(tp,support,out=np.zeros(9,dtype=float),where=support>0)
    f1 = np.divide(2*tp,support+positives,out=np.zeros(9,dtype=float),where=(support+positives)>0)
    normal = int(support[NORMAL])
    return {'n':len(y),'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f1.mean()),'class_order':list(CLASSES),
            'per_class':{name:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i]),'recall_defined':bool(support[i])} for i,name in enumerate(CLASSES)},
            'confusion_matrix':cm.tolist(),'scratch_precision':float(precision[SCRATCH]),'scratch_recall':float(recall[SCRATCH]),
            'normal_count':normal,'normal_any_defect_count':normal-int(tp[NORMAL]),'normal_to_scratch_count':int(cm[NORMAL,SCRATCH]),
            'normal_to_scratch_per_1000':1000*int(cm[NORMAL,SCRATCH])/normal if normal else None}


def scratch_pr(truth, probabilities, precision_floor=.8):
    """OVR curve/AP and validation-only threshold choice, including tied scores.

    Caller must supply a development validation partition, never final test.
    The precision floor is an illustrative assumption, not a factory cost rule.
    No qualifying threshold is represented as None, not an invented threshold.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve
    p = validate_probabilities(probabilities)
    y = np.asarray(truth) == SCRATCH
    if len(y) != len(p) or not y.any() or y.all() or not 0 < precision_floor <= 1:
        raise ValueError("PR selection needs positive and negative validation examples")
    precision, recall, thresholds = precision_recall_curve(y,p[:,SCRATCH])
    # Integer/Fraction arithmetic makes the precision-floor and tied-score
    # policy exact (for P80: 5*TP >= 4*predicted positives, positives > 0).
    from fractions import Fraction
    floor = Fraction(str(precision_floor))
    candidates = []
    for threshold in thresholds:
        predicted = p[:,SCRATCH] >= threshold
        positives, tp = int(predicted.sum()), int((predicted & y).sum())
        if positives and tp*floor.denominator >= positives*floor.numerator:
            candidates.append((tp, Fraction(tp,positives), float(threshold), positives))
    chosen = None
    if candidates:
        tp, ratio, threshold, positives = max(candidates, key=lambda r:(r[0],r[1],r[2]))
        chosen = {'threshold':threshold,'precision':float(ratio),'recall':tp/int(y.sum()),
                  'true_positive':tp,'predicted_positive':positives,'positive_support':int(y.sum())}
    return {'average_precision':float(average_precision_score(y,p[:,SCRATCH])), 'precision':precision.tolist(),
            'recall':recall.tolist(),'thresholds':thresholds.tolist(),'precision_floor':precision_floor,
            'selected':chosen,'assumption':'Illustrative precision floor, not a confirmed production requirement'}
