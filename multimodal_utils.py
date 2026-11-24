"""Small, auditable NumPy metrics for a fixed multiclass experiment.

Class columns always follow the notebook's declared class order. Argmax ties
select the lowest class index. Softmax scores are not assumed to be calibrated
probabilities of correctness, even though each row sums to one.
"""

from collections.abc import Mapping
from math import ceil

import numpy as np


def _matrix(values, name):
    """Convert numeric input and reject malformed or non-finite matrices."""
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ValueError(f"{name} must have nonempty shape (examples, classes)")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _probabilities(probabilities):
    probabilities = _matrix(probabilities, "probabilities")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    # A small tolerance accommodates float32 softmax output without silently
    # normalizing arbitrary scores into a different experiment.
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    return probabilities


def _labels(y_true, n_examples, n_classes):
    labels = np.asarray(y_true)
    if labels.ndim != 1 or len(labels) != n_examples:
        raise ValueError("y_true must have one label per example")
    if labels.dtype.kind not in "iu":
        raise ValueError("y_true must contain integer class indices")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("y_true contains an out-of-range class index")
    return labels.astype(np.int64, copy=False)


def _validated(y_true, probabilities):
    probabilities = _probabilities(probabilities)
    labels = _labels(y_true, *probabilities.shape)
    return labels, probabilities


def _unit_scalar(value, name, lower_inclusive=True):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    lower_ok = value >= 0 if lower_inclusive else value > 0
    if not np.isfinite(value) or not lower_ok or value > 1:
        bounds = "[0, 1]" if lower_inclusive else "(0, 1]"
        raise ValueError(f"{name} must lie in {bounds}")
    return value


def softmax(logits):
    """Return stable float64 softmax scores for a nonempty N-by-K matrix."""
    logits = _matrix(logits, "logits")
    # Subtracting each row's maximum leaves softmax unchanged and avoids
    # overflow when logits are large. Extreme negative differences may become
    # -inf; their exponentials correctly become zero.
    with np.errstate(over="ignore", under="ignore"):
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def classification_metrics(y_true, probabilities):
    """Accuracy, fixed-class macro F1, and confusion counts.

Absent classes have recall None; macro F1 includes every declared probability
column, assigning zero F1 when both precision and recall denominators are zero.
The confusion matrix uses actual classes as rows and predictions as columns.
"""
    labels, probabilities = _validated(y_true, probabilities)
    predictions = probabilities.argmax(axis=1)
    n_examples, n_classes = probabilities.shape
    confusion = np.bincount(
        n_classes * labels + predictions, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)
    correct = np.diag(confusion)
    actual_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    # F1 = 2 TP / (actual positives + predicted positives).
    f1_denominator = actual_counts + predicted_counts
    f1 = np.divide(
        2.0 * correct, f1_denominator,
        out=np.zeros(n_classes, dtype=np.float64), where=f1_denominator != 0,
    )
    recall = [
        float(correct[i] / actual_counts[i]) if actual_counts[i] else None
        for i in range(n_classes)
    ]
    return {
        "n_examples": int(n_examples),
        "accuracy": float(correct.sum() / n_examples),
        "macro_f1": float(f1.mean()),
        "per_class_recall": recall,
        "confusion_matrix": confusion.tolist(),
        "n_errors": int(n_examples - correct.sum()),
        "prediction_tie_break": "lowest_class_index",
    }


def choose_method(metrics_by_name, method_order):
    """Select highest development accuracy; ties use declared method order.

Only candidates in method_order are considered. The same order must be fixed
before evaluating development data, so tie handling is not selected afterward.
"""
    order = list(method_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("method_order must be nonempty and contain no duplicates")
    if not isinstance(metrics_by_name, Mapping):
        raise ValueError("metrics_by_name must be a mapping")
    scores = []
    for name in order:
        if name not in metrics_by_name or "accuracy" not in metrics_by_name[name]:
            raise ValueError(f"missing accuracy for method {name!r}")
        scores.append(_unit_scalar(metrics_by_name[name]["accuracy"], "accuracy"))
    return order[int(np.argmax(scores))]


def coverage_threshold(confidence, target):
    """Choose a calibration-only cutoff reaching at least target coverage.

Take the ceil(target * N)-th largest score and retain scores >= the cutoff.
Ties can make observed calibration coverage exceed the requested target. This
does not guarantee the same coverage, or any accuracy, on a future/test set.
"""
    scores = np.asarray(confidence, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("confidence must be a nonempty finite vector")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("confidence must lie in [0, 1]")
    target = _unit_scalar(target, "target", lower_inclusive=False)
    k = ceil(target * len(scores))
    return float(np.partition(scores, len(scores) - k)[len(scores) - k])


def selective_metrics(y_true, probs, threshold):
    """Measure retained and referred predictions at a fixed cutoff.

All class-specific quantities group by the true class. Referred errors are
errors routed for review, not errors proven resolved by a human. Accuracy is
undefined (None) if no examples are retained; error capture is undefined if
the original predictions contain no errors.
"""
    labels, probabilities = _validated(y_true, probs)
    threshold = _unit_scalar(threshold, "threshold")
    predictions = probabilities.argmax(axis=1)
    accepted = probabilities.max(axis=1) >= threshold
    errors = predictions != labels
    n_examples, n_classes = probabilities.shape
    accepted_count = int(accepted.sum())
    accepted_errors = int(np.sum(accepted & errors))
    referred_errors = int(np.sum(~accepted & errors))
    total_errors = int(errors.sum())
    per_class_coverage = []
    per_class_accuracy = []
    for class_id in range(n_classes):
        member = labels == class_id
        class_count = int(member.sum())
        retained_member = member & accepted
        retained_count = int(retained_member.sum())
        per_class_coverage.append(
            float(retained_count / class_count) if class_count else None
        )
        per_class_accuracy.append(
            float(np.sum(retained_member & ~errors) / retained_count)
            if retained_count else None
        )
    return {
        "threshold": threshold,
        "n_examples": int(n_examples),
        "accepted_count": accepted_count,
        "referred_count": int(n_examples - accepted_count),
        "coverage": float(accepted_count / n_examples),
        "selective_accuracy": (
            float(1 - accepted_errors / accepted_count) if accepted_count else None
        ),
        "selective_error_rate": (
            float(accepted_errors / accepted_count) if accepted_count else None
        ),
        "accepted_errors": accepted_errors,
        "referred_errors": referred_errors,
        "caught_error_fraction": (
            float(referred_errors / total_errors) if total_errors else None
        ),
        "correctly_accepted_fraction_total": float(
            (accepted_count - accepted_errors) / n_examples
        ),
        "per_class_coverage": per_class_coverage,
        "per_class_accepted_accuracy": per_class_accuracy,
    }


def paired_accuracy_bootstrap(y_true, probs_a, probs_b, resamples=2000, seed=2026):
    """Paired image-bootstrap descriptive interval for accuracy(B)-accuracy(A).

Each image contributes -1, 0, or +1 to the paired correctness difference.
Sampling the three category counts with a multinomial is exactly equivalent
in distribution to resampling N image indices with replacement. It preserves
pairing and avoids allocating a resamples-by-N index array. The interval is
conditional on these examples/models, not a domain-shift or causal guarantee.
"""
    labels, a = _validated(y_true, probs_a)
    b = _probabilities(probs_b)
    if a.shape != b.shape:
        raise ValueError("both methods must have matching example/class shapes")
    if isinstance(resamples, (bool, np.bool_)) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    a_correct = (a.argmax(axis=1) == labels).astype(np.int8)
    b_correct = (b.argmax(axis=1) == labels).astype(np.int8)
    differences = b_correct - a_correct
    counts = np.bincount(differences + 1, minlength=3)
    sampled_counts = np.random.default_rng(seed).multinomial(
        len(labels), counts / len(labels), size=resamples
    )
    sampled_deltas = (sampled_counts[:, 2] - sampled_counts[:, 0]) / len(labels)
    interval = np.quantile(sampled_deltas, [0.025, 0.975], method="linear")
    return {
        "mean_delta_b_minus_a": float(differences.mean()),
        "ci95": interval.tolist(),
        "b_better_count": int(counts[2]),
        "a_better_count": int(counts[0]),
        "tied_count": int(counts[1]),
        "resamples": int(resamples),
        "seed": int(seed),
        "resampling_unit": "paired_image",
    }
