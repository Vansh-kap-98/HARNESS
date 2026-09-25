"""Metrics for the evaluation harness.

Everything here is pure: sequences in, numbers out, no model and no I/O, so it
runs and is tested without a GPU.

The rules from the master plan are encoded, not left to the person reading the
table:

* every accuracy carries a 95% Wilson interval (section 8 rule 11)
* ECE is marked unreportable below 200 decisions (rule 12)
* the majority-class floor is a first-class metric, not an afterthought
  (section 3)
* validity is a bug counter, never a score (rule 1)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ScoreError",
    "Proportion",
    "FieldScore",
    "CalibrationBin",
    "Calibration",
    "LatencySummary",
    "wilson_interval",
    "per_field_accuracy",
    "exact_match",
    "majority_class",
    "majority_class_predictions",
    "calibration",
    "validity_bug_count",
    "latency_summary",
    "percentile",
    "summary_notes",
    "Z95",
    "MIN_CALIBRATION_N",
    "MIN_TREND_N",
    "CALIBRATION_BINS",
]

# 97.5th percentile of the standard normal; gives a two-sided 95% interval.
Z95 = 1.959963984540054
CALIBRATION_BINS = 10
MIN_CALIBRATION_N = 200
MIN_TREND_N = 50

_MISSING = object()


class ScoreError(Exception):
    """The inputs to a metric are malformed."""


@dataclass(frozen=True)
class Proportion:
    """A count out of a total, with its 95% Wilson interval.

    Wilson interval: a binomial confidence interval that stays inside [0, 1]
    and stays sensible at small n and near 0 or 1, unlike the normal
    ("Wald") approximation, which can produce intervals below 0 or above 1.
    """

    successes: int
    total: int

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else float("nan")

    @property
    def interval(self) -> Tuple[float, float]:
        return wilson_interval(self.successes, self.total)

    @property
    def low(self) -> float:
        return self.interval[0]

    @property
    def high(self) -> float:
        return self.interval[1]

    def as_dict(self) -> Dict[str, Any]:
        low, high = self.interval
        return {
            "value": self.value,
            "ci95_low": low,
            "ci95_high": high,
            "successes": self.successes,
            "total": self.total,
        }

    def __str__(self) -> str:
        if not self.total:
            return "n/a (n=0)"
        low, high = self.interval
        return "%.3f [%.3f, %.3f] (n=%d)" % (self.value, low, high, self.total)


@dataclass(frozen=True)
class FieldScore:
    """Accuracy for one field, plus how often the prediction was absent."""

    field: str
    accuracy: Proportion
    missing: int

    def as_dict(self) -> Dict[str, Any]:
        payload = {"field": self.field, "missing": self.missing}
        payload.update(self.accuracy.as_dict())
        return payload


@dataclass(frozen=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bin": "[%.1f, %.1f)" % (self.low, self.high),
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
            "gap": self.accuracy - self.mean_confidence,
        }


@dataclass(frozen=True)
class Calibration:
    """Expected Calibration Error plus the reliability table it came from.

    ECE: bin the decisions by stated confidence, and average
    |accuracy - mean confidence| over bins, weighted by bin size. It answers
    "when this model says 0.9, is it right 90% of the time?".

    ``reportable`` is False below ``MIN_CALIBRATION_N`` decisions. At that size
    the estimate is noise-dominated and biased low, so it may appear as a
    diagnostic but never as a result (section 8 rule 12).
    """

    ece: float
    bins: Tuple[CalibrationBin, ...]
    total: int

    @property
    def reportable(self) -> bool:
        return self.total >= MIN_CALIBRATION_N

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ece": self.ece,
            "total_decisions": self.total,
            "reportable": self.reportable,
            "bins": [b.as_dict() for b in self.bins],
        }


@dataclass(frozen=True)
class LatencySummary:
    """Seconds per example. Warmup exclusion is the runner's job."""

    p50: float
    p95: float
    examples: int
    repeats: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "p50_seconds": self.p50,
            "p95_seconds": self.p95,
            "examples": self.examples,
            "repeats": self.repeats,
        }


def wilson_interval(
    successes: int, total: int, z: float = Z95
) -> Tuple[float, float]:
    if total < 0 or successes < 0 or successes > total:
        raise ScoreError(
            "invalid counts: %d successes out of %d" % (successes, total)
        )
    if total == 0:
        return (float("nan"), float("nan"))
    n = float(total)
    phat = successes / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    spread = (
        z
        * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def per_field_accuracy(
    golds: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Dict[str, FieldScore]:
    """Accuracy per field. A missing prediction counts as wrong, and is also
    counted separately -- it means an invalid or truncated instance, which is
    a pipeline bug worth seeing rather than a rounding error in the accuracy.
    """
    _check_pairs(golds, predictions)
    scores: Dict[str, FieldScore] = {}
    for name in fields:
        correct = 0
        missing = 0
        for gold, prediction in zip(golds, predictions):
            if name not in gold:
                raise ScoreError("gold record is missing field %r" % name)
            value = prediction.get(name, _MISSING)
            if value is _MISSING:
                missing += 1
                continue
            if value == gold[name] and _same_type(value, gold[name]):
                correct += 1
        scores[name] = FieldScore(
            field=name,
            accuracy=Proportion(successes=correct, total=len(golds)),
            missing=missing,
        )
    return scores


def exact_match(
    golds: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Proportion:
    """Whole-instance accuracy: every field right, none missing."""
    _check_pairs(golds, predictions)
    hits = 0
    for gold, prediction in zip(golds, predictions):
        if all(
            name in prediction
            and prediction[name] == gold[name]
            and _same_type(prediction[name], gold[name])
            for name in fields
        ):
            hits += 1
    return Proportion(successes=hits, total=len(golds))


def majority_class(
    golds: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> Dict[str, Any]:
    """The most common gold value per field.

    Fitted on the same records it is scored against, so it is an optimistic
    floor -- which is the point: a model that cannot beat an optimistic floor
    has shown nothing. Ties break on the JSON rendering of the value, so the
    result does not depend on record order.
    """
    if not golds:
        raise ScoreError("no gold records")
    chosen: Dict[str, Any] = {}
    for name in fields:
        counts: Dict[str, int] = {}
        values: Dict[str, Any] = {}
        for gold in golds:
            if name not in gold:
                raise ScoreError("gold record is missing field %r" % name)
            key = json.dumps(gold[name], sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
            values[key] = gold[name]
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        chosen[name] = values[best]
    return chosen


def majority_class_predictions(
    golds: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> List[Dict[str, Any]]:
    """The floor baseline as a prediction list, scorable like any model."""
    constant = majority_class(golds, fields)
    return [dict(constant) for _ in golds]


def calibration(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = CALIBRATION_BINS,
) -> Calibration:
    if len(confidences) != len(correct):
        raise ScoreError(
            "%d confidences but %d outcomes" % (len(confidences), len(correct))
        )
    if bins < 1:
        raise ScoreError("need at least one bin")
    for value in confidences:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScoreError("confidence must be numeric, got %r" % (value,))
        if math.isnan(value) or not 0.0 <= float(value) <= 1.0:
            raise ScoreError("confidence %r is outside [0, 1]" % (value,))

    total = len(confidences)
    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(bins)]
    for value, hit in zip(confidences, correct):
        index = min(int(float(value) * bins), bins - 1)
        buckets[index].append((float(value), bool(hit)))

    table: List[CalibrationBin] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        low = index / bins
        high = (index + 1) / bins
        if not bucket:
            table.append(CalibrationBin(low, high, 0, float("nan"), float("nan")))
            continue
        mean_confidence = math.fsum(value for value, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, hit in bucket if hit) / len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - mean_confidence)
        table.append(
            CalibrationBin(low, high, len(bucket), mean_confidence, accuracy)
        )
    return Calibration(ece=ece, bins=tuple(table), total=total)


def validity_bug_count(validities: Sequence[bool]) -> int:
    """Invalid instances. Never a score: by construction this must be 0.

    A non-zero count is a compiler or assembler bug to fix, not a model
    weakness to report (section 8 rule 1).
    """
    return sum(1 for valid in validities if not valid)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        raise ScoreError("no values")
    if not 0.0 <= q <= 100.0:
        raise ScoreError("percentile must lie in [0, 100], got %r" % (q,))
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(repeats: Sequence[Sequence[float]]) -> LatencySummary:
    """p50 = median of per-example medians across repeats; p95 = pooled.

    ``repeats`` is one list of per-example seconds per full pass over the set,
    every pass covering the same examples in the same order (section 6).
    """
    if not repeats:
        raise ScoreError("no repeats")
    lengths = {len(repeat) for repeat in repeats}
    if len(lengths) != 1:
        raise ScoreError("repeats cover different numbers of examples: %s" % sorted(lengths))
    examples = lengths.pop()
    if examples == 0:
        raise ScoreError("repeats contain no examples")
    for repeat in repeats:
        for value in repeat:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ScoreError("latency must be numeric, got %r" % (value,))
            if value < 0 or math.isnan(value) or math.isinf(value):
                raise ScoreError("latency %r is not a finite, non-negative time" % (value,))

    per_example_median = [
        percentile([repeat[i] for repeat in repeats], 50.0) for i in range(examples)
    ]
    pooled = [value for repeat in repeats for value in repeat]
    return LatencySummary(
        p50=percentile(per_example_median, 50.0),
        p95=percentile(pooled, 95.0),
        examples=examples,
        repeats=len(repeats),
    )


def summary_notes(examples: int, decisions: int) -> Tuple[str, ...]:
    """Caveats this run must be published with, derived from its own size."""
    notes: List[str] = []
    if examples < MIN_TREND_N:
        notes.append(
            "n=%d examples is below the %d-example floor: no trend may be "
            "claimed from this run (rule 11)" % (examples, MIN_TREND_N)
        )
    if decisions < MIN_CALIBRATION_N:
        notes.append(
            "%d decisions is below the %d needed for calibration: ECE and the "
            "reliability table are diagnostics only, and no knee point may be "
            "read from this run (rule 12)" % (decisions, MIN_CALIBRATION_N)
        )
    return tuple(notes)


def _check_pairs(
    golds: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> None:
    if len(golds) != len(predictions):
        raise ScoreError(
            "%d gold records but %d predictions" % (len(golds), len(predictions))
        )
    if not golds:
        raise ScoreError("no records to score")


def _same_type(left: Any, right: Any) -> bool:
    """`True == 1` in Python; in JSON they are different values."""
    return isinstance(left, bool) == isinstance(right, bool)
