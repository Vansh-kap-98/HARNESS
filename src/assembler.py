"""Turn model answers back into a JSON instance.

Pipeline position: schema -> compiler -> typed questions -> model -> [assembler]
-> JSON instance -> Blaze validation.

The assembler is the only place that decides what the instance looks like, so
it is also the place that must refuse to emit something that cannot validate.
It never guesses a value, never fills a field it has no answer for, and never
puts confidence inside the instance (master prompt section 4.3).

Runners produce the canonical ``Answer`` type below. Model-specific output
shapes (Laya's per-`[MASK]` softmax, a constrained decoder's JSON) are adapted
in ``src/runners/*``, not here, so that the assembler stays testable without a
GPU and identical across every model in the ladder.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.compiler import CompiledSchema, FieldSpec

__all__ = [
    "AssemblyError",
    "Answer",
    "Assembled",
    "assemble",
    "escalate_fields",
    "PROBABILITY_SUM_TOLERANCE",
]

PROBABILITY_SUM_TOLERANCE = 1e-3


class AssemblyError(Exception):
    """The answers cannot be turned into an instance."""


@dataclass(frozen=True)
class Answer:
    """One model answer, in the compiler's answer space.

    ``index`` indexes ``FieldSpec.values``: for a `choice` it is the option
    position, for a `score` the level (0 = lowest), for a `noul` 0 = no,
    1 = yes. ``confidence`` is the probability the model assigned to the
    option it picked, in [0, 1] -- not a similarity, not a logit.
    """

    index: int
    confidence: float
    distribution: Optional[Tuple[float, ...]] = None

    @classmethod
    def from_distribution(cls, probabilities: Sequence[float]) -> "Answer":
        """Build an answer by argmax over a probability vector.

        Ties go to the lowest index, which keeps runs reproducible.
        """
        probs = tuple(float(p) for p in probabilities)
        if not probs:
            raise AssemblyError("empty probability vector")
        _check_probabilities(probs, "distribution")
        best = max(range(len(probs)), key=lambda i: probs[i])
        return cls(index=best, confidence=probs[best], distribution=probs)


@dataclass(frozen=True)
class Assembled:
    """A JSON instance plus per-field confidence, kept strictly apart."""

    instance: Dict[str, Any]
    confidence: Dict[str, float]

    @property
    def min_confidence(self) -> float:
        return min(self.confidence.values()) if self.confidence else 1.0

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable view. Confidence stays outside the instance."""
        return {"instance": self.instance, "confidence": self.confidence}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)


def assemble(
    compiled: CompiledSchema, answers: Mapping[str, Answer]
) -> Assembled:
    """Build the instance for exactly the compiled questions.

    Field order follows schema property order, so two runs of the same model
    produce byte-identical instances.
    """
    if compiled.blocking_fields:
        raise AssemblyError(
            "schema requires fields the compiler refused (%s); no instance "
            "assembled from these questions can ever validate -- this schema "
            "is out of scope for the decision track"
            % list(compiled.blocking_fields)
        )

    missing = [name for name in compiled.questions if name not in answers]
    if missing:
        raise AssemblyError("no answer for questions %s" % missing)
    unknown = [name for name in answers if name not in compiled.questions]
    if unknown:
        raise AssemblyError(
            "answers for fields that were never asked: %s" % sorted(unknown)
        )

    instance: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    for name in compiled.questions:
        spec = compiled.specs[name]
        answer = answers[name]
        _check_answer(name, spec, answer)
        instance[name] = spec.values[answer.index]
        confidence[name] = float(answer.confidence)
    return Assembled(instance=instance, confidence=confidence)


def escalate_fields(
    assembled: Assembled, threshold: float
) -> Tuple[str, ...]:
    """Fields whose confidence is below ``threshold``, in instance order.

    There is deliberately no default threshold. Like temperature, it is a
    fitted quantity: Phase 4 fits it on `val` and records it in the run
    config. A number invented here would silently become a result.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise AssemblyError("threshold must be a number, got %r" % (threshold,))
    if not 0.0 <= float(threshold) <= 1.0:
        raise AssemblyError("threshold must lie in [0, 1], got %r" % (threshold,))
    return tuple(
        name
        for name in assembled.instance
        if assembled.confidence[name] < float(threshold)
    )


def _check_answer(name: str, spec: FieldSpec, answer: Answer) -> None:
    if not isinstance(answer, Answer):
        raise AssemblyError(
            "field %r: expected an Answer, got %s" % (name, type(answer).__name__)
        )
    if isinstance(answer.index, bool) or not isinstance(answer.index, int):
        raise AssemblyError(
            "field %r: answer index must be an int, got %r" % (name, answer.index)
        )
    if not 0 <= answer.index < len(spec.values):
        raise AssemblyError(
            "field %r: answer index %d is outside the %d option(s) the compiler "
            "produced" % (name, answer.index, len(spec.values))
        )
    _check_probabilities((answer.confidence,), "field %r confidence" % name)
    if answer.distribution is not None:
        if len(answer.distribution) != len(spec.values):
            raise AssemblyError(
                "field %r: distribution has %d entries but the question has %d "
                "options" % (name, len(answer.distribution), len(spec.values))
            )
        _check_probabilities(answer.distribution, "field %r distribution" % name)
        total = math.fsum(answer.distribution)
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise AssemblyError(
                "field %r: distribution sums to %.6f, not 1" % (name, total)
            )
        top = answer.distribution[answer.index]
        if abs(top - answer.confidence) > PROBABILITY_SUM_TOLERANCE:
            raise AssemblyError(
                "field %r: confidence %.6f does not match distribution[%d]=%.6f"
                % (name, answer.confidence, answer.index, top)
            )


def _check_probabilities(values: Sequence[float], what: str) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssemblyError("%s must be numeric, got %r" % (what, value))
        if math.isnan(value) or math.isinf(value):
            raise AssemblyError("%s is not a finite number (%r)" % (what, value))
        if not 0.0 <= float(value) <= 1.0:
            raise AssemblyError("%s must lie in [0, 1], got %r" % (what, value))
