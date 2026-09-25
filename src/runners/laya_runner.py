"""Adapt Laya's output to the harness's canonical `Answer` type.

Everything model-specific lives here. The compiler, assembler and scorer never
see a Laya-shaped dict, so they stay testable without a GPU and identical
across every model in the ladder.

What the repo's own `rl_agent_api.py` returns, per question type:

    choice -> {"type", "choice", "probabilities", "confidence", "rl_agent"}
    score  -> {"type", "score", "legend", "probabilities", "confidence", "rl_agent"}
    noul   -> {"type", "noul", "rl_agent"}          # the value IS p(yes)

Two deliberate decisions:

* **Confidence comes from `probabilities`, not from their `confidence` key.**
  That key is computed by `confidence_from_probs(p, k)` and we do not know
  whether it is the max probability or something adjusted for the option
  count. ECE only means anything when the number compared against accuracy is
  the probability of the option the model actually chose, so we take
  `probabilities[index]` and keep their figure alongside for comparison.
* **`score` is read from `probabilities`, not from the `score` float.** The
  card does not define whether that float is a level index, a 0-1 scale, or
  something else, and guessing would silently shift every ordinal answer by
  one. The argmax over `probabilities` is unambiguous.

Anything this module cannot verify, it asserts at runtime and fails loudly on.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Tuple

from src.assembler import Answer
from src.compiler import CompiledSchema

__all__ = [
    "LayaRunnerError",
    "Predictor",
    "load_router",
    "to_answers",
    "reported_confidence",
    "NOUL_DECISION_BOUNDARY",
]

# Converting p(yes) to a boolean happens at the midpoint. This is the
# definition of the decision boundary, not a tuned quantity -- the escalation
# threshold is a separate, fitted number and lives in src/assembler.py.
NOUL_DECISION_BOUNDARY = 0.5

PROBABILITY_SUM_TOLERANCE = 1e-3

Predictor = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class LayaRunnerError(Exception):
    """Laya returned something this adapter cannot map to an Answer."""


def load_router(device: str = "cuda", preload: bool = True) -> Predictor:
    """Build the real router and hand back its `predict`.

    Kept separate from `to_answers` so every test runs without the package
    installed. Which checkpoint the router picks (root / multilingual /
    typed-decisions) is its own routing decision, reported back in
    `result["routing"]["model"]`; record that in the run config.
    """
    from laya import Router  # imported lazily: not installed locally

    router = Router(preload=preload, device=device)
    return router.predict


def to_answers(
    result: Mapping[str, Any], compiled: CompiledSchema
) -> Dict[str, Answer]:
    """Convert one Laya result into canonical answers for the assembler."""
    answers_block = result.get("answers")
    if not isinstance(answers_block, Mapping):
        raise LayaRunnerError("result has no `answers` mapping")

    answers: Dict[str, Answer] = {}
    for name, question in compiled.questions.items():
        if name not in answers_block:
            raise LayaRunnerError("no answer for question %r" % name)
        raw = answers_block[name]
        if not isinstance(raw, Mapping):
            raise LayaRunnerError("answer for %r is not a mapping" % name)
        values = compiled.specs[name].values
        qtype = question["type"]
        if qtype == "choice":
            answers[name] = _from_choice(name, raw, values)
        elif qtype == "score":
            answers[name] = _from_score(name, raw, values)
        elif qtype == "noul":
            answers[name] = _from_noul(name, raw)
        else:  # pragma: no cover - the compiler emits no other type
            raise LayaRunnerError("unknown question type %r for %r" % (qtype, name))
    return answers


def reported_confidence(
    result: Mapping[str, Any], compiled: CompiledSchema
) -> Dict[str, float]:
    """Laya's own `confidence` field, for comparison against max-probability.

    Not used to build answers. Worth logging: if the two diverge, the
    calibration story depends on which one a run reported.
    """
    answers_block = result.get("answers", {})
    reported: Dict[str, float] = {}
    for name in compiled.questions:
        raw = answers_block.get(name)
        if isinstance(raw, Mapping) and "confidence" in raw:
            reported[name] = float(raw["confidence"])
    return reported


def _from_choice(
    name: str, raw: Mapping[str, Any], values: Tuple[Any, ...]
) -> Answer:
    probabilities = _probabilities(name, raw, values)
    index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    chosen = raw.get("choice")
    if chosen is None:
        raise LayaRunnerError("choice answer for %r has no `choice`" % name)
    if chosen not in values:
        raise LayaRunnerError(
            "%r: model returned option %r, which was never offered %r"
            % (name, chosen, list(values))
        )
    # The alignment of `probabilities` with the option order is an assumption
    # about the package, so check it rather than trust it.
    if values[index] != chosen:
        raise LayaRunnerError(
            "%r: `choice` is %r but the highest probability is on %r. The "
            "probability vector is not aligned with the option order; do not "
            "score this run." % (name, chosen, values[index])
        )
    return Answer(
        index=index,
        confidence=probabilities[index],
        distribution=tuple(probabilities),
    )


def _from_score(
    name: str, raw: Mapping[str, Any], values: Tuple[Any, ...]
) -> Answer:
    probabilities = _probabilities(name, raw, values)
    index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return Answer(
        index=index,
        confidence=probabilities[index],
        distribution=tuple(probabilities),
    )


def _from_noul(name: str, raw: Mapping[str, Any]) -> Answer:
    if "noul" not in raw:
        raise LayaRunnerError("noul answer for %r has no `noul` value" % name)
    probability = raw["noul"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise LayaRunnerError(
            "%r: `noul` must be a probability, got %r" % (name, probability)
        )
    probability = float(probability)
    if math.isnan(probability) or not 0.0 <= probability <= 1.0:
        raise LayaRunnerError(
            "%r: `noul` probability %r is outside [0, 1]" % (name, probability)
        )
    index = 1 if probability >= NOUL_DECISION_BOUNDARY else 0
    confidence = probability if index == 1 else 1.0 - probability
    return Answer(
        index=index,
        confidence=confidence,
        distribution=(1.0 - probability, probability),
    )


def _probabilities(
    name: str, raw: Mapping[str, Any], values: Tuple[Any, ...]
) -> List[float]:
    probabilities = raw.get("probabilities")
    if probabilities is None:
        raise LayaRunnerError(
            "%r: no `probabilities`, so no usable confidence. Laya's own "
            "`confidence` field is not a substitute (see module docstring)."
            % name
        )
    try:
        numbers = [float(value) for value in probabilities]
    except (TypeError, ValueError):
        raise LayaRunnerError("%r: `probabilities` is not a list of numbers" % name)
    if len(numbers) != len(values):
        raise LayaRunnerError(
            "%r: %d probabilities for %d options. The option set the model saw "
            "is not the one the compiler produced."
            % (name, len(numbers), len(values))
        )
    for value in numbers:
        if math.isnan(value) or not 0.0 <= value <= 1.0:
            raise LayaRunnerError("%r: probability %r is outside [0, 1]" % (name, value))
    total = math.fsum(numbers)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise LayaRunnerError(
            "%r: probabilities sum to %.6f, not 1" % (name, total)
        )
    return numbers
