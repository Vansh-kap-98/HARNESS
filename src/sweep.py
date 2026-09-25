"""Build routing questions at k = 4, 8, 16, ... options, to find the option wall.

The question this answers: a decision model scores every option in one forward
pass, but all the options share one context budget. Where does accuracy fall
off as the number of candidate API operations grows?

Why this does not go through `src/compiler.py`: the compiler refuses a `choice`
with more than 20 options (`MAX_CHOICE_OPTIONS`), and that 20 is currently an
assumption -- arithmetic on a 512-token context, not a measurement. This sweep
is the experiment that either validates it or moves it, so it builds questions
directly and reports, per k, whether the compiler would have refused. For every
k the compiler *does* accept, the question built here is identical to the one
the compiler produces from the equivalent schema; `tests/test_sweep.py` asserts
that, so the two paths cannot drift.

Three things the design has to get right, or the curve measures the wrong thing:

* **Nested option sets.** The k=8 set contains the k=4 set. Otherwise each k is
  a different task and a drop cannot be attributed to the option count.
* **A fixed seed per example.** Same distractors, same order, every run and
  every model (section 8 rule 8).
* **Chance moves with k.** Random guessing is 1/k, so raw accuracy *must* fall
  as k grows. Every result carries its chance level; a curve without it says
  nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.score import Proportion

__all__ = [
    "SweepError",
    "Operation",
    "RoutingExample",
    "BuiltQuestion",
    "KResult",
    "DEFAULT_KS",
    "ROUTING_INSTRUCTIONS",
    "load_catalogue",
    "load_examples",
    "build_question",
    "build_schema",
    "build_sweep",
    "approx_tokens_by_chars",
    "question_tokens",
    "score_k",
]

DEFAULT_KS: Tuple[int, ...] = (4, 8, 16, 32, 64, 128)

# One blessed instruction string: it is part of the input, so changing it
# between models or between k would break rule 8.
ROUTING_INSTRUCTIONS = (
    "Which API operation should be called to satisfy this request?"
)


class SweepError(Exception):
    """The sweep cannot be built from these inputs."""


@dataclass(frozen=True)
class Operation:
    """One candidate API operation: the unit the model chooses between."""

    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise SweepError("operation id must be a non-empty string")
        if not self.description or not self.description.strip():
            raise SweepError("operation %r has no description" % self.id)


@dataclass(frozen=True)
class RoutingExample:
    """A request, and the one operation that is correct for it."""

    id: str
    state: str
    gold: str


@dataclass(frozen=True)
class BuiltQuestion:
    """One example rendered at one k."""

    example_id: str
    k: int
    state: str
    question: Dict[str, Any]
    values: Tuple[str, ...]
    gold_index: int

    @property
    def gold(self) -> str:
        return self.values[self.gold_index]


@dataclass(frozen=True)
class KResult:
    """Accuracy at one k, next to the chance level it has to beat."""

    k: int
    accuracy: Proportion
    chance: float
    compiler_would_refuse: bool
    median_tokens: Optional[int] = None
    context_overflows: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"k": self.k, "chance": self.chance}
        payload.update(self.accuracy.as_dict())
        payload["lift_over_chance"] = self.accuracy.value - self.chance
        payload["compiler_would_refuse"] = self.compiler_would_refuse
        if self.median_tokens is not None:
            payload["median_tokens"] = self.median_tokens
        if self.context_overflows is not None:
            payload["context_overflows"] = self.context_overflows
        return payload


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_catalogue(path: str) -> List[Operation]:
    """Read the operation catalogue: {"id": ..., "description": ...} per line."""
    operations: List[Operation] = []
    seen = set()
    for number, payload in _read_jsonl(path):
        for key in ("id", "description"):
            if key not in payload:
                raise SweepError("catalogue line %d has no %r" % (number, key))
        if payload["id"] in seen:
            raise SweepError("duplicate operation id %r on line %d" % (payload["id"], number))
        seen.add(payload["id"])
        operations.append(Operation(id=payload["id"], description=payload["description"]))
    if not operations:
        raise SweepError("%s contains no operations" % path)
    return operations


def load_examples(path: str) -> List[RoutingExample]:
    """Read routing examples: {"id": ..., "state": ..., "gold": ...} per line."""
    examples: List[RoutingExample] = []
    seen = set()
    for number, payload in _read_jsonl(path):
        for key in ("id", "state", "gold"):
            if key not in payload:
                raise SweepError("example line %d has no %r" % (number, key))
        if payload["id"] in seen:
            raise SweepError("duplicate example id %r on line %d" % (payload["id"], number))
        seen.add(payload["id"])
        examples.append(
            RoutingExample(id=payload["id"], state=payload["state"], gold=payload["gold"])
        )
    if not examples:
        raise SweepError("%s contains no examples" % path)
    return examples


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def build_question(
    example: RoutingExample,
    catalogue: Sequence[Operation],
    k: int,
    seed: int,
) -> BuiltQuestion:
    """Render one example as a k-option `choice`.

    The option set is the gold operation plus the k-1 distractors ranking
    highest in a selection permutation seeded by `(seed, example.id)`. Display
    order comes from a second, independent permutation. Neither depends on k,
    so the k=8 set contains the k=4 set, options keep their relative order as
    k grows, and the gold option does not sit at a fixed position.
    """
    if k < 2:
        raise SweepError("k must be at least 2, got %r" % (k,))
    if len(catalogue) < k:
        raise SweepError(
            "catalogue has %d operations, cannot build a %d-option question"
            % (len(catalogue), k)
        )
    by_id = {operation.id: operation for operation in catalogue}
    if example.gold not in by_id:
        raise SweepError(
            "example %r has gold %r, which is not in the catalogue"
            % (example.id, example.gold)
        )

    # Selection and display use two independent permutations. With one shared
    # permutation the k-1 distractors are by construction the highest-ranked
    # ones, so the gold option sorts after all of them and lands last in almost
    # every question -- a model could score 100% by always picking the last
    # option. Keep these apart.
    ranked = sorted(catalogue, key=lambda op: _rank(seed, example.id, op.id, "select"))
    chosen: List[Operation] = []
    for operation in ranked:
        if operation.id == example.gold:
            continue
        chosen.append(operation)
        if len(chosen) == k - 1:
            break
    chosen.append(by_id[example.gold])
    # Display order does not depend on k, so options keep their relative order
    # as k grows, and gold's position varies freely across examples.
    chosen.sort(key=lambda op: _rank(seed, example.id, op.id, "display"))

    values = tuple(operation.id for operation in chosen)
    question = {
        "type": "choice",
        "instructions": ROUTING_INSTRUCTIONS,
        "criteria": {operation.id: operation.description for operation in chosen},
    }
    return BuiltQuestion(
        example_id=example.id,
        k=k,
        state=example.state,
        question=question,
        values=values,
        gold_index=values.index(example.gold),
    )


def build_schema(built: BuiltQuestion) -> Dict[str, Any]:
    """The same options as a JSON Schema, for the constrained-decoding baseline.

    Rule 14: the generative baseline sees exactly the text the decision model
    sees, in exactly the same order. `tests/test_sweep.py` checks that
    compiling this schema reproduces `built.question` wherever the compiler
    accepts it.
    """
    criteria = built.question["criteria"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["operation"],
        "properties": {
            "operation": {
                "description": ROUTING_INSTRUCTIONS,
                "type": "string",
                "oneOf": [
                    {"const": value, "description": criteria[value]}
                    for value in built.values
                ],
            }
        },
    }


def build_sweep(
    examples: Sequence[RoutingExample],
    catalogue: Sequence[Operation],
    ks: Sequence[int] = DEFAULT_KS,
    seed: int = 0,
) -> Dict[int, List[BuiltQuestion]]:
    """Every example at every k, in input order."""
    if not ks:
        raise SweepError("no values of k")
    if len(set(ks)) != len(ks):
        raise SweepError("duplicate values of k: %s" % list(ks))
    return {
        k: [build_question(example, catalogue, k, seed) for example in examples]
        for k in ks
    }


# --------------------------------------------------------------------------
# token budget -- the thing the wall is made of
# --------------------------------------------------------------------------


def approx_tokens_by_chars(text: str) -> int:
    """Rough stand-in: ~4 characters per token.

    Only for planning before a real tokenizer is available. Replace it with
    the model's own tokenizer as soon as the Kaggle probe reports one; any
    token count reported from this function must say so.
    """
    return (len(text) + 3) // 4


def question_tokens(built: BuiltQuestion, count_tokens: Callable[[str], int]) -> int:
    """Total budget the model must fit: state + instructions + every option.

    `count_tokens` is required, with no default, so a run can never quietly
    report estimated tokens as measured ones.
    """
    parts = [built.state, built.question["instructions"]]
    for value, description in built.question["criteria"].items():
        parts.append(value)
        parts.append(description)
    return sum(count_tokens(part) for part in parts)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score_k(
    built: Sequence[BuiltQuestion],
    predictions: Sequence[str],
    compiler_max_options: int,
    count_tokens: Optional[Callable[[str], int]] = None,
    context_limit: Optional[int] = None,
) -> KResult:
    """Accuracy at one k, against the 1/k chance level.

    A prediction is the operation id the model chose. `compiler_max_options`
    is passed in (rather than imported) so the result records the limit that
    was in force when the run happened.
    """
    if not built:
        raise SweepError("no questions to score")
    if len(built) != len(predictions):
        raise SweepError(
            "%d questions but %d predictions" % (len(built), len(predictions))
        )
    ks = {item.k for item in built}
    if len(ks) != 1:
        raise SweepError("questions mix several values of k: %s" % sorted(ks))
    k = ks.pop()

    hits = sum(1 for item, choice in zip(built, predictions) if choice == item.gold)
    unknown = [
        choice
        for item, choice in zip(built, predictions)
        if choice not in item.values
    ]
    if unknown:
        raise SweepError(
            "predictions include options that were never offered: %s" % unknown[:3]
        )

    median_tokens = None
    overflows = None
    if count_tokens is not None:
        counts = sorted(question_tokens(item, count_tokens) for item in built)
        median_tokens = counts[len(counts) // 2]
        if context_limit is not None:
            overflows = sum(1 for count in counts if count > context_limit)

    return KResult(
        k=k,
        accuracy=Proportion(successes=hits, total=len(built)),
        chance=1.0 / k,
        compiler_would_refuse=k > compiler_max_options,
        median_tokens=median_tokens,
        context_overflows=overflows,
    )


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _rank(seed: int, example_id: str, operation_id: str, purpose: str) -> str:
    """Stable pseudo-random order, identical on every machine and run.

    `purpose` ("select" or "display") gives two independent orderings from the
    same seed, which is what keeps the gold option off a fixed position.
    Python's `hash()` is salted per process, so it cannot be used here.
    """
    key = "%d|%s|%s|%s" % (seed, example_id, operation_id, purpose)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SweepError("line %d is not valid JSON: %s" % (number, exc))
            if not isinstance(payload, Mapping):
                raise SweepError("line %d is not a JSON object" % number)
            yield number, payload
