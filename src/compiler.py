"""Compile a JSON Schema into Laya-style typed decision questions.

Pipeline position: schema -> [compiler] -> typed questions -> model -> answers
-> assembler -> JSON instance -> Blaze validation.

The compiler is deliberately narrow. Every schema feature that cannot be
expressed as a `choice` / `score` / `noul` question is REFUSED and reported,
never approximated (master prompt section 4.2).

Two refusal levels:

* ``CompilerError``  -- the schema as a whole cannot be compiled (it is
  malformed, unbundled, or uses object-level keywords that change *which*
  properties exist, so there is no fixed question set to compile at all).
* skipped field      -- one property cannot be turned into a question. The
  rest of the schema still compiles; the property is reported with a reason
  and a route (``generative-track``).

Vocabulary note: JSON Schema has no keyword for "this enum is ordered", so
this compiler reads one non-standard annotation, ``x-ordered: true``, to turn
a string enum into a `score` instead of a `choice`. Unknown keywords are
ignored by validators, so a schema carrying it still validates normally.
This is the only extension keyword the compiler reads.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CompilerError",
    "UnsupportedField",
    "FieldSpec",
    "SkippedField",
    "CompiledSchema",
    "compile_schema",
    "compile_schema_file",
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_STEPS",
    "ORDERED_MARKER",
]

# --- one blessed configuration; these are constants, not knobs -------------
MAX_CHOICE_OPTIONS = 20
MAX_SCORE_STEPS = 10
ORDERED_MARKER = "x-ordered"
GENERATIVE_ROUTE = "generative-track"

# Annotation keywords that may sit next to a `$ref` (they do not affect
# validation, so carrying them over is exact, not an approximation).
_REF_SIBLING_ANNOTATIONS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
        ORDERED_MARKER,
    }
)

# Keywords at the object level that change which properties exist. With these
# present there is no single fixed question set, so we refuse the schema.
_ROOT_FORBIDDEN = (
    "if",
    "then",
    "else",
    "not",
    "allOf",
    "anyOf",
    "oneOf",
    "dependentSchemas",
    "dependentRequired",
    "patternProperties",
    "propertyNames",
    "unevaluatedProperties",
)

# Keywords inside a property that we cannot honour when picking a question.
_PROPERTY_FORBIDDEN = (
    "if",
    "then",
    "else",
    "not",
    "allOf",
    "anyOf",
    "dependentSchemas",
    "dependentRequired",
)


class CompilerError(Exception):
    """The schema cannot be compiled at all."""


class UnsupportedField(Exception):
    """One property cannot become a question. Carries the reported reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class FieldSpec:
    """How to turn one model answer back into a JSON value.

    ``values`` is aligned with the question's answer space:
      * choice -- ``values[i]`` is the JSON value for option key
        ``tuple(question["criteria"])[i]``
      * score  -- ``values[i]`` is the JSON value for criteria label ``i``
        (index 0 = lowest)
      * noul   -- ``(False, True)``
    """

    field: str
    qtype: str
    values: Tuple[Any, ...]
    required: bool


@dataclass(frozen=True)
class SkippedField:
    """A refused property (master prompt section 4.2 report row)."""

    field: str
    reason: str
    route: str = GENERATIVE_ROUTE

    def as_dict(self) -> Dict[str, str]:
        return {"field": self.field, "reason": self.reason, "route": self.route}


@dataclass(frozen=True)
class CompiledSchema:
    questions: Dict[str, Dict[str, Any]]
    specs: Dict[str, FieldSpec]
    skipped: Tuple[SkippedField, ...]
    required: FrozenSet[str] = dataclass_field(default_factory=frozenset)

    @property
    def unsupported_report(self) -> List[Dict[str, str]]:
        return [s.as_dict() for s in self.skipped]

    @property
    def blocking_fields(self) -> Tuple[str, ...]:
        """Refused properties that the schema *requires*.

        If this is non-empty, no assembled instance can ever validate: the
        decision track cannot produce these fields and the schema will not
        accept an instance without them. Callers must treat the schema as out
        of scope rather than emit a knowingly invalid instance.
        """
        return tuple(s.field for s in self.skipped if s.field in self.required)

    @property
    def optional_fields(self) -> Tuple[str, ...]:
        """Compiled properties that the schema does not require.

        A decision model answers every question it is given, so these fields
        are always present in the assembled instance -- the pipeline cannot
        express "absent". A schema where absence carries meaning is out of
        scope for the decision track; callers must check this list rather than
        discover it from a confusing diff against gold.
        """
        return tuple(name for name in self.questions if name not in self.required)

    def to_json(self) -> str:
        return json.dumps(
            {
                "questions": self.questions,
                "specs": {
                    name: {
                        "field": spec.field,
                        "type": spec.qtype,
                        "values": list(spec.values),
                        "required": spec.required,
                    }
                    for name, spec in self.specs.items()
                },
                "unsupported": self.unsupported_report,
                "blocking_fields": list(self.blocking_fields),
                "always_filled_optional_fields": list(self.optional_fields),
            },
            indent=2,
        )


# --------------------------------------------------------------------------
# $ref resolution (local pointers only; bundle remote refs before compiling)
# --------------------------------------------------------------------------


def _resolve_pointer(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#"):
        raise CompilerError(
            "non-local $ref %r: bundle the schema first "
            "(`jsonschema bundle <schema>`); the compiler resolves only "
            "same-document JSON Pointers" % ref
        )
    fragment = ref[1:]
    if fragment == "":
        return root
    if not fragment.startswith("/"):
        raise CompilerError(
            "$ref %r uses a plain-name anchor; only JSON Pointer fragments "
            "(#/$defs/...) are supported" % ref
        )
    node = root  # type: Any
    for token in fragment.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and key in node:
            node = node[key]
        elif isinstance(node, list):
            try:
                node = node[int(key)]
            except (ValueError, IndexError):
                raise CompilerError("$ref %r does not resolve" % ref)
        else:
            raise CompilerError("$ref %r does not resolve" % ref)
    if not isinstance(node, Mapping):
        raise CompilerError("$ref %r does not point at a schema object" % ref)
    return node


def _deref(subschema: Any, root: Mapping[str, Any]) -> Dict[str, Any]:
    """Follow ``$ref`` chains inside one document and return a plain dict."""
    if isinstance(subschema, bool):
        raise CompilerError("boolean schemas are not supported")
    if not isinstance(subschema, Mapping):
        raise CompilerError(
            "expected a schema object, got %s" % type(subschema).__name__
        )

    annotations: Dict[str, Any] = {}
    seen: List[str] = []
    current: Mapping[str, Any] = subschema
    while "$ref" in current:
        siblings = set(current) - {"$ref"} - _REF_SIBLING_ANNOTATIONS
        if siblings:
            raise CompilerError(
                "$ref with validation siblings %s is not supported; inline the "
                "reference instead" % sorted(siblings)
            )
        for key in _REF_SIBLING_ANNOTATIONS:
            if key in current and key not in annotations:
                annotations[key] = current[key]
        ref = current["$ref"]
        if not isinstance(ref, str):
            raise CompilerError("$ref must be a string, got %r" % (ref,))
        if ref in seen:
            raise CompilerError("circular $ref chain through %r" % ref)
        seen.append(ref)
        current = _resolve_pointer(root, ref)

    resolved = dict(current)
    # An annotation written next to the `$ref` is the more specific one, so it
    # wins over the same annotation on the target (and over anything deeper in
    # the chain, since `annotations` keeps the outermost occurrence).
    resolved.update(annotations)
    return resolved


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------


def compile_schema(schema: Mapping[str, Any]) -> CompiledSchema:
    """Compile an object schema into typed questions plus a refusal report.

    Question order follows property order in the schema, so the same schema
    always produces the same questions in the same order.
    """
    if not isinstance(schema, Mapping):
        raise CompilerError("schema must be a JSON object")

    root = schema
    top = _deref(schema, root)

    declared_type = top.get("type")
    if declared_type != "object":
        raise CompilerError(
            'root schema must declare `type: "object"`, got %r' % (declared_type,)
        )
    present = [keyword for keyword in _ROOT_FORBIDDEN if keyword in top]
    if present:
        raise CompilerError(
            "root keywords %s change which properties exist; the decision "
            "track needs a fixed question set (section 4.2)" % present
        )
    if isinstance(top.get("additionalProperties"), Mapping):
        raise CompilerError(
            "`additionalProperties` as a schema describes an open-ended map; "
            "route this schema to the generative track (section 4.2)"
        )

    properties = top.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise CompilerError("root schema has no `properties` to compile")

    raw_required = top.get("required", [])
    if isinstance(raw_required, str) or not isinstance(raw_required, Sequence):
        raise CompilerError("`required` must be an array of property names")
    required = frozenset(raw_required)
    unknown_required = sorted(required - set(properties))
    if unknown_required:
        raise CompilerError(
            "`required` names properties that do not exist: %s" % unknown_required
        )

    questions: Dict[str, Dict[str, Any]] = {}
    specs: Dict[str, FieldSpec] = {}
    skipped: List[SkippedField] = []

    for name, raw in properties.items():
        subschema = _deref(raw, root)
        try:
            question, values = _compile_property(name, subschema)
        except UnsupportedField as exc:
            skipped.append(SkippedField(field=name, reason=exc.reason))
            continue
        questions[name] = question
        specs[name] = FieldSpec(
            field=name,
            qtype=question["type"],
            values=values,
            required=name in required,
        )

    return CompiledSchema(
        questions=questions,
        specs=specs,
        skipped=tuple(skipped),
        required=required,
    )


def compile_schema_file(path: str) -> CompiledSchema:
    with open(path, "r", encoding="utf-8") as handle:
        return compile_schema(json.load(handle))


def _instructions(name: str, subschema: Mapping[str, Any]) -> str:
    for key in ("description", "title"):
        text = subschema.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return "What is the value of '%s'?" % name


def _compile_property(
    name: str, subschema: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
    blocked = [keyword for keyword in _PROPERTY_FORBIDDEN if keyword in subschema]
    if blocked:
        raise UnsupportedField(
            "unsupported keywords %s: conditional or combined subschemas cannot "
            "be expressed as one typed question" % blocked
        )
    if "enum" in subschema and "oneOf" in subschema:
        raise CompilerError(
            "property %r has both `enum` and `oneOf`; ambiguous option source" % name
        )

    declared_type = subschema.get("type")
    if isinstance(declared_type, list):
        raise UnsupportedField(
            "union type %s (including nullability) has no typed question form"
            % declared_type
        )

    ordered = subschema.get(ORDERED_MARKER, False)
    if not isinstance(ordered, bool):
        raise CompilerError(
            "property %r: `%s` must be a boolean, got %r"
            % (name, ORDERED_MARKER, ordered)
        )

    if "enum" in subschema:
        return _compile_enum(name, subschema, ordered)
    if "oneOf" in subschema:
        return _compile_one_of_consts(name, subschema, ordered)
    if declared_type == "boolean":
        question = {"type": "noul", "instructions": _instructions(name, subschema)}
        return question, (False, True)
    if declared_type == "integer":
        return _compile_bounded_integer(name, subschema)
    if declared_type == "number":
        raise UnsupportedField(
            "`number` is continuous; the decision track has no question type for "
            "it (section 4.2)"
        )
    if declared_type == "string":
        raise UnsupportedField(
            "free-form `string` with no `enum` / `oneOf` of `const`s (section 4.2)"
        )
    if declared_type in ("array", "object"):
        raise UnsupportedField(
            "`%s` values are not produced by a fixed set of typed questions "
            "(section 4.2)" % declared_type
        )
    raise UnsupportedField(
        "no supported question type for this property (type=%r, no `enum` or "
        "`oneOf` of `const`s)" % (declared_type,)
    )


def _as_choice_or_score(
    name: str,
    subschema: Mapping[str, Any],
    values: Sequence[Any],
    labels: Sequence[str],
    ordered: bool,
) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
    instructions = _instructions(name, subschema)
    if ordered:
        if len(values) > MAX_SCORE_STEPS:
            raise UnsupportedField(
                "ordered field has %d levels, over the %d-step `score` limit"
                % (len(values), MAX_SCORE_STEPS)
            )
        question = {
            "type": "score",
            "instructions": instructions,
            "criteria": list(labels),
        }
        return question, tuple(values)

    if len(values) > MAX_CHOICE_OPTIONS:
        raise UnsupportedField(
            "%d options, over the %d-option `choice` limit; split into a "
            "coarse-to-fine hierarchy to support it (section 4.2)"
            % (len(values), MAX_CHOICE_OPTIONS)
        )
    criteria: Dict[str, str] = {}
    for value, label in zip(values, labels):
        criteria[str(value)] = label
    if len(criteria) != len(values):
        raise CompilerError(
            "property %r: option keys collide after string conversion" % name
        )
    question = {
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }
    return question, tuple(values)


def _compile_enum(
    name: str, subschema: Mapping[str, Any], ordered: bool
) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
    values = subschema.get("enum")
    if not isinstance(values, list) or not values:
        raise CompilerError("property %r: `enum` must be a non-empty array" % name)
    if len({_hashable(value) for value in values}) != len(values):
        raise CompilerError("property %r: `enum` contains duplicate values" % name)
    non_strings = [value for value in values if not isinstance(value, str)]
    if non_strings:
        raise UnsupportedField(
            "`enum` contains non-string values (%r); only string enums map to "
            "`choice` / `score`" % non_strings[:3]
        )
    return _as_choice_or_score(name, subschema, values, list(values), ordered)


def _compile_one_of_consts(
    name: str, subschema: Mapping[str, Any], ordered: bool
) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
    branches = subschema.get("oneOf")
    if not isinstance(branches, list) or not branches:
        raise CompilerError("property %r: `oneOf` must be a non-empty array" % name)

    values: List[Any] = []
    labels: List[str] = []
    for branch in branches:
        if not isinstance(branch, Mapping) or "const" not in branch:
            raise UnsupportedField(
                "`oneOf` branches must each be a single `const`; branch %r is a "
                "general subschema" % (branch,)
            )
        extra = set(branch) - {"const"} - _REF_SIBLING_ANNOTATIONS
        if extra:
            raise UnsupportedField(
                "`oneOf` branch carries validation keywords %s beyond `const`"
                % sorted(extra)
            )
        value = branch["const"]
        if not isinstance(value, str):
            raise UnsupportedField(
                "`const` %r is not a string; only string options map to "
                "`choice` / `score`" % (value,)
            )
        text = branch.get("description") or branch.get("title") or value
        values.append(value)
        labels.append(str(text).strip())
    if len(set(values)) != len(values):
        raise CompilerError("property %r: duplicate `const` values in `oneOf`" % name)
    return _as_choice_or_score(name, subschema, values, labels, ordered)


def _compile_bounded_integer(
    name: str, subschema: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
    if "multipleOf" in subschema:
        raise UnsupportedField(
            "`multipleOf` changes the allowed integer set; not supported"
        )
    lower_bounds = [
        bound
        for bound in (
            _integer_bound(subschema, "minimum", 0),
            _integer_bound(subschema, "exclusiveMinimum", 1),
        )
        if bound is not None
    ]
    upper_bounds = [
        bound
        for bound in (
            _integer_bound(subschema, "maximum", 0),
            _integer_bound(subschema, "exclusiveMaximum", -1),
        )
        if bound is not None
    ]
    if not lower_bounds or not upper_bounds:
        raise UnsupportedField(
            "unbounded `integer` (needs a lower and an upper bound spanning at "
            "most %d steps to become a `score`)" % MAX_SCORE_STEPS
        )
    low, high = max(lower_bounds), min(upper_bounds)
    if high < low:
        raise CompilerError(
            "property %r: empty integer range [%d, %d]" % (name, low, high)
        )
    steps = high - low + 1
    if steps > MAX_SCORE_STEPS:
        raise UnsupportedField(
            "`integer` range [%d, %d] spans %d steps, over the %d-step `score` "
            "limit (section 4.2)" % (low, high, steps, MAX_SCORE_STEPS)
        )
    values = list(range(low, high + 1))
    question = {
        "type": "score",
        "instructions": _instructions(name, subschema),
        "criteria": [str(value) for value in values],
    }
    return question, tuple(values)


def _integer_bound(
    subschema: Mapping[str, Any], keyword: str, offset: int
) -> Optional[int]:
    if keyword not in subschema:
        return None
    raw = subschema[keyword]
    if isinstance(raw, bool) or not isinstance(raw, int):
        # draft-4 style boolean exclusiveMinimum, or a non-integral bound
        raise UnsupportedField(
            "`%s` must be an integer for an integer `score`, got %r" % (keyword, raw)
        )
    return raw + offset


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return json.dumps(value, sort_keys=True)


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m src.compiler <schema.json>", file=sys.stderr)
        return 2
    compiled = compile_schema_file(argv[1])
    print(compiled.to_json())
    if compiled.blocking_fields:
        print(
            "BLOCKED: required fields cannot be compiled: %s"
            % list(compiled.blocking_fields),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
