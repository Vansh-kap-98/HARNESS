"""Unit tests for src/compiler.py.

Run: py -3.12 -m pytest -q
"""

from __future__ import annotations

import json
import os

import pytest

from src.compiler import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_STEPS,
    CompilerError,
    CompiledSchema,
    compile_schema,
    compile_schema_file,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_SCHEMA = os.path.join(REPO_ROOT, "schemas", "support_ticket.json")


def object_schema(properties, required=None, **extra):
    """Minimal object schema wrapper so each test shows only what it tests."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
    }
    if required is not None:
        schema["required"] = required
    schema.update(extra)
    return schema


# --------------------------------------------------------------------------
# the smoke-test schema
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke() -> CompiledSchema:
    return compile_schema_file(SMOKE_SCHEMA)


def test_smoke_schema_compiles_every_field(smoke):
    assert smoke.skipped == ()
    assert smoke.blocking_fields == ()
    assert list(smoke.questions) == [
        "department",
        "urgency",
        "refund_requested",
        "churn_risk",
    ]


def test_smoke_department_is_a_choice_with_per_option_criteria(smoke):
    question = smoke.questions["department"]
    assert question["type"] == "choice"
    assert question["instructions"] == "Which team should own this ticket?"
    assert list(question["criteria"]) == ["billing", "technical", "sales", "other"]
    assert question["criteria"]["technical"].startswith("bugs, outages")
    assert smoke.specs["department"].values == (
        "billing",
        "technical",
        "sales",
        "other",
    )


def test_smoke_urgency_is_an_ordered_score(smoke):
    question = smoke.questions["urgency"]
    assert question["type"] == "score"
    # criteria is a list, low -> high
    assert question["criteria"] == ["not_urgent", "soon", "critical"]
    assert smoke.specs["urgency"].values == ("not_urgent", "soon", "critical")


def test_smoke_booleans_are_nouls(smoke):
    for name in ("refund_requested", "churn_risk"):
        question = smoke.questions[name]
        assert question["type"] == "noul"
        assert "criteria" not in question
        assert smoke.specs[name].values == (False, True)


def test_smoke_every_field_is_marked_required(smoke):
    assert all(spec.required for spec in smoke.specs.values())


def test_questions_and_skips_partition_the_properties(smoke):
    with open(SMOKE_SCHEMA, "r", encoding="utf-8") as handle:
        properties = set(json.load(handle)["properties"])
    covered = set(smoke.questions) | {s.field for s in smoke.skipped}
    assert covered == properties
    assert set(smoke.specs) == set(smoke.questions)


def test_compilation_is_deterministic(smoke):
    assert compile_schema_file(SMOKE_SCHEMA).to_json() == smoke.to_json()


def test_to_json_round_trips(smoke):
    payload = json.loads(smoke.to_json())
    assert payload["unsupported"] == []
    assert payload["specs"]["urgency"]["values"] == [
        "not_urgent",
        "soon",
        "critical",
    ]


# --------------------------------------------------------------------------
# choice
# --------------------------------------------------------------------------


def test_plain_string_enum_becomes_a_choice_keyed_by_value():
    compiled = compile_schema(
        object_schema({"colour": {"type": "string", "enum": ["red", "green"]}})
    )
    question = compiled.questions["colour"]
    assert question["type"] == "choice"
    # no per-option text in the schema, so the value is its own criterion
    assert question["criteria"] == {"red": "red", "green": "green"}


def test_instructions_fall_back_to_title_then_to_the_field_name():
    compiled = compile_schema(
        object_schema(
            {
                "titled": {"type": "string", "title": "Pick one", "enum": ["a", "b"]},
                "bare": {"type": "string", "enum": ["a", "b"]},
            }
        )
    )
    assert compiled.questions["titled"]["instructions"] == "Pick one"
    assert compiled.questions["bare"]["instructions"] == "What is the value of 'bare'?"


def test_one_of_consts_uses_branch_description_as_criteria_text():
    compiled = compile_schema(
        object_schema(
            {
                "plan": {
                    "oneOf": [
                        {"const": "free", "description": "no card on file"},
                        {"const": "pro", "title": "paid monthly"},
                        {"const": "enterprise"},
                    ]
                }
            }
        )
    )
    assert compiled.questions["plan"]["criteria"] == {
        "free": "no card on file",
        "pro": "paid monthly",
        "enterprise": "enterprise",
    }


def test_choice_over_the_option_limit_is_refused():
    options = ["opt%02d" % i for i in range(MAX_CHOICE_OPTIONS + 1)]
    compiled = compile_schema(object_schema({"big": {"type": "string", "enum": options}}))
    assert compiled.questions == {}
    (skip,) = compiled.skipped
    assert skip.field == "big"
    assert "coarse-to-fine" in skip.reason
    assert skip.route == "generative-track"


def test_choice_at_exactly_the_option_limit_is_accepted():
    options = ["opt%02d" % i for i in range(MAX_CHOICE_OPTIONS)]
    compiled = compile_schema(object_schema({"big": {"type": "string", "enum": options}}))
    assert compiled.questions["big"]["type"] == "choice"


def test_non_string_enum_is_refused():
    compiled = compile_schema(object_schema({"level": {"enum": [1, 2, 3]}}))
    (skip,) = compiled.skipped
    assert "non-string" in skip.reason


def test_one_of_with_a_general_subschema_branch_is_refused():
    compiled = compile_schema(
        object_schema(
            {"mixed": {"oneOf": [{"const": "a"}, {"type": "string", "minLength": 2}]}}
        )
    )
    (skip,) = compiled.skipped
    assert "single `const`" in skip.reason


def test_enum_and_one_of_together_is_a_compiler_error():
    with pytest.raises(CompilerError, match="ambiguous"):
        compile_schema(
            object_schema({"x": {"enum": ["a"], "oneOf": [{"const": "a"}]}})
        )


def test_duplicate_enum_values_are_a_compiler_error():
    with pytest.raises(CompilerError, match="duplicate"):
        compile_schema(object_schema({"x": {"type": "string", "enum": ["a", "a"]}}))


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def test_ordered_marker_turns_an_enum_into_a_score():
    compiled = compile_schema(
        object_schema(
            {"size": {"type": "string", "enum": ["s", "m", "l"], "x-ordered": True}}
        )
    )
    assert compiled.questions["size"]["criteria"] == ["s", "m", "l"]
    assert compiled.questions["size"]["type"] == "score"


def test_ordered_marker_on_one_of_consts_keeps_branch_order():
    compiled = compile_schema(
        object_schema(
            {
                "tier": {
                    "x-ordered": True,
                    "oneOf": [
                        {"const": "low", "description": "can wait"},
                        {"const": "high", "description": "act now"},
                    ],
                }
            }
        )
    )
    assert compiled.questions["tier"]["criteria"] == ["can wait", "act now"]
    assert compiled.specs["tier"].values == ("low", "high")


def test_non_boolean_ordered_marker_is_a_compiler_error():
    with pytest.raises(CompilerError, match="x-ordered"):
        compile_schema(
            object_schema({"x": {"enum": ["a", "b"], "x-ordered": "yes"}})
        )


def test_bounded_integer_becomes_a_score():
    compiled = compile_schema(
        object_schema({"stars": {"type": "integer", "minimum": 1, "maximum": 5}})
    )
    question = compiled.questions["stars"]
    assert question["type"] == "score"
    assert question["criteria"] == ["1", "2", "3", "4", "5"]
    assert compiled.specs["stars"].values == (1, 2, 3, 4, 5)


def test_exclusive_integer_bounds_are_converted_exactly():
    compiled = compile_schema(
        object_schema(
            {
                "n": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 4,
                }
            }
        )
    )
    assert compiled.specs["n"].values == (1, 2, 3)


def test_integer_range_over_the_step_limit_is_refused():
    compiled = compile_schema(
        object_schema({"score": {"type": "integer", "minimum": 0, "maximum": 100}})
    )
    (skip,) = compiled.skipped
    assert "101 steps" in skip.reason
    assert str(MAX_SCORE_STEPS) in skip.reason


def test_unbounded_integer_is_refused():
    compiled = compile_schema(object_schema({"count": {"type": "integer"}}))
    (skip,) = compiled.skipped
    assert "unbounded" in skip.reason


def test_half_bounded_integer_is_refused():
    compiled = compile_schema(
        object_schema({"count": {"type": "integer", "minimum": 0}})
    )
    (skip,) = compiled.skipped
    assert "unbounded" in skip.reason


def test_integer_with_multiple_of_is_refused():
    compiled = compile_schema(
        object_schema(
            {"n": {"type": "integer", "minimum": 0, "maximum": 8, "multipleOf": 2}}
        )
    )
    (skip,) = compiled.skipped
    assert "multipleOf" in skip.reason


def test_more_ordered_levels_than_score_steps_is_refused():
    levels = ["l%d" % i for i in range(MAX_SCORE_STEPS + 1)]
    compiled = compile_schema(
        object_schema({"x": {"type": "string", "enum": levels, "x-ordered": True}})
    )
    (skip,) = compiled.skipped
    assert "over the %d-step" % MAX_SCORE_STEPS in skip.reason


# --------------------------------------------------------------------------
# refusals (section 4.2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subschema,needle",
    [
        ({"type": "string"}, "free-form"),
        ({"type": "string", "pattern": "^a+$"}, "free-form"),
        ({"type": "number", "minimum": 0, "maximum": 1}, "continuous"),
        ({"type": "array", "items": {"type": "string"}}, "`array`"),
        ({"type": "object", "properties": {}}, "`object`"),
        ({"type": ["string", "null"]}, "union type"),
        ({}, "no supported question type"),
        ({"const": "fixed"}, "no supported question type"),
        (
            {"type": "string", "enum": ["a"], "allOf": [{"minLength": 1}]},
            "unsupported keywords",
        ),
        (
            {"if": {"const": "a"}, "then": {"const": "b"}},
            "unsupported keywords",
        ),
    ],
)
def test_unsupported_property_shapes_are_reported_not_approximated(subschema, needle):
    compiled = compile_schema(object_schema({"x": subschema}))
    assert compiled.questions == {}
    (skip,) = compiled.skipped
    assert skip.field == "x"
    assert needle in skip.reason
    assert skip.route == "generative-track"


def test_unsupported_report_row_shape():
    compiled = compile_schema(object_schema({"note": {"type": "string"}}))
    assert compiled.unsupported_report == [
        {
            "field": "note",
            "reason": compiled.skipped[0].reason,
            "route": "generative-track",
        }
    ]


def test_supported_fields_still_compile_next_to_a_refused_one():
    compiled = compile_schema(
        object_schema(
            {
                "summary": {"type": "string"},
                "urgent": {"type": "boolean"},
            }
        )
    )
    assert list(compiled.questions) == ["urgent"]
    assert [s.field for s in compiled.skipped] == ["summary"]


def test_a_required_but_unsupported_field_blocks_the_schema():
    compiled = compile_schema(
        object_schema(
            {"summary": {"type": "string"}, "urgent": {"type": "boolean"}},
            required=["summary", "urgent"],
        )
    )
    # A required field we cannot answer means no assembled instance can ever
    # validate; the caller must not pretend otherwise.
    assert compiled.blocking_fields == ("summary",)


def test_optional_compiled_fields_are_flagged_as_always_filled():
    compiled = compile_schema(
        object_schema(
            {"urgent": {"type": "boolean"}, "vip": {"type": "boolean"}},
            required=["urgent"],
        )
    )
    # The model answers both questions, so `vip` is always present in the
    # instance even though the schema does not require it.
    assert compiled.optional_fields == ("vip",)
    assert json.loads(compiled.to_json())["always_filled_optional_fields"] == ["vip"]


def test_smoke_schema_has_no_always_filled_optional_fields(smoke):
    assert smoke.optional_fields == ()


def test_an_optional_unsupported_field_does_not_block():
    compiled = compile_schema(
        object_schema(
            {"summary": {"type": "string"}, "urgent": {"type": "boolean"}},
            required=["urgent"],
        )
    )
    assert compiled.blocking_fields == ()


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("if", {"required": ["a"]}),
        ("allOf", [{"required": ["a"]}]),
        ("anyOf", [{"required": ["a"]}]),
        ("oneOf", [{"required": ["a"]}]),
        ("not", {"required": ["a"]}),
        ("dependentSchemas", {"a": {"required": ["a"]}}),
        ("dependentRequired", {"a": ["a"]}),
        ("patternProperties", {"^x-": {"type": "string"}}),
        ("propertyNames", {"pattern": "^a"}),
        ("unevaluatedProperties", False),
    ],
)
def test_root_keywords_that_change_field_presence_are_refused(keyword, value):
    schema = object_schema({"a": {"type": "boolean"}})
    schema[keyword] = value
    with pytest.raises(CompilerError, match="fixed question set"):
        compile_schema(schema)


def test_additional_properties_as_a_schema_is_refused():
    schema = object_schema({"a": {"type": "boolean"}})
    schema["additionalProperties"] = {"type": "string"}
    with pytest.raises(CompilerError, match="open-ended map"):
        compile_schema(schema)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_additional_properties_is_fine(value):
    schema = object_schema({"a": {"type": "boolean"}})
    schema["additionalProperties"] = value
    assert list(compile_schema(schema).questions) == ["a"]


def test_non_object_root_is_refused():
    with pytest.raises(CompilerError, match="type"):
        compile_schema({"type": "array", "items": {"type": "string"}})


def test_root_without_properties_is_refused():
    with pytest.raises(CompilerError, match="no `properties`"):
        compile_schema({"type": "object"})


def test_required_naming_a_missing_property_is_refused():
    with pytest.raises(CompilerError, match="do not exist"):
        compile_schema(
            object_schema({"a": {"type": "boolean"}}, required=["a", "ghost"])
        )


# --------------------------------------------------------------------------
# $ref handling
# --------------------------------------------------------------------------


def test_local_pointer_refs_are_resolved():
    schema = object_schema(
        {"urgency": {"$ref": "#/$defs/urgency"}},
        **{
            "$defs": {
                "urgency": {
                    "type": "string",
                    "enum": ["low", "high"],
                    "x-ordered": True,
                    "description": "how fast",
                }
            }
        }
    )
    compiled = compile_schema(schema)
    assert compiled.questions["urgency"]["type"] == "score"
    assert compiled.questions["urgency"]["instructions"] == "how fast"


def test_annotation_next_to_a_ref_overrides_the_target_annotation():
    schema = object_schema(
        {"urgency": {"$ref": "#/$defs/u", "description": "local text"}},
        **{"$defs": {"u": {"type": "boolean", "description": "shared text"}}}
    )
    compiled = compile_schema(schema)
    assert compiled.questions["urgency"]["instructions"] == "local text"


def test_validation_siblings_next_to_a_ref_are_refused():
    schema = object_schema(
        {"x": {"$ref": "#/$defs/u", "minLength": 2}},
        **{"$defs": {"u": {"type": "string", "enum": ["a", "bb"]}}}
    )
    with pytest.raises(CompilerError, match="validation siblings"):
        compile_schema(schema)


def test_remote_refs_are_refused_with_a_bundling_hint():
    schema = object_schema({"x": {"$ref": "https://example.com/u.json"}})
    with pytest.raises(CompilerError, match="bundle the schema first"):
        compile_schema(schema)


def test_unresolvable_local_ref_is_refused():
    schema = object_schema({"x": {"$ref": "#/$defs/missing"}})
    with pytest.raises(CompilerError, match="does not resolve"):
        compile_schema(schema)


def test_anchor_refs_are_refused():
    schema = object_schema({"x": {"$ref": "#urgency"}})
    with pytest.raises(CompilerError, match="anchor"):
        compile_schema(schema)


def test_circular_refs_are_refused():
    schema = object_schema(
        {"x": {"$ref": "#/$defs/a"}},
        **{"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}}
    )
    with pytest.raises(CompilerError, match="circular"):
        compile_schema(schema)
