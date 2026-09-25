"""Unit tests for src/assembler.py."""

from __future__ import annotations

import json
import os

import pytest

from src.assembler import (
    Answer,
    Assembled,
    AssemblyError,
    assemble,
    escalate_fields,
)
from src.compiler import compile_schema, compile_schema_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_SCHEMA = os.path.join(REPO_ROOT, "schemas", "support_ticket.json")


@pytest.fixture(scope="module")
def smoke():
    return compile_schema_file(SMOKE_SCHEMA)


@pytest.fixture
def smoke_answers():
    return {
        "department": Answer(index=0, confidence=0.91),
        "urgency": Answer(index=2, confidence=0.72),
        "refund_requested": Answer(index=1, confidence=0.88),
        "churn_risk": Answer(index=0, confidence=0.55),
    }


def object_schema(properties, required=None):
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
    }
    if required is not None:
        schema["required"] = required
    return schema


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_answers_become_the_schema_values(smoke, smoke_answers):
    result = assemble(smoke, smoke_answers)
    assert result.instance == {
        "department": "billing",
        "urgency": "critical",
        "refund_requested": True,
        "churn_risk": False,
    }


def test_field_order_follows_the_schema(smoke, smoke_answers):
    assert list(assemble(smoke, smoke_answers).instance) == list(smoke.questions)


def test_confidence_is_kept_outside_the_instance(smoke, smoke_answers):
    result = assemble(smoke, smoke_answers)
    assert "confidence" not in result.instance
    assert set(result.confidence) == set(result.instance)
    payload = json.loads(result.to_json())
    assert payload["instance"]["department"] == "billing"
    assert payload["confidence"]["department"] == pytest.approx(0.91)


def test_min_confidence(smoke, smoke_answers):
    assert assemble(smoke, smoke_answers).min_confidence == pytest.approx(0.55)


def test_assembly_is_deterministic(smoke, smoke_answers):
    first = assemble(smoke, smoke_answers)
    second = assemble(smoke, smoke_answers)
    assert first.to_json() == second.to_json()


def test_noul_index_zero_is_false(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["refund_requested"] = Answer(index=0, confidence=0.6)
    result = assemble(smoke, answers)
    assert result.instance["refund_requested"] is False


def test_optional_fields_are_still_filled():
    compiled = compile_schema(
        object_schema(
            {"urgent": {"type": "boolean"}, "vip": {"type": "boolean"}},
            required=["urgent"],
        )
    )
    result = assemble(
        compiled,
        {
            "urgent": Answer(index=1, confidence=0.9),
            "vip": Answer(index=0, confidence=0.9),
        },
    )
    # The model answers every question, so `vip` is present even though the
    # schema does not require it. compiled.optional_fields warns about this.
    assert result.instance == {"urgent": True, "vip": False}
    assert compiled.optional_fields == ("vip",)


# --------------------------------------------------------------------------
# Answer.from_distribution
# --------------------------------------------------------------------------


def test_from_distribution_takes_the_argmax():
    answer = Answer.from_distribution([0.1, 0.7, 0.2])
    assert answer.index == 1
    assert answer.confidence == pytest.approx(0.7)
    assert answer.distribution == (0.1, 0.7, 0.2)


def test_from_distribution_breaks_ties_on_the_lowest_index():
    assert Answer.from_distribution([0.5, 0.5]).index == 0


def test_from_distribution_rejects_out_of_range_probabilities():
    with pytest.raises(AssemblyError, match=r"\[0, 1\]"):
        Answer.from_distribution([1.4, -0.4])


def test_from_distribution_rejects_an_empty_vector():
    with pytest.raises(AssemblyError, match="empty"):
        Answer.from_distribution([])


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_blocked_schema_is_never_assembled():
    compiled = compile_schema(
        object_schema(
            {"summary": {"type": "string"}, "urgent": {"type": "boolean"}},
            required=["summary", "urgent"],
        )
    )
    with pytest.raises(AssemblyError, match="out of scope"):
        assemble(compiled, {"urgent": Answer(index=1, confidence=0.9)})


def test_a_missing_answer_is_an_error_not_a_guess(smoke, smoke_answers):
    answers = dict(smoke_answers)
    del answers["urgency"]
    with pytest.raises(AssemblyError, match="no answer for questions"):
        assemble(smoke, answers)


def test_an_answer_for_an_unasked_field_is_an_error(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["invented"] = Answer(index=0, confidence=1.0)
    with pytest.raises(AssemblyError, match="never asked"):
        assemble(smoke, answers)


@pytest.mark.parametrize("index", [-1, 4, 99])
def test_an_out_of_range_index_is_an_error(smoke, smoke_answers, index):
    answers = dict(smoke_answers)
    answers["department"] = Answer(index=index, confidence=0.9)
    with pytest.raises(AssemblyError, match="outside the 4 option"):
        assemble(smoke, answers)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_a_confidence_outside_zero_to_one_is_an_error(
    smoke, smoke_answers, confidence
):
    answers = dict(smoke_answers)
    answers["churn_risk"] = Answer(index=0, confidence=confidence)
    with pytest.raises(AssemblyError):
        assemble(smoke, answers)


def test_a_boolean_index_is_an_error(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["churn_risk"] = Answer(index=True, confidence=0.9)
    with pytest.raises(AssemblyError, match="must be an int"):
        assemble(smoke, answers)


def test_a_distribution_of_the_wrong_length_is_an_error(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["department"] = Answer(
        index=0, confidence=0.9, distribution=(0.9, 0.1)
    )
    with pytest.raises(AssemblyError, match="2 entries but the question has 4"):
        assemble(smoke, answers)


def test_a_distribution_that_does_not_sum_to_one_is_an_error(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["urgency"] = Answer(
        index=0, confidence=0.5, distribution=(0.5, 0.2, 0.1)
    )
    with pytest.raises(AssemblyError, match="sums to"):
        assemble(smoke, answers)


def test_confidence_must_match_the_distribution(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["urgency"] = Answer(
        index=0, confidence=0.9, distribution=(0.2, 0.3, 0.5)
    )
    with pytest.raises(AssemblyError, match="does not match distribution"):
        assemble(smoke, answers)


def test_a_consistent_distribution_is_accepted(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["urgency"] = Answer(
        index=2, confidence=0.5, distribution=(0.2, 0.3, 0.5)
    )
    assert assemble(smoke, answers).instance["urgency"] == "critical"


def test_a_non_answer_value_is_an_error(smoke, smoke_answers):
    answers = dict(smoke_answers)
    answers["churn_risk"] = 1  # type: ignore[assignment]
    with pytest.raises(AssemblyError, match="expected an Answer"):
        assemble(smoke, answers)


# --------------------------------------------------------------------------
# escalation
# --------------------------------------------------------------------------


def test_escalate_fields_returns_low_confidence_fields_in_instance_order(
    smoke, smoke_answers
):
    result = assemble(smoke, smoke_answers)
    assert escalate_fields(result, 0.8) == ("urgency", "churn_risk")


def test_escalate_fields_is_strictly_below_the_threshold(smoke, smoke_answers):
    result = assemble(smoke, smoke_answers)
    assert escalate_fields(result, 0.55) == ()


@pytest.mark.parametrize("threshold", [-0.1, 1.1, "0.5", None, True])
def test_escalate_fields_rejects_a_bad_threshold(smoke, smoke_answers, threshold):
    result = assemble(smoke, smoke_answers)
    with pytest.raises(AssemblyError):
        escalate_fields(result, threshold)


def test_escalate_fields_has_no_default_threshold():
    # A default here would quietly become a published result; Phase 4 fits it
    # on val and records it in the run config.
    with pytest.raises(TypeError):
        escalate_fields(Assembled(instance={}, confidence={}))  # type: ignore[call-arg]
