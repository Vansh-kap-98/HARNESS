"""Unit tests for src/runners/laya_runner.py.

The adapter is tested against fake Laya results, so none of this needs the
`laya` package, a download, or a GPU. `load_router` is the only part that
touches the real package and it is a two-line wrapper.
"""

from __future__ import annotations

import pytest

from src.assembler import assemble
from src.compiler import compile_schema_file
from src.runners.laya_runner import (
    NOUL_DECISION_BOUNDARY,
    LayaRunnerError,
    reported_confidence,
    to_answers,
)

import os

SMOKE_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "schemas",
    "support_ticket.json",
)


@pytest.fixture(scope="module")
def compiled():
    return compile_schema_file(SMOKE_SCHEMA)


def laya_result(
    department_probs=(0.91, 0.05, 0.03, 0.01),
    urgency_probs=(0.1, 0.2, 0.7),
    refund=0.88,
    churn=0.12,
    department_choice="billing",
):
    """A result shaped like `rl_agent_api.py` builds them."""
    return {
        "answers": {
            "department": {
                "type": "choice",
                "choice": department_choice,
                "probabilities": list(department_probs),
                "confidence": 0.94,
                "rl_agent": "english",
            },
            "urgency": {
                "type": "score",
                "score": 2.0,
                "legend": ["not_urgent", "soon", "critical"],
                "probabilities": list(urgency_probs),
                "confidence": 0.81,
                "rl_agent": "english",
            },
            "refund_requested": {"type": "noul", "noul": refund, "rl_agent": "english"},
            "churn_risk": {"type": "noul", "noul": churn, "rl_agent": "english"},
        },
        "routing": {"model": "english", "repo": "convaiinnovations/laya"},
    }


# --------------------------------------------------------------------------
# the happy path, all the way to an instance
# --------------------------------------------------------------------------


def test_a_full_result_becomes_an_instance(compiled):
    answers = to_answers(laya_result(), compiled)
    assembled = assemble(compiled, answers)
    assert assembled.instance == {
        "department": "billing",
        "urgency": "critical",
        "refund_requested": True,
        "churn_risk": False,
    }


def test_choice_confidence_is_the_chosen_options_probability(compiled):
    answers = to_answers(laya_result(), compiled)
    assert answers["department"].confidence == pytest.approx(0.91)
    # not laya's own `confidence` field, which says 0.94
    assert answers["department"].distribution == (0.91, 0.05, 0.03, 0.01)


def test_score_is_read_from_probabilities_not_the_score_float(compiled):
    # `score` is 2.0 here, and the argmax is also index 2, but the point is
    # that a differently scaled float cannot shift the answer.
    answers = to_answers(laya_result(urgency_probs=(0.7, 0.2, 0.1)), compiled)
    assert answers["urgency"].index == 0
    assert answers["urgency"].confidence == pytest.approx(0.7)


@pytest.mark.parametrize(
    "probability,expected_index,expected_confidence",
    [
        (0.88, 1, 0.88),
        (0.12, 0, 0.88),
        (0.5, 1, 0.5),
        (0.49, 0, 0.51),
        (0.0, 0, 1.0),
        (1.0, 1, 1.0),
    ],
)
def test_noul_probability_becomes_a_boolean_and_a_confidence(
    compiled, probability, expected_index, expected_confidence
):
    answers = to_answers(laya_result(refund=probability), compiled)
    answer = answers["refund_requested"]
    assert answer.index == expected_index
    assert answer.confidence == pytest.approx(expected_confidence)
    assert answer.distribution == pytest.approx((1 - probability, probability))


def test_the_decision_boundary_is_the_midpoint():
    assert NOUL_DECISION_BOUNDARY == 0.5


def test_reported_confidence_is_kept_for_comparison(compiled):
    reported = reported_confidence(laya_result(), compiled)
    assert reported == {"department": 0.94, "urgency": 0.81}
    assert "churn_risk" not in reported  # noul answers carry no confidence key


# --------------------------------------------------------------------------
# things that must fail loudly rather than score
# --------------------------------------------------------------------------


def test_a_misaligned_probability_vector_is_refused(compiled):
    """The gold check: `choice` says billing, argmax says technical."""
    result = laya_result(department_probs=(0.05, 0.91, 0.03, 0.01))
    with pytest.raises(LayaRunnerError, match="not aligned with the option order"):
        to_answers(result, compiled)


def test_the_wrong_number_of_probabilities_is_refused(compiled):
    result = laya_result(department_probs=(0.5, 0.5))
    with pytest.raises(LayaRunnerError, match="2 probabilities for 4 options"):
        to_answers(result, compiled)


def test_probabilities_that_do_not_sum_to_one_are_refused(compiled):
    result = laya_result(department_probs=(0.5, 0.2, 0.1, 0.1))
    with pytest.raises(LayaRunnerError, match="sum to"):
        to_answers(result, compiled)


def test_a_missing_probabilities_key_is_refused(compiled):
    result = laya_result()
    del result["answers"]["department"]["probabilities"]
    with pytest.raises(LayaRunnerError, match="no usable confidence"):
        to_answers(result, compiled)


def test_an_option_that_was_never_offered_is_refused(compiled):
    result = laya_result(department_choice="marketing")
    with pytest.raises(LayaRunnerError, match="never offered"):
        to_answers(result, compiled)


def test_a_missing_answer_is_refused(compiled):
    result = laya_result()
    del result["answers"]["churn_risk"]
    with pytest.raises(LayaRunnerError, match="no answer for question"):
        to_answers(result, compiled)


def test_a_result_without_answers_is_refused(compiled):
    with pytest.raises(LayaRunnerError, match="no `answers` mapping"):
        to_answers({"routing": {"model": "english"}}, compiled)


@pytest.mark.parametrize("bad", [1.5, -0.1, "yes", True, None])
def test_a_bad_noul_value_is_refused(compiled, bad):
    result = laya_result()
    result["answers"]["churn_risk"]["noul"] = bad
    with pytest.raises(LayaRunnerError):
        to_answers(result, compiled)


def test_a_noul_answer_without_its_value_is_refused(compiled):
    result = laya_result()
    del result["answers"]["churn_risk"]["noul"]
    with pytest.raises(LayaRunnerError, match="no `noul` value"):
        to_answers(result, compiled)
