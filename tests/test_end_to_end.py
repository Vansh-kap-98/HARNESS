"""The whole harness on three examples, with a stub model.

schema -> compiler -> questions -> [stub model] -> answers -> assembler
-> instance -> validation -> metrics

The stub is a keyword matcher, not a model, and its accuracy means nothing.
What this test pins down is the wiring: that the pieces agree on the answer
space, that every assembled instance validates, and that the metrics come out
with the caveats this sample size demands.
"""

from __future__ import annotations

import os
from typing import Dict

import pytest

from src import validate as V
from src.assembler import Answer, assemble, escalate_fields
from src.compiler import CompiledSchema, compile_schema_file
from src.dataset import check_records, load_jsonl
from src.score import (
    calibration,
    exact_match,
    majority_class_predictions,
    per_field_accuracy,
    summary_notes,
    validity_bug_count,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_SCHEMA = os.path.join(REPO_ROOT, "schemas", "support_ticket.json")
MINI = os.path.join(REPO_ROOT, "tests", "fixtures", "mini.jsonl")


def stub_model(state: str, compiled: CompiledSchema) -> Dict[str, Answer]:
    """A deterministic keyword matcher standing in for a real runner."""
    text = state.lower()
    scores = {
        "billing": sum(word in text for word in ("charge", "refund", "invoice", "bill")),
        "technical": sum(word in text for word in ("api", "503", "outage", "error", "down")),
        "sales": sum(word in text for word in ("upgrade", "seat", "pricing", "contract")),
        "other": 0,
    }
    options = compiled.specs["department"].values
    weights = [max(scores[option], 0.01) for option in options]
    total = sum(weights)
    department = Answer.from_distribution([w / total for w in weights])

    urgency_level = 2 if any(w in text for w in ("down", "today", "outage")) else 1
    refund = any(word in text for word in ("refund", "back", "credit"))
    churn = any(word in text for word in ("cancel", "competitor", "leave"))
    return {
        "department": department,
        "urgency": Answer(index=urgency_level, confidence=0.66),
        "refund_requested": Answer(index=int(refund), confidence=0.81),
        "churn_risk": Answer(index=int(churn), confidence=0.74),
    }


@pytest.fixture(scope="module")
def compiled():
    return compile_schema_file(SMOKE_SCHEMA)


@pytest.fixture(scope="module")
def records():
    return load_jsonl(MINI)


@pytest.fixture(scope="module")
def run(compiled, records):
    return [(record, assemble(compiled, stub_model(record.state, compiled))) for record in records]


def test_the_data_agrees_with_the_compiled_schema(compiled, records):
    assert check_records(records, compiled) == ()
    assert compiled.skipped == ()
    assert compiled.blocking_fields == ()


def test_every_instance_validates(monkeypatch, stub_cli, run):
    """Validity is 100% by construction. A failure here is a pipeline bug."""
    monkeypatch.setenv(V.CLI_ENV_VAR, stub_cli())
    V._probe_cache.clear()
    validities = []
    for _record, assembled in run:
        result = V.require_authoritative(V.validate(SMOKE_SCHEMA, assembled.instance))
        validities.append(result.valid)
    V._probe_cache.clear()
    assert validity_bug_count(validities) == 0


def test_the_dev_engine_agrees(run):
    for _record, assembled in run:
        assert V.validate_with_python_jsonschema(SMOKE_SCHEMA, assembled.instance).valid


def test_instances_carry_every_field_in_schema_order(compiled, run):
    for _record, assembled in run:
        assert list(assembled.instance) == list(compiled.questions)


def test_confidence_never_leaks_into_the_instance(run):
    for _record, assembled in run:
        assert all(not isinstance(v, dict) for v in assembled.instance.values())
        assert set(assembled.confidence) == set(assembled.instance)


def test_metrics_run_over_the_whole_set(compiled, run):
    fields = list(compiled.questions)
    golds = [record.gold for record, _ in run]
    predictions = [assembled.instance for _, assembled in run]

    scores = per_field_accuracy(golds, predictions, fields)
    assert set(scores) == set(fields)
    assert all(score.missing == 0 for score in scores.values())
    assert all(0 <= score.accuracy.value <= 1 for score in scores.values())

    overall = exact_match(golds, predictions, fields)
    assert 0 <= overall.value <= 1
    assert overall.low <= overall.value <= overall.high


def test_the_floor_baseline_is_scored_on_the_same_records(compiled, run):
    fields = list(compiled.questions)
    golds = [record.gold for record, _ in run]
    floor = majority_class_predictions(golds, fields)
    scores = per_field_accuracy(golds, floor, fields)
    # every model in the ladder has to be compared against this
    assert all(score.accuracy.total == len(golds) for score in scores.values())


def test_calibration_over_this_run_is_not_reportable(compiled, run):
    fields = list(compiled.questions)
    confidences = []
    correct = []
    for record, assembled in run:
        for name in fields:
            confidences.append(assembled.confidence[name])
            correct.append(assembled.instance[name] == record.gold[name])
    result = calibration(confidences, correct)
    assert result.total == len(fields) * len(run)
    assert result.reportable is False  # 12 decisions


def test_the_run_carries_its_caveats(compiled, run):
    notes = summary_notes(examples=len(run), decisions=len(run) * len(compiled.questions))
    assert len(notes) == 2
    assert any("no trend may be claimed" in note for note in notes)
    assert any("knee point" in note for note in notes)


def test_escalation_picks_the_least_confident_fields(run):
    _record, assembled = run[0]
    escalated = escalate_fields(assembled, 0.70)
    assert "urgency" in escalated  # stub confidence 0.66
    assert "refund_requested" not in escalated  # 0.81
