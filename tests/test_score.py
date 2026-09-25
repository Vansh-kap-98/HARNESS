"""Unit tests for src/score.py."""

from __future__ import annotations

import math

import pytest

from src.score import (
    MIN_CALIBRATION_N,
    MIN_TREND_N,
    Proportion,
    ScoreError,
    calibration,
    exact_match,
    latency_summary,
    majority_class,
    majority_class_predictions,
    per_field_accuracy,
    percentile,
    summary_notes,
    validity_bug_count,
    wilson_interval,
)

FIELDS = ["department", "urgency", "refund_requested", "churn_risk"]


def record(department="billing", urgency="soon", refund=True, churn=False):
    return {
        "department": department,
        "urgency": urgency,
        "refund_requested": refund,
        "churn_risk": churn,
    }


# --------------------------------------------------------------------------
# Wilson intervals
# --------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(7, 10)
    assert low < 0.7 < high


def test_wilson_interval_stays_inside_zero_and_one():
    assert wilson_interval(0, 5) == (0.0, pytest.approx(0.4344, abs=1e-3))
    assert wilson_interval(5, 5)[1] == 1.0
    assert wilson_interval(0, 5)[0] == 0.0


def test_wilson_interval_narrows_as_n_grows():
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_matches_a_known_value():
    # 50/100 -> 0.5 [0.4038, 0.5962] (standard textbook value)
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_on_no_data_is_nan():
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ScoreError):
        wilson_interval(5, 3)


def test_proportion_renders_with_its_interval():
    assert str(Proportion(7, 10)).startswith("0.700 [")


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------


def test_per_field_accuracy_counts_each_field_separately():
    golds = [record(), record(department="technical")]
    predictions = [record(), record()]
    scores = per_field_accuracy(golds, predictions, FIELDS)
    assert scores["department"].accuracy.successes == 1
    assert scores["urgency"].accuracy.successes == 2
    assert scores["department"].accuracy.total == 2


def test_a_missing_prediction_counts_as_wrong_and_is_flagged():
    golds = [record(), record()]
    predictions = [record(), {"department": "billing"}]
    scores = per_field_accuracy(golds, predictions, FIELDS)
    assert scores["urgency"].accuracy.successes == 1
    assert scores["urgency"].missing == 1
    assert scores["department"].missing == 0


def test_true_does_not_match_one():
    # In Python `True == 1`; in JSON they are different values, and a model
    # that emits 1 for a boolean field has not got it right.
    golds = [record(refund=True)]
    predictions = [record(refund=1)]
    scores = per_field_accuracy(golds, predictions, FIELDS)
    assert scores["refund_requested"].accuracy.successes == 0


def test_exact_match_needs_every_field():
    golds = [record(), record()]
    predictions = [record(), record(churn=True)]
    assert exact_match(golds, predictions, FIELDS).successes == 1


def test_exact_match_fails_on_a_missing_field():
    golds = [record()]
    predictions = [{"department": "billing", "urgency": "soon", "churn_risk": False}]
    assert exact_match(golds, predictions, FIELDS).successes == 0


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ScoreError, match="1 gold records but 2"):
        exact_match([record()], [record(), record()], FIELDS)


def test_scoring_nothing_is_an_error():
    with pytest.raises(ScoreError, match="no records"):
        exact_match([], [], FIELDS)


def test_gold_missing_a_field_is_a_dataset_error():
    with pytest.raises(ScoreError, match="gold record is missing"):
        per_field_accuracy([{"department": "billing"}], [record()], FIELDS)


# --------------------------------------------------------------------------
# the majority-class floor
# --------------------------------------------------------------------------


def test_majority_class_picks_the_most_common_value():
    golds = [
        record(department="billing"),
        record(department="billing"),
        record(department="sales"),
    ]
    assert majority_class(golds, FIELDS)["department"] == "billing"


def test_majority_class_is_order_independent():
    a = [record(department="billing"), record(department="sales")]
    b = list(reversed(a))
    assert majority_class(a, FIELDS) == majority_class(b, FIELDS)


def test_majority_class_predictions_are_scorable():
    golds = [record(), record(), record(department="sales")]
    predictions = majority_class_predictions(golds, FIELDS)
    scores = per_field_accuracy(golds, predictions, FIELDS)
    # the floor this run has to beat: 2/3 on department, 3/3 on the rest
    assert scores["department"].accuracy.successes == 2
    assert scores["churn_risk"].accuracy.successes == 3


def test_majority_class_needs_records():
    with pytest.raises(ScoreError, match="no gold records"):
        majority_class([], FIELDS)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def test_a_perfectly_calibrated_model_has_near_zero_ece():
    # 10 decisions at confidence 0.9, 9 of them right
    confidences = [0.9] * 10
    correct = [True] * 9 + [False]
    assert calibration(confidences, correct).ece == pytest.approx(0.0, abs=1e-9)


def test_an_overconfident_model_has_ece_equal_to_the_gap():
    confidences = [0.95] * 10
    correct = [True] * 5 + [False] * 5
    assert calibration(confidences, correct).ece == pytest.approx(0.45)


def test_calibration_is_not_reportable_below_the_minimum():
    small = calibration([0.9] * 10, [True] * 10)
    assert small.total == 10
    assert small.reportable is False
    big = calibration([0.9] * MIN_CALIBRATION_N, [True] * MIN_CALIBRATION_N)
    assert big.reportable is True


def test_reliability_table_covers_the_unit_interval():
    result = calibration([0.05, 0.95], [False, True])
    assert len(result.bins) == 10
    assert result.bins[0].count == 1
    assert result.bins[-1].count == 1
    assert math.isnan(result.bins[5].accuracy)


def test_confidence_of_one_lands_in_the_last_bin():
    result = calibration([1.0], [True])
    assert result.bins[-1].count == 1


def test_calibration_rejects_mismatched_inputs():
    with pytest.raises(ScoreError, match="2 confidences but 1"):
        calibration([0.5, 0.5], [True])


def test_calibration_rejects_out_of_range_confidence():
    with pytest.raises(ScoreError, match=r"\[0, 1\]"):
        calibration([1.5], [True])


# --------------------------------------------------------------------------
# validity and latency
# --------------------------------------------------------------------------


def test_validity_counts_bugs_not_score():
    assert validity_bug_count([True, True, False]) == 1
    assert validity_bug_count([True, True]) == 0


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


def test_percentile_of_one_value():
    assert percentile([7.0], 95) == 7.0


def test_latency_summary_uses_per_example_medians_for_p50():
    repeats = [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 9.0],  # one slow outlier in one repeat
        [0.9, 1.9, 3.1],
    ]
    summary = latency_summary(repeats)
    assert summary.examples == 3
    assert summary.repeats == 3
    # per-example medians are 1.0, 2.0, 3.1 -> median 2.0
    assert summary.p50 == pytest.approx(2.0)
    # p95 comes from the pooled times, so the outlier still shows up
    assert summary.p95 > 3.1


def test_latency_summary_rejects_ragged_repeats():
    with pytest.raises(ScoreError, match="different numbers of examples"):
        latency_summary([[1.0, 2.0], [1.0]])


def test_latency_summary_rejects_negative_times():
    with pytest.raises(ScoreError, match="non-negative"):
        latency_summary([[-1.0]])


def test_latency_summary_needs_repeats():
    with pytest.raises(ScoreError, match="no repeats"):
        latency_summary([])


# --------------------------------------------------------------------------
# the caveats a run must be published with
# --------------------------------------------------------------------------


def test_a_fifty_example_four_field_run_clears_both_thresholds():
    # 50 examples x 4 fields = 200 decisions, exactly the two floors
    assert summary_notes(examples=50, decisions=200) == ()


def test_a_small_run_carries_both_caveats():
    notes = summary_notes(examples=20, decisions=80)
    assert len(notes) == 2
    assert str(MIN_TREND_N) in notes[0]
    assert "rule 12" in notes[1]


def test_a_forty_example_smoke_run_is_flagged():
    notes = summary_notes(examples=40, decisions=160)
    assert any("no trend may be claimed" in note for note in notes)
