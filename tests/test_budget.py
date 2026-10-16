"""Unit tests for src/budget.py.

The arithmetic replicates laya/common.py. The reference points below are the
ones measured against the real library on Banking77 and recorded in
results/banking77_k_sweep.md, so a change that breaks them breaks agreement
with the model we are predicting.
"""

from __future__ import annotations

import pytest

from src.budget import (
    HEAD_SLACK,
    MIN_TOKENS_PER_OPTION,
    BudgetError,
    option_budget,
    render_option,
    usable_option_count,
)

# A stand-in tokenizer: one token per 4 characters, rounded up. Deterministic
# and monotonic, which is all the arithmetic needs. Real runs pass the model's
# own tokenizer.
def chars4(text):
    return (len(text) + 3) // 4


def banking_criteria(count):
    """Banking77-shaped options: `label: label with spaces`."""
    names = ["card_arrival", "card_not_working", "card_delivery_estimate",
             "pending_cash_withdrawal", "transaction_charged_twice",
             "beneficiary_not_allowed", "terminate_account", "pin_blocked",
             "exchange_rate", "unable_to_verify_identity", "card_linking",
             "verify_source_of_funds"]
    out = {}
    index = 0
    while len(out) < count:
        base = names[index % len(names)]
        key = base if index < len(names) else "%s_%d" % (base, index)
        out[key] = key.replace("_", " ")
        index += 1
    return out


# --------------------------------------------------------------------------
# the arithmetic the library actually uses
# --------------------------------------------------------------------------


def test_options_that_fit_are_untouched():
    report = option_budget(banking_criteria(4), 192, chars4)
    assert report.truncated is False
    assert report.cut == ()
    assert report.collisions == ()
    assert report.usable is True


def test_truncation_begins_only_when_slack_runs_out():
    """The library truncates when fewer than HEAD_SLACK tokens remain."""
    criteria = {"a": "x" * 40}
    exact = option_budget(criteria, chars4(render_option("a", "x" * 40)) + HEAD_SLACK, chars4)
    assert exact.truncated is False
    tight = option_budget(criteria, chars4(render_option("a", "x" * 40)) + HEAD_SLACK - 1, chars4)
    assert tight.truncated is True


def long_criteria(count):
    """Options long enough that they always overflow the head budget."""
    return {"option_%03d" % i: "a fairly wordy description of option %03d "
                               "that will not fit alongside the others" % i
            for i in range(count)}


def test_tokens_per_option_follows_the_library_formula():
    """max(4, (head_max_len - 16) // k), the library's own arithmetic."""
    for count, head, expected in [(32, 192, 5), (64, 192, 4), (77, 192, 4),
                                  (32, 384, 11), (64, 384, 5), (77, 384, 4)]:
        report = option_budget(long_criteria(count), head, chars4)
        assert report.truncated is True, (count, head)
        assert report.tokens_per_option == expected, (count, head)


def test_the_floor_is_never_breached():
    report = option_budget(banking_criteria(200), 192, chars4)
    assert report.tokens_per_option == MIN_TOKENS_PER_OPTION


# --------------------------------------------------------------------------
# collisions, which are the thing that actually breaks a run
# --------------------------------------------------------------------------


def test_options_sharing_a_prefix_collide_once_truncated():
    """What happened on Banking77: card_not_working became card_not_."""
    criteria = {
        "card_not_working_quickly": "card not working quickly",
        "card_not_working_at_all": "card not working at all",
        "exchange_rate": "exchange rate",
    }
    # head 20 with 3 options puts every option on the 4-token floor, which is
    # 16 characters here, shorter than the prefix these two share.
    report = option_budget(criteria, 20, chars4)
    assert report.truncated is True
    assert report.usable is False
    assert any(set(group) == {"card_not_working_quickly", "card_not_working_at_all"}
               for group in report.collisions)


def test_distinct_counts_collapsed_groups_once():
    criteria = {"shared_prefix_longer_aaa": "x",
                "shared_prefix_longer_bbb": "y",
                "other": "z"}
    report = option_budget(criteria, 20, chars4)
    assert report.collisions
    assert report.distinct == report.options - 1


def test_a_usable_report_explains_nothing():
    assert option_budget(banking_criteria(4), 192, chars4).reason() == ""


def test_an_unusable_report_shows_its_arithmetic():
    criteria = {"card_not_working_quickly": "card not working quickly",
                "card_not_working_at_all": "card not working at all"}
    reason = option_budget(criteria, 20, chars4).reason()
    assert "tokens each" in reason
    assert "identical strings" in reason


# --------------------------------------------------------------------------
# usable_option_count
# --------------------------------------------------------------------------


def test_usable_option_count_is_the_whole_set_when_everything_fits():
    criteria = banking_criteria(4)
    assert usable_option_count(criteria, 512, chars4) == 4


def test_usable_option_count_shrinks_as_the_budget_shrinks():
    criteria = banking_criteria(12)
    generous = usable_option_count(criteria, 512, chars4)
    tight = usable_option_count(criteria, 48, chars4)
    assert tight <= generous


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_no_options_is_an_error():
    with pytest.raises(BudgetError, match="no options"):
        option_budget({}, 192, chars4)


@pytest.mark.parametrize("head", [0, -1])
def test_a_non_positive_budget_is_an_error(head):
    with pytest.raises(BudgetError, match="must be positive"):
        option_budget(banking_criteria(4), head, chars4)


def test_render_option_handles_an_empty_description():
    assert render_option("card_arrival", "") == "card_arrival:"
    assert render_option("card_arrival", "card arrival") == "card_arrival: card arrival"


def test_report_serialises():
    import json
    payload = json.loads(json.dumps(option_budget(banking_criteria(32), 192, chars4).as_dict()))
    assert payload["tokens_per_option"] == 5
    assert payload["truncated"] is True
