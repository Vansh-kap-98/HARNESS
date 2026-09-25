"""Unit tests for src/sweep.py.

The properties that matter most are the methodological ones: nested option
sets, a fixed order across runs, and agreement with the compiler wherever the
compiler is willing to compile.
"""

from __future__ import annotations

import json

import pytest

from src.compiler import MAX_CHOICE_OPTIONS, compile_schema
from src.sweep import (
    DEFAULT_KS,
    Operation,
    RoutingExample,
    SweepError,
    approx_tokens_by_chars,
    build_question,
    build_schema,
    build_sweep,
    load_catalogue,
    load_examples,
    question_tokens,
    score_k,
)

SEED = 0


def catalogue(size=140):
    """A synthetic catalogue, for exercising the machinery only.

    Not evaluation data: the real sweep needs a real operation corpus (MCP
    tool listings or an OpenAPI set), because synthetic distractors are much
    easier to tell apart than real ones.
    """
    services = ["billing", "identity", "storage", "messaging", "calendar", "payroll", "search"]
    resources = ["invoice", "customer", "file", "thread", "event"]
    verbs = ["create", "get", "list", "update", "delete"]
    operations = []
    for service in services:
        for resource in resources:
            for verb in verbs:
                operations.append(
                    Operation(
                        id="%s.%s_%s" % (service, verb, resource),
                        description="%s a %s in the %s service" % (verb, resource, service),
                    )
                )
                if len(operations) == size:
                    return operations
    return operations


def examples(count=6):
    pool = catalogue()
    return [
        RoutingExample(
            id="r%03d" % index,
            state="please %s" % pool[(index * 7) % len(pool)].description,
            gold=pool[(index * 7) % len(pool)].id,
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------
# the three properties the curve depends on
# --------------------------------------------------------------------------


def test_option_sets_are_nested_as_k_grows():
    example = examples(1)[0]
    pool = catalogue()
    previous = None
    for k in DEFAULT_KS:
        current = set(build_question(example, pool, k, SEED).values)
        assert len(current) == k
        if previous is not None:
            assert previous < current, "k=%d set is not a superset of the smaller one" % k
        previous = current


def test_relative_order_is_preserved_as_k_grows():
    example = examples(1)[0]
    pool = catalogue()
    small = build_question(example, pool, 4, SEED).values
    large = build_question(example, pool, 32, SEED).values
    positions = [large.index(value) for value in small]
    assert positions == sorted(positions)


def test_building_is_deterministic_across_runs():
    example, pool = examples(1)[0], catalogue()
    first = build_question(example, pool, 16, SEED)
    second = build_question(example, pool, 16, SEED)
    assert first.values == second.values
    assert first.question == second.question


def test_a_different_seed_gives_different_distractors():
    example, pool = examples(1)[0], catalogue()
    assert (
        build_question(example, pool, 16, 0).values
        != build_question(example, pool, 16, 1).values
    )


def test_the_gold_option_is_always_present_and_indexed():
    pool = catalogue()
    for example in examples():
        for k in (4, 16, 128):
            built = build_question(example, pool, k, SEED)
            assert built.gold == example.gold
            assert built.values[built.gold_index] == example.gold


def test_gold_is_not_always_in_the_same_position():
    """Position bias is real, so the gold answer must move around."""
    pool = catalogue()
    positions = {build_question(e, pool, 16, SEED).gold_index for e in examples(6)}
    assert len(positions) > 1


def test_gold_lands_in_every_position_roughly_evenly():
    """Regression guard.

    An earlier version used one permutation for both choosing the distractors
    and ordering them, which put the gold option last in almost every
    question: a model could have scored 100% by always picking the last
    option. Gold must be spread across all k positions.
    """
    pool = catalogue()
    counts = [0] * 4
    total = 80
    for example in examples(total):
        counts[build_question(example, pool, 4, SEED).gold_index] += 1
    assert all(counts), "gold never appeared in some position: %s" % counts
    # no position may take more than half of all examples
    assert max(counts) < total / 2, counts


# --------------------------------------------------------------------------
# agreement with the compiler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 8, 16, MAX_CHOICE_OPTIONS])
def test_the_schema_compiles_to_the_same_question(k):
    built = build_question(examples(1)[0], catalogue(), k, SEED)
    compiled = compile_schema(build_schema(built))
    assert compiled.skipped == ()
    assert compiled.questions["operation"] == built.question
    assert compiled.specs["operation"].values == built.values


def test_above_the_limit_the_compiler_refuses_and_the_sweep_says_so():
    k = MAX_CHOICE_OPTIONS + 1
    built = build_question(examples(1)[0], catalogue(), k, SEED)
    compiled = compile_schema(build_schema(built))
    assert compiled.questions == {}
    assert "coarse-to-fine" in compiled.skipped[0].reason

    result = score_k([built], [built.gold], compiler_max_options=MAX_CHOICE_OPTIONS)
    assert result.compiler_would_refuse is True


def test_at_or_below_the_limit_nothing_is_flagged():
    built = build_question(examples(1)[0], catalogue(), MAX_CHOICE_OPTIONS, SEED)
    result = score_k([built], [built.gold], compiler_max_options=MAX_CHOICE_OPTIONS)
    assert result.compiler_would_refuse is False


# --------------------------------------------------------------------------
# building the whole sweep
# --------------------------------------------------------------------------


def test_build_sweep_covers_every_example_at_every_k():
    sweep = build_sweep(examples(6), catalogue(), DEFAULT_KS, SEED)
    assert set(sweep) == set(DEFAULT_KS)
    assert all(len(items) == 6 for items in sweep.values())
    assert [item.example_id for item in sweep[8]] == ["r%03d" % i for i in range(6)]


def test_a_catalogue_smaller_than_k_is_an_error():
    with pytest.raises(SweepError, match="cannot build a 128-option"):
        build_question(examples(1)[0], catalogue(20), 128, SEED)


def test_gold_outside_the_catalogue_is_an_error():
    bad = RoutingExample(id="x", state="do a thing", gold="nope.not_real")
    with pytest.raises(SweepError, match="not in the catalogue"):
        build_question(bad, catalogue(), 4, SEED)


@pytest.mark.parametrize("k", [0, 1, -4])
def test_k_below_two_is_an_error(k):
    with pytest.raises(SweepError, match="at least 2"):
        build_question(examples(1)[0], catalogue(), k, SEED)


def test_duplicate_ks_are_an_error():
    with pytest.raises(SweepError, match="duplicate values of k"):
        build_sweep(examples(1), catalogue(), (4, 4), SEED)


def test_an_operation_without_a_description_is_an_error():
    with pytest.raises(SweepError, match="no description"):
        Operation(id="a.b", description="  ")


# --------------------------------------------------------------------------
# token budget
# --------------------------------------------------------------------------


def test_the_token_estimate_grows_with_k():
    example, pool = examples(1)[0], catalogue()
    counts = [
        question_tokens(build_question(example, pool, k, SEED), approx_tokens_by_chars)
        for k in (4, 16, 64)
    ]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_a_512_token_context_is_blown_well_before_k_128():
    """The arithmetic behind the ~20-option claim, made checkable."""
    example, pool = examples(1)[0], catalogue()
    over = [
        k
        for k in DEFAULT_KS
        if question_tokens(build_question(example, pool, k, SEED), approx_tokens_by_chars) > 512
    ]
    assert over, "expected some k to exceed a 512-token context"
    assert min(over) <= 64


def test_question_tokens_requires_a_counter():
    built = build_question(examples(1)[0], catalogue(), 4, SEED)
    with pytest.raises(TypeError):
        question_tokens(built)  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_score_k_reports_accuracy_against_chance():
    pool = catalogue()
    built = [build_question(e, pool, 4, SEED) for e in examples(4)]
    predictions = [built[0].gold, built[1].gold, built[2].values[0], built[3].values[0]]
    result = score_k(built, predictions, compiler_max_options=MAX_CHOICE_OPTIONS)
    assert result.k == 4
    assert result.chance == 0.25
    assert result.accuracy.total == 4
    assert 0 <= result.accuracy.value <= 1
    assert result.accuracy.low <= result.accuracy.value <= result.accuracy.high


def test_a_perfect_run_has_lift_over_chance():
    pool = catalogue()
    built = [build_question(e, pool, 16, SEED) for e in examples(4)]
    result = score_k(built, [b.gold for b in built], compiler_max_options=MAX_CHOICE_OPTIONS)
    assert result.accuracy.value == 1.0
    assert result.as_dict()["lift_over_chance"] == pytest.approx(1.0 - 1 / 16)


def test_score_k_reports_the_token_budget_and_overflows():
    pool = catalogue()
    built = [build_question(e, pool, 64, SEED) for e in examples(4)]
    result = score_k(
        built,
        [b.gold for b in built],
        compiler_max_options=MAX_CHOICE_OPTIONS,
        count_tokens=approx_tokens_by_chars,
        context_limit=512,
    )
    assert result.median_tokens > 512
    assert result.context_overflows == 4


def test_a_prediction_that_was_never_offered_is_an_error():
    built = [build_question(examples(1)[0], catalogue(), 4, SEED)]
    with pytest.raises(SweepError, match="never offered"):
        score_k(built, ["billing.create_invoice_typo"], compiler_max_options=20)


def test_mixed_k_in_one_score_call_is_an_error():
    pool = catalogue()
    built = [
        build_question(examples(2)[0], pool, 4, SEED),
        build_question(examples(2)[1], pool, 8, SEED),
    ]
    with pytest.raises(SweepError, match="mix several values of k"):
        score_k(built, [built[0].gold, built[1].gold], compiler_max_options=20)


def test_result_serialises():
    pool = catalogue()
    built = [build_question(e, pool, 8, SEED) for e in examples(3)]
    payload = json.loads(
        json.dumps(
            score_k(built, [b.gold for b in built], compiler_max_options=20).as_dict()
        )
    )
    assert payload["k"] == 8
    assert payload["chance"] == 0.125
    assert payload["compiler_would_refuse"] is False


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def write_jsonl(tmp_path, rows, name):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_catalogue_and_examples_round_trip(tmp_path):
    catalogue_path = write_jsonl(
        tmp_path,
        [
            {"id": "billing.create_invoice", "description": "raise a new invoice"},
            {"id": "billing.refund_charge", "description": "refund a charge"},
        ],
        "catalogue.jsonl",
    )
    examples_path = write_jsonl(
        tmp_path,
        [{"id": "r1", "state": "give them their money back", "gold": "billing.refund_charge"}],
        "examples.jsonl",
    )
    operations = load_catalogue(catalogue_path)
    loaded = load_examples(examples_path)
    built = build_question(loaded[0], operations, 2, SEED)
    assert built.gold == "billing.refund_charge"
    assert set(built.values) == {"billing.create_invoice", "billing.refund_charge"}


def test_a_duplicate_operation_id_is_an_error(tmp_path):
    path = write_jsonl(
        tmp_path,
        [{"id": "a.b", "description": "one"}, {"id": "a.b", "description": "two"}],
        "catalogue.jsonl",
    )
    with pytest.raises(SweepError, match="duplicate operation id"):
        load_catalogue(path)


def test_a_missing_field_is_an_error(tmp_path):
    path = write_jsonl(tmp_path, [{"id": "a.b"}], "catalogue.jsonl")
    with pytest.raises(SweepError, match="'description'"):
        load_catalogue(path)


def test_an_empty_catalogue_is_an_error(tmp_path):
    path = tmp_path / "catalogue.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SweepError, match="no operations"):
        load_catalogue(str(path))
