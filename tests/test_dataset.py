"""Unit tests for src/dataset.py."""

from __future__ import annotations

import json
import os

import pytest

from src.compiler import compile_schema_file
from src.dataset import (
    DatasetError,
    check_records,
    label_distribution,
    load_jsonl,
    sha256_file,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_SCHEMA = os.path.join(REPO_ROOT, "schemas", "support_ticket.json")
MINI = os.path.join(REPO_ROOT, "tests", "fixtures", "mini.jsonl")


@pytest.fixture(scope="module")
def compiled():
    return compile_schema_file(SMOKE_SCHEMA)


def write_jsonl(tmp_path, records, name="data.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return str(path)


def good_record(record_id="t001", **overrides):
    gold = {
        "department": "billing",
        "urgency": "soon",
        "refund_requested": True,
        "churn_risk": False,
    }
    gold.update(overrides)
    return {"id": record_id, "state": "some ticket text", "gold": gold}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_the_fixture_loads_in_file_order():
    records = load_jsonl(MINI)
    assert [r.id for r in records] == ["t001", "t002", "t003"]
    assert records[1].gold["urgency"] == "critical"


def test_the_fixture_agrees_with_the_schema(compiled):
    assert check_records(load_jsonl(MINI), compiled) == ()


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(good_record()) + "\n\n" + json.dumps(good_record("t002")) + "\n",
        encoding="utf-8",
    )
    assert len(load_jsonl(str(path))) == 2


@pytest.mark.parametrize(
    "line,needle",
    [
        ("{not json}", "not valid JSON"),
        ('["a"]', "not a JSON object"),
        ('{"state": "x", "gold": {}}', "'id'"),
        ('{"id": "a", "gold": {}}', "'state'"),
        ('{"id": "a", "state": "x"}', "'gold'"),
        ('{"id": "", "state": "x", "gold": {}}', "empty id"),
        ('{"id": "a", "state": "  ", "gold": {}}', "empty state"),
        ('{"id": "a", "state": "x", "gold": []}', "non-object gold"),
    ],
)
def test_malformed_lines_are_rejected(tmp_path, line, needle):
    path = tmp_path / "data.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=needle):
        load_jsonl(str(path))


def test_duplicate_ids_name_both_lines(tmp_path):
    path = write_jsonl(tmp_path, [good_record("t001"), good_record("t001")])
    with pytest.raises(DatasetError, match="lines 1 and 2"):
        load_jsonl(path)


def test_an_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="no records"):
        load_jsonl(str(path))


# --------------------------------------------------------------------------
# checking gold against the compiled schema
# --------------------------------------------------------------------------


def test_a_typo_in_a_label_is_caught(tmp_path, compiled):
    path = write_jsonl(tmp_path, [good_record(urgency="not urgent")])
    problems = check_records(load_jsonl(path), compiled)
    assert len(problems) == 1
    assert "urgency = 'not urgent' is not one of" in problems[0]


def test_a_missing_gold_field_is_caught(tmp_path, compiled):
    record = good_record()
    del record["gold"]["churn_risk"]
    problems = check_records(load_jsonl(write_jsonl(tmp_path, [record])), compiled)
    assert any("churn_risk" in problem for problem in problems)


def test_an_extra_gold_field_is_caught(tmp_path, compiled):
    record = good_record()
    record["gold"]["sentiment"] = "angry"
    problems = check_records(load_jsonl(write_jsonl(tmp_path, [record])), compiled)
    assert any("no question covers" in problem for problem in problems)


def test_one_for_a_boolean_is_caught(tmp_path, compiled):
    # `1` passes a naive `in (False, True)` check because `True == 1`.
    path = write_jsonl(tmp_path, [good_record(refund_requested=1)])
    problems = check_records(load_jsonl(path), compiled)
    assert any("refund_requested" in problem for problem in problems)


def test_every_problem_is_reported_in_one_pass(tmp_path, compiled):
    records = [
        good_record("t001", urgency="nope"),
        good_record("t002", department="marketing"),
    ]
    problems = check_records(load_jsonl(write_jsonl(tmp_path, records)), compiled)
    assert len(problems) == 2


# --------------------------------------------------------------------------
# distribution and freezing
# --------------------------------------------------------------------------


def test_label_distribution_counts_and_sorts_by_frequency():
    records = load_jsonl(MINI)
    distribution = label_distribution(records, ["department", "churn_risk"])
    assert distribution["department"] == {'"billing"': 2, '"technical"': 1}
    assert list(distribution["churn_risk"]) == ["false", "true"]


def test_sha256_is_stable_and_content_sensitive(tmp_path):
    first = write_jsonl(tmp_path, [good_record()], name="a.jsonl")
    same = write_jsonl(tmp_path, [good_record()], name="b.jsonl")
    different = write_jsonl(tmp_path, [good_record(urgency="critical")], name="c.jsonl")
    assert sha256_file(first) == sha256_file(same)
    assert sha256_file(first) != sha256_file(different)
    assert len(sha256_file(first)) == 64
