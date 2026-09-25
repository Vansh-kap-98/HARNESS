"""Load, check and freeze an evaluation set.

One JSON object per line:

    {"id": "t001", "state": "<the ticket text>", "gold": {"department": "billing", ...}}

"Freeze" means: the file stops changing, and its sha256 goes into every run's
config snapshot. If the hash in a results folder does not match the file, the
numbers in that folder were produced on different data and cannot be compared
with anything else (section 6).

Gold values are checked against the compiled schema, so a typo like
``"not urgent"`` for ``"not_urgent"`` is caught at freeze time rather than
showing up later as a model that mysteriously never gets that field right.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.compiler import CompiledSchema, compile_schema_file

__all__ = [
    "DatasetError",
    "Record",
    "load_jsonl",
    "check_records",
    "label_distribution",
    "sha256_file",
]


class DatasetError(Exception):
    """The evaluation file is malformed or disagrees with the schema."""


@dataclass(frozen=True)
class Record:
    id: str
    state: str
    gold: Dict[str, Any]


def load_jsonl(path: str) -> List[Record]:
    """Read the file, preserving order. Order is part of the frozen data."""
    records: List[Record] = []
    seen: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError("line %d is not valid JSON: %s" % (number, exc))
            if not isinstance(payload, dict):
                raise DatasetError("line %d is not a JSON object" % number)
            for key in ("id", "state", "gold"):
                if key not in payload:
                    raise DatasetError("line %d has no %r" % (number, key))
            record_id = payload["id"]
            if not isinstance(record_id, str) or not record_id.strip():
                raise DatasetError("line %d has a non-string or empty id" % number)
            if not isinstance(payload["state"], str) or not payload["state"].strip():
                raise DatasetError(
                    "line %d (%s) has a non-string or empty state" % (number, record_id)
                )
            if not isinstance(payload["gold"], dict):
                raise DatasetError("line %d (%s) has a non-object gold" % (number, record_id))
            if record_id in seen:
                raise DatasetError(
                    "duplicate id %r on lines %d and %d"
                    % (record_id, seen[record_id], number)
                )
            seen[record_id] = number
            records.append(
                Record(id=record_id, state=payload["state"], gold=dict(payload["gold"]))
            )
    if not records:
        raise DatasetError("%s contains no records" % path)
    return records


def check_records(
    records: Sequence[Record], compiled: CompiledSchema
) -> Tuple[str, ...]:
    """Return every problem found, so one pass fixes the whole file."""
    problems: List[str] = []
    asked = list(compiled.questions)
    for record in records:
        missing = [name for name in asked if name not in record.gold]
        if missing:
            problems.append("%s: gold has no %s" % (record.id, missing))
        extra = [name for name in record.gold if name not in asked]
        if extra:
            problems.append(
                "%s: gold has fields no question covers: %s" % (record.id, sorted(extra))
            )
        for name in asked:
            if name not in record.gold:
                continue
            allowed = compiled.specs[name].values
            value = record.gold[name]
            if not any(
                value == candidate
                and isinstance(value, bool) == isinstance(candidate, bool)
                for candidate in allowed
            ):
                problems.append(
                    "%s: %s = %r is not one of %r"
                    % (record.id, name, value, list(allowed))
                )
    return tuple(problems)


def label_distribution(
    records: Sequence[Record], fields: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """Per-field value counts, for spotting a set that is mostly one label."""
    distribution: Dict[str, Dict[str, int]] = {}
    for name in fields:
        counts: Dict[str, int] = {}
        for record in records:
            if name not in record.gold:
                continue
            key = json.dumps(record.gold[name], sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
        distribution[name] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return distribution


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m src.dataset <data.jsonl> <schema.json>", file=sys.stderr)
        return 2
    data_path, schema_path = argv[1], argv[2]
    compiled = compile_schema_file(schema_path)
    records = load_jsonl(data_path)
    problems = check_records(records, compiled)
    report: Dict[str, Any] = {
        "path": data_path,
        "sha256": sha256_file(data_path),
        "records": len(records),
        "decisions": len(records) * len(compiled.questions),
        "fields": list(compiled.questions),
        "label_distribution": label_distribution(records, list(compiled.questions)),
        "problems": list(problems),
    }
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
