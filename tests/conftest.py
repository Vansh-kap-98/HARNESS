"""Shared test fixtures."""

from __future__ import annotations

import os
import sys

import pytest

# A stand-in for Sourcemeta's CLI with the same contract:
# `<cli> validate <schema> <instance>`, exit 0 when valid, non-zero when not.
# It exercises the Blaze code path on a machine without Blaze installed. It is
# never a substitute for running the real CLI -- test_real_blaze_cli_if_present
# does that wherever the binary exists.
_STUB_SOURCE = '''\
import json
import sys

from jsonschema.validators import Draft202012Validator

command, schema_path, instance_path = sys.argv[1], sys.argv[2], sys.argv[3]
if command != "validate":
    sys.exit(64)
with open(schema_path, encoding="utf-8") as handle:
    schema = json.load(handle)
with open(instance_path, encoding="utf-8") as handle:
    instance = json.load(handle)
errors = list(Draft202012Validator(schema).iter_errors(instance))
if errors:
    print(errors[0].message, file=sys.stderr)
    sys.exit(1)
print("ok")
'''


@pytest.fixture
def stub_cli(tmp_path):
    """Factory returning the path to a well-behaved fake `jsonschema` CLI."""

    def _make(name: str = "jsonschema") -> str:
        script = tmp_path / "stub_cli.py"
        script.write_text(_STUB_SOURCE, encoding="utf-8")
        if os.name == "nt":
            wrapper = tmp_path / (name + ".bat")
            wrapper.write_text(
                '@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable, script),
                encoding="utf-8",
            )
        else:
            wrapper = tmp_path / name
            wrapper.write_text(
                '#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, script),
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
        return str(wrapper)

    return _make
