"""Unit tests for src/validate.py.

The interesting case is the name collision: the Python `jsonschema` package
installs a console script called `jsonschema`, and on a machine where that is
what PATH finds, name-based discovery would validate with the wrong engine and
never say so. The probe has to catch that.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from src import validate as V

SMOKE_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "schemas",
    "support_ticket.json",
)

GOOD_INSTANCE = {
    "department": "billing",
    "urgency": "critical",
    "refund_requested": True,
    "churn_risk": False,
}
BAD_INSTANCE = {
    "department": "billing",
    "urgency": "extremely",  # not in the enum
    "refund_requested": True,
    "churn_risk": False,
}


@pytest.fixture(autouse=True)
def clear_probe_cache(monkeypatch):
    monkeypatch.delenv(V.CLI_ENV_VAR, raising=False)
    V._probe_cache.clear()
    yield
    V._probe_cache.clear()


# --------------------------------------------------------------------------
# discovery and the impostor CLI
# --------------------------------------------------------------------------


def test_env_var_pointing_at_nothing_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(V.CLI_ENV_VAR, str(tmp_path / "nope"))
    with pytest.raises(V.BlazeUnavailable, match="not a file"):
        V.find_cli()


def test_missing_cli_names_the_env_var(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(V.BlazeUnavailable, match=V.CLI_ENV_VAR):
        V.find_cli()


def test_python_jsonschemas_own_cli_is_rejected_by_the_probe(monkeypatch):
    """The real collision, on this machine's own venv."""
    suffix = ".exe" if os.name == "nt" else ""
    impostor = os.path.join(os.path.dirname(sys.executable), "jsonschema" + suffix)
    if not os.path.isfile(impostor):
        pytest.skip("python-jsonschema's console script is not installed")
    monkeypatch.setenv(V.CLI_ENV_VAR, impostor)
    with pytest.raises(V.BlazeUnavailable, match="probably not"):
        V.probe_cli()


def test_a_cli_that_accepts_everything_is_rejected(monkeypatch, tmp_path):
    always_ok = tmp_path / ("yes.bat" if os.name == "nt" else "yes")
    if os.name == "nt":
        always_ok.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    else:
        always_ok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        always_ok.chmod(0o755)
    monkeypatch.setenv(V.CLI_ENV_VAR, str(always_ok))
    with pytest.raises(V.BlazeUnavailable, match="cannot be trusted"):
        V.probe_cli()


def test_a_well_behaved_cli_passes_the_probe(monkeypatch, tmp_path, stub_cli):
    monkeypatch.setenv(V.CLI_ENV_VAR, stub_cli())
    assert V.probe_cli().endswith(("jsonschema", "jsonschema.bat"))


# --------------------------------------------------------------------------
# the Blaze code path, exercised against the stub
# --------------------------------------------------------------------------


def test_valid_instance_through_the_cli_path(monkeypatch, tmp_path, stub_cli):
    monkeypatch.setenv(V.CLI_ENV_VAR, stub_cli())
    result = V.validate(SMOKE_SCHEMA, GOOD_INSTANCE)
    assert result.valid is True
    assert result.authoritative is True
    assert result.errors == ()


def test_invalid_instance_through_the_cli_path(monkeypatch, tmp_path, stub_cli):
    monkeypatch.setenv(V.CLI_ENV_VAR, stub_cli())
    result = V.validate(SMOKE_SCHEMA, BAD_INSTANCE)
    assert result.valid is False
    assert result.authoritative is True
    assert result.errors and result.errors[0]


def test_a_missing_schema_file_is_reported(monkeypatch, tmp_path, stub_cli):
    monkeypatch.setenv(V.CLI_ENV_VAR, stub_cli())
    with pytest.raises(V.ValidationError, match="does not exist"):
        V.validate(str(tmp_path / "absent.json"), GOOD_INSTANCE)


# --------------------------------------------------------------------------
# the dev fallback, which may never reach results/
# --------------------------------------------------------------------------


def test_dev_fallback_agrees_on_a_valid_instance():
    result = V.validate_with_python_jsonschema(SMOKE_SCHEMA, GOOD_INSTANCE)
    assert result.valid is True
    assert result.engine == V.DEV_ENGINE
    assert result.authoritative is False


def test_dev_fallback_reports_the_failing_field():
    result = V.validate_with_python_jsonschema(SMOKE_SCHEMA, BAD_INSTANCE)
    assert result.valid is False
    assert any("urgency" in error for error in result.errors)


def test_require_authoritative_blocks_the_dev_engine():
    result = V.validate_with_python_jsonschema(SMOKE_SCHEMA, GOOD_INSTANCE)
    with pytest.raises(V.ValidationError, match="only Blaze is authoritative"):
        V.require_authoritative(result)


def test_require_authoritative_passes_blaze_through(monkeypatch, tmp_path, stub_cli):
    monkeypatch.setenv(V.CLI_ENV_VAR, stub_cli())
    result = V.validate(SMOKE_SCHEMA, GOOD_INSTANCE)
    assert V.require_authoritative(result) is result


def test_result_serialises(monkeypatch, tmp_path, stub_cli):
    monkeypatch.setenv(V.CLI_ENV_VAR, stub_cli())
    payload = json.loads(json.dumps(V.validate(SMOKE_SCHEMA, GOOD_INSTANCE).as_dict()))
    assert payload["engine"] == V.BLAZE_ENGINE
    assert payload["authoritative"] is True


# --------------------------------------------------------------------------
# the real thing, when it is installed
# --------------------------------------------------------------------------


def test_real_blaze_cli_if_present():
    """Runs only where Sourcemeta's CLI is actually installed.

    This is the test that confirms the exit-code contract on the real binary,
    rather than on our stand-in.
    """
    try:
        V.probe_cli()
    except V.BlazeUnavailable as exc:
        pytest.skip("Sourcemeta jsonschema CLI not available: %s" % exc)
    assert V.validate(SMOKE_SCHEMA, GOOD_INSTANCE).valid is True
    assert V.validate(SMOKE_SCHEMA, BAD_INSTANCE).valid is False
