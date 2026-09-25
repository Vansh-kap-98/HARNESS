"""Validate an assembled instance with Blaze, via Sourcemeta's `jsonschema` CLI.

Blaze is the only authoritative engine (master prompt section 4.3). The
python-jsonschema fallback exists for a local loop without the CLI installed;
it marks its results ``authoritative=False`` and `results/` must refuse those
(section 8 rule 15).

**The CLI name collides.** The Python package `jsonschema` installs its own
deprecated `jsonschema` console script. On this machine that script is the one
on PATH. Name-based discovery would therefore validate with the wrong engine
and never say so, which is exactly the kind of silent wrong answer this
project cannot afford.

So the CLI is not trusted because of its name. Before the first validation,
``probe_cli`` runs a known-valid and a known-invalid pair through whatever
binary was found and requires exit code 0 then non-zero. That both rejects the
impostor and confirms the exit-code contract we rely on, on the machine we are
actually running on.

Point ``SOURCEMETA_JSONSCHEMA_CLI`` at the binary to skip PATH lookup:

    export SOURCEMETA_JSONSCHEMA_CLI=/path/to/sourcemeta/jsonschema
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ValidationError",
    "BlazeUnavailable",
    "ValidationResult",
    "validate",
    "validate_with_blaze",
    "validate_with_python_jsonschema",
    "require_authoritative",
    "find_cli",
    "probe_cli",
    "BLAZE_ENGINE",
    "DEV_ENGINE",
    "CLI_ENV_VAR",
]

CLI_ENV_VAR = "SOURCEMETA_JSONSCHEMA_CLI"
CLI_NAME = "jsonschema"
BLAZE_ENGINE = "blaze"
DEV_ENGINE = "python-jsonschema"
CLI_TIMEOUT_SECONDS = 60

_PROBE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}
_PROBE_VALID: Dict[str, Any] = {"ok": True}
_PROBE_INVALID: Dict[str, Any] = {"ok": "yes"}

_probe_cache: Dict[str, bool] = {}


class ValidationError(Exception):
    """The validator could not be run (not: the instance was invalid)."""


class BlazeUnavailable(ValidationError):
    """No usable Sourcemeta `jsonschema` CLI was found."""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    engine: str
    authoritative: bool
    errors: Tuple[str, ...] = ()
    command: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "engine": self.engine,
            "authoritative": self.authoritative,
            "errors": list(self.errors),
            "command": self.command,
        }


def find_cli() -> str:
    """Locate the CLI binary. Does not check that it is the right one."""
    configured = os.environ.get(CLI_ENV_VAR)
    if configured:
        if not os.path.isfile(configured):
            raise BlazeUnavailable(
                "%s points at %r, which is not a file" % (CLI_ENV_VAR, configured)
            )
        return configured
    found = shutil.which(CLI_NAME)
    if not found:
        raise BlazeUnavailable(
            "no `jsonschema` CLI on PATH. Install Sourcemeta's CLI (see the "
            "sourcemeta/jsonschema README) or set %s to its path" % CLI_ENV_VAR
        )
    return found


def probe_cli(cli: Optional[str] = None) -> str:
    """Prove the binary behaves like Sourcemeta's `jsonschema validate`.

    Runs one valid and one invalid instance through it and requires exit code
    0 then non-zero. Cached per binary per process.
    """
    path = cli or find_cli()
    if _probe_cache.get(path):
        return path

    with tempfile.TemporaryDirectory(prefix="schema-native-probe-") as workdir:
        schema_path = os.path.join(workdir, "probe.schema.json")
        _write_json(schema_path, _PROBE_SCHEMA)
        good = _run_cli(path, schema_path, _PROBE_VALID, workdir, "valid.json")
        bad = _run_cli(path, schema_path, _PROBE_INVALID, workdir, "invalid.json")

    if good.returncode != 0:
        raise BlazeUnavailable(
            "%r rejected a known-valid instance (exit %d). It is probably not "
            "Sourcemeta's CLI -- the Python `jsonschema` package installs a "
            "script with the same name. Set %s to the right binary.\n%s"
            % (path, good.returncode, CLI_ENV_VAR, _output(good))
        )
    if bad.returncode == 0:
        raise BlazeUnavailable(
            "%r accepted a known-invalid instance, so its exit code cannot be "
            "trusted as a verdict.\n%s" % (path, _output(bad))
        )
    _probe_cache[path] = True
    return path


def validate(
    schema_path: str, instance: Mapping[str, Any]
) -> ValidationResult:
    """Authoritative validation. Raises if Blaze is not available."""
    return validate_with_blaze(schema_path, instance)


def validate_with_blaze(
    schema_path: str, instance: Mapping[str, Any]
) -> ValidationResult:
    cli = probe_cli()
    if not os.path.isfile(schema_path):
        raise ValidationError("schema %r does not exist" % schema_path)
    with tempfile.TemporaryDirectory(prefix="schema-native-") as workdir:
        completed = _run_cli(cli, schema_path, instance, workdir, "instance.json")
    valid = completed.returncode == 0
    return ValidationResult(
        valid=valid,
        engine=BLAZE_ENGINE,
        authoritative=True,
        errors=() if valid else (_output(completed),),
        command="%s validate %s <instance>" % (os.path.basename(cli), schema_path),
    )


def validate_with_python_jsonschema(
    schema_path: str, instance: Mapping[str, Any]
) -> ValidationResult:
    """Dev-loop fallback. Never authoritative, never allowed in `results/`."""
    try:
        from jsonschema.validators import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ValidationError("python-jsonschema is not installed: %s" % exc)

    with open(schema_path, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    errors: List[str] = [
        "%s: %s" % ("/".join(str(p) for p in error.absolute_path) or "<root>", error.message)
        for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    ]
    return ValidationResult(
        valid=not errors,
        engine=DEV_ENGINE,
        authoritative=False,
        errors=tuple(errors),
        command=None,
    )


def require_authoritative(result: ValidationResult) -> ValidationResult:
    """Gate for anything that writes to `results/` (section 8 rule 15)."""
    if not result.authoritative:
        raise ValidationError(
            "refusing to record a verdict from %r: only Blaze is authoritative"
            % result.engine
        )
    return result


def _run_cli(
    cli: str,
    schema_path: str,
    instance: Mapping[str, Any],
    workdir: str,
    filename: str,
) -> "subprocess.CompletedProcess[str]":
    instance_path = os.path.join(workdir, filename)
    _write_json(instance_path, instance)
    try:
        return subprocess.run(
            [cli, "validate", schema_path, instance_path],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError as exc:
        raise BlazeUnavailable("could not run %r: %s" % (cli, exc))
    except subprocess.TimeoutExpired:
        raise ValidationError(
            "%r did not finish within %d seconds" % (cli, CLI_TIMEOUT_SECONDS)
        )


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _output(completed: "subprocess.CompletedProcess[str]") -> str:
    return ((completed.stdout or "") + (completed.stderr or "")).strip()


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m src.validate <schema.json> <instance.json>", file=sys.stderr)
        return 2
    with open(argv[2], "r", encoding="utf-8") as handle:
        instance = json.load(handle)
    try:
        result = validate(argv[1], instance)
    except BlazeUnavailable as exc:
        print("blaze unavailable: %s" % exc, file=sys.stderr)
        return 3
    print(json.dumps(result.as_dict(), indent=2))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
