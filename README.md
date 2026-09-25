# schema-native — decision track

A JSON Schema is compiled into typed decision questions (`choice` / `score` / `noul`), answered by
a small non-autoregressive decision model in one forward pass, then reassembled into a JSON
instance and validated with Blaze.

Read [PLAN.md](PLAN.md) first. It is the master plan and the list of things that are banned.

```
schema -> bundle $refs -> compiler -> typed questions -> model -> answers -> assembler -> instance -> Blaze
```

## Setup

The system `python` on this machine is 3.8 (end of life; current `torch`/`transformers` will not
install on it). Use the checked-in 3.12 virtualenv, or recreate it:

```bash
py -3.12 -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

## What is built

| Module | Does | Tested |
|---|---|---|
| `src/compiler.py` | schema → questions + refusal report | yes |
| `src/assembler.py` | answers → instance + per-field confidence | yes |
| `src/validate.py` | Blaze via the `jsonschema` CLI | yes (stub CLI; real CLI test skips until installed) |
| `src/dataset.py` | load / check / freeze an eval set | yes |
| `src/score.py` | accuracy, Wilson intervals, ECE, latency, floors | yes |
| `src/runners/*` | model adapters | **not written** — needs the real Laya and Outlines/XGrammar APIs, which must be read from the installed packages, not guessed |

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q
```

Compile a schema to questions + the unsupported-field report:

```bash
.venv/Scripts/python.exe -m src.compiler schemas/support_ticket.json
```

Exit code 1 means the schema has `blocking_fields`: a property the schema *requires* could not be
compiled, so no assembled instance can ever validate and the schema is out of scope for this track.

Check and freeze an evaluation file (prints the sha256 that goes into the run config):

```bash
.venv/Scripts/python.exe -m src.dataset tests/fixtures/mini.jsonl schemas/support_ticket.json
```

Validate one instance with Blaze:

```bash
.venv/Scripts/python.exe -m src.validate schemas/support_ticket.json instance.json
```

## The `jsonschema` CLI name collision

The Python `jsonschema` package installs a console script called `jsonschema`, and on this machine
that is what PATH finds — *not* Sourcemeta's CLI. `src/validate.py` therefore never trusts the
binary by name. Before the first validation it runs one known-valid and one known-invalid instance
through whatever it found and requires exit 0 then non-zero; anything else raises `BlazeUnavailable`
rather than silently validating with the wrong engine.

Point it at the right binary to skip PATH lookup:

```bash
export SOURCEMETA_JSONSCHEMA_CLI=/path/to/sourcemeta/jsonschema
```

`python-jsonschema` is available as a dev fallback. It marks its verdicts `authoritative=False`, and
`require_authoritative()` refuses to let them reach `results/`.

## What the compiler will not do

It refuses free strings, unbounded or wide numeric ranges, arrays, open-ended maps, conditional
subschemas, and anything else that cannot be one typed question — see PLAN.md §4.2. Refusals are
reported as `{"field", "reason", "route"}`; they are never approximated.
