# Master plan v2 — JSON Schema-native small model (decision track)

Supersedes the v1 master prompt. Paste this whole file at the start of a new session.
Last edited 23 Sep 2026. Status: early exploration, nothing measured yet, no parameter limit decided.

---

## 0. What changed from v1, and why

| # | Change | Why | Status |
|---|---|---|---|
| 1 | `x-ordered: true` is the blessed marker for an ordered enum (§4.1) | v1 said "an enum explicitly marked ordered" without defining the marker; JSON Schema has no such keyword, so the compiler cannot read intent that isn't written down | **applied in code**; spelling needs Juan's sign-off (§9) |
| 2 | A refused property that the schema *requires* makes the whole schema out of scope (`blocking_fields`, §4.3) | v1's "validity is 100% by construction" is unachievable there — the field cannot be produced and the instance can never validate. Without this rule it looks like a compiler bug with no fix | **applied in code** |
| 3 | Optional compiled properties are reported as always-filled (`optional_fields`, §4.3) | A decision model cannot answer "absent". Silent semantic change otherwise | **applied in code** |
| 4 | Third generative size added to the ladder; decision and generative models plotted as **two lines, never one fitted curve** (§6) | 2 decision points + 2 generative points is not a scaling curve. A knee read off a mixed-family curve cannot license a parameter ceiling | plan only |
| 5 | Majority-class-per-field floor added to every results table (§3, §6) | With 4 / 3 / 2 / 2 options per field, per-field accuracy looks respectable from noise alone | plan only |
| 6 | Smoke set moved to `data/smoke/`, and **no ECE and no knee point may be claimed from it** (§6, §8 rules 12–13) | n=50 × 4 fields ≈ 200 decisions over ~10 confidence bins ≈ 20 per bin. ECE is noise-dominated and biased low at that n | plan only |
| 7 | Phase 2 eval set sized at 400 (200 val / 200 test) (§5) | ECE, temperature scaling and the knee point all need n the smoke set does not have | plan only |
| 8 | No `val` split in the smoke test, and therefore **no temperature scaling in the smoke test** (§6) | v1 required val-only temperature fitting but gave the smoke test no val split. Raw ECE reported as uncalibrated diagnostic only | plan only |
| 9 | Latency protocol fixed: batch 1, 5 warmup, 3 repeats, median, GPU-sharing recorded (§6) | A free shared T4 swamps the differences we care about; p95 is otherwise unreportable | plan only |
| 10 | Every vendor/competitor number in §2 is marked **unverified**, with the command to check it | Claude's knowledge cutoff is May 2026; Laya, Jev and the Sep 2026 claims post-date it. They are Vansh's notes, not confirmed facts | plan only |
| 11 | Python 3.12 venv is the blessed local environment (§7) | `python` on this machine was 3.8 (EOL); current `torch`/`transformers` will not install on it | **applied in repo** (`.venv`) |
| 12 | The generative baseline gets the **same schema text**, the same option order, greedy decoding, fixed seed (§8 rule 14) | Descriptions are what make the decision questions informative. Giving the baseline less is a rigged comparison; position bias makes option order a real variable | plan only |
| 13 | `python-jsonschema` output may never enter `results/` (§4.3) | It disagrees with Blaze on edge cases (format assertions, unknown keywords) | plan only |

---

## 1. Who you are working with and how

You are helping **Vansh** (B.Tech CSE AI/ML, Sourcemeta Core/Blaze contributor) on an early-stage research project.

- **Teach as you go.** One-line definition next to every technical term. Vansh is preparing for technical interviews.
- **Concrete, copy-paste-ready outputs.** Full files, full commands, exact package names. No pseudocode unless asked.
- **Evidence over assumptions.** Separate measured facts, self-reported claims (model cards, blogs), and your own predictions. Label predictions as predictions.
- **Push back** when something is wrong or out of scope. Don't just agree.
- **Keep scope narrow.** If a task drifts into the generative/API track, flag it and stop.

---

## 2. Project context

**Team**

- **Sri Hariharan Sharma (Hari):** leads the decision-model track (Laya-like).
- **Juan Cruz Viotti:** Sourcemeta maintainer. Proposed the "JSON Schema-native model" and the API-consumption hypothesis.
- **Harsh Popat:** context on the API-usage side.
- **Vansh:** decision track. Owns the **shared JSON Schema layer** and the **evaluation harness**.

**The idea.** Build a tiny open-weight model where JSON Schema is part of how the model works, not a validator run on its output afterward. Core question (Hari): *how small can a model be and still be really good at JSON, JSON Schema, and decision/API tasks?* Findings may feed back into the JSON Schema standard.

**Two tracks, one shared JSON Schema layer**

1. **Decision track (THIS PROJECT):** a non-autoregressive decision model (Laya/Jev style) — it scores a fixed set of typed options in one forward pass and never generates text.
2. **API track (NOT this project):** a small autoregressive model that writes request bodies for APIs described by JSON Schema. Owned by Juan/Harsh.

Combining the two comes later, only if both work on their own.

**Reference points — ALL UNVERIFIED.** Everything below is Vansh's notes. An assistant with a May 2026 cutoff cannot confirm any of it and must not treat it as established. Verify before any of it enters a write-up:

```bash
# model cards, files, and real parameter counts
python -c "from huggingface_hub import model_info; i=model_info('convaiinnovations/laya'); print(i.id, i.downloads, i.tags); print([s.rfilename for s in i.siblings][:40])"
```

- **TypeSafe Jev:** closed, API-only "System One" decision model. Typed answers `choice`, `score`, `noul` (calibrated yes/no). Trained with RLCD. Internals undisclosed. $40M seed, Sep 2026. *Vendor-reported, never measured by us.*
- **Laya (ConvAI Innovations):** Apache-2.0 open weights. `convaiinnovations/laya` = ModernBERT-large backbone + 2-layer decision head, 421M, 512 context; `multilingual` subfolder = mmBERT-base, 322M. Each option scored at its own `[MASK]` token, then softmaxed. Self-reported weaknesses: near chance on typed decisions zero-shot; options share a fixed `head_max_len` budget so many-option tasks degrade (Banking77 0.425 vs Jev 0.870); ships overconfident, needs temperature scaling; English checkpoint collapses on non-Latin scripts while staying confident.
- **xLAM-2-1b (Salesforce):** purpose-built 1B tool-calling model reported at ~3× a general 1B on BFCL v4; weak multi-turn.
- **JSONSchemaBench:** 10K real-world schemas for evaluating constrained decoding; framework coverage varies ~2×.
- **"Let Me Speak Freely" (EMNLP 2024):** format restrictions can hurt reasoning while helping classification.
- **MCP** defaults to JSON Schema 2020-12 for tool schemas.

---

## 3. What counts as evidence

**Baselines, both reported in every table:**

1. **Majority class per field** — predict the most common gold value for each field. Parameter count 0. This is the floor; a model that does not clearly beat it has shown nothing.
2. **A small general instruct LLM + constrained decoding** filling the same schema. Constrained decoding (grammar-restricted sampling that makes schema-invalid tokens impossible) already guarantees validity for the features it supports, so validity alone proves nothing.

**The claim we want to test:** a small decision model fills schema fields

- more accurately (per field and whole instance),
- much faster,
- with more trustworthy confidence

than that LLM baseline, at a smaller parameter count.

**Every accuracy number carries a 95% Wilson interval** (a binomial confidence interval that stays sensible at small n and near 0 or 1, unlike the normal approximation). A difference whose intervals overlap is not a difference.

**A null result is a valid result.** If the baseline wins, report it.

---

## 4. Architecture: the shared JSON Schema layer

Pipeline: **schema → (bundle `$ref`s) → compiler → typed questions → model → answers → assembler → JSON instance → Blaze validation**

### 4.1 Compiler mapping rules (target: Laya question format)

| Schema pattern | Question type | Rule |
|---|---|---|
| `enum` of strings, or `oneOf` of `const`s | `choice` | Options = the values; criteria text from each branch's `description`/`title`, else the value itself |
| `type: boolean` | `noul` | Instructions from `description`/`title` |
| The above, plus `x-ordered: true` | `score` | `criteria` = the labels in schema order, low → high |
| `integer` with both bounds spanning ≤ 10 steps | `score` | `criteria` = `["1","2",...]`; `exclusiveMinimum`/`exclusiveMaximum` converted exactly |

`x-ordered` is the **only** extension keyword the compiler reads. JSON Schema has no ordered-enum keyword; unknown keywords are ignored by validators, so a schema carrying it still validates. See §9 — this is the `$vocabulary` question and needs Juan's sign-off before it spreads into the eval set.

Laya question format (from its model card, unverified):

```python
questions = {
  "department": {"type": "choice", "instructions": "...", "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages"}},
  "urgency":    {"type": "score",  "instructions": "...", "criteria": ["not urgent", "soon", "critical"]},
  "refund":     {"type": "noul",   "instructions": "..."}
}
result = agent.predict(state, questions)   # one forward pass for all questions
```

### 4.2 UNSUPPORTED: the compiler refuses these, never approximates them

- free `string` (no `enum`/`const`)
- `number`; `integer` with an unbounded or > 10-step range; `multipleOf`
- arrays of free values; `additionalProperties`/`patternProperties` maps
- `choice` with more than 20 options, unless split into a coarse-to-fine hierarchy (explicit opt-in, not built yet)
- `if/then/else`, `dependentSchemas`, `dependentRequired`, `not`, `allOf`/`anyOf`/`oneOf` at the object level — anything that changes which fields exist
- union types including nullability (`["string","null"]`)
- `$ref` that is remote (bundle first) or carries validation siblings

Two refusal levels:

- **`CompilerError`** — the whole schema is out of scope (malformed, unbundled, or object-level keywords above). There is no fixed question set to compile.
- **skipped field** — one property is refused; the rest still compiles. Reported as `{"field": ..., "reason": ..., "route": "generative-track"}`.

### 4.3 Assembler and validation

- Build the instance from answers, in schema property order.
- Keep per-field confidence **alongside** the instance, never inside it.
- Validate with **Blaze** via Sourcemeta's `jsonschema` CLI (`jsonschema validate <schema> <instance>`), installed per the sourcemeta/jsonschema README.
- **The CLI name collides.** The Python `jsonschema` package installs its own `jsonschema` console script, and on this machine that is what PATH finds. `src/validate.py` therefore never trusts the binary by name: before the first validation it runs a known-valid and a known-invalid instance through whatever it found and requires exit 0 then non-zero. Set `SOURCEMETA_JSONSCHEMA_CLI` to the real binary to skip PATH lookup.
- `python-jsonschema` is a local dev convenience only. **Its verdicts may never appear in `results/`** — it disagrees with Blaze on format assertions and unknown keywords.
- Validity must be 100% by construction. An invalid instance is a **compiler/assembler bug**, not a model error. Log it and fix it.
- **`blocking_fields`** — if a *required* property was refused, no assembled instance can ever validate. That schema is out of scope for the decision track: report it and stop. Do not emit a knowingly invalid instance.
- **`optional_fields`** — compiled properties the schema does not require are still always answered and therefore always present. If absence carries meaning in a schema, that schema is out of scope too.

---

## 5. Phase plan

| Phase | Needs parameter limit? | Content | n |
|---|---|---|---|
| 0. Align with Hari | — | Parameter limit, success metric, ownership | — |
| 1. Schema layer | No | Compiler + assembler + unsupported report + tests | — |
| 1.5 Smoke test | No | Phases 1–3 in miniature, pipeline shakedown (§6) | 50 |
| 2. Eval set | No | Real schemas, messy inputs, correct instances, frozen splits | 400 = 200 val + 200 test |
| 3. Baselines (zero-shot) | No | Full size ladder, ECE, knee point | 200 test |
| 4. Fine-tune | **Yes** | Two sizes under the limit; full FT vs LoRA; temperature scaling on **val only** | 200 val / 200 test |
| 5. Demo | — | Schema + text → instance, per-field confidence, escalate flag, validation badge, latency, comparison chart | — |

The **parameter limit proposal comes out of Phase 3, not the smoke test.** The smoke set is too small to support a knee point (§8 rule 12).

**Current step: the smoke test.**

---

## 6. Smoke test spec

**Purpose: prove the pipeline runs end to end and get a first directional signal.** It is not evidence for a parameter limit and not a calibration measurement.

**Schema** — `schemas/support_ticket.json` (written, compiles clean):

- `department`: `oneOf` of consts (billing, technical, sales, other) → `choice`
- `urgency`: ordered enum (not_urgent, soon, critical) → `score`
- `refund_requested`: boolean → `noul`
- `churn_risk`: boolean → `noul`

**Data** — `data/smoke/tickets.jsonl`, ~50 messy ticket texts with gold instances, one `{"id": ..., "state": ..., "gold": {...}}` per line.

- Lives in `data/smoke/`, **not** `data/test/`. `train`/`val`/`test` stay reserved for the Phase 2 set so the smoke set can never be silently reused as a test split.
- If a frontier model labels them, hand-check at least 20% and report the label-agreement rate (the teacher ceiling — the accuracy no student can be shown to exceed).
- Freeze before running any model: `python -m src.dataset data/smoke/tickets.jsonl schemas/support_ticket.json` checks every gold value against the compiled answer space, prints the label distribution, and prints the `sha256` that goes into every run's config snapshot.
- **No val split, therefore no temperature scaling in the smoke test.** Any ECE printed is an uncalibrated diagnostic, labelled as such.

**Ladder — same inputs, same order, fixed seeds, every model:**

| # | Model | Params | Family |
|---|---|---|---|
| B0 | majority class per field | 0 | trivial floor |
| D1 | `convaiinnovations/laya`, subfolder `multilingual` | 322M | decision |
| D2 | `convaiinnovations/laya`, root (English) | 421M | decision |
| G1 | `Qwen/Qwen2.5-0.5B-Instruct` + constrained decoding | 0.5B | generative |
| G2 | `Qwen/Qwen2.5-1.5B-Instruct` + constrained decoding | 1.5B | generative |
| G3 | `Qwen/Qwen2.5-3B-Instruct` + constrained decoding | 3B | generative |

- G3 exists to give the generative family **three** points, which is the minimum for reading a knee within one family. Check its licence on the model card before anything is published — my recollection is that Qwen2.5-3B is *not* Apache-2.0 while 0.5B/1.5B are; if that blocks publication, swap in `Qwen/Qwen2.5-7B-Instruct` (fp16 ≈ 14 GB, tight on a 16 GB T4) or an Apache-2.0 model of similar size.
- **Plot decision and generative models as two separate lines on one accuracy-vs-parameters chart. Never fit one curve through both.** They differ in architecture, pretraining and task format; a knee across families measures none of those.
- Verify every HF model ID and library version before use. Don't guess.
- **Constrained decoding:** Outlines or XGrammar, the exact same JSON Schema, greedy decoding (`do_sample=False`), fixed seed, one prompt template recorded verbatim in the run config.
- **Hardware:** free Kaggle/Colab T4. Set `USE_TF=0` if `laya.load()` hangs.

**Metrics per model**

- per-field accuracy, with 95% Wilson interval
- exact match (all four fields right), with 95% Wilson interval
- validity rate — a **bug counter**, never a win (§8 rule 1)
- reliability table for the decision models — diagnostic only at this n, never headlined
- latency: **batch size 1, 5 discarded warmup examples, 3 full repeats, report the median of per-example medians and the p95 of the pooled runs**; record whether the GPU was shared and exclude model/tokenizer load time
- parameter count, measured (`sum(p.numel() for p in model.parameters())`), not copied from a card

**Output** — one folder under `results/<UTC timestamp>-smoke/` containing: `config.json` (model IDs + revisions, library versions, data sha256, seeds, prompt template, GPU), `per_example.csv`, `summary.csv`, and the two-line chart.

---

## 7. Repo layout and environment

```
HARNESS/
  PLAN.md                  # this file
  README.md                # setup + commands
  .venv/                   # Python 3.12 (system `python` is 3.8, EOL)
  requirements-dev.txt     # local: pytest (+ python-jsonschema, dev fallback only)
  requirements.txt         # Kaggle: pinned after the first successful run via `pip freeze`
  conftest.py              # puts the repo root on sys.path for tests
  schemas/                 # input schemas, bundled, no remote $ref
  data/smoke/              # frozen smoke JSONL (50)
  data/{train,val,test}/   # Phase 2 splits, frozen
  src/compiler.py          # schema -> questions + unsupported report
  src/assembler.py         # answers -> instance + confidences
  src/validate.py          # Blaze via the jsonschema CLI, with an impostor probe
  src/dataset.py           # load, check against the schema, freeze (sha256)
  src/score.py             # Wilson intervals, ECE, majority floor, latency
  src/runners/laya_runner.py  # NOT WRITTEN: needs the real Laya API verified
  src/runners/llm_runner.py   # NOT WRITTEN: needs the real Outlines/XGrammar API
  notebooks/smoke_test.ipynb  # Kaggle-ready
  results/                 # one folder per run, config snapshot included
  tests/                   # unit tests + one end-to-end wiring test
  tests/fixtures/mini.jsonl   # 3 records, test-only, never an eval set
```

Local commands:

```bash
.venv/Scripts/python.exe -m pytest -q
```

```bash
.venv/Scripts/python.exe -m src.compiler schemas/support_ticket.json
```

```bash
.venv/Scripts/python.exe -m src.dataset tests/fixtures/mini.jsonl schemas/support_ticket.json
```

```bash
.venv/Scripts/python.exe -m src.validate schemas/support_ticket.json <instance.json>
```

---

## 8. BANNED: never do these

1. **Never report schema validity as a model win.** It is guaranteed by construction or by constrained decoding.
2. **Never train, tune hyperparameters, or fit temperature on the test split.** Temperature scaling uses `val` only.
3. **Never compare our numbers to Jev** as if measured side by side, unless we actually ran Jev. Cite vendor numbers as self-reported.
4. **Never invent** benchmark numbers, model IDs, package APIs, or CLI flags. If unsure, say so and give the command to check.
5. **Never silently approximate** an unsupported schema feature. Refuse it and route it (§4.2).
6. **Never mix tracks.** No generative argument-filling inside the decision demo.
7. **Never use frontier-model labels as gold** without the hand-checked sample and the reported label-agreement rate (the teacher ceiling).
8. **Never change inputs between models** in a comparison. Same schemas, same examples, same order, fixed seeds.
9. **Never hardcode API keys or tokens.** Read them from environment variables.
10. **Never add config flags or options "just in case."** One blessed configuration. Also applies to anything proposed upstream to Sourcemeta.
11. **Never claim a trend from fewer than ~50 examples,** and never report an accuracy without its 95% Wilson interval.
12. **Never report ECE or a knee point from the smoke set.** n≈200 decisions over ~10 bins is noise, and ECE is biased low at that n. Both are Phase 3 outputs.
13. **Never fit one scaling curve across decision and generative models.** Two lines, always.
14. **Never give the generative baseline less than the decision model gets.** Same schema including all `description` text, same option order, greedy decoding, prompt template recorded in the run config.
15. **Never let `python-jsonschema` verdicts into `results/`.** Blaze decides.

---

## 9. Open questions — do not assume answers

- **Parameter ceiling.** To be proposed from the Phase 3 knee point, within the generative family, with the decision line shown alongside.
- **The success metric** Hari and Vansh agree on. Candidate: exact match on the test split, with per-field accuracy and ECE as secondary.
- **Is `x-ordered` the right spelling?** Needs Juan. Options: a custom keyword in a Sourcemeta `$vocabulary`, reuse of an existing annotation, or requiring `oneOf` branch order to imply ordering. Blocking before the Phase 2 schemas are written.
- **Does "custom vocabularies" (Hari) mean JSON Schema `$vocabulary` keyword sets?** Confirm before designing for it.
- **Full fine-tuning vs LoRA** at this size. Test both in Phase 4.
- **Where Blaze fits beyond validation:** data filter, or RL reward.
- **Laya's `head_max_len` option budget.** Known failure mode on many-option tasks; the smoke schema's 4-option `choice` will not surface it. Add a per-question token-budget diagnostic to the compiler before Phase 3, once the tokenizer is known.

---

## 10. Current state

**Done — 180 tests passing, 1 skipped (the real-Blaze test, which needs the CLI installed)**

- `src/compiler.py` — schema to questions, two-level refusals, `$ref` resolution.
- `src/assembler.py` — canonical `Answer` type, instance building, escalation with no default threshold.
- `src/validate.py` — Blaze via the CLI, functional probe against the name collision, non-authoritative dev fallback.
- `src/dataset.py` — JSONL loading, gold checked against the compiled answer space, label distribution, sha256 freeze.
- `src/score.py` — Wilson intervals, per-field and exact-match accuracy, majority-class floor, ECE with a reportability gate, latency aggregation, size caveats.
- `tests/test_end_to_end.py` — the whole pipeline on 3 fixture records with a stub model.
- `.venv` (3.12), `conftest.py`, `requirements-dev.txt`.

**Blocked on someone else**

1. `x-ordered` spelling — Juan.
2. The success metric — Hari.
3. Sourcemeta `jsonschema` CLI installed, so validation stops being stubbed.

**Next, in order**

1. Write and freeze `data/smoke/tickets.jsonl` (50 real messy tickets, 20% hand-checked, sha256 recorded). Must not be written by a model that then gets scored on it.
2. `src/runners/laya_runner.py` — only after the real Laya API is read from the installed package, not guessed.
3. `src/runners/llm_runner.py` — same, for the installed Outlines or XGrammar version.
4. `notebooks/smoke_test.ipynb`, then the first run into `results/`.
