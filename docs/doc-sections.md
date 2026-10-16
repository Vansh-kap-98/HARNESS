# Sections for *Laya + Jev + JSON Schema*

Drafted for Hari's document. §16.4 is the section he tagged; §19.2 is the empty one.
Every number cited here has a run behind it in `results/`.

---

## 16.4 Policy: turning a probability into an action

**The rule as written is correct.** Automate when the expected cost of being wrong
is below the cost of a review:

```
act if (1 - p) · c_w < c_h        →        p > 1 - c_h / c_w
```

Checked against a brute-force search over p at four cost ratios; closed form and
search agree to 1e-4 every time. The worked example holds: c_w = 50, c_h = 5
gives p* = 0.90, so 0.94 automates and 0.86 reviews.

It is valid under three conditions, and two of them are currently violated in
practice.

### Condition 1: p must be calibrated — and above ten options it isn't

The rule consumes a probability, not a score. Three different numbers come back
from Laya and only one of them belongs here:

| field | what it is |
|---|---|
| `probabilities` | the per-option distribution |
| `confidence` | `1 - H(p)/log k`, normalised entropy. The library's own docstring: *"not calibrated: it is not what temperature scaling fits and not what the reported ECE measures"* |
| `answer_confidence` | the calibrated max-p, and the only one this rule may use |

Worse, the calibration that `answer_confidence` depends on is bucketed by
`(question type, option count)`, and the shipped checkpoint's `choice:11+`
temperature is 0.1006 — outside the library's own accepted range of [0.5, 5], so
it is clamped at load time with the warning *"treat confidence from the affected
entries as uncalibrated."*

**So for any choice with eleven or more options, the input this rule requires
does not exist on the shipped checkpoint.** Either refit temperatures on your own
data (the fine-tuning recipe does this and replaces the per-bucket table with one
value per question type), or keep k ≤ 10 where the shipped calibration applies.

### Condition 2: every wrong option must cost about the same

`(1 - p) · c_w` lumps all the ways of being wrong into one price. The honest
version is `Σ_{i ≠ chosen} p_i · c_i`, and the two disagree in the unsafe
direction:

```
p = (0.95, 0.04, 0.01),  c_h = 5
flat rule    : (1 - 0.95) × 50 = 2.50   →  AUTOMATE
per-outcome  : 0.04×10 + 0.01×2000 = 20.40  →  REVIEW
```

A rare catastrophic option is invisible to the flat rule. Use per-outcome costs
wherever the options differ in kind — which, for API routing, they almost always
do.

### Condition 3: review is assumed perfect

If a human reviewer is also wrong with probability e, reviewing costs
`c_h + e·c_w` and the threshold drops: at c_w = 50, c_h = 5, a 5% reviewer error
rate moves p* from 0.90 to 0.85. This errs toward reviewing too much, so it is
the safe direction, but it is worth knowing the rule is conservative rather than
exact.

### And the part the cost rule cannot express

A threshold decides *how confident is confident enough*. It cannot decide *what
must never be automatic regardless of confidence*. Those are different questions
and they want different machinery:

- **Rules own the operation class.** Whether an operation is irreversible, moves
  money, or is externally visible is a static property of the API. It is known
  before runtime, it is auditable, and a model will never beat a lookup at it.
- **The model owns the situation.** The same `DELETE` is routine against a test
  tenant and catastrophic against production. No static rule sees that; it is in
  the state text, which is what a `noul` reads.
- **The invariant that makes it safe: the model may only ever add a gate, never
  remove one.** Rules set a floor, the model can escalate above it, and when the
  model is wrong you fall back to the floor, which fails closed.

On existing vocabulary for the floor: RFC 9110 already classifies HTTP methods as
safe and idempotent, and MCP has carried tool annotations since the 2025-03-26
revision — `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`,
with `destructive` and `openWorld` defaulting to true, so absent annotations fail
closed. The catch is that the MCP spec calls them *hints* and says clients
"should never make tool use decisions based on annotations from untrusted
servers." So they are an input to a policy table we own, not the authority.

Nothing standardised exists for the semantic half. That is the gap.

---

## 19.2 Where this goes next

§16.2 lists eight open questions about which parts of JSON Schema map to which
decision type. Those are the right questions, and the harness is the instrument
that answers them. The proposal is to work the table row by row, each row ending
in a measurement rather than an opinion.

| Schema feature | Question | Status |
|---|---|---|
| `enum` → Choice | how many options before it breaks? | **answered** |
| `description` → question text | how much does description quality matter? | partial |
| `boolean` → Noul | how to word the yes/no statement? | next, cheap |
| `minimum`/`maximum` → Score | when is the scale really ordered? | next, and it is the vocabulary question |
| `required` → input contract | should missing evidence mean abstain? | after the gate work |
| nested object, array, `oneOf`/`anyOf` | — | currently refused by the compiler; revisit last |

### Row 1 is closed

The usable option count is not a constant. Options share `head_max_len`, and the
library gives each `max(4, (head_max_len - 16) // k)` tokens, truncating the rest.
Past that point labels are cut mid-word and distinct options collapse into
identical strings — at k=32 on Banking77, five of the thirty-two options became
byte-identical.

Measured: accuracy falls 0.234 from k=16 to k=32 (five seeds, mean 0.239,
sd 0.013, McNemar p < 1e-20 each). Raising `head_max_len` from 192 to 384 recovers
0.134 of that, with disjoint intervals. k=64 and k=77 do not move because 384
still leaves them at the 4-token floor.

So roughly 57% of the cliff is a configuration artefact and the rest is genuine
discrimination cost. `src/budget.py` computes this from a schema and a model
config before a forward pass, and refuses a `choice` whose options would collapse.

### The three next runs, and what each decides

1. **Re-score calibration on `answer_confidence`.** Our ECE of 0.497 was computed
   on raw max-probability, which is not the field the card's 0.466 → 0.081 claim
   refers to. Until this is redone we cannot say whether Laya's calibration claim
   holds. One scoring pass, no retraining. *Decides whether the gate in §16.4 has
   a usable input at all.*

2. **State truncation under a wide option budget.** `max_len` is 512 and options
   and state share it, so raising `head_max_len` to 384 leaves ~128 tokens for the
   state. Our Banking77 queries are 15–25 tokens so we never saw this; a real
   support email is 200–500. The question is not only whether it truncates but
   from which end — losing the opening of an email loses who and what, losing the
   end loses the request itself. *Decides whether "raise the budget" is advice we
   can actually give.*

3. **Ordinality.** JSON Schema cannot express that an enum's members have a
   natural order. It does not affect validation, which is why it is reasonably
   absent from core, but every consumer that is not a validator needs it — model
   compilers, UI generators, anything that sorts. We needed it on day one and had
   to invent a marker to get past it. *Decides whether there is a vocabulary
   contribution here, and it is the smallest possible proposal: one annotation
   keyword, no validation semantics.*

### What this is not

Every number above is from Banking77 intent classification, which is a proxy for
API operations. Real API descriptions are more confusable than banking intents,
so the wall is likely lower, not higher. And none of it tests the project's
original claim — that making JSON Schema native to the pipeline improves accuracy
over a small LLM with constrained decoding on a *schema-shaped* task. That
experiment has not been run.
