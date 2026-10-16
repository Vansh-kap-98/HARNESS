# Banking77: accuracy vs option count (Laya, English checkpoint, zero-shot)

First measured results of the project. Every number here was produced by this
repo's harness on Google Colab (T4), not quoted from a model card.

## Configuration

| | |
|---|---|
| Model | `convaiinnovations/laya`, English checkpoint, zero-shot, `model="english"` |
| laya version | 0.3.20 (as installed by `pip install laya`, Sep 2026) |
| Data | Banking77 test split, `PolyAI-LDN/task-specific-datasets` `banking_data/test.csv`, 3,080 queries, 77 intents |
| Examples per k | 500, drawn by `random.Random(0).shuffle()` then head — the CSV is grouped by category, so an unshuffled head would cover ~12 intents |
| Option sets | `src/sweep.py`, nested across k, selection and display permutations independent |
| Criteria text | the label name with underscores replaced by spaces (`card_arrival` → `card arrival`) |
| Instructions | "Which customer-service intent does this message express?" |
| Seeds | 0 for the curve; 0 and 1 for the cliff replication |
| Tokenizer | `AutoTokenizer.from_pretrained("convaiinnovations/laya", subfolder="tokenizer")` |

## The curve

| k | accuracy | 95% Wilson | chance (1/k) | real tokens (median) | over 512 |
|---|---|---|---|---|---|
| 4 | 0.906 | [0.877, 0.929] | 0.250 | 61 | 0/500 |
| 8 | 0.850 | [0.816, 0.879] | 0.125 | 99 | 0/500 |
| 16 | 0.760 | [0.721, 0.795] | 0.063 | 179 | 0/500 |
| 32 | 0.526 | [0.482, 0.569] | 0.031 | 339 | 0/500 |
| 64 | 0.420 | [0.378, 0.464] | 0.016 | 656 | 500/500 |
| 77 | 0.368 | [0.327, 0.411] | 0.013 | 783 | 500/500 |

Full test split at k=77 (n=3,080): **0.381 [0.364, 0.398]**. The card claims
0.425; our interval excludes it. Candidate causes, untested: our criteria text
is only the label name, and the split/checkpoint they used is not stated.

## Finding: the wall is governed by tokens per option

The largest drop is 16 → 32, **−0.234**. Three candidate explanations were
tested and two were eliminated.

**Not the context filling up.** Using Laya's own tokenizer and its own
`build_sequence`, the full sequence at k=32 is 182 tokens of a 512-token
`max_len`. At k=77 it is 325. The context is never close to full.

**Not options being dropped.** `build_sequence` returns one `[MASK]` marker per
option, and the count equals k at every k up to 77. Every option is scored.

**It is the per-option token budget.** `head_max_len` is 192, shared across all
k options, so each option gets roughly `192/k` tokens. Past k≈20 that is less
than a label needs, and the labels are cut mid-word:

```
k=16   '[MASK] terminate_account: terminate account'
k=32   '[MASK] card_not_'
k=64   '[MASK] unable_to'
```

Truncation collapses distinct options into identical strings — unanswerable by
construction, since the model is not shown different things:

| k | options | identical after truncation | distinct |
|---|---|---|---|
| 16 | 16 | 0 | 16 |
| 32 | 32 | 5 | 28 |
| 64 | 64 | 11 | 56 |
| 77 | 77 | 15 | 67 |

Collisions begin exactly where the cliff is.

### Confirmed by raising the budget

Re-running with `head_max_len` set to 384 (still inside `max_len` 512), nothing
else changed:

| k | 192 | 384 | change |
|---|---|---|---|
| 32 | 0.526 [0.482, 0.569] | **0.660 [0.617, 0.700]** | **+0.134**, intervals disjoint |
| 64 | 0.420 [0.378, 0.464] | 0.406 [0.364, 0.450] | none |
| 77 | 0.368 [0.327, 0.411] | 0.372 [0.331, 0.415] | none |

Lined up by tokens per option rather than by k, the pattern is one variable:

| budget ÷ k | tokens/option | accuracy |
|---|---|---|
| 192 ÷ 16 | 12 | 0.760 |
| 384 ÷ 32 | 12 | 0.660 |
| 192 ÷ 32 | 6 | 0.526 |
| 384 ÷ 64 | 6 | 0.406 |
| 192 ÷ 64 | 3 | 0.420 |
| 384 ÷ 77 | 5 | 0.372 |

k=64 and k=77 did not move because 384 still leaves them at 5–6 tokens per
option, the same starvation. Giving k=64 twelve tokens each needs 768, past
`max_len` 512 — which is where the context ceiling finally binds, indirectly.

**So the 0.234 cliff decomposes into −0.134 of token-budget artefact and −0.100
of genuine discrimination cost.** The practical rule: a decision model of this
shape needs roughly 12 tokens per option, and the usable option count is
`head_max_len / 12`, not a fixed number.

### Two claims withdrawn

1. *"The option wall is discrimination, not context."* Too strong. Most of the
   cliff was a budget artefact. The corrected claim is above.
2. *"Our redundant option text (`card_arrival: card arrival`) was costing us."*
   Tested by stripping the description: **+0.036 at k=16, +0.008 at k=32, +0.000
   at k=64.** Real but marginal, and it does not survive past k=16 because even a
   bare label does not fit at 6 tokens.

### Replication: five seeds, paired test

The 500 examples are identical at both k, so the comparison is paired and
McNemar's exact test is the right one — comparing two independent Wilson
intervals is valid but conservative. Suggested by Harsh Popat.

| seed | k=16 | k=32 | drop | right at 16 only | right at 32 only | McNemar p |
|---|---|---|---|---|---|---|
| 0 | 0.760 | 0.526 | 0.234 | 142 | 25 | 4.65e-21 |
| 1 | 0.752 | 0.522 | 0.230 | 126 | 11 | 6.65e-26 |
| 2 | 0.784 | 0.534 | 0.250 | 136 | 11 | 1.44e-28 |
| 3 | 0.758 | 0.534 | 0.224 | 134 | 22 | 8.71e-21 |
| 4 | 0.766 | 0.508 | 0.258 | 142 | 13 | 1.37e-28 |

**Mean drop 0.239, sd 0.013, range 0.224–0.258.** Every seed is significant at
p < 1e-20. The cliff does not depend on which distractors were sampled.

Seed 0 at k=16 also reproduced identically across Colab and Kaggle sessions,
confirming the pipeline is deterministic end to end.

## Calibration: measured, but on a checkpoint that warns about itself

ECE at k=77 (n=3,080, 10 bins): **0.497**, computed on `max(probabilities)`.

**That number is not comparable to the card's 0.466, and an earlier version of
this file wrongly said it corroborated it.** `laya/common.py` carries three
different quantities and we measured the one nobody acts on:

| field | what it is |
|---|---|
| `probabilities` | the per-option distribution; our ECE used its max |
| `confidence` | `1 - H(p)/log k`, normalised entropy. The library's own docstring: *"not calibrated: it is not what temperature scaling fits and not what the reported ECE measures"* |
| `answer_confidence` | the calibrated max-p, and the field a gate should threshold |

This also explains the divergence captured in the worked example: `urgency`
reported `confidence` 0.0145 against a max-p of 0.4119. That is normalised
entropy on a near-uniform distribution, not a probability.

**Outstanding:** re-score ECE on `answer_confidence`. Only that number speaks to
the card's 0.466 → 0.081 claim, and only that number is relevant to using a
decision model as a safety gate.

**Caveat that must travel with that number.** At load time the library emits:

```
laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
using choice:11+=0.10058280825614929 -> 0.5.
Treat confidence from the affected entries as uncalibrated.
```

Laya temperature-scales per (question type, option-count) bucket. The shipped
temperature for `choice:11+` is out of range and gets clamped. Every k from 16
up is in that bucket, so these confidences are uncalibrated by the library's own
admission — and that is precisely the many-option regime routing lives in.

## Baseline: Qwen2.5-0.5B-Instruct, same examples, same k

Scored by picking the option with the highest total log-probability as a
continuation of a prompt listing all k options. For a closed set this is
*stronger* than grammar-constrained greedy decoding, which picks token by token
and can walk into a suboptimal option — so the baseline gets the better version
of the task.

| k | Laya 421M | Qwen 0.5B | gap | chance |
|---|---|---|---|---|
| 4 | 0.906 [0.877, 0.929] | 0.404 [0.362, 0.448] | 0.502 | 0.250 |
| 8 | 0.850 [0.816, 0.879] | 0.304 [0.265, 0.346] | 0.546 | 0.125 |
| 16 | 0.760 [0.721, 0.795] | 0.258 [0.222, 0.298] | 0.502 | 0.063 |
| 32 | 0.526 [0.482, 0.569] | 0.286 [0.248, 0.327] | 0.240 | 0.031 |
| 64 | 0.420 [0.378, 0.464] | 0.266 [0.229, 0.306] | 0.154 | 0.016 |
| 77 | 0.368 [0.327, 0.411] | 0.230 [0.195, 0.269] | 0.138 | 0.013 |

**The decision model wins at every k, and its lead collapses.** Laya falls 0.538
across the range; Qwen falls 0.174. The convergence is driven entirely by Laya
degrading, not by the baseline improving. Whether the lines cross beyond k=77 is
not licensed by this data.

Qwen is non-monotonic between k=16 and k=32 (0.258 → 0.286), within overlapping
intervals, so consistent with noise at n=500.

### Length bias: measured, and it changes nothing

Options are ranked by *unnormalised* summed log-probability, which favours short
strings, and Banking77 labels run from `card_arrival` to
`verify_source_of_funds`. Re-scored both ways from the same forward passes
(Kaggle, T4, same examples and seed):

| k | summed | length-normalised | picked a shortest option | chance |
|---|---|---|---|---|
| 4 | 0.404 [0.362, 0.448] | 0.390 [0.348, 0.433] | 34.4% | 25.0% |
| 8 | 0.304 [0.265, 0.346] | 0.308 [0.269, 0.350] | 24.0% | 12.5% |
| 16 | 0.258 [0.222, 0.298] | 0.240 [0.205, 0.279] | 18.6% | 6.2% |

The bias is real: the summed scorer picks a shortest option about 3x chance by
k=16. But normalising recovers no accuracy — every pair of intervals overlaps
heavily and two of the three move slightly the wrong way. The short options it
over-picks were not costing it correct answers.

**So the baseline was not handicapped by the scoring choice.** Qwen-0.5B's
weakness on this task is genuine, and the gap to Laya stands as measured.

### One caveat that remains

**No latency comparison may be drawn from this run.** The baseline scorer
   does k chunked forward passes per example (43 minutes for 500 examples at
   k=77, against roughly 25 seconds for Laya). That is an artefact of scoring
   every option exhaustively, not a property of the model. A latency claim needs
   single-pass constrained decoding and the §6 protocol.

Banking77 intents are also a proxy for API operations: real API descriptions are
more confusable than banking intents, so the real wall is likely lower, not
higher.

## Consequence for the compiler

`MAX_CHOICE_OPTIONS = 20` was derived from token arithmetic. The data says the
arithmetic was irrelevant — but the number lands in the right place for a
different reason: accuracy is still 0.760 at k=16 and has collapsed to 0.526 by
k=32. Keep 20; replace the justification.
