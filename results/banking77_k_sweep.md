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

## Finding: the option wall is discrimination, not context

The largest drop is 16 → 32, **−0.234**, at 339 tokens of a 512-token context
with nothing truncated. The step that *does* overflow, 32 → 64, costs only
−0.106.

Token counts are the real tokenizer summing state + instructions + each option
id + each description. They omit Laya's per-option `[MASK]` and separators, so
true usage is higher — but even allowing two or three tokens of framing per
option, k=32 lands near 400 and stays inside the budget. The conclusion is
robust to that uncertainty.

**Replicated across distractor seeds:**

| seed | k=16 | k=32 | drop |
|---|---|---|---|
| 0 | 0.760 [0.721, 0.795] | 0.526 [0.482, 0.569] | 0.234 |
| 1 | 0.752 [0.712, 0.788] | 0.522 [0.478, 0.565] | 0.230 |

So the cliff is not an artifact of which distractors were sampled. Seed 0 at
k=16 reproduced identically across two sessions, confirming determinism.

## Calibration: measured, but on a checkpoint that warns about itself

ECE at k=77 (n=3,080, 10 bins): **0.497**. The card reports 0.466 uncalibrated
and 0.081 after temperature scaling, so our uncalibrated figure corroborates
theirs even though our accuracy does not.

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
