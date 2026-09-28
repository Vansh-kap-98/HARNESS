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

## What this does NOT show

There is **no baseline line yet**. "Accuracy falls as options multiply" is
expected and proves nothing on its own. The claim worth testing is whether a
small generative LLM degrades more slowly than a 421M decision model, at the
same k, on the same examples. Until that line exists, this is one curve, not a
comparison.

Banking77 intents are also a proxy for API operations: real API descriptions are
more confusable than banking intents, so the real wall is likely lower, not
higher.

## Consequence for the compiler

`MAX_CHOICE_OPTIONS = 20` was derived from token arithmetic. The data says the
arithmetic was irrelevant — but the number lands in the right place for a
different reason: accuracy is still 0.760 at k=16 and has collapsed to 0.526 by
k=32. Keep 20; replace the justification.
