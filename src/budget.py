"""Will the options survive the model's option budget?

A decision model scores each option at its own marker, and the options share a
fixed token budget (`head_max_len`). When they do not fit, the model does not
refuse -- it silently truncates each option, and past a point distinct options
become identical strings that no model can tell apart.

This module answers that question before a forward pass, from the schema and
the model's config alone.

The arithmetic replicates `laya/common.py`, which does:

    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]

Measured against that library at `head_max_len=192` on Banking77: k=16 fits
untouched, k=32 gets 5 tokens per option (predicted 5), k=64 and k=77 both hit
the floor of 4 (predicted 4). Raising the budget to 384 takes k=32 to 11 tokens
and lifts accuracy 0.526 -> 0.660, while k=64 gains one token and does not move.
See `results/banking77_k_sweep.md`.

One thing this cannot do on its own: the caller supplies `count_tokens`, because
only the model's own tokenizer knows the real lengths. A character estimate will
mislead near the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "BudgetError",
    "OptionBudget",
    "HEAD_SLACK",
    "MIN_TOKENS_PER_OPTION",
    "render_option",
    "option_budget",
    "usable_option_count",
]

# Both constants are the library's, not ours: truncation begins when fewer than
# HEAD_SLACK tokens remain, and no option is cut below MIN_TOKENS_PER_OPTION.
HEAD_SLACK = 16
MIN_TOKENS_PER_OPTION = 4


class BudgetError(Exception):
    """The budget cannot be computed from these inputs."""


@dataclass(frozen=True)
class OptionBudget:
    """What the model will actually see for one `choice` question."""

    options: int
    head_max_len: int
    rendered_tokens: int
    truncated: bool
    tokens_per_option: int
    cut: Tuple[str, ...]
    collisions: Tuple[Tuple[str, ...], ...]

    @property
    def distinct(self) -> int:
        collided = sum(len(group) for group in self.collisions)
        return self.options - collided + len(self.collisions)

    @property
    def usable(self) -> bool:
        """False once the model can no longer tell every option apart."""
        return not self.collisions

    def as_dict(self) -> Dict[str, Any]:
        return {
            "options": self.options,
            "head_max_len": self.head_max_len,
            "rendered_tokens": self.rendered_tokens,
            "truncated": self.truncated,
            "tokens_per_option": self.tokens_per_option,
            "cut": list(self.cut),
            "collisions": [list(group) for group in self.collisions],
            "distinct": self.distinct,
            "usable": self.usable,
        }

    def reason(self) -> str:
        """Why this question is refused, with the arithmetic that refuses it."""
        if self.usable:
            return ""
        groups = ", ".join("/".join(group) for group in self.collisions[:3])
        return (
            "%d options share a %d-token budget, leaving %d tokens each; after "
            "truncation only %d remain distinct (%s). The model cannot "
            "distinguish options it is shown as identical strings."
            % (self.options, self.head_max_len, self.tokens_per_option,
               self.distinct, groups)
        )


def render_option(key: str, description: str) -> str:
    """How an option is laid out before its marker: `key: description`."""
    text = (description or "").strip()
    return "%s: %s" % (key, text) if text else "%s:" % key


def option_budget(
    criteria: Mapping[str, str],
    head_max_len: int,
    count_tokens: Callable[[str], int],
) -> OptionBudget:
    """Compute what survives, using the model's own truncation arithmetic.

    `count_tokens` must be the model's tokenizer. Truncation here is applied to
    the rendered string by token count, which is what the library does to the
    token ids; the two agree except where a cut lands mid-token, so treat a
    result within a token of the boundary as uncertain.
    """
    if not isinstance(criteria, Mapping) or not criteria:
        raise BudgetError("no options to budget")
    if head_max_len <= 0:
        raise BudgetError("head_max_len must be positive, got %r" % (head_max_len,))

    keys = list(criteria)
    rendered = [render_option(key, criteria[key]) for key in keys]
    lengths = [count_tokens(text) for text in rendered]
    total = sum(lengths)

    if head_max_len - total >= HEAD_SLACK:
        return OptionBudget(
            options=len(keys),
            head_max_len=head_max_len,
            rendered_tokens=total,
            truncated=False,
            tokens_per_option=max(lengths),
            cut=(),
            collisions=(),
        )

    per = max(MIN_TOKENS_PER_OPTION,
              (head_max_len - HEAD_SLACK) // max(1, len(keys)))
    cut = tuple(key for key, length in zip(keys, lengths) if length > per)

    # What the model is left with. Truncating by token count is approximated
    # here by taking the longest prefix whose token count fits.
    surviving: List[str] = []
    for text in rendered:
        surviving.append(_prefix_within(text, per, count_tokens))

    seen: Dict[str, List[str]] = {}
    for key, text in zip(keys, surviving):
        seen.setdefault(text, []).append(key)
    collisions = tuple(
        tuple(group) for group in seen.values() if len(group) > 1
    )

    return OptionBudget(
        options=len(keys),
        head_max_len=head_max_len,
        rendered_tokens=total,
        truncated=True,
        tokens_per_option=per,
        cut=cut,
        collisions=collisions,
    )


def usable_option_count(
    criteria: Mapping[str, str],
    head_max_len: int,
    count_tokens: Callable[[str], int],
) -> int:
    """The largest prefix of these options that stays mutually distinguishable.

    Options are dropped from the end, which is why option order matters: put
    the ones you cannot afford to lose first.
    """
    keys = list(criteria)
    for size in range(len(keys), 1, -1):
        subset = {key: criteria[key] for key in keys[:size]}
        if option_budget(subset, head_max_len, count_tokens).usable:
            return size
    return 0


def _prefix_within(
    text: str, budget: int, count_tokens: Callable[[str], int]
) -> str:
    if count_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]
