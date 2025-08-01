"""Combine the verdicts of a pair judged in both orders."""

from __future__ import annotations

from dataclasses import dataclass

_MIRROR = {"A": "B", "B": "A", "tie": "tie"}


@dataclass(frozen=True)
class Combined:
    verdict: str  # A | B | tie, in the original order
    flipped: bool


def combine(forward: str, backward: str) -> Combined:
    """``forward`` judged (a, b); ``backward`` judged (b, a), in that call's own labels.

    The backward verdict is mapped back to the original order. If the two agree, that is the
    verdict. Any disagreement is a flip, including a tie in one order and a win in the other,
    and the pair is scored as a tie.
    """
    back = _MIRROR[backward]
    if back == forward:
        return Combined(forward, False)
    return Combined("tie", True)
