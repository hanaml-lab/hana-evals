"""Position, length and self-preference bias, measured on pairs with a known human verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from judge_cal.swap import combine


@dataclass(frozen=True)
class Audit:
    n: int
    position_flip_rate: float  # share of pairs whose verdict changed with the order
    first_position_win_rate: float  # among decisive single-order verdicts, how often slot A won
    length_win_rate: float  # on human-tied pairs of different length; 0.5 is unbiased
    length_pairs: int
    self_preference: float = 0.0  # judge's win rate for its own family's answers minus the people's; 0 is unbiased
    self_pairs: int = 0


def load_pairs(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


_WIN = {"A": 1.0, "B": 0.0, "tie": 0.5}  # the share of a pair won by answer A


def audit(judge, pairs: str | Path | list[dict], family: str | None = None) -> Audit:
    """``family`` names the judge's own model family; pairs whose ``families`` give exactly one answer from it
    measure self-preference against the human verdict."""
    rows = load_pairs(pairs) if isinstance(pairs, str | Path) else pairs
    flips = first_wins = decisive = 0
    longer_wins = 0.0
    length_pairs = 0
    own_judge = own_human = 0.0
    self_pairs = 0
    for row in rows:
        forward = judge.compare_once(row["question"], row["a"], row["b"])
        backward = judge.compare_once(row["question"], row["b"], row["a"])
        result = combine(forward, backward)
        flips += result.flipped
        for v in (forward, backward):
            if v != "tie":
                decisive += 1
                first_wins += v == "A"
        if row.get("human") == "tie" and len(row["a"]) != len(row["b"]):
            length_pairs += 1
            longer = "A" if len(row["a"]) > len(row["b"]) else "B"
            verdict = result.verdict if judge.swap else forward
            longer_wins += 1.0 if verdict == longer else 0.5 if verdict == "tie" else 0.0
        families = row.get("families") or {}
        if family and row.get("human") in _WIN and [families.get("a"), families.get("b")].count(family) == 1:
            self_pairs += 1
            own = _WIN if families.get("a") == family else {v: 1.0 - share for v, share in _WIN.items()}
            own_judge += own[result.verdict if judge.swap else forward]
            own_human += own[row["human"]]
    n = len(rows)
    return Audit(
        n=n,
        position_flip_rate=flips / n if n else 0.0,
        first_position_win_rate=first_wins / decisive if decisive else 0.5,
        length_win_rate=longer_wins / length_pairs if length_pairs else 0.5,
        length_pairs=length_pairs,
        self_preference=(own_judge - own_human) / self_pairs if self_pairs else 0.0,
        self_pairs=self_pairs,
    )
