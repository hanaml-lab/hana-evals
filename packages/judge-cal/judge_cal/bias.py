"""Position and length bias, measured on pairs with a known human verdict."""

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


def load_pairs(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(judge, pairs: str | Path | list[dict]) -> Audit:
    rows = load_pairs(pairs) if isinstance(pairs, str | Path) else pairs
    flips = first_wins = decisive = 0
    longer_wins = 0.0
    length_pairs = 0
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
    n = len(rows)
    return Audit(
        n=n,
        position_flip_rate=flips / n if n else 0.0,
        first_position_win_rate=first_wins / decisive if decisive else 0.5,
        length_win_rate=longer_wins / length_pairs if length_pairs else 0.5,
        length_pairs=length_pairs,
    )
