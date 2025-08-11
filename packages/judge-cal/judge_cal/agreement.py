"""Agreement between a judge and human raters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def majority(labels: Sequence[str]) -> str:
    """Majority vote; a split vote is a tie."""
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "tie"
    return counts[0][0]


def pairwise_agreement(judge: Sequence[str], human: Sequence[str]) -> float:
    if len(judge) != len(human) or not judge:
        raise ValueError("need two non-empty sequences of equal length")
    return float(np.mean([j == h for j, h in zip(judge, human, strict=True)]))


def rank(x: Sequence[float]) -> np.ndarray:
    """Ranks starting at 1, ties given their average rank."""
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr))
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two sequences of equal length >= 2")
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty sequences of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
