"""Map judge scores to the probability that a human would accept the answer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from judge_cal.agreement import spearman


def pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic (non-decreasing) regression by pool-adjacent-violators.

    Returns the distinct x values and the fitted value at each.
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    ux, inverse = np.unique(xs, return_inverse=True)
    sums = np.bincount(inverse, weights=ys)
    weights = np.bincount(inverse).astype(float)
    blocks: list[list[float]] = []  # [value, weight, count of distinct x]
    for s, w in zip(sums, weights, strict=True):
        blocks.append([s / w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    fitted = np.concatenate([[v] * int(c) for v, _, c in blocks])
    return ux, fitted


def platt(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 100) -> tuple[float, float]:
    """Fit p = sigmoid(a * x + b) by Newton's method with a small ridge penalty."""
    a, b = 0.0, 0.0
    X = np.stack([x, np.ones_like(x)], axis=1).astype(float)
    w = np.array([a, b])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) + l2 * w
        hess = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(total)


@dataclass
class Agreement:
    """How the calibrated judge's accept/reject decisions compare with the human labels."""

    n: int
    accuracy: float
    sensitivity: float  # P(judge accepts | human accepts)
    specificity: float  # P(judge rejects | human rejects)
    positives: int
    negatives: int
    spearman: float | None = None


@dataclass
class Calibrator:
    method: str
    raw: np.ndarray  # judge scores on the calibration set
    labels: np.ndarray  # 1 if the human accepted
    scale: tuple[int, int] = (1, 5)
    params: dict = field(default_factory=dict)
    agreement: Agreement | None = None
    ece_before: float = float("nan")
    ece_after: float = float("nan")

    def fit(self) -> Calibrator:
        if self.method == "isotonic":
            self.params["x"], self.params["y"] = pav(self.raw, self.labels)
        elif self.method == "platt":
            self.params["a"], self.params["b"] = platt(self.raw, self.labels)
        else:
            raise ValueError("method is isotonic or platt")
        return self

    def predict(self, raw) -> np.ndarray:
        raw = np.asarray(raw, float)
        if self.method == "isotonic":
            return np.interp(raw, self.params["x"], self.params["y"])
        return 1 / (1 + np.exp(-(self.params["a"] * raw + self.params["b"])))

    def refit(self, raw: np.ndarray, labels: np.ndarray) -> Calibrator:
        return Calibrator(self.method, raw, labels, self.scale).fit()

    def score(self, judge, items: str | Path | list[dict]) -> Scores:
        rows = _load(items)
        raw = np.array([judge.score(r["question"], r["answer"], r.get("reference")) for r in rows], float)
        return Scores(raw, self.predict(raw), self)


@dataclass
class Scores:
    raw: np.ndarray
    probs: np.ndarray
    calibrator: Calibrator

    @property
    def mean(self) -> float:
        return float(self.probs.mean())


def _load(source: str | Path | list[dict]) -> list[dict]:
    if isinstance(source, list):
        return source
    return [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]


def _accepted(row: dict, scale: tuple[int, int]) -> int:
    if "human" in row:
        return int(bool(row["human"]))
    return int(row["human_score"] >= scale[0] + 0.75 * (scale[1] - scale[0]))


def fit(judge, human_labels: str | Path | list[dict], method: str = "isotonic", folds: int = 5,
        seed: int = 0) -> Calibrator:
    """Fit a calibrator on held-out human labels.

    Each label row has ``question``, ``answer`` and either ``human`` (1 accept, 0 reject) or
    ``human_score`` on the rubric scale. ``ece_after`` is measured by cross-validation, so it is
    not flattered by fitting and scoring on the same rows.
    """
    rows = _load(human_labels)
    scale = judge.rubric.scale
    raw = np.array([judge.score(r["question"], r["answer"], r.get("reference")) for r in rows], float)
    labels = np.array([_accepted(r, scale) for r in rows], float)
    cal = Calibrator(method, raw, labels, scale).fit()

    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(rows)) % folds
    held_out = np.empty(len(rows))
    for k in range(folds):
        test = fold == k
        if test.any() and (~test).any():
            held_out[test] = cal.refit(raw[~test], labels[~test]).predict(raw[test])
    cal.ece_before = ece((raw - scale[0]) / (scale[1] - scale[0]), labels)
    cal.ece_after = ece(held_out, labels)

    decided = cal.predict(raw) >= 0.5
    pos, neg = labels == 1, labels == 0
    human_scores = [r["human_score"] for r in rows if "human_score" in r]
    cal.agreement = Agreement(
        n=len(rows),
        accuracy=float((decided == pos).mean()),
        sensitivity=float(decided[pos].mean()) if pos.any() else float("nan"),
        specificity=float((~decided[neg]).mean()) if neg.any() else float("nan"),
        positives=int(pos.sum()),
        negatives=int(neg.sum()),
        spearman=spearman(raw, human_scores) if len(human_scores) == len(rows) else None,
    )
    return cal
