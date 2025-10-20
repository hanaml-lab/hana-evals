"""An interval for the human-accept rate that accounts for the judge's own error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from judge_cal.calibrate import Agreement, Scores


@dataclass(frozen=True)
class Summary:
    mean: float
    low: float
    high: float
    naive_low: float  # interval if the judge were ground truth
    naive_high: float
    n_items: int
    n_labels: int
    agreement: Agreement | None

    def __str__(self) -> str:
        line = (f"human-accept rate {self.mean:.3f} (95% CI {self.low:.3f}-{self.high:.3f}; "
                f"judge-as-truth CI {self.naive_low:.3f}-{self.naive_high:.3f}; "
                f"{self.n_items} items, {self.n_labels} human labels)")
        if self.agreement:
            line += f"\njudge vs human: accuracy {self.agreement.accuracy:.3f}"
        return line


def summary(scores: Scores, human_agreement: Agreement | None = None, n_boot: int = 2000,
            level: float = 0.95, seed: int = 0) -> Summary:
    """Bootstrap both sources of error: which items were sampled, and which human labels the
    calibrator was fitted on. The second term is what widens the interval when the judge
    disagrees with people often or when there are few labels."""
    rng = np.random.default_rng(seed)
    cal = scores.calibrator
    n, m = len(scores.raw), len(cal.raw)
    threshold = cal.scale[0] + 0.75 * (cal.scale[1] - cal.scale[0])
    full, naive = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        items = scores.raw[rng.integers(0, n, n)]
        labels_idx = rng.integers(0, m, m)
        refit = cal.refit(cal.raw[labels_idx], cal.labels[labels_idx])
        full[i] = refit.predict(items).mean()
        naive[i] = (items >= threshold).mean()
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(full, [tail, 100 - tail])
    nlo, nhi = np.percentile(naive, [tail, 100 - tail])
    return Summary(scores.mean, float(lo), float(hi), float(nlo), float(nhi), n, m,
                   human_agreement or cal.agreement)
