import numpy as np
import pytest

from judge_cal import Judge, bias, calibrate, report
from judge_cal.agreement import cohen_kappa, majority, spearman
from judge_cal.swap import combine

PAIRS = "data/controlled-length-pairs.jsonl"


def biased_backend(prompt: str) -> str:
    """Prefers the longer answer, and slot A when lengths are close."""
    if "[Answer A]" in prompt:
        a = prompt.split("[Answer A]\n")[1].split("\n\n[Answer B]")[0]
        b = prompt.split("[Answer B]\n")[1].split("\n\nWhich")[0]
        return "[[A]]" if len(a) >= len(b) * 0.8 else "[[B]]"
    answer = prompt.split("Answer:\n")[1].split("\n\nEnd with")[0]
    return f"[[{1 + min(4, len(answer) // 40)}]]"


def test_swap_counts_tie_against_win_as_flip():
    assert combine("A", "B") == combine("A", "B")
    assert combine("A", "B").verdict == "A" and not combine("A", "B").flipped
    assert combine("A", "A").flipped
    assert combine("tie", "B").flipped and combine("tie", "B").verdict == "tie"
    assert not combine("tie", "tie").flipped


def test_audit_finds_length_bias():
    judge = Judge(backend=biased_backend, rubric="rubrics/helpfulness.yaml", swap=False)
    audit = bias.audit(judge, pairs=PAIRS)
    assert audit.length_pairs == 12
    assert audit.length_win_rate > 0.9
    unbiased = Judge(backend=lambda p: "[[tie]]")
    assert bias.audit(unbiased, PAIRS).length_win_rate == 0.5


def test_spearman_and_kappa():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 2, 3], [3, 2, 2, 1]) == pytest.approx(-1.0)
    assert cohen_kappa(list("AABB"), list("AABB")) == 1.0
    assert majority(["A", "B"]) == "tie"


def test_pav_is_monotone_and_exact_on_sorted_data():
    x = np.array([1, 2, 3, 4, 5], float)
    _, fit = calibrate.pav(x, np.array([0, 1, 0, 1, 1], float))
    assert np.all(np.diff(fit) >= 0)
    assert fit.tolist() == [0, 0.5, 0.5, 1, 1]


def test_calibration_reduces_ece_and_report_interval_is_wider():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(300):
        score = int(rng.integers(1, 6))
        # the judge is overconfident: humans accept far less often than its score implies
        accept = rng.random() < 0.1 * score
        rows.append({"question": f"q{i}", "answer": "x" * (40 * (score - 1)), "human": int(accept)})
    judge = Judge(backend=biased_backend, rubric="rubrics/helpfulness.yaml")
    cal = calibrate.fit(judge, rows, method="isotonic")
    assert cal.ece_after < cal.ece_before
    platt = calibrate.fit(judge, rows, method="platt")
    assert platt.params["a"] > 0
    scores = cal.score(judge, rows[:100])
    s = report.summary(scores, n_boot=300)
    assert s.low <= s.mean <= s.high
    assert (s.high - s.low) > 0
    assert "95% CI" in str(s)
