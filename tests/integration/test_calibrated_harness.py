"""judge-cal calibrates a judge; eval-harness uses it as the check on a judged stage."""

from pathlib import Path

import pytest

from eval_harness import Runner, Suite
from hana_evals import ReferenceOverlapJudge, calibrated_check
from judge_cal import Judge, report

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def calibrated():
    judge = Judge(backend=ReferenceOverlapJudge(), rubric=ROOT / "packages/judge-cal/rubrics/faithfulness.yaml")
    check, cal = calibrated_check(judge, ROOT / "examples/data/email-labels.jsonl")
    return judge, check, cal


def test_calibration_improves_and_agrees(calibrated):
    _, _, cal = calibrated
    assert cal.ece_after < cal.ece_before
    assert cal.agreement.accuracy > 0.75


def test_judge_check_runs_on_the_email_stage(calibrated):
    judge, check, cal = calibrated
    suite = Suite.load(ROOT / "packages/eval-harness/suites/back-office")
    result = Runner(agent={"model": "simulated", "error_rate": 0.3, "seed": 4}, repeats=4,
                    review_budget=1.0, judge_check=check).run(suite)
    judged = [r for r in result.stages if r["check_by"] == "judge"]
    assert judged and all(r["stage"] == "draft_email" for r in judged)
    # a truncated draft is rejected by the calibrated judge and redrafted
    assert any(r["check_passed"] is False for r in judged)
    wrong_kept = [r for r in judged if r["check_passed"] and not r["correct"]]
    assert len(wrong_kept) <= len(judged) * 0.1


def test_report_interval_includes_judge_error(calibrated):
    judge, _, cal = calibrated
    items = [{"question": "q", "answer": f"Hello, invoices INV-2026{i:04d} differ.",
              "reference": f"Hello, invoices INV-2026{i:04d} differ."} for i in range(40)]
    s = report.summary(cal.score(judge, items), n_boot=300)
    assert s.low <= s.mean <= s.high
