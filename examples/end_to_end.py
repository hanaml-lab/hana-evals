"""Calibrate a judge, then let eval-harness use it on a long back-office job. Offline.

    python examples/end_to_end.py

1. judge-cal fits the judge's 1-5 scores to 160 human accept/reject labels on email drafts;
2. eval-harness runs the six-job reconciliation suite eight times per job with five agent seeds
   (240 runs per placement), checking three of the nine stages of each job: code checks on
   figures and the calibrated judge on the supplier email. The same budget goes to the last
   stages, to evenly spaced stages, or to the stages whose errors cost most downstream;
3. the report gives pass^k, undetected critical errors, and an interval for the email
   acceptance rate that includes the judge's own error.
"""

from pathlib import Path

from eval_harness import Runner, Suite, metrics
from hana_evals import ReferenceOverlapJudge, calibrated_check
from judge_cal import Judge, report

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "packages/eval-harness/suites/back-office"
RUBRIC = ROOT / "packages/judge-cal/rubrics/faithfulness.yaml"


def main() -> None:
    judge = Judge(backend=ReferenceOverlapJudge(), rubric=RUBRIC)
    check, cal = calibrated_check(judge, ROOT / "examples/data/email-labels.jsonl")
    a = cal.agreement
    print(f"judge calibration: ECE {cal.ece_before:.3f} -> {cal.ece_after:.3f}, "
          f"agreement with humans {a.accuracy:.2f} (n={a.n})")

    suite = Suite.load(SUITE)
    for placement in ("last", "uniform", "dependency"):
        runs = [run for seed in range(1, 6)
                for run in Runner(agent={"model": "simulated", "error_rate": 0.12, "seed": seed}, repeats=8,
                                  review_budget=0.3, placement=placement, judge_check=check).run(suite).runs]
        print(f"{placement:>10}: pass^1 {metrics.pass_k(runs, 1):.2f}  pass^8 {metrics.pass_k(runs, 8):.2f}  "
              f"undetected critical per 100 runs {metrics.undetected_critical_per_100(runs):.1f}")

    emails = [{"question": "draft", "answer": r["output"], "reference": r["reference"]}
              for r in _final_emails(suite)]
    scores = cal.score(judge, emails)
    print(report.summary(scores, n_boot=500))


def _final_emails(suite):
    from eval_harness.agents import SimulatedAgent

    agent = SimulatedAgent(error_rate=0.3, seed=2)
    for job in suite.jobs:
        ref = job.reference()
        stage = job.stage("draft_email")
        for attempt in range(8):
            yield {"output": agent.act(job, stage, ref, attempt), "reference": ref["draft_email"]}


if __name__ == "__main__":
    main()
