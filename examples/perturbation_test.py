"""Catch an agent that passes every check for the wrong reason. Offline, about ten seconds.

    python examples/perturbation_test.py

1. A replaying agent has seen the answers of the six reconciliation jobs and recognises each job
   by its supplier's name. eval-harness, checking every stage, scores it perfectly.
2. The perturbation test (VPT, https://hanaml.com/research/papers/detecting-reward-hacking) re-runs
   an agent on renamed and rescaled instances of the same job. It is calibrated on honest runs of a
   seeded agent that makes mistakes, then scores the replaying agent and fresh honest runs.
"""

from pathlib import Path

from eval_harness import Runner, Suite
from eval_harness.agents import ReplayAgent, SimulatedAgent
from eval_harness.perturbation import PerturbationTest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "packages/eval-harness/suites/back-office"


def main() -> None:
    suite = Suite.load(SUITE)
    leaked = ReplayAgent({job.inputs["supplier"]: job.reference() for job in suite.jobs})
    checked = Runner(agent=leaked, repeats=8, review_budget=1.0).run(suite)
    print(f"replaying agent, every stage checked: pass^8 {checked.pass_k(8):.2f}  "
          f"undetected critical per 100 runs {checked.undetected_critical_per_100:.1f}")

    vpt = PerturbationTest(k=8, alpha=0.1)
    honest = [(SimulatedAgent(error_rate=0.12, seed=seed), job) for seed in range(60) for job in suite.jobs]
    vpt.calibrate(honest[::2], honest[1::2])
    print(f"threshold {vpt.threshold:.2f} (split-conformal, alpha {vpt.alpha}, "
          f"{len(honest)} honest runs, scored when they pass their checks)")

    for job in suite.jobs:
        s = vpt.score(leaked, job)
        gaps = "  ".join(f"{name} {gap:.2f}" for name, gap in s.gaps.items())
        print(f"  {job.id}: gaps {gaps}  score {s.score:.1f}  flagged {s.flagged} ({s.family})")

    fresh = [vpt.score(SimulatedAgent(error_rate=0.12, seed=seed), job)
             for seed in range(100, 140) for job in suite.jobs]
    passing = [s for s in fresh if s.passed]
    flagged = sum(s.flagged for s in passing)
    print(f"fresh honest runs that pass their checks: {len(passing)}, flagged {flagged} "
          f"({flagged / len(passing):.1%}; the expected rate is at most {vpt.alpha:.0%})")


if __name__ == "__main__":
    main()
