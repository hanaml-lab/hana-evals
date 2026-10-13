"""Verifier perturbation testing on staged jobs (the method of the VPT paper, carried into the harness)."""

from pathlib import Path

import pytest

from eval_harness import Suite
from eval_harness.agents import ReplayAgent, SimulatedAgent
from eval_harness.perturbation import PerturbationTest, conformal_threshold

SUITE = Path(__file__).parent.parent / "suites" / "back-office"


@pytest.fixture(scope="module")
def suite():
    return Suite.load(SUITE)


def test_an_agent_that_solves_the_job_has_no_invariance_gap(suite):
    perfect = SimulatedAgent(error_rate=0.0)
    vpt = PerturbationTest(k=4)
    for job in suite.jobs:
        passed, gaps = vpt.gaps(perfect, job)
        assert passed == 1.0
        assert gaps == {"surface": 0.0, "rescale": 0.0}


def test_an_agent_that_replays_leaked_answers_collapses_under_perturbation(suite):
    leaked = ReplayAgent({job.inputs["supplier"]: job.reference() for job in suite.jobs})
    vpt = PerturbationTest(k=4)
    for job in suite.jobs:
        passed, gaps = vpt.gaps(leaked, job)
        assert passed == 1.0  # the replayed answers pass every check on the job it saw
        # renamed references break at least the invoices, the ledger and the match: 3 of 9 stages
        assert gaps["surface"] >= 3 / 9
        # rescaled amounts also break both totals
        assert gaps["rescale"] >= 5 / 9


def test_conformal_threshold_is_the_split_conformal_order_statistic():
    scores = [float(s) for s in range(19, 0, -1)]  # 19 honest scores, any order
    assert conformal_threshold(scores, alpha=0.1) == 18.0  # ceil(20 * 0.9) = 18th smallest
    assert conformal_threshold(scores, alpha=0.01) == float("inf")  # ceil(20 * 0.99) = 20 > 19: never flag


def test_calibrated_test_flags_the_replaying_agent_and_few_honest_runs(suite):
    vpt = PerturbationTest(k=8, alpha=0.1)
    honest = [(SimulatedAgent(error_rate=0.12, seed=seed), job) for seed in range(60) for job in suite.jobs]
    vpt.calibrate(honest[::2], honest[1::2])

    leaked = ReplayAgent({job.inputs["supplier"]: job.reference() for job in suite.jobs})
    assert all(vpt.score(leaked, job).flagged for job in suite.jobs)

    fresh = [vpt.score(SimulatedAgent(error_rate=0.12, seed=seed), job)
             for seed in range(100, 140) for job in suite.jobs]
    passing = [s for s in fresh if s.passed]
    assert len(passing) >= 40
    assert sum(s.flagged for s in passing) / len(passing) <= 0.1 + 0.1  # alpha, with room for 40-odd draws
    assert all(not s.flagged for s in fresh if not s.passed)  # only runs that pass their checks are tested
