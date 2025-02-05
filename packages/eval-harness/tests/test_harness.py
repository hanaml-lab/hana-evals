import json
from pathlib import Path

import pytest

from eval_harness import Runner, Suite
from eval_harness.cli import main
from eval_harness.faults import Fault
from eval_harness.metrics import pass_hat_k
from eval_harness.placement import place
from eval_harness.stages import Job, Stage

SUITE = Path(__file__).parent.parent / "suites" / "back-office"


@pytest.fixture(scope="module")
def suite():
    return Suite.load(SUITE)


def test_suite_loads_and_reference_is_consistent(suite):
    assert len(suite.jobs) == 6
    ref = suite.jobs[1].reference()
    assert ref["flag_mismatches"]  # job 2 has a missing ledger entry
    assert ref["approve_payment"] == "hold"


def test_cycle_rejected():
    with pytest.raises(ValueError, match="cycle"):
        Job("j", (Stage("a", "load", deps=("b",)), Stage("b", "load", deps=("a",))))


def test_dependency_placement_prefers_wide_stages(suite):
    job = suite.jobs[0]
    picked = place(job, 0.2, "dependency")
    assert len(picked) == 2
    # the payment decision is critical and costly to undo; a wrong invoice read reaches six stages
    assert picked == {"approve_payment", "invoices"}
    assert place(job, 0.2, "last") == {"draft_email", "approve_payment"}


def test_pass_hat_k():
    assert pass_hat_k(8, 8, 8) == 1.0
    assert pass_hat_k(7, 8, 8) == 0.0
    assert pass_hat_k(4, 8, 1) == 0.5


def test_perfect_agent_passes_everything(suite):
    r = Runner(agent={"model": "simulated", "error_rate": 0.0}, repeats=3).run(suite)
    assert r.pass_k(3) == 1.0 and r.undetected_critical_per_100 == 0.0


def test_dependency_placement_beats_last(suite):
    cfg = {"model": "simulated", "error_rate": 0.15, "seed": 3}
    dep = Runner(agent=cfg, repeats=8, placement="dependency", review_budget=0.3).run(suite)
    last = Runner(agent=cfg, repeats=8, placement="last", review_budget=0.3).run(suite)
    assert dep.pass_k(1) > last.pass_k(1)


def test_fault_recovery_with_and_without_a_check(suite):
    cfg = {"model": "simulated", "error_rate": 0.0}
    checked = Runner(agent=cfg, repeats=2, placement="dependency", review_budget=0.2,
                     faults=[Fault("invoices")]).run(suite)
    assert checked.median_recovery == 1
    unchecked = Runner(agent=cfg, repeats=2, placement="last", review_budget=0.1,
                       faults=[Fault("invoices")]).run(suite)
    assert unchecked.median_recovery is None
    assert unchecked.pass_k(1) < checked.pass_k(1)


def test_cli_run_and_report(tmp_path, capsys):
    agent = tmp_path / "agent.yaml"
    agent.write_text("model: simulated\nerror_rate: 0.1\nreview_budget: 0.2\nplacement: dependency\n")
    assert main(["run", str(SUITE), "--agent", str(agent), "--repeats", "2", "--out", str(tmp_path / "runs")]) == 0
    run_dir = capsys.readouterr().out.strip()
    settings = json.loads((Path(run_dir) / "settings.json").read_text())
    assert settings["repeats"] == 2 and settings["agent"]["error_rate"] == 0.1
    assert main(["report", run_dir, "--metric", "pass^2", "--metric", "undetected-critical"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "pass^2" in out and "undetected_critical_per_100" in out
