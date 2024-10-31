"""Run every job of a suite several times, checking the stages the placement picks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_harness import metrics
from eval_harness.agents import Agent, load_agent
from eval_harness.checks import Check, CodeCheck, equal
from eval_harness.faults import Fault, recovery_steps
from eval_harness.placement import place
from eval_harness.stages import Job, Suite
from eval_harness.store import Store, trace_id


@dataclass
class Result:
    stages: list[dict]
    runs: list[dict]
    settings: dict
    path: Path | None = None

    def pass_k(self, k: int) -> float:
        return metrics.pass_k(self.runs, k)

    @property
    def undetected_critical_per_100(self) -> float:
        return metrics.undetected_critical_per_100(self.runs)

    @property
    def checks_per_completed_job(self) -> float:
        return metrics.reviewer_load(self.runs)

    @property
    def median_recovery(self) -> float | None:
        return metrics.median_recovery(self.runs)


@dataclass
class Runner:
    agent: str | Path | dict | Agent = field(default_factory=lambda: {"model": "simulated"})
    repeats: int = 8
    review_budget: float = 0.2
    placement: str = "dependency"
    max_retries: int = 2
    judge_check: Check | None = None
    faults: list[Fault] = field(default_factory=list)
    out: str | Path | None = None

    def __post_init__(self):
        self._agent, self._agent_settings = load_agent(self.agent)
        self._code = CodeCheck()

    def settings(self, suite: Suite) -> dict:
        return {"suite": suite.name, "agent": self._agent_settings, "repeats": self.repeats,
                "review_budget": self.review_budget, "placement": self.placement, "max_retries": self.max_retries,
                "judge_check": type(self.judge_check).__name__ if self.judge_check else None,
                "faults": [f.__dict__ for f in self.faults]}

    def run(self, suite: Suite) -> Result:
        stage_rows, run_rows = [], []
        for job in suite.jobs:
            checked = place(job, self.review_budget, self.placement)
            reference = job.reference()
            for repeat in range(self.repeats):
                rows, run = self._run_once(job, repeat, checked, reference)
                stage_rows += rows
                run_rows.append(run)
        result = Result(stage_rows, run_rows, self.settings(suite))
        if self.out is not None:
            result.path = Store(self.out).save(result.settings, stage_rows, run_rows)
        return result

    def _check(self, job: Job, stage, output: Any, local_expected: Any):
        check = self.judge_check if stage.check == "judge" and self.judge_check else self._code
        return check(job, stage, output, local_expected)

    def _run_once(self, job: Job, repeat: int, checked: set[str], reference: dict) -> tuple[list[dict], dict]:
        tid = trace_id()
        upstream: dict[str, Any] = {}
        trace: list[dict] = []
        flagged: set[str] = set()
        steps = checks = 0
        for stage in job.order():
            attempt = 0
            while True:
                out = self._agent.act(job, stage, upstream, attempt + 100 * repeat)
                steps += 1
                fault = next((f for f in self.faults if f.stage == stage.id), None)
                if fault and attempt == 0:
                    out = fault.apply(out)
                local = job.run_tool(stage, upstream)  # what this stage should give on its actual inputs
                verdict = None
                if stage.id in checked:
                    verdict = self._check(job, stage, out, local)
                    checks += 1
                trace.append({"job": job.id, "repeat": repeat, "trace_id": tid, "stage": stage.id,
                              "attempt": attempt, "correct": equal(out, reference[stage.id]),
                              "checked": verdict is not None, "check_passed": verdict.passed if verdict else None,
                              "check_by": verdict.by if verdict else None})
                if verdict is None or verdict.passed or attempt >= self.max_retries:
                    if verdict is not None and not verdict.passed:
                        flagged.add(stage.id)
                    break
                attempt += 1
            upstream[stage.id] = out
        final = {s.id: equal(upstream[s.id], reference[s.id]) for s in job.stages}
        critical_wrong = {s.id for s in job.stages if s.critical and not final[s.id]}
        success = all(final.values())
        run = {"job": job.id, "repeat": repeat, "trace_id": tid, "success": success, "steps": steps,
               "checks": checks, "flagged": sorted(flagged),
               "undetected_critical": bool(critical_wrong) and not flagged,
               "recovery_steps": recovery_steps(trace, self.faults[0].stage) if self.faults else None}
        return trace, run
