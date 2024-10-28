"""Checks placed on stages. A code check compares with the reference; a judge check asks a
calibrated judge (hana-evals' judge-cal) whether a person would accept the output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from eval_harness.stages import Job, Stage


@dataclass(frozen=True)
class Verdict:
    passed: bool
    confidence: float
    by: str


class Check(Protocol):
    def __call__(self, job: Job, stage: Stage, output: Any, expected: Any) -> Verdict: ...


def equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return abs(a - b) <= tol * max(1.0, abs(b))
    return a == b


class CodeCheck:
    def __call__(self, job: Job, stage: Stage, output: Any, expected: Any) -> Verdict:
        ok = equal(output, expected)
        return Verdict(ok, 1.0, "code")


class JudgeCheck:
    """Uses a ``judge_cal.Judge`` and a fitted ``judge_cal.calibrate.Calibrator``.

    The judge scores the output against the stage instructions with the reference as the
    reference answer. The calibrated probability that a human would accept it must reach
    ``threshold``.
    """

    def __init__(self, judge, calibrator, threshold: float = 0.5):
        self.judge, self.calibrator, self.threshold = judge, calibrator, threshold

    def __call__(self, job: Job, stage: Stage, output: Any, expected: Any) -> Verdict:
        as_text = output if isinstance(output, str) else json.dumps(output)
        ref = expected if isinstance(expected, str) else json.dumps(expected)
        raw = self.judge.score(stage.instructions or stage.id, as_text, ref)
        p = float(self.calibrator.predict([raw])[0])
        return Verdict(p >= self.threshold, p, "judge")
