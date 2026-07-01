"""Wire a judge, calibrated with judge-cal, into eval-harness as a stage check."""

from __future__ import annotations

import re
from pathlib import Path

from eval_harness.checks import JudgeCheck
from judge_cal import Judge, calibrate
from judge_cal.calibrate import Calibrator


class ReferenceOverlapJudge:
    """An offline judge backend: scores an answer 1-5 by how much of the reference's specific
    content (identifiers, figures, capitalised names) it contains. Used for tests and the
    example; in production, point ``Judge`` at ``hanaml/judge-7b`` instead."""

    _KEY = re.compile(r"[A-Z][A-Z0-9-]{2,}\d|\d[\d,.]*|[A-Z][a-z]+(?: [A-Z][a-z]+)+")

    def __call__(self, prompt: str) -> str:
        answer = prompt.split("Answer:\n", 1)[1].rsplit("\n\nEnd with", 1)[0]
        reference = prompt.split("Reference answer:\n", 1)[1].split("\n\nAnswer:", 1)[0] \
            if "Reference answer:" in prompt else ""
        keys = set(self._KEY.findall(reference))
        if not keys:
            return "[[3]]"
        found = sum(1 for k in keys if k in answer) / len(keys)
        return f"[[{1 + round(4 * found)}]]"


def calibrated_check(judge: Judge, human_labels: str | Path | list[dict], method: str = "isotonic",
                     threshold: float = 0.5) -> tuple[JudgeCheck, Calibrator]:
    """Fit the judge on held-out human labels and return the eval-harness check that uses it."""
    cal = calibrate.fit(judge, human_labels, method=method)
    return JudgeCheck(judge, cal, threshold), cal
